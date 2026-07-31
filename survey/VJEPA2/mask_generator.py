"""
V-JEPA 2.1 3D スパシオテンポラルマスク生成 - 簡略化疑似コード
==============================================================

動画の時空間パッチを3Dブロック状にマスクする機構。
V-JEPAのマスキング戦略のポイント:
  - 空間的に連続した矩形ブロックをマスク
  - 時間方向に全フレームを貫通するチューブ状マスク (temporal_scale=1.0)
  - 複数ブロック (npred個) を生成してAND合成
  - エンコーダ用マスク (可視): コンテキストトークンのインデックス
  - Predictor用マスク (予測): ターゲットトークンのインデックス

V-JEPA 2.1の標準マスク設定:
  - 8個の小ブロック (spatial_scale=0.15, temporal_scale=1.0)
  - 2個の大ブロック (spatial_scale=0.70, temporal_scale=1.0)

対応する公式実装:
  - src/masks/multiseq_multiblock3d.py
  - src/masks/utils.py
"""

import math
import random
from multiprocessing import Value

import torch


# ============================================================
# 3D ブロックマスクジェネレータ
# ============================================================

class MaskGenerator:
    """
    3D スパシオテンポラルブロックマスクを生成するクラス。

    1バッチ分の (masks_enc, masks_pred) インデックスを生成する。

    masks_enc  (エンコーダ用): コンテキスト(可視)パッチのインデックス
    masks_pred (Predictor用): ターゲット(マスク)パッチのインデックス

    注意: マスクされたパッチ = 0, 可視パッチ = 1 でまず3Dグリッドを作り、
          その後それぞれのインデックスに変換する。
    """

    def __init__(
        self,
        crop_size: int = 224,                    # 入力解像度 (px)
        num_frames: int = 16,                    # クリップのフレーム数
        spatial_patch_size: int = 16,            # 空間パッチサイズ (px)
        temporal_patch_size: int = 2,            # チューブレットサイズ (フレーム)
        spatial_pred_mask_scale: tuple = (0.15, 0.15),  # 空間マスク割合 [min, max]
        temporal_pred_mask_scale: tuple = (1.0, 1.0),   # 時間マスク割合 [min, max]
        aspect_ratio: tuple = (0.75, 1.5),       # マスクブロックのアスペクト比 [min, max]
        npred: int = 8,                          # マスクブロック数
        max_context_frames_ratio: float = 1.0,  # コンテキストが使える最大フレーム割合
        max_keep: int = None,                    # コンテキストパッチの最大保持数
    ):
        if not isinstance(crop_size, tuple):
            crop_size = (crop_size, crop_size)

        # パッチグリッドサイズ (パッチ単位)
        self.height = crop_size[0] // spatial_patch_size   # 空間高さ (パッチ数)
        self.width  = crop_size[1] // spatial_patch_size   # 空間幅   (パッチ数)
        self.duration = num_frames // temporal_patch_size  # 時間方向 (チューブレット数)

        self.spatial_pred_mask_scale = spatial_pred_mask_scale
        self.temporal_pred_mask_scale = temporal_pred_mask_scale
        self.aspect_ratio = aspect_ratio
        self.npred = npred
        self.max_keep = max_keep
        self.max_context_duration = max(1, int(self.duration * max_context_frames_ratio))

        # マルチプロセス対応のカウンタ (シードの一貫性確保)
        self._itr_counter = Value("i", -1)

    def step(self) -> int:
        """イテレーションカウンタをインクリメント (シード管理用)"""
        i = self._itr_counter
        with i.get_lock():
            i.value += 1
            return i.value

    def _sample_block_size(self, seed: int) -> tuple:
        """
        マスクブロックの時空間サイズをサンプリングする。

        処理:
          1. 時間スケールをランダムサンプリング → t (チューブレット数)
          2. 空間スケールをランダムサンプリング → 面積
          3. アスペクト比をランダムサンプリング → h, w

        出力:
            (t, h, w): チューブレット数, 空間高さ(パッチ), 空間幅(パッチ)
        """
        g = torch.Generator()
        g.manual_seed(seed)

        # ---- 時間方向ブロックサイズ
        min_t, max_t = self.temporal_pred_mask_scale
        rand_t = torch.rand(1, generator=g).item()
        temporal_scale = min_t + rand_t * (max_t - min_t)
        t = max(1, int(self.duration * temporal_scale))

        # ---- 空間ブロック面積
        min_s, max_s = self.spatial_pred_mask_scale
        rand_s = torch.rand(1, generator=g).item()
        spatial_scale = min_s + rand_s * (max_s - min_s)
        target_area = int(self.height * self.width * spatial_scale)

        # ---- アスペクト比
        min_ar, max_ar = self.aspect_ratio
        rand_ar = torch.rand(1, generator=g).item()
        ar = min_ar + rand_ar * (max_ar - min_ar)

        # ---- h, w を面積とアスペクト比から算出
        # h * w = target_area, h/w = ar
        # → h = sqrt(target_area * ar), w = sqrt(target_area / ar)
        h = int(round(math.sqrt(target_area * ar)))
        w = int(round(math.sqrt(target_area / ar)))
        h = min(h, self.height)
        w = min(w, self.width)

        return (t, h, w)

    def _sample_block_mask(self, block_size: tuple) -> torch.Tensor:
        """
        ランダム位置に3Dブロックマスクを生成する。

        入力:
            block_size: (t, h, w)

        出力:
            mask: (duration, height, width) int32
                  0 = マスク (予測対象), 1 = 可視 (コンテキスト)

        処理:
          1. start, top, left をランダムサンプリング
          2. 対応する3D領域を0に設定
          3. max_context_duration 以降のフレームも0に設定
             (コンテキストは最初のmax_context_duration フレームのみ使用可)
        """
        t, h, w = block_size

        # ランダム位置
        top   = torch.randint(0, self.height - h + 1, (1,)).item()
        left  = torch.randint(0, self.width  - w + 1, (1,)).item()
        start = torch.randint(0, self.duration - t + 1, (1,)).item()

        # 全体を1 (可視) で初期化
        mask = torch.ones((self.duration, self.height, self.width), dtype=torch.int32)

        # ブロック部分を0 (マスク) に設定
        mask[start:start + t, top:top + h, left:left + w] = 0

        # コンテキストは最初のmax_context_durationフレームのみ
        if self.max_context_duration < self.duration:
            mask[self.max_context_duration:, :, :] = 0

        return mask

    def __call__(self, batch_size: int) -> tuple:
        """
        バッチ全体のマスクを生成する。

        処理:
          1. シードからブロックサイズをサンプリング (バッチ内で共通)
          2. 各サンプルについてブロック位置をサンプリング
          3. npred個のブロックをAND合成 (0のピクセルがマスク)
          4. コンテキストとターゲットのインデックスに変換
          5. バッチ内で最小のパッチ数に揃える (同一バッチサイズのため)

        入力:
            batch_size: バッチサイズ B

        出力:
            masks_enc:  (B, N_ctx)   コンテキストパッチインデックス
            masks_pred: (B, N_pred)  ターゲットパッチインデックス

            N_ctx, N_pred はバッチ内最小値に揃えられる
        """
        seed = self.step()
        block_size = self._sample_block_size(seed)

        collated_masks_pred = []
        collated_masks_enc  = []

        min_keep_enc  = self.duration * self.height * self.width
        min_keep_pred = self.duration * self.height * self.width

        for _ in range(batch_size):
            # コンテキストが空にならないまでリトライ
            empty_context = True
            while empty_context:
                # 全体を1 (可視) で初期化
                mask_enc = torch.ones(
                    (self.duration, self.height, self.width), dtype=torch.int32
                )

                # npred個のブロックマスクをAND合成
                # → 複数のブロックのうちいずれかに入ればマスク
                for _ in range(self.npred):
                    mask_enc = mask_enc * self._sample_block_mask(block_size)

                # 3Dマスクを1Dインデックスに変換
                # mask_enc.flatten(): (N_total,)  各要素が0(マスク) or 1(可視)
                mask_flat = mask_enc.flatten()
                mask_pred = torch.argwhere(mask_flat == 0).squeeze()  # マスクインデックス
                mask_enc  = torch.nonzero(mask_flat).squeeze()        # 可視インデックス

                empty_context = len(mask_enc) == 0
                if not empty_context:
                    min_keep_pred = min(min_keep_pred, len(mask_pred))
                    min_keep_enc  = min(min_keep_enc, len(mask_enc))
                    collated_masks_pred.append(mask_pred)
                    collated_masks_enc.append(mask_enc)

        # max_keepがある場合は可視パッチ数を制限
        if self.max_keep is not None:
            min_keep_enc = min(min_keep_enc, self.max_keep)

        # バッチ内で最小のパッチ数に揃える (スタック可能にするため)
        collated_masks_enc  = [cm[:min_keep_enc]  for cm in collated_masks_enc]
        collated_masks_pred = [cm[:min_keep_pred] for cm in collated_masks_pred]

        # (B, N_ctx), (B, N_pred) にスタック
        masks_enc  = torch.stack(collated_masks_enc,  dim=0)  # (B, N_ctx)
        masks_pred = torch.stack(collated_masks_pred, dim=0)  # (B, N_pred)

        return masks_enc, masks_pred


# ============================================================
# MaskCollator: 複数FPS/マスク設定を管理するコレーター
# ============================================================

class MaskCollator:
    """
    DataLoaderのcollate_fnとして使用するマスクコレーター。

    複数FPS(frames per clip)のデータセットと
    複数マスク設定に対応。

    各マスク設定に対してMaskGeneratorを生成し、
    バッチ組み立て時にマスクを生成してデータに追加する。

    __call__の出力:
        list of (collated_batch, masks_enc_list, masks_pred_list)
        - 各要素がFPCが同じサンプルのバッチ
        - masks_enc_list:  list of (B, N_ctx)   各マスク設定のエンコーダマスク
        - masks_pred_list: list of (B, N_pred)  各マスク設定のPredictorマスク
    """

    def __init__(
        self,
        cfgs_mask: list,          # マスク設定のリスト (各要素が1つのマスク戦略)
        dataset_fpcs: list,       # データセットのFPCリスト (例: [16, 16])
        crop_size: int = 224,
        patch_size: int = 16,
        tubelet_size: int = 2,
    ):
        # FPC (frames per clip) ごとにMaskGeneratorを生成
        self.mask_generators = {}
        for fpc in set(dataset_fpcs):
            self.mask_generators[fpc] = []
            for m in cfgs_mask:
                generator = MaskGenerator(
                    crop_size=crop_size,
                    num_frames=fpc,
                    spatial_patch_size=patch_size,
                    temporal_patch_size=tubelet_size,
                    spatial_pred_mask_scale=m.get("spatial_scale", (0.15, 0.15)),
                    temporal_pred_mask_scale=m.get("temporal_scale", (1.0, 1.0)),
                    aspect_ratio=m.get("aspect_ratio", (0.75, 1.5)),
                    npred=m.get("num_blocks", 8),
                    max_context_frames_ratio=m.get("max_temporal_keep", 1.0),
                    max_keep=m.get("max_keep", None),
                )
                self.mask_generators[fpc].append(generator)

    def step(self):
        """全ジェネレータのカウンタを進める"""
        for fpc in self.mask_generators:
            for gen in self.mask_generators[fpc]:
                gen.step()

    def __call__(self, batch: list) -> list:
        """
        バッチをFPCごとにグループ化し、各グループにマスクを付加する。

        入力:
            batch: list of (video_tensor, label, clip_indices)
                   video_tensor: (3, T, H, W) または (3, H, W)

        出力:
            list of (collated_batch, masks_enc_list, masks_pred_list)
            - collated_batch: 通常のcollatされたバッチ
            - masks_enc_list:  list of (B, N_ctx)  マスク設定数分
            - masks_pred_list: list of (B, N_pred) マスク設定数分
        """
        # FPCでサンプルをグループ化
        filtered_batches = {fpc: [] for fpc in self.mask_generators}
        for sample in batch:
            # サンプルのFPCを判定
            if len(sample) >= 3 and isinstance(sample[-1], (list, tuple)):
                try:
                    fpc = len(sample[-1][-1])  # clip_indices から取得
                except (TypeError, IndexError):
                    fpc = 1
            else:
                fpc = 1  # 画像

            if fpc in filtered_batches:
                filtered_batches[fpc].append(sample)

        # FPCごとにcollateしマスクを生成
        fpc_collations = []
        for fpc, fpc_batch in filtered_batches.items():
            if len(fpc_batch) == 0:
                continue

            collated_batch = torch.utils.data.default_collate(fpc_batch)
            batch_size = len(fpc_batch)

            masks_enc_list  = []
            masks_pred_list = []
            for gen in self.mask_generators[fpc]:
                masks_enc, masks_pred = gen(batch_size)
                masks_enc_list.append(masks_enc)    # (B, N_ctx)
                masks_pred_list.append(masks_pred)  # (B, N_pred)

            fpc_collations.append((collated_batch, masks_enc_list, masks_pred_list))

        return fpc_collations


# ============================================================
# 距離重み計算 (L_context用)
# ============================================================

def compute_mask_distance(
    masks_pred: list,
    masks_enc: list,
    grid_size: int,
    offset: bool = False,
) -> list:
    """
    各コンテキストパッチと最近傍マスクパッチの距離を計算する。

    V-JEPA 2.1のコンテキスト損失の距離重み λ_i = λ / sqrt(d_min(i, M)) で使用。

    入力:
        masks_pred: list of (B, N_pred)  ターゲットパッチインデックス
        masks_enc:  list of (B, N_ctx)   コンテキストパッチインデックス
        grid_size:  H/patch_size = W/patch_size (空間グリッドサイズ)

    出力:
        distance_weights: list of list of (B, N_ctx)
                          各マスク設定・バッチ要素について
                          コンテキストパッチの最小距離テンソル

    距離の定義:
        2D空間グリッド上のマンハッタン距離 or チェビシェフ距離
        (公式実装はブロック単位の距離を計算)

    注意:
        この簡略化実装では空間距離のみ計算する (時間方向を無視)。
        公式実装 app/vjepa_2_1/models/utils/masks_dist.py では
        完全な時空間距離を計算している。
    """
    all_d_weights = []

    for mask_p_batch, mask_e_batch in zip(masks_pred, masks_enc):
        # mask_p_batch: (B, N_pred)
        # mask_e_batch: (B, N_ctx)
        B = mask_p_batch.shape[0]
        d_batch = []

        for b in range(B):
            pred_idx = mask_p_batch[b]  # (N_pred,) フラットインデックス
            enc_idx  = mask_e_batch[b]  # (N_ctx,)  フラットインデックス

            # フラットインデックスを (row, col) に変換
            # 注意: 時空間の場合は (t, row, col) だが、ここでは空間のみ簡略化
            pred_row = pred_idx // grid_size  # (N_pred,)
            pred_col = pred_idx %  grid_size  # (N_pred,)
            enc_row  = enc_idx  // grid_size  # (N_ctx,)
            enc_col  = enc_idx  %  grid_size  # (N_ctx,)

            # 各コンテキストパッチと全ターゲットパッチの距離を計算
            # enc: (N_ctx, 1, 2), pred: (1, N_pred, 2)
            enc_pos  = torch.stack([enc_row,  enc_col],  dim=1).float()   # (N_ctx, 2)
            pred_pos = torch.stack([pred_row, pred_col], dim=1).float()   # (N_pred, 2)

            # ユークリッド距離: (N_ctx, N_pred)
            diff = enc_pos.unsqueeze(1) - pred_pos.unsqueeze(0)   # (N_ctx, N_pred, 2)
            dist = torch.norm(diff, dim=-1)                        # (N_ctx, N_pred)

            # 最小距離 (最近傍のマスクパッチとの距離)
            d_min = dist.min(dim=1).values  # (N_ctx,)
            d_min = d_min.clamp(min=1.0)    # 距離0を防ぐ (自分自身との距離)

            d_batch.append(d_min)

        all_d_weights.append(d_batch)

    return all_d_weights


# ============================================================
# 動作確認 example
# ============================================================

if __name__ == "__main__":
    import torch

    print("=" * 60)
    print("V-JEPA 2.1 マスクジェネレータ 動作確認")
    print("=" * 60)

    # ----------------------------------------
    # 標準的なViT-L設定 (256px, 16フレーム)
    # ----------------------------------------
    print("\n[1] 標準マスク生成 (V-JEPA 2.1 設定)")
    B = 4
    # 設定: 256px, 16フレーム, patch=16, tubelet=2
    # グリッド: 16x16空間 × 8時間 = 2048パッチ
    gen_small = MaskGenerator(
        crop_size=256,
        num_frames=16,
        spatial_patch_size=16,
        temporal_patch_size=2,
        spatial_pred_mask_scale=(0.15, 0.15),  # 空間の15%マスク
        temporal_pred_mask_scale=(1.0, 1.0),   # 全時間フレームマスク
        aspect_ratio=(0.75, 1.5),
        npred=8,                                # 8ブロック
    )

    masks_enc, masks_pred = gen_small(B)
    print(f"  グリッドサイズ: {gen_small.duration}t × {gen_small.height}h × {gen_small.width}w = {gen_small.duration * gen_small.height * gen_small.width} patches")
    print(f"  masks_enc (コンテキスト): {masks_enc.shape}  (インデックス値 range: {masks_enc.min()}~{masks_enc.max()})")
    print(f"  masks_pred (ターゲット):  {masks_pred.shape}")
    print(f"  コンテキスト割合: {masks_enc.shape[1] / (gen_small.duration*gen_small.height*gen_small.width):.2%}")
    print(f"  ターゲット割合:   {masks_pred.shape[1] / (gen_small.duration*gen_small.height*gen_small.width):.2%}")
    # コンテキスト + ターゲット ≤ 総パッチ数 (重複がある場合は < になる)
    assert masks_enc.shape[0] == B
    assert masks_pred.shape[0] == B

    # ----------------------------------------
    # 大ブロックマスク
    # ----------------------------------------
    print("\n[2] 大ブロックマスク (空間70%)")
    gen_large = MaskGenerator(
        crop_size=256,
        num_frames=16,
        spatial_patch_size=16,
        temporal_patch_size=2,
        spatial_pred_mask_scale=(0.70, 0.70),  # 空間の70%マスク
        temporal_pred_mask_scale=(1.0, 1.0),
        aspect_ratio=(0.75, 1.5),
        npred=2,  # 2ブロック
    )
    masks_enc_l, masks_pred_l = gen_large(B)
    print(f"  masks_enc (コンテキスト): {masks_enc_l.shape}")
    print(f"  masks_pred (ターゲット):  {masks_pred_l.shape}")
    print(f"  コンテキスト割合: {masks_enc_l.shape[1] / (gen_large.duration*gen_large.height*gen_large.width):.2%}")

    # ----------------------------------------
    # MaskCollator: 複数マスク設定
    # ----------------------------------------
    print("\n[3] MaskCollator (V-JEPA 2.1 標準設定)")
    cfgs_mask = [
        {"num_blocks": 8,  "spatial_scale": (0.15, 0.15), "temporal_scale": (1.0, 1.0), "aspect_ratio": (0.75, 1.5)},
        {"num_blocks": 2,  "spatial_scale": (0.70, 0.70), "temporal_scale": (1.0, 1.0), "aspect_ratio": (0.75, 1.5)},
    ]
    collator = MaskCollator(
        cfgs_mask=cfgs_mask,
        dataset_fpcs=[16],
        crop_size=256,
        patch_size=16,
        tubelet_size=2,
    )
    print(f"  マスク設定数: {len(cfgs_mask)}")
    print(f"  FPCごとのジェネレータ数: {len(collator.mask_generators[16])}")

    # ----------------------------------------
    # 距離重み計算
    # ----------------------------------------
    print("\n[4] 距離重み計算 (L_context用 λ_i)")
    grid_size = 256 // 16  # = 16
    d_weights = compute_mask_distance([masks_pred], [masks_enc], grid_size)
    # d_weights: list(len=1 mask) of list(len=B) of (N_ctx,)
    d_sample = d_weights[0][0]  # 最初のマスク設定、最初のバッチ要素
    print(f"  距離重み形状: {d_sample.shape}")
    print(f"  距離範囲: min={d_sample.min():.2f}, max={d_sample.max():.2f}, mean={d_sample.mean():.2f}")
    # マスク近傍は小さい距離 → λ_i = λ/sqrt(d_min) で大きい重みになる
    # マスクから遠いパッチは大きい距離 → 小さい重みになる
    lambda_val = 0.5
    weights = lambda_val / d_sample.sqrt()  # λ / sqrt(d_min)
    print(f"  λ_i = λ/sqrt(d_min): min={weights.min():.4f}, max={weights.max():.4f}")

    print("\n全テスト通過!")
