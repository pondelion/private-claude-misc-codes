"""
MiniCPM-V 4.5 - 視覚エンコーダ & 画像パーティショニング
================================================

EVA02-Enormous / SigLip 視覚エンコーダと
LLaVA-UHD 方式の画像パーティショニング戦略の実装。

論文: MiniCPM-V 4.5: Cooking Efficient MLLMs via Architecture, Data, and Training Recipes
公式実装:
    - omnilmm/model/omnilmm.py: create_vision_module()
    - omnilmm/model/utils.py: build_transform(), RandomAugment
    - finetune/dataset.py: slice_image(), find_best_resize()

処理の流れ:
1. 入力画像をLLaVA-UHD方式でスライスに分割
2. 各スライスを正規化・リサイズ
3. EVA02/SigLipで特徴抽出
4. CLS/prefixトークンを除去して特徴列を返す
"""

"""
============================================================
Shape Convention (形状表記規則)
============================================================
N       : スライス数 (1 + grid_x * grid_y)
H, W    : 画像の高さ、幅 (ピクセル)
H_s, W_s: スライスの高さ、幅 (patch_sizeの倍数)
P       : パッチサイズ (14)
L_vis   : パッチトークン数 = (H_s/P) * (W_s/P)
D_vis   : 視覚エンコーダの出力次元 (1792 for EVA02-Enormous)
"""

import math
import os
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from timm.data.transforms import RandomResizedCropAndInterpolation


# ========================================
# 正規化定数
# ========================================
OPENAI_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
OPENAI_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

IMAGENET_INCEPTION_MEAN = (0.5, 0.5, 0.5)
IMAGENET_INCEPTION_STD = (0.5, 0.5, 0.5)


# ========================================
# 画像変換パイプライン
# ========================================
def build_transform(
    is_train: bool,
    randaug: bool = True,
    input_size: int = 448,
    std_mode: str = "OPENAI_CLIP",
) -> transforms.Compose:
    """
    画像の前処理パイプラインを構築する

    公式実装: omnilmm/model/utils.py: build_transform()

    ========================================
    入力:
        is_train: 学習時かどうか
        randaug: ランダム拡張を適用するか
        input_size: 出力画像サイズ (ピクセル)
        std_mode: 正規化モード ("OPENAI_CLIP" or "IMAGENET_INCEPTION")

    出力:
        transform: torchvision.transforms.Compose
            PIL Image → (3, input_size, input_size) torch.Tensor
    ========================================
    """
    if std_mode == "OPENAI_CLIP":
        mean, std = OPENAI_CLIP_MEAN, OPENAI_CLIP_STD
    elif std_mode == "IMAGENET_INCEPTION":
        mean, std = IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
    else:
        raise NotImplementedError(f"Unknown std_mode: {std_mode}")

    if is_train:
        # 学習用: ランダムクロップ + オプションの拡張
        crop_scale = float(os.environ.get("TRAIN_CROP_SCALE", 0.9999))
        t = [
            RandomResizedCropAndInterpolation(
                input_size, scale=(crop_scale, 1.0), interpolation="bicubic"
            ),
        ]
        if randaug and os.environ.get("TRAIN_DO_AUG", "False") == "True":
            t.append(RandomAugment(
                N=2, M=7, isPIL=True,
                augs=["Identity", "AutoContrast", "Equalize",
                       "Brightness", "Sharpness", "ShearX", "ShearY",
                       "TranslateX", "TranslateY", "Rotate"],
            ))
        t += [transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)]
        return transforms.Compose(t)
    else:
        # 推論用: リサイズ + 正規化のみ
        return transforms.Compose([
            transforms.Resize(
                (input_size, input_size),
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])


class RandomAugment:
    """
    ランダムデータ拡張

    公式実装: omnilmm/model/utils.py: RandomAugment

    N個のランダムな拡張を確率0.5でそれぞれ適用する。
    拡張の種類: Identity, AutoContrast, Equalize, Rotate,
               Solarize, Color, Contrast, Brightness, Sharpness,
               ShearX, ShearY, TranslateX, TranslateY, Posterize

    ========================================
    入力: PIL Image or numpy array
    出力: 拡張された numpy array
    ========================================
    """

    def __init__(self, N: int = 2, M: int = 10, isPIL: bool = False, augs: list = None):
        """
        Args:
            N: 適用する拡張の数
            M: 拡張の強度レベル (0-10)
            isPIL: 入力がPIL Imageかどうか
            augs: 使用する拡張のリスト
        """
        self.N = N
        self.M = M
        self.isPIL = isPIL
        self.augs = augs or [
            "Identity", "AutoContrast", "Equalize", "Rotate",
            "Solarize", "Color", "Contrast", "Brightness",
            "Sharpness", "ShearX", "TranslateX", "TranslateY",
            "Posterize", "ShearY",
        ]

    def __call__(self, img):
        """
        入力:
            img: PIL Image or numpy array (H, W, 3)

        出力:
            img: numpy array (H, W, 3) - 拡張後
        """
        if self.isPIL:
            img = np.array(img)

        # N個のランダムな拡張を選択
        sampled_ops = np.random.choice(self.augs, self.N)
        for name in sampled_ops:
            if np.random.random() > 0.5:
                continue
            # 各拡張関数を適用 (公式実装の func_dict[name] に対応)
            # 詳細な拡張関数は omnilmm/model/utils.py を参照
            # ここでは主要な処理フローを示すため省略
            pass

        return img


# ========================================
# LLaVA-UHD 画像パーティショニング
# ========================================
def ensure_divide(length: int, patch_size: int) -> int:
    """
    長さをpatch_sizeの倍数に丸める

    公式実装: finetune/dataset.py: ensure_divide()

    ========================================
    入力:
        length: 元の長さ (ピクセル)
        patch_size: パッチサイズ (14)

    出力:
        adjusted_length: patch_sizeの倍数に丸められた長さ
    ========================================
    """
    return max(round(length / patch_size) * patch_size, patch_size)


def find_best_resize(
    original_size: Tuple[int, int],
    scale_resolution: int,
    patch_size: int,
    allow_upscale: bool = False,
) -> Tuple[int, int]:
    """
    画像の最適リサイズ先を計算する

    公式実装: finetune/dataset.py: find_best_resize()

    ========================================
    入力:
        original_size: (W, H) - 元の画像サイズ
        scale_resolution: ターゲット解像度 (448)
        patch_size: パッチサイズ (14)
        allow_upscale: アップスケールを許可するか

    出力:
        (best_width, best_height): patch_sizeの倍数に丸められたサイズ
    ========================================
    """
    width, height = original_size

    if (width * height > scale_resolution * scale_resolution) or allow_upscale:
        r = width / height
        height = int(scale_resolution / math.sqrt(r))
        width = int(height * r)

    best_width = ensure_divide(width, patch_size)
    best_height = ensure_divide(height, patch_size)

    return (best_width, best_height)


def get_refine_size(
    original_size: Tuple[int, int],
    grid: List[int],
    scale_resolution: int,
    patch_size: int,
    allow_upscale: bool = False,
) -> Tuple[int, int]:
    """
    グリッド分割後の画像の精密サイズを計算する

    公式実装: finetune/dataset.py: get_refine_size()

    ========================================
    入力:
        original_size: (W, H) - 元の画像サイズ
        grid: [grid_x, grid_y] - グリッド分割数
        scale_resolution: ターゲット解像度
        patch_size: パッチサイズ

    出力:
        (refine_width, refine_height): グリッド全体のサイズ
            各セルがscale_resolution相当になるよう調整
    ========================================
    """
    width, height = original_size
    grid_x, grid_y = grid

    refine_width = ensure_divide(width, grid_x)
    refine_height = ensure_divide(height, grid_y)

    grid_width = refine_width / grid_x
    grid_height = refine_height / grid_y

    best_grid_size = find_best_resize(
        (grid_width, grid_height),
        scale_resolution,
        patch_size,
        allow_upscale=allow_upscale,
    )

    refine_size = (best_grid_size[0] * grid_x, best_grid_size[1] * grid_y)
    return refine_size


def split_to_patches(
    image: Image.Image,
    grid: List[int],
) -> List[List[Image.Image]]:
    """
    画像をグリッドに従って分割する

    公式実装: finetune/dataset.py: split_to_patches()

    ========================================
    入力:
        image: PIL Image (refine_sizeにリサイズ済み)
        grid: [grid_x, grid_y] - 横×縦のグリッド数

    出力:
        patches: List[List[PIL Image]]
            patches[row][col] = (grid_width, grid_height) のスライス
            len(patches) = grid_y (行数)
            len(patches[0]) = grid_x (列数)
    ========================================
    """
    patches = []
    width, height = image.size
    grid_x = int(width / grid[0])
    grid_y = int(height / grid[1])

    for i in range(0, height, grid_y):
        row_patches = []
        for j in range(0, width, grid_x):
            box = (j, i, j + grid_x, i + grid_y)
            patch = image.crop(box)
            row_patches.append(patch)
        patches.append(row_patches)

    return patches


def slice_image(
    image: Image.Image,
    max_slice_nums: int = 9,
    scale_resolution: int = 448,
    patch_size: int = 14,
    never_split: bool = False,
) -> Tuple[Image.Image, List[List[Image.Image]], Optional[List[int]]]:
    """
    LLaVA-UHD方式で高解像度画像をスライスに分割する

    公式実装: finetune/dataset.py: slice_image()

    ========================================
    処理の流れ:
    1. 画像のアスペクト比から最適なスライス数を推定
    2. 候補グリッド（例: 2x3, 3x2, 1x4, ...）から
       アスペクト比が最も近いものを選択
    3. 画像をグリッドに沿って分割
    4. ソース画像（全体の縮小版）も別途作成

    入力:
        image: PIL Image - 元画像 (任意サイズ)
        max_slice_nums: 最大スライス数 (9)
        scale_resolution: 各スライスのターゲット解像度 (448)
        patch_size: ViTのパッチサイズ (14)
        never_split: Trueの場合、分割せずリサイズのみ

    出力:
        source_image: PIL Image - 全体の縮小版
            サイズ: (best_width, best_height) ※patch_sizeの倍数
        patches: List[List[PIL Image]] - グリッド分割されたスライス
            patches[row][col] = 各スライス
            len = 0 (分割不要の場合) or grid_y * grid_x
        best_grid: [grid_x, grid_y] or None - 選択されたグリッド
    ========================================
    """
    original_width, original_height = image.size
    log_ratio = math.log(original_width / original_height)
    ratio = (original_width * original_height) / (scale_resolution * scale_resolution)
    multiple = min(math.ceil(ratio), max_slice_nums)

    source_image = None
    best_grid = None
    patches = []

    if multiple <= 1 or never_split:
        # --- 分割不要: リサイズのみ ---
        best_size = find_best_resize(
            (original_width, original_height),
            scale_resolution, patch_size, allow_upscale=True
        )
        source_image = image.resize(best_size, Image.Resampling.BICUBIC)
        # source_image: (best_width, best_height)
        #   best_width, best_height は patch_size(14) の倍数
    else:
        # --- グリッド分割 ---

        # 候補スライス数の列挙
        candidate_split_grids_nums = []
        for i in [multiple - 1, multiple, multiple + 1]:
            if i == 1 or i > max_slice_nums:
                continue
            candidate_split_grids_nums.append(i)

        # ソース画像（全体の縮小版）
        best_resize = find_best_resize(
            (original_width, original_height), scale_resolution, patch_size
        )
        source_image = image.copy().resize(best_resize, Image.Resampling.BICUBIC)

        # 候補グリッドの列挙 (例: 6 → [1,6], [2,3], [3,2], [6,1])
        candidate_grids = []
        for split_grids_nums in candidate_split_grids_nums:
            m = 1
            while m <= split_grids_nums:
                if split_grids_nums % m == 0:
                    candidate_grids.append([m, split_grids_nums // m])
                m += 1

        # アスペクト比が最も近いグリッドを選択
        best_grid = [1, 1]
        min_error = float("inf")
        for grid in candidate_grids:
            error = abs(log_ratio - math.log(grid[0] / grid[1]))
            if error < min_error:
                best_grid = grid
                min_error = error
        # best_grid: 例 [2, 3] (横2列×縦3行)

        # グリッドに合わせてリサイズ
        refine_size = get_refine_size(
            (original_width, original_height),
            best_grid, scale_resolution, patch_size, allow_upscale=True
        )
        refine_image = image.resize(refine_size, Image.Resampling.BICUBIC)
        # refine_image: (refine_width, refine_height)
        #   refine_width = best_grid_cell_width * grid_x
        #   refine_height = best_grid_cell_height * grid_y

        # グリッドに沿って分割
        patches = split_to_patches(refine_image, best_grid)
        # patches: List[List[PIL Image]]
        #   patches[row][col]: 各スライス

    return source_image, patches, best_grid


# ========================================
# 視覚エンコーダ本体
# ========================================
class EVA02VisionEncoder(nn.Module):
    """
    EVA02-Enormous Vision Transformer

    公式実装: omnilmm/model/omnilmm.py: create_vision_module()

    アーキテクチャ:
        - timm の EVA02-Enormous (patch14, CLIP-224)
        - 動的画像サイズ対応 (dynamic_img_size=True, dynamic_img_pad=True)
        - 最終ブロックを Identity に置換 → 2番目最後の層の出力を使用
        - attn_pool がある場合も Identity に置換

    ========================================
    入力:
        pixel_values: (N, 3, H, W)
            - N: バッチ/スライス数
            - 3: RGB
            - H, W: 解像度 (patch_size=14 の倍数)

    出力:
        features: (N, L_vis, D_vis)
            - L_vis = (H/14) * (W/14): パッチトークン数
            - D_vis = 1792: EVA02-Enormous の隠れ次元
    ========================================
    """

    def __init__(
        self,
        model_name: str = "eva02_enormous_patch14_clip_224.laion2b_plus",
        patch_size: int = 14,
        d_vis: int = 1792,
    ):
        super().__init__()

        self.patch_size = patch_size
        self.d_vis = d_vis

        # ========================================
        # 1. Vision Transformer の構築
        # ========================================
        # 公式: timm.create_model(model_name, pretrained=False,
        #         num_classes=0, dynamic_img_size=True, dynamic_img_pad=True)
        #
        # EVA02-Enormous の構造:
        # - パッチ埋め込み: Conv2d(3, d_vis, kernel_size=14, stride=14)
        # - 48層の Transformer ブロック
        # - 各ブロック: LayerNorm → MultiHeadAttention → LayerNorm → MLP
        # - MLP: Linear(d_vis, 4*d_vis) → GELU → Linear(4*d_vis, d_vis)

        # パッチ埋め込み
        self.patch_embed = nn.Conv2d(
            3, d_vis, kernel_size=patch_size, stride=patch_size
        )
        # patch_embed: (N, 3, H, W) → (N, D_vis, H/P, W/P)

        # 位置埋め込み (学習可能)
        # 動的サイズのため、推論時にbicubic補間で調整
        self.pos_embed = nn.Parameter(
            torch.zeros(1, 256 + 1, d_vis)  # 16x16 + CLS
        )
        # pos_embed: (1, L_pos+1, D_vis) ※CLSトークン分+1

        # CLS トークン
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_vis))

        # Transformer ブロック (48層、最終層はIdentity)
        self.num_blocks = 48
        self.blocks = nn.ModuleList([
            TransformerBlock(d_vis, num_heads=d_vis // 64)
            for _ in range(self.num_blocks - 1)  # 47層 (最終層はIdentity)
        ])
        # 最終ブロックは Identity に置換済み
        # → 2番目最後の層の出力を特徴として使用

        # 最終LayerNorm
        self.norm = nn.LayerNorm(d_vis)

        # num_prefix_tokens: CLSトークン等のプレフィックス数
        self.num_prefix_tokens = 1  # CLS のみ

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        ========================================
        入力:
            pixel_values: (N, 3, H, W)
                - N: バッチ/スライス数
                - H, W: 解像度 (patch_size の倍数)

        出力:
            features: (N, L_vis, D_vis)
                - L_vis = (H/14) * (W/14)
                - D_vis = 1792

        処理の流れ:
            1. パッチ埋め込み
            2. 位置埋め込み追加
            3. 47層の Transformer ブロック (最終層はスキップ)
            4. CLS/prefix トークンを除去
        ========================================
        """
        N, C, H, W = pixel_values.shape
        # pixel_values: (N, 3, H, W)

        # --- 1. パッチ埋め込み ---
        x = self.patch_embed(pixel_values)
        # x: (N, D_vis, H/P, W/P) = (N, 1792, H/14, W/14)

        x = x.flatten(2).transpose(1, 2)
        # x: (N, L_vis, D_vis)
        #   L_vis = (H/14) * (W/14)

        # --- 2. CLSトークン + 位置埋め込み ---
        cls_tokens = self.cls_token.expand(N, -1, -1)
        # cls_tokens: (N, 1, D_vis)
        x = torch.cat([cls_tokens, x], dim=1)
        # x: (N, 1+L_vis, D_vis)

        # 位置埋め込み (動的サイズ対応: bicubic補間)
        pos_embed = self._interpolate_pos_embed(x.shape[1])
        # pos_embed: (1, 1+L_vis, D_vis)
        x = x + pos_embed

        # --- 3. Transformer ブロック (47層) ---
        for block in self.blocks:
            x = block(x)
        # x: (N, 1+L_vis, D_vis)

        x = self.norm(x)
        # x: (N, 1+L_vis, D_vis)

        # --- 4. CLS/prefix トークンを除去 ---
        if self.num_prefix_tokens > 0:
            x = x[:, self.num_prefix_tokens:]
        # x: (N, L_vis, D_vis)
        #   L_vis = (H/14) * (W/14)
        #   D_vis = 1792

        return x

    def _interpolate_pos_embed(self, target_len: int) -> torch.Tensor:
        """
        位置埋め込みを動的サイズに補間する

        ========================================
        入力:
            target_len: ターゲットのトークン数 (1 + L_vis)

        出力:
            pos_embed: (1, target_len, D_vis)
        ========================================
        """
        pos_embed = self.pos_embed  # (1, L_orig+1, D_vis)

        if pos_embed.shape[1] == target_len:
            return pos_embed

        # CLS と spatial を分離
        cls_pos = pos_embed[:, :1, :]  # (1, 1, D_vis)
        spatial_pos = pos_embed[:, 1:, :]  # (1, L_orig, D_vis)

        src_size = int(math.sqrt(spatial_pos.shape[1]))
        tgt_size = int(math.sqrt(target_len - 1))

        # bicubic 補間
        spatial_pos = spatial_pos.reshape(1, src_size, src_size, -1).permute(0, 3, 1, 2)
        # spatial_pos: (1, D_vis, src_size, src_size)
        spatial_pos = F.interpolate(
            spatial_pos.float(),
            size=(tgt_size, tgt_size),
            mode="bicubic",
            align_corners=False,
        ).to(spatial_pos.dtype)
        # spatial_pos: (1, D_vis, tgt_size, tgt_size)
        spatial_pos = spatial_pos.permute(0, 2, 3, 1).flatten(1, 2)
        # spatial_pos: (1, tgt_size*tgt_size, D_vis)

        pos_embed = torch.cat([cls_pos, spatial_pos], dim=1)
        # pos_embed: (1, 1 + tgt_size^2, D_vis)

        return pos_embed


class TransformerBlock(nn.Module):
    """
    Vision Transformerの1ブロック

    ========================================
    入力/出力: (N, L, D_vis)
    ========================================
    """

    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        入力: x: (N, L, D_vis)
        出力: x: (N, L, D_vis)
        """
        # Pre-Norm + Self-Attention + Residual
        h = self.norm1(x)
        h, _ = self.attn(h, h, h)
        x = x + h
        # x: (N, L, D_vis)

        # Pre-Norm + MLP + Residual
        h = self.norm2(x)
        h = self.mlp(h)
        x = x + h
        # x: (N, L, D_vis)

        return x


# ========================================
# 使用例
# ========================================
def example_usage():
    """
    画像パーティショニング → 視覚エンコーディングのデモ
    """
    # --- 画像読み込み ---
    image = Image.open("example.jpg").convert("RGB")
    # image.size: (1200, 800) ← 任意サイズ

    # --- Step 1: LLaVA-UHD パーティショニング ---
    source_image, patches, best_grid = slice_image(
        image,
        max_slice_nums=9,
        scale_resolution=448,
        patch_size=14,
    )
    # source_image: (448, 294) ← 全体の縮小版 (patch_sizeの倍数)
    # patches: [[patch_00, patch_01], [patch_10, patch_11], [patch_20, patch_21]]
    #   → best_grid = [2, 3] (横2×縦3)
    # 合計スライス数: 1 (source) + 2*3 = 7

    # --- Step 2: 全スライスを収集 ---
    all_images = [source_image]
    for row in patches:
        for patch in row:
            all_images.append(patch)
    # all_images: 7枚の PIL Image

    # --- Step 3: 画像変換（正規化） ---
    transform = build_transform(is_train=False, input_size=448, std_mode="OPENAI_CLIP")
    pixel_values = torch.stack([transform(img) for img in all_images])
    # pixel_values: (7, 3, 448, 448)

    # --- Step 4: 視覚エンコーダ ---
    encoder = EVA02VisionEncoder()
    features = encoder(pixel_values)
    # features: (7, 1024, 1792)
    #   7: スライス数
    #   1024: (448/14)*(448/14) = 32*32 パッチトークン
    #   1792: EVA02-Enormous の出力次元

    print(f"入力画像サイズ: {image.size}")
    print(f"スライス数: {len(all_images)}")
    print(f"視覚特徴形状: {features.shape}")
    # → 後段の Resampler で (7, 1024, 1792) → (7, 64, 4096) に圧縮


if __name__ == "__main__":
    example_usage()
