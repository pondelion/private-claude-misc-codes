"""
Qwen2.5-Omni Vision Encoder - 簡略化疑似コード
================================================

ViT ベースの画像/動画エンコーダ (Qwen2.5-VL と共通)
画像/動画 → パッチ埋め込み → Transformer → PatchMerger → 特徴ベクトル系列

公式実装: modeling_qwen2_5_omni_low_VRAM_mode.py (Lines 1216-1377)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Optional, Tuple


class VisionRotaryEmbedding(nn.Module):
    """
    Vision 用 2D Rotary Position Embedding

    画像/動画のパッチに対して、height と width の独立した
    回転位置エンコーディングを計算
    """

    def __init__(self, dim: int, theta: float = 10000.0):
        """
        パラメータ:
            dim: 回転埋め込みの次元 (= head_dim // 2)
            theta: 基底周波数
        """
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, seq_len: int) -> torch.Tensor:
        """
        入力:
            seq_len: シーケンス長

        出力:
            freqs: (seq_len, dim) - 回転周波数
        """
        t = torch.arange(seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        return freqs


class VisionPatchEmbed(nn.Module):
    """
    画像/動画のパッチ埋め込み

    3D畳み込みでパッチを特徴ベクトルに変換
    """

    def __init__(
        self,
        patch_size: int = 14,
        temporal_patch_size: int = 2,
        in_channels: int = 3,
        embed_dim: int = 1024,
    ):
        """
        パラメータ:
            patch_size: 空間パッチサイズ (14x14 ピクセル)
            temporal_patch_size: 時間パッチサイズ (2フレーム)
            in_channels: 入力チャンネル数 (3 = RGB)
            embed_dim: 出力埋め込み次元 (1024)
        """
        super().__init__()
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size

        # 3D畳み込み: (T, H, W) → パッチ埋め込み
        self.proj = nn.Conv3d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=(temporal_patch_size, patch_size, patch_size),
            stride=(temporal_patch_size, patch_size, patch_size),
            bias=False,
        )
        # 入力: (B, 3, T, H, W)
        # 出力: (B, embed_dim, T//2, H//14, W//14)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        入力:
            hidden_states: (N_patches, patch_dim) - 平坦化されたパッチ
                N_patches: 全パッチ数
                patch_dim: 3 * temporal_patch_size * patch_size * patch_size
                         = 3 * 2 * 14 * 14 = 1176

        出力:
            embeddings: (N_patches, embed_dim) - パッチ埋め込み
                embed_dim: 1024
        """
        # 実際の実装では3D Convを適用
        # ここでは概念的にLinear射影として簡略化
        target_dtype = self.proj.weight.dtype
        hidden_states = hidden_states.to(dtype=target_dtype)

        # パッチを3D形状に復元して畳み込み
        # (N, 1176) → (N, 3, 2, 14, 14) → Conv3D → (N, 1024, 1, 1, 1) → (N, 1024)
        N = hidden_states.shape[0]
        hidden_states = hidden_states.reshape(
            N, 3, self.temporal_patch_size, self.patch_size, self.patch_size
        )
        hidden_states = self.proj(hidden_states.unsqueeze(0))
        hidden_states = hidden_states.squeeze(0).flatten(1)
        # hidden_states: (N_patches, embed_dim=1024)

        return hidden_states


class PatchMerger(nn.Module):
    """
    隣接パッチの統合

    spatial_merge_size × spatial_merge_size のパッチグループを
    1つのトークンにマージしてトークン数を削減

    例: spatial_merge_size=2 → 2×2=4パッチ → 1トークン (1/4に削減)
    """

    def __init__(
        self,
        dim: int,
        context_dim: int,
        spatial_merge_size: int = 2,
    ):
        """
        パラメータ:
            dim: 出力次元 (= Thinker LLMの入力次元に合わせる)
            context_dim: 入力次元 (= Vision Encoderの隠れ次元, 1024)
            spatial_merge_size: 統合サイズ (2 → 2×2=4パッチを1トークンに)
        """
        super().__init__()
        self.hidden_size = context_dim * (spatial_merge_size ** 2)
        # hidden_size = 1024 * 4 = 4096 (4パッチ分を結合)

        self.ln_q = nn.LayerNorm(context_dim, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, dim),
        )
        # 4096 → 4096 → GELU → 4096 → dim

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        入力:
            hidden_states: (N_patches, context_dim) - ViT出力パッチ
                N_patches: 全パッチ数 (spatial_merge_size^2 の倍数)
                context_dim: 1024

                パッチは既にマージ順に並び替え済み:
                [patch_00, patch_01, patch_10, patch_11,  ← 1つ目のグループ (2x2)
                 patch_02, patch_03, patch_12, patch_13,  ← 2つ目のグループ
                 ...]

        出力:
            merged: (N_merged, dim) - マージ後トークン
                N_merged = N_patches // (spatial_merge_size^2)
                dim: 出力次元
        """
        hidden_states = self.ln_q(hidden_states)
        # hidden_states: (N_patches, 1024)

        # spatial_merge_size^2 個ずつグループ化して結合
        # (N_patches, 1024) → (N_merged, 4, 1024) → (N_merged, 4096)
        hidden_states = hidden_states.view(-1, self.hidden_size)
        # hidden_states: (N_merged, 4096)

        hidden_states = self.mlp(hidden_states)
        # hidden_states: (N_merged, dim)

        return hidden_states


class VisionBlock(nn.Module):
    """
    Vision Transformer の単一ブロック

    ウィンドウアテンション (局所) またはフルアテンション (大域) を使用
    """

    def __init__(
        self,
        hidden_size: int = 1024,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        # Layer Normalization (Pre-Norm)
        self.norm1 = nn.LayerNorm(hidden_size, eps=1e-6)
        self.norm2 = nn.LayerNorm(hidden_size, eps=1e-6)

        # Self-Attention
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            batch_first=False,
        )

        # MLP (SwiGLU)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, hidden_size),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: torch.Tensor,
    ) -> torch.Tensor:
        """
        入力:
            hidden_states: (total_tokens, hidden_size) - パック済みトークン
                total_tokens: 全画像/動画の有効トークン数合計
                hidden_size: 1024

            cu_seqlens: (num_sequences + 1,) - 累積シーケンス長
                ウィンドウアテンション時: ウィンドウ境界
                フルアテンション時: 画像/動画境界

            rotary_pos_emb: (total_tokens, head_dim) - 2D RoPE

        出力:
            hidden_states: (total_tokens, hidden_size)
        """

        # ----------------------------------------
        # 1. Pre-Norm + Self-Attention + RoPE
        # ----------------------------------------
        residual = hidden_states
        hidden_states = self.norm1(hidden_states)
        # hidden_states: (total_tokens, 1024)

        # Flash Attention with RoPE and cu_seqlens
        # 実際には flash_attn_varlen_func を使用
        # cu_seqlens によりウィンドウ/フルアテンションを切り替え
        hidden_states_3d = hidden_states.unsqueeze(1)
        attn_out, _ = self.attn(
            hidden_states_3d, hidden_states_3d, hidden_states_3d
        )
        hidden_states = attn_out.squeeze(1)
        # hidden_states: (total_tokens, 1024)

        hidden_states = residual + hidden_states

        # ----------------------------------------
        # 2. Pre-Norm + MLP
        # ----------------------------------------
        residual = hidden_states
        hidden_states = self.norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        # hidden_states: (total_tokens, 1024)

        return hidden_states


class VisionEncoder(nn.Module):
    """
    Qwen2.5-Omni Vision Encoder

    Qwen2.5-VL と同一の ViT ベースエンコーダ (~675M パラメータ)

    アーキテクチャ:
        パッチ埋め込み (patch_size=14, temporal_patch_size=2)
        → 2D RoPE 位置エンコーディング (height, width 独立)
        → ウィンドウアテンション ViT ブロック × depth
          (一部のレイヤーでフルアテンション: fullatt_block_indexes)
        → PatchMerger (2×2 パッチ → 1 トークン)
        → 出力特徴

    ウィンドウアテンション:
        - 画像/動画を空間的にウィンドウに分割
        - 各ウィンドウ内でのみアテンション計算 (計算効率向上)
        - fullatt_block_indexes で指定されたレイヤーではフルアテンション
          (大域的な文脈を捕捉)
    """

    def __init__(
        self,
        hidden_size: int = 1024,
        depth: int = 24,
        num_heads: int = 16,
        patch_size: int = 14,
        temporal_patch_size: int = 2,
        spatial_merge_size: int = 2,
        in_channels: int = 3,
        mlp_ratio: float = 4.0,
        window_size: int = 112,
        fullatt_block_indexes: Optional[List[int]] = None,
        out_hidden_size: Optional[int] = None,
    ):
        """
        パラメータ:
            hidden_size: ViT隠れ次元 (1024)
            depth: ViTレイヤー数 (24)
            num_heads: アテンションヘッド数 (16)
            patch_size: 空間パッチサイズ (14)
            temporal_patch_size: 時間パッチサイズ (2)
            spatial_merge_size: PatchMerger統合サイズ (2)
            in_channels: 入力チャンネル (3=RGB)
            mlp_ratio: MLP拡張比 (4.0)
            window_size: ウィンドウアテンションのウィンドウサイズ (112ピクセル)
            fullatt_block_indexes: フルアテンションを使うレイヤーインデックス
            out_hidden_size: PatchMerger出力次元 (None → hidden_size)
        """
        super().__init__()

        self.hidden_size = hidden_size
        self.depth = depth
        self.spatial_merge_size = spatial_merge_size
        self.spatial_merge_unit = spatial_merge_size ** 2  # 4
        self.window_size = window_size
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size

        if fullatt_block_indexes is None:
            # デフォルト: 一部のレイヤーでフルアテンション
            fullatt_block_indexes = [7, 15, 23]
        self.fullatt_block_indexes = set(fullatt_block_indexes)

        if out_hidden_size is None:
            out_hidden_size = hidden_size

        # パッチ埋め込み
        self.patch_embed = VisionPatchEmbed(
            patch_size=patch_size,
            temporal_patch_size=temporal_patch_size,
            in_channels=in_channels,
            embed_dim=hidden_size,
        )

        # 2D RoPE
        head_dim = hidden_size // num_heads
        self.rotary_pos_emb = VisionRotaryEmbedding(dim=head_dim // 2)

        # ViT ブロック
        self.blocks = nn.ModuleList([
            VisionBlock(
                hidden_size=hidden_size,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
            )
            for _ in range(depth)
        ])

        # PatchMerger
        self.merger = PatchMerger(
            dim=out_hidden_size,
            context_dim=hidden_size,
            spatial_merge_size=spatial_merge_size,
        )

    def rot_pos_emb(self, grid_thw: torch.Tensor) -> torch.Tensor:
        """
        2D 回転位置埋め込みの計算

        入力:
            grid_thw: (num_images_or_videos, 3) - 各画像/動画の [T, H, W] パッチ数

        出力:
            rotary_pos_emb: (total_tokens, head_dim) - 位置埋め込み

        処理:
            各画像/動画について:
            1. (H, W) の位置グリッドを作成
            2. spatial_merge_size で並び替え (マージ順)
            3. T回繰り返し (時間方向)
            4. height と width の位置IDを結合
            5. 回転周波数を計算
        """

        all_pos_ids = []

        for t, h, w in grid_thw:
            t, h, w = t.item(), h.item(), w.item()

            # height と width の位置グリッドを作成
            hpos_ids = torch.arange(h).unsqueeze(1).expand(h, w)
            wpos_ids = torch.arange(w).unsqueeze(0).expand(h, w)
            # hpos_ids: (H, W) - 各位置のheight座標
            # wpos_ids: (H, W) - 各位置のwidth座標

            # spatial_merge_size で並び替え
            # (H, W) → (H//2, 2, W//2, 2) → (H//2, W//2, 2, 2) → flatten
            merge_size = self.spatial_merge_size
            hpos_ids = hpos_ids.reshape(h // merge_size, merge_size,
                                         w // merge_size, merge_size)
            hpos_ids = hpos_ids.permute(0, 2, 1, 3).flatten()
            # hpos_ids: (H*W,) - マージ順に並び替え済み

            wpos_ids = wpos_ids.reshape(h // merge_size, merge_size,
                                         w // merge_size, merge_size)
            wpos_ids = wpos_ids.permute(0, 2, 1, 3).flatten()
            # wpos_ids: (H*W,)

            # height と width の位置IDを結合
            pos_ids = torch.stack([hpos_ids, wpos_ids], dim=-1)
            # pos_ids: (H*W, 2)

            # T回繰り返し (時間方向: 各フレームで同じ空間位置)
            pos_ids = pos_ids.repeat(t, 1)
            # pos_ids: (T*H*W, 2)

            all_pos_ids.append(pos_ids)

        all_pos_ids = torch.cat(all_pos_ids, dim=0)
        # all_pos_ids: (total_patches, 2)

        # 回転周波数の計算
        max_grid = all_pos_ids.max() + 1
        rotary_pos_emb_full = self.rotary_pos_emb(max_grid)
        # rotary_pos_emb_full: (max_grid, head_dim//2)

        # 各パッチの位置に対応する回転周波数を取得
        rotary_pos_emb_h = rotary_pos_emb_full[all_pos_ids[:, 0]]
        rotary_pos_emb_w = rotary_pos_emb_full[all_pos_ids[:, 1]]
        # 各: (total_patches, head_dim//2)

        # height と width の周波数を結合
        rotary_pos_emb_out = torch.cat([rotary_pos_emb_h, rotary_pos_emb_w], dim=-1)
        # rotary_pos_emb_out: (total_patches, head_dim)

        return rotary_pos_emb_out

    def get_window_index(self, grid_thw: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        ウィンドウアテンション用のインデックス計算

        入力:
            grid_thw: (num, 3) - 各画像/動画の [T, H, W]

        出力:
            window_index: (total_tokens,) - ウィンドウ順の並び替えインデックス
            cu_window_seqlens: (num_windows + 1,) - ウィンドウ境界の累積長

        処理:
            1. 各画像/動画の空間グリッドをウィンドウに分割
            2. ウィンドウ内のトークンが連続するように並び替え
            3. cu_seqlens を計算 (Flash Attention用)
        """
        # ウィンドウサイズ (LLMグリッド座標)
        vit_merger_window_size = (
            self.window_size // self.spatial_merge_size // self.patch_size
        )
        # 例: 112 // 2 // 14 = 4 (4×4パッチのウィンドウ)

        all_window_indices = []
        cu_seqlens_list = [0]
        offset = 0

        for t, h, w in grid_thw:
            t, h, w = t.item(), h.item(), w.item()

            llm_grid_h = h // self.spatial_merge_size
            llm_grid_w = w // self.spatial_merge_size
            # LLMグリッドサイズ (PatchMerger後のサイズ)

            # 各フレームのインデックスグリッド
            for frame in range(t):
                # (llm_grid_h, llm_grid_w) のインデックスを作成
                index_grid = torch.arange(llm_grid_h * llm_grid_w).reshape(
                    llm_grid_h, llm_grid_w
                )

                # ウィンドウサイズにパディング
                pad_h = (vit_merger_window_size - llm_grid_h % vit_merger_window_size) % vit_merger_window_size
                pad_w = (vit_merger_window_size - llm_grid_w % vit_merger_window_size) % vit_merger_window_size
                index_grid = F.pad(index_grid, (0, pad_w, 0, pad_h), value=-1)

                # ウィンドウに分割
                padded_h, padded_w = index_grid.shape
                num_win_h = padded_h // vit_merger_window_size
                num_win_w = padded_w // vit_merger_window_size

                index_grid = index_grid.reshape(
                    num_win_h, vit_merger_window_size,
                    num_win_w, vit_merger_window_size
                ).permute(0, 2, 1, 3).reshape(-1)
                # ウィンドウ順に並び替え

                # パディング(-1)を除去
                valid_mask = index_grid >= 0
                valid_indices = index_grid[valid_mask] + offset

                all_window_indices.append(valid_indices)

                # ウィンドウごとのcu_seqlens
                for wh in range(num_win_h):
                    for ww in range(num_win_w):
                        win_size = min(vit_merger_window_size, llm_grid_h - wh * vit_merger_window_size) * \
                                   min(vit_merger_window_size, llm_grid_w - ww * vit_merger_window_size)
                        cu_seqlens_list.append(cu_seqlens_list[-1] + win_size * self.spatial_merge_unit)

                offset += llm_grid_h * llm_grid_w

        window_index = torch.cat(all_window_indices)
        cu_window_seqlens = torch.tensor(cu_seqlens_list, dtype=torch.int32)

        return window_index, cu_window_seqlens

    def forward(
        self,
        hidden_states: torch.Tensor,
        grid_thw: torch.Tensor,
    ) -> torch.Tensor:
        """
        Vision Encoder のフォワードパス

        入力:
            hidden_states: (total_patches, patch_dim) - 全画像/動画のパッチ
                total_patches: 全パッチ数
                patch_dim: 3 * temporal_patch_size * patch_size^2 = 1176

            grid_thw: (num_images_or_videos, 3) - 各画像/動画の [T, H, W]
                T: 時間パッチ数 (画像は T=1, 動画は T=フレーム数//2)
                H: heightパッチ数 (H_pixels // patch_size)
                W: widthパッチ数 (W_pixels // patch_size)

        出力:
            features: (total_merged_tokens, out_hidden_size) - マージ後特徴
                total_merged_tokens = total_patches // spatial_merge_unit
                out_hidden_size: 出力特徴次元 (1024)

        処理フロー:
            1. パッチ埋め込み
            2. 2D RoPE 位置エンコーディング計算
            3. ウィンドウインデックス計算
            4. トークンをウィンドウ順に並び替え
            5. ViT ブロック × depth (ウィンドウ/フルアテンション)
            6. PatchMerger (2×2 統合)
            7. 元の順序に復元
        """

        # ========================================
        # Step 1: パッチ埋め込み
        # ========================================
        hidden_states = self.patch_embed(hidden_states)
        # hidden_states: (total_patches, 1024)

        # ========================================
        # Step 2: 2D RoPE 計算
        # ========================================
        rotary_pos_emb = self.rot_pos_emb(grid_thw)
        # rotary_pos_emb: (total_patches, head_dim)
        # height と width の独立した回転位置エンコーディング

        # ========================================
        # Step 3: ウィンドウインデックス計算
        # ========================================
        window_index, cu_window_seqlens = self.get_window_index(grid_thw)
        # window_index: (total_patches,) - ウィンドウ順の並び替えインデックス
        # cu_window_seqlens: (num_windows + 1,) - ウィンドウ境界

        # フルアテンション用の cu_seqlens
        cu_seqlens = torch.zeros(grid_thw.shape[0] + 1, dtype=torch.int32)
        for i, (t, h, w) in enumerate(grid_thw):
            cu_seqlens[i + 1] = cu_seqlens[i] + t * h * w
        # cu_seqlens: (num_images + 1,) - 画像/動画境界

        # ========================================
        # Step 4: ウィンドウ順に並び替え
        # ========================================
        # spatial_merge_unit ごとにグループ化してから並び替え
        hidden_states = hidden_states.reshape(
            -1, self.spatial_merge_unit, self.hidden_size
        )
        # hidden_states: (total_patches // 4, 4, 1024)

        hidden_states = hidden_states[window_index]
        # ウィンドウ順に並び替え

        hidden_states = hidden_states.reshape(-1, self.hidden_size)
        # hidden_states: (total_patches, 1024)

        # RoPE も同様に並び替え
        rotary_pos_emb = rotary_pos_emb.reshape(
            -1, self.spatial_merge_unit, rotary_pos_emb.shape[-1]
        )[window_index].reshape(-1, rotary_pos_emb.shape[-1])

        # ========================================
        # Step 5: ViT ブロック
        # ========================================
        for i, block in enumerate(self.blocks):
            if i in self.fullatt_block_indexes:
                # フルアテンション: 画像/動画全体でアテンション
                current_cu_seqlens = cu_seqlens
            else:
                # ウィンドウアテンション: ウィンドウ内でのみアテンション
                current_cu_seqlens = cu_window_seqlens

            hidden_states = block(
                hidden_states=hidden_states,
                cu_seqlens=current_cu_seqlens,
                rotary_pos_emb=rotary_pos_emb,
            )
            # hidden_states: (total_patches, 1024)

        # ========================================
        # Step 6: PatchMerger
        # ========================================
        hidden_states = self.merger(hidden_states)
        # hidden_states: (total_merged_tokens, out_hidden_size)
        # total_merged_tokens = total_patches // 4

        # ========================================
        # Step 7: 元の順序に復元
        # ========================================
        reverse_index = torch.argsort(window_index)
        hidden_states = hidden_states[reverse_index]
        # hidden_states: (total_merged_tokens, out_hidden_size)

        return hidden_states


# ============================================
# 使用例
# ============================================

def example_vision_encoder():
    """
    Vision Encoder の使用例

    実際にモジュールをインスタンス化し、ダミー入力で
    フォワードパスを実行して各ステージの形状を確認する
    """

    # --- 初期化 (depth=4に縮小して高速化) ---
    encoder = VisionEncoder(
        hidden_size=1024,
        depth=4,                    # 実モデルは24、ここでは4層に縮小
        num_heads=16,
        patch_size=14,
        temporal_patch_size=2,
        spatial_merge_size=2,
        window_size=112,
        fullatt_block_indexes=[3],  # 最後の層のみフルアテンション
    )
    encoder.eval()

    patch_dim = 3 * 2 * 14 * 14  # = 1176

    # ========================================
    # 例1: 単一画像 (504×504)
    # ========================================
    H, W = 504, 504
    H_patches = H // 14  # 36
    W_patches = W // 14  # 36
    T_patches = 1         # 画像は T=1 (2同一フレーム→temporal_patch_size=2で割る)

    N_patches = T_patches * H_patches * W_patches  # 1 × 36 × 36 = 1296
    pixel_values = torch.randn(N_patches, patch_dim)
    grid_thw = torch.tensor([[T_patches, H_patches, W_patches]])  # (1, 3)

    # パッチ埋め込み
    with torch.no_grad():
        patch_embed_out = encoder.patch_embed(pixel_values)
    assert patch_embed_out.shape == (N_patches, 1024)

    # 2D RoPE
    rotary_pos_emb = encoder.rot_pos_emb(grid_thw)
    head_dim = 1024 // 16  # = 64
    assert rotary_pos_emb.shape == (N_patches, head_dim)

    # ウィンドウインデックス
    window_index, cu_window_seqlens = encoder.get_window_index(grid_thw)
    assert window_index.shape[0] == N_patches // (2 * 2)  # merge_unit で割る

    # フルフォワードパス
    with torch.no_grad():
        output = encoder(pixel_values, grid_thw)

    N_merged = N_patches // (2 * 2)  # 1296 // 4 = 324
    assert output.shape == (N_merged, 1024)

    print(f"[Vision Encoder 使用例]")
    print(f"  例1: 単一画像 ({H}×{W})")
    print(f"    入力:         pixel_values {pixel_values.shape}  (N_patches, patch_dim)")
    print(f"                  grid_thw     {grid_thw.tolist()}")
    print(f"    パッチ埋め込み: {patch_embed_out.shape}  → (N_patches, 1024)")
    print(f"    2D RoPE:       {rotary_pos_emb.shape}  → (N_patches, head_dim)")
    print(f"    ViTブロック×{encoder.depth}:  ({N_patches}, 1024)")
    print(f"    PatchMerger:   {output.shape}  → ({N_merged}, 1024)  1/4に削減")
    print()

    # ========================================
    # 例2: 動画 (8フレーム, 280×280)
    # ========================================
    T_frames = 8
    H_v, W_v = 280, 280
    T_patches_v = T_frames // 2   # 4
    H_patches_v = H_v // 14      # 20
    W_patches_v = W_v // 14      # 20

    N_patches_v = T_patches_v * H_patches_v * W_patches_v  # 4 × 20 × 20 = 1600
    pixel_values_v = torch.randn(N_patches_v, patch_dim)
    grid_thw_v = torch.tensor([[T_patches_v, H_patches_v, W_patches_v]])

    with torch.no_grad():
        output_v = encoder(pixel_values_v, grid_thw_v)

    N_merged_v = N_patches_v // 4  # 400
    assert output_v.shape == (N_merged_v, 1024)

    print(f"  例2: 動画 ({T_frames}フレーム, {H_v}×{W_v})")
    print(f"    入力:         pixel_values {pixel_values_v.shape}")
    print(f"                  grid_thw     {grid_thw_v.tolist()}")
    print(f"    パッチ数:     T={T_patches_v} × H={H_patches_v} × W={W_patches_v} = {N_patches_v}")
    print(f"    PatchMerger後: {output_v.shape}  ({N_merged_v} トークン)")
    print()

    # ========================================
    # 例3: バッチ入力 (画像2枚)
    # ========================================
    # 画像A: 280×280 → T=1, H=20, W=20 → 400 patches
    # 画像B: 420×280 → T=1, H=30, W=20 → 600 patches
    N_a = 1 * 20 * 20   # 400
    N_b = 1 * 30 * 20   # 600
    N_total = N_a + N_b  # 1000

    pixel_values_batch = torch.randn(N_total, patch_dim)
    grid_thw_batch = torch.tensor([
        [1, 20, 20],  # 画像A
        [1, 30, 20],  # 画像B
    ])

    with torch.no_grad():
        output_batch = encoder(pixel_values_batch, grid_thw_batch)

    N_merged_batch = N_total // 4  # 250
    assert output_batch.shape == (N_merged_batch, 1024)

    print(f"  例3: バッチ入力 (画像2枚)")
    print(f"    画像A: 280×280 → {N_a} patches → {N_a//4} merged tokens")
    print(f"    画像B: 420×280 → {N_b} patches → {N_b//4} merged tokens")
    print(f"    結合入力:  pixel_values {pixel_values_batch.shape}")
    print(f"    結合出力:  {output_batch.shape}  (total={N_merged_batch})")
    print()

    # ウィンドウアテンションの詳細
    window_pixel = 112
    vit_merger_window = window_pixel // 2 // 14  # = 4
    print(f"  [ウィンドウアテンション]")
    print(f"    ウィンドウサイズ: {window_pixel}px → {vit_merger_window}×{vit_merger_window} LLMグリッド")
    print(f"    ウィンドウ内トークン: {vit_merger_window**2 * 4} (merge_unit含む)")
    print(f"    フルアテンション層: {sorted(encoder.fullatt_block_indexes)}")
    print(f"    ウィンドウアテンション層: 残り{encoder.depth - len(encoder.fullatt_block_indexes)}層")


if __name__ == "__main__":
    example_vision_encoder()
