"""
Qwen3-Omni Vision Encoder: SigLIP2-So400m ベース
==================================================

SigLIP2-So400m (Tschannen et al., 2025) で初期化された
ViT ベースの画像/動画エンコーダ。Qwen3-VL から採用。

Qwen2.5-Omni の ViT (~675M) を完全に置き換える。

主な差分 (vs Qwen2.5-Omni):
    - バックボーン: 独自 ViT (~675M) → SigLIP2-So400m (~540M)
    - 初期化: Qwen2.5-VL 由来 → SigLIP2 事前学習済み (Qwen3-VL 由来)
    - パラメータ数: ~675M → ~540M (約20%削減)
    - PatchMerger: 同一 (spatial_merge_size=2, 4パッチ→1トークン)
    - patch_size: 同一 (14)
    - temporal_patch_size: 同一 (2)
    - ウィンドウ/フルアテンションパターン: 同一構造
    - 動画フレームレート: 動的サンプリング (音声12.5Hzと時間的にアライン)

アーキテクチャ:
    PatchEmbed (Conv3D, patch_size=14, temporal_patch_size=2)
    → 2D RoPE 位置エンコーディング
    → ViTブロック × depth (ウィンドウ + フルアテンション)
    → PatchMerger (2×2 → 1, トークン数1/4)
    → 出力特徴

入力:
    - 画像: (N_patches, patch_dim) - 平坦化パッチ列
      patch_dim = 3 × temporal_patch_size × patch_size² = 3 × 2 × 14 × 14 = 1176
    - 動画: 同形式。フレームは動的レートでサンプリングされ、
      temporal_patch_size=2 で2フレームずつペア化
    - grid_thw: (num_images_or_videos, 3) - 各入力の [T, H, W] パッチ数

出力:
    - features: (total_merged_tokens, hidden_size)
      total_merged_tokens = total_patches // 4 (PatchMerger による)

動画の時間アライメント:
    音声エンコーダ (AuT) が12.5Hzでトークンを出力するため、
    動画フレームも動的フレームレートでサンプリングし、
    temporal_patch_size=2 で割った後のトークン列が
    音声トークンと時間的に対応するようにする。

パラメータ数: ~540M (0.54B)
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================
# Rotary Position Embedding (2D)
# ============================================

class VisionRotaryEmbedding(nn.Module):
    """
    Vision 用 2D Rotary Position Embedding

    画像/動画パッチの height と width に対して独立した
    回転位置エンコーディングを計算。

    ※ Qwen2.5-Omni と同一の仕組み。SigLIP2 自体は
      学習可能位置埋め込みを使うが、Qwen3-VL / Qwen3-Omni
      では RoPE に置き換えている。
    """

    def __init__(self, dim, theta=10000.0):
        """
        パラメータ:
            dim: 回転埋め込み次元 (= head_dim // 2)
            theta: 基底周波数
        """
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, seq_len):
        """
        入力: seq_len (int)
        出力: (seq_len, dim) - 回転周波数
        """
        t = torch.arange(seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        return freqs


# ============================================
# Patch Embedding (Conv3D)
# ============================================

class PatchEmbed(nn.Module):
    """
    画像/動画のパッチ埋め込み (Conv3D)

    3D畳み込みで (T, H, W) パッチを特徴ベクトルに変換。

    ※ Qwen2.5-Omni と同一構造。
      SigLIP2 はもともと2Dだが、Qwen3-VL で動画対応のため
      3D Conv に拡張されている (temporal_patch_size=2)。
    """

    def __init__(
        self,
        patch_size=14,
        temporal_patch_size=2,
        in_channels=3,
        embed_dim=1152,
    ):
        """
        パラメータ:
            patch_size: 空間パッチサイズ (14×14 ピクセル)
            temporal_patch_size: 時間パッチサイズ (2フレーム → 1時間トークン)
            in_channels: 入力チャンネル数 (3 = RGB)
            embed_dim: 出力埋め込み次元
                ※ SigLIP2-So400m では 1152 (Qwen2.5-Omni の ViT は 1024)
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

    def forward(self, hidden_states):
        """
        入力:
            hidden_states: (N_patches, patch_dim)
                N_patches: 全パッチ数
                patch_dim: 3 × temporal_patch_size × patch_size²
                         = 3 × 2 × 14 × 14 = 1176

        出力:
            embeddings: (N_patches, embed_dim)

        ※ Qwen2.5-Omni と同一のインターフェース。
          embed_dim が 1024 → 1152 に変更 (SigLIP2-So400m)。
        """
        target_dtype = self.proj.weight.dtype
        hidden_states = hidden_states.to(dtype=target_dtype)

        N = hidden_states.shape[0]
        hidden_states = hidden_states.reshape(
            N, 3, self.temporal_patch_size, self.patch_size, self.patch_size
        )
        hidden_states = self.proj(hidden_states.unsqueeze(0))
        hidden_states = hidden_states.squeeze(0).flatten(1)
        # hidden_states: (N_patches, embed_dim)

        return hidden_states


# ============================================
# ViT Block (ウィンドウ/フルアテンション)
# ============================================

class ViTBlock(nn.Module):
    """
    Vision Transformer ブロック

    Pre-Norm + Self-Attention + MLP
    ウィンドウアテンション (局所) またはフルアテンション (大域) を使用。

    ※ Qwen2.5-Omni と同一構造だが、SigLIP2 ベースのため
      hidden_size が 1024 → 1152、num_heads が 16 → 16 に変更。
      MLP も SwiGLU ベースになっている可能性がある。
    """

    def __init__(self, hidden_size=1152, num_heads=16, mlp_ratio=4.0):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        # Pre-Norm
        self.norm1 = nn.LayerNorm(hidden_size, eps=1e-6)
        self.norm2 = nn.LayerNorm(hidden_size, eps=1e-6)

        # Self-Attention
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            batch_first=False,
        )

        # MLP
        # ※ SigLIP2 は SwiGLU を使用する可能性があるが、
        #   ここでは簡略化のため GELU で表現
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, hidden_size),
        )

    def forward(self, hidden_states, cu_seqlens, rotary_pos_emb):
        """
        入力:
            hidden_states: (total_tokens, hidden_size) - パック済みトークン
            cu_seqlens:    (num_seqs + 1,) - 累積シーケンス長
                ウィンドウアテンション時: ウィンドウ境界
                フルアテンション時: 画像/動画境界
            rotary_pos_emb: (total_tokens, head_dim) - 2D RoPE

        出力:
            hidden_states: (total_tokens, hidden_size)

        ※ 実モデルでは flash_attn_varlen_func + RoPE を使用
        """
        # Self-Attention
        residual = hidden_states
        hidden_states = self.norm1(hidden_states)
        hidden_states_3d = hidden_states.unsqueeze(1)
        attn_out, _ = self.attn(
            hidden_states_3d, hidden_states_3d, hidden_states_3d
        )
        hidden_states = attn_out.squeeze(1)
        hidden_states = residual + hidden_states

        # MLP
        residual = hidden_states
        hidden_states = self.norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


# ============================================
# PatchMerger (空間パッチ統合)
# ============================================

class PatchMerger(nn.Module):
    """
    隣接パッチの統合 (spatial_merge_size=2)

    2×2 = 4パッチを1トークンに統合し、トークン数を1/4に削減。

    ※ Qwen2.5-Omni と完全に同一の仕組み。
      context_dim が 1024 → 1152 に変更された点のみ異なる。
      hidden_size = context_dim × spatial_merge_size² = 1152 × 4 = 4608
    """

    def __init__(self, dim, context_dim, spatial_merge_size=2):
        """
        パラメータ:
            dim: 出力次元 (Thinker LLM の入力次元に合わせる)
            context_dim: 入力次元 (= ViT の hidden_size, 1152)
            spatial_merge_size: 統合サイズ (2 → 2×2=4パッチを1トークンに)
        """
        super().__init__()
        self.spatial_merge_size = spatial_merge_size
        self.hidden_size = context_dim * (spatial_merge_size ** 2)
        # hidden_size = 1152 * 4 = 4608

        self.ln_q = nn.LayerNorm(context_dim, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, dim),
        )
        # 4608 → 4608 → GELU → 4608 → dim

    def forward(self, hidden_states):
        """
        入力:
            hidden_states: (N_patches, context_dim)
                N_patches: 全パッチ数 (spatial_merge_size² の倍数)

                パッチはマージ順に並び替え済み:
                [patch_00, patch_01, patch_10, patch_11,  ← 1グループ (2×2)
                 patch_02, patch_03, patch_12, patch_13,  ← 2グループ目
                 ...]

        出力:
            merged: (N_merged, dim)
                N_merged = N_patches // 4
        """
        hidden_states = self.ln_q(hidden_states)
        # (N_patches, context_dim)

        # 4パッチずつグループ化して結合
        # (N_patches, 1152) → (N_merged, 4, 1152) → (N_merged, 4608)
        hidden_states = hidden_states.view(-1, self.hidden_size)

        hidden_states = self.mlp(hidden_states)
        # (N_merged, dim)

        return hidden_states


# ============================================
# Vision Encoder (完全モデル)
# ============================================

class VisionEncoder(nn.Module):
    """
    Qwen3-Omni Vision Encoder (SigLIP2-So400m ベース)

    Qwen3-VL と同一の ViT エンコーダ (~540M パラメータ)
    SigLIP2-So400m (Tschannen et al., 2025) で初期化。

    アーキテクチャ:
        PatchEmbed (patch_size=14, temporal_patch_size=2)
        → 2D RoPE 位置エンコーディング
        → ViTブロック × depth (ウィンドウ + フルアテンション混合)
        → PatchMerger (2×2=4パッチ → 1トークン)
        → 出力特徴

    vs Qwen2.5-Omni:
        - hidden_size: 1024 → 1152 (SigLIP2-So400m)
        - パラメータ: ~675M → ~540M
        - 初期化: Qwen2.5-VL → SigLIP2 事前学習 (Qwen3-VL 経由)
        - それ以外の構造 (PatchMerger, ウィンドウアテンション等) は同一

    ウィンドウアテンション:
        - 大部分のレイヤー: ウィンドウ内のみアテンション (計算効率)
        - fullatt_block_indexes で指定したレイヤー: フルアテンション (大域文脈)
    """

    def __init__(
        self,
        hidden_size=1152,
        depth=27,
        num_heads=16,
        patch_size=14,
        temporal_patch_size=2,
        spatial_merge_size=2,
        in_channels=3,
        mlp_ratio=4.0,
        window_size=112,
        fullatt_block_indexes=None,
        out_hidden_size=None,
    ):
        """
        パラメータ:
            hidden_size: ViT 隠れ次元 (1152, SigLIP2-So400m)
                ※ Qwen2.5-Omni は 1024
            depth: ViT レイヤー数 (27, SigLIP2-So400m)
                ※ Qwen2.5-Omni は 24
            num_heads: アテンションヘッド数 (16)
            patch_size: 空間パッチサイズ (14)
            temporal_patch_size: 時間パッチサイズ (2)
            spatial_merge_size: PatchMerger 統合サイズ (2)
            in_channels: 入力チャンネル (3=RGB)
            mlp_ratio: MLP 拡張比 (4.0)
            window_size: ウィンドウアテンションのウィンドウサイズ (112ピクセル)
            fullatt_block_indexes: フルアテンション層インデックス
            out_hidden_size: PatchMerger 出力次元 (None → hidden_size)
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
            # SigLIP2-So400m の depth=27 に対応したデフォルト
            # ※ 実際の値は公式設定に依存
            fullatt_block_indexes = [8, 17, 26]
        self.fullatt_block_indexes = set(fullatt_block_indexes)

        if out_hidden_size is None:
            out_hidden_size = hidden_size

        # パッチ埋め込み
        self.patch_embed = PatchEmbed(
            patch_size=patch_size,
            temporal_patch_size=temporal_patch_size,
            in_channels=in_channels,
            embed_dim=hidden_size,
        )

        # 2D RoPE
        head_dim = hidden_size // num_heads
        self.rotary_pos_emb = VisionRotaryEmbedding(dim=head_dim // 2)

        # ViT ブロック × depth
        self.blocks = nn.ModuleList([
            ViTBlock(
                hidden_size=hidden_size,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
            )
            for _ in range(depth)
        ])

        # PatchMerger (2×2 → 1)
        self.merger = PatchMerger(
            dim=out_hidden_size,
            context_dim=hidden_size,
            spatial_merge_size=spatial_merge_size,
        )

    def rot_pos_emb(self, grid_thw):
        """
        2D 回転位置埋め込みの計算

        入力:
            grid_thw: (num_images_or_videos, 3) - 各入力の [T, H, W]

        出力:
            rotary_pos_emb: (total_patches, head_dim) - 位置埋め込み

        処理:
            1. (H, W) 位置グリッド作成
            2. spatial_merge_size で並び替え (マージ順)
            3. T回繰り返し (時間方向)
            4. height / width 位置IDを結合 → 回転周波数計算

        ※ Qwen2.5-Omni と完全に同一のアルゴリズム
        """
        all_pos_ids = []

        for t, h, w in grid_thw:
            t, h, w = t.item(), h.item(), w.item()

            hpos_ids = torch.arange(h).unsqueeze(1).expand(h, w)
            wpos_ids = torch.arange(w).unsqueeze(0).expand(h, w)

            merge_size = self.spatial_merge_size
            hpos_ids = hpos_ids.reshape(
                h // merge_size, merge_size, w // merge_size, merge_size
            ).permute(0, 2, 1, 3).flatten()
            wpos_ids = wpos_ids.reshape(
                h // merge_size, merge_size, w // merge_size, merge_size
            ).permute(0, 2, 1, 3).flatten()

            pos_ids = torch.stack([hpos_ids, wpos_ids], dim=-1)
            pos_ids = pos_ids.repeat(t, 1)
            all_pos_ids.append(pos_ids)

        all_pos_ids = torch.cat(all_pos_ids, dim=0)

        max_grid = all_pos_ids.max() + 1
        rotary_pos_emb_full = self.rotary_pos_emb(max_grid)

        rotary_pos_emb_h = rotary_pos_emb_full[all_pos_ids[:, 0]]
        rotary_pos_emb_w = rotary_pos_emb_full[all_pos_ids[:, 1]]
        rotary_pos_emb_out = torch.cat([rotary_pos_emb_h, rotary_pos_emb_w], dim=-1)

        return rotary_pos_emb_out

    def get_window_index(self, grid_thw):
        """
        ウィンドウアテンション用インデックス計算

        入力:
            grid_thw: (num, 3) - 各入力の [T, H, W]

        出力:
            window_index: (total_merged_tokens,) - ウィンドウ順並び替えインデックス
            cu_window_seqlens: (num_windows + 1,) - ウィンドウ境界累積長

        ※ Qwen2.5-Omni と同一のアルゴリズム
        """
        vit_merger_window_size = (
            self.window_size // self.spatial_merge_size // self.patch_size
        )
        # 112 // 2 // 14 = 4

        all_window_indices = []
        cu_seqlens_list = [0]
        offset = 0

        for t, h, w in grid_thw:
            t, h, w = t.item(), h.item(), w.item()

            llm_grid_h = h // self.spatial_merge_size
            llm_grid_w = w // self.spatial_merge_size

            for frame in range(t):
                index_grid = torch.arange(llm_grid_h * llm_grid_w).reshape(
                    llm_grid_h, llm_grid_w
                )

                pad_h = (vit_merger_window_size - llm_grid_h % vit_merger_window_size) % vit_merger_window_size
                pad_w = (vit_merger_window_size - llm_grid_w % vit_merger_window_size) % vit_merger_window_size
                index_grid = F.pad(index_grid, (0, pad_w, 0, pad_h), value=-1)

                padded_h, padded_w = index_grid.shape
                num_win_h = padded_h // vit_merger_window_size
                num_win_w = padded_w // vit_merger_window_size

                index_grid = index_grid.reshape(
                    num_win_h, vit_merger_window_size,
                    num_win_w, vit_merger_window_size
                ).permute(0, 2, 1, 3).reshape(-1)

                valid_mask = index_grid >= 0
                valid_indices = index_grid[valid_mask] + offset
                all_window_indices.append(valid_indices)

                for wh in range(num_win_h):
                    for ww in range(num_win_w):
                        win_size = (
                            min(vit_merger_window_size, llm_grid_h - wh * vit_merger_window_size)
                            * min(vit_merger_window_size, llm_grid_w - ww * vit_merger_window_size)
                        )
                        cu_seqlens_list.append(
                            cu_seqlens_list[-1] + win_size * self.spatial_merge_unit
                        )

                offset += llm_grid_h * llm_grid_w

        window_index = torch.cat(all_window_indices)
        cu_window_seqlens = torch.tensor(cu_seqlens_list, dtype=torch.int32)

        return window_index, cu_window_seqlens

    def forward(self, hidden_states, grid_thw):
        """
        Vision Encoder フォワードパス

        入力:
            hidden_states: (total_patches, patch_dim)
                patch_dim = 3 × 2 × 14 × 14 = 1176
            grid_thw: (num_images_or_videos, 3) - [T, H, W]

        出力:
            features: (total_merged_tokens, out_hidden_size)
                total_merged_tokens = total_patches // 4

        処理フロー:
            1. PatchEmbed: (N, 1176) → (N, 1152)
            2. 2D RoPE 計算
            3. ウィンドウインデックス計算
            4. ウィンドウ順に並び替え
            5. ViTブロック × depth (ウィンドウ/フルアテンション)
            6. PatchMerger: (N, 1152) → (N//4, out_hidden_size)
            7. 元の順序に復元

        ※ Qwen2.5-Omni と同一のフローだが、
          hidden_size=1152 (SigLIP2-So400m) を使用。
        """

        # ========================================
        # Step 1: パッチ埋め込み
        # ========================================
        hidden_states = self.patch_embed(hidden_states)
        # (total_patches, 1152)
        # ※ Qwen2.5-Omni: (total_patches, 1024)

        # ========================================
        # Step 2: 2D RoPE 計算
        # ========================================
        rotary_pos_emb = self.rot_pos_emb(grid_thw)
        # (total_patches, head_dim)

        # ========================================
        # Step 3: ウィンドウインデックス計算
        # ========================================
        window_index, cu_window_seqlens = self.get_window_index(grid_thw)

        # フルアテンション用 cu_seqlens
        cu_seqlens = torch.zeros(grid_thw.shape[0] + 1, dtype=torch.int32)
        for i, (t, h, w) in enumerate(grid_thw):
            cu_seqlens[i + 1] = cu_seqlens[i] + t * h * w

        # ========================================
        # Step 4: ウィンドウ順に並び替え
        # ========================================
        hidden_states = hidden_states.reshape(
            -1, self.spatial_merge_unit, self.hidden_size
        )
        hidden_states = hidden_states[window_index]
        hidden_states = hidden_states.reshape(-1, self.hidden_size)

        rotary_pos_emb = rotary_pos_emb.reshape(
            -1, self.spatial_merge_unit, rotary_pos_emb.shape[-1]
        )[window_index].reshape(-1, rotary_pos_emb.shape[-1])

        # ========================================
        # Step 5: ViTブロック × depth
        # ========================================
        for i, block in enumerate(self.blocks):
            if i in self.fullatt_block_indexes:
                current_cu_seqlens = cu_seqlens
            else:
                current_cu_seqlens = cu_window_seqlens

            hidden_states = block(
                hidden_states=hidden_states,
                cu_seqlens=current_cu_seqlens,
                rotary_pos_emb=rotary_pos_emb,
            )
        # (total_patches, 1152)

        # ========================================
        # Step 6: PatchMerger (2×2 → 1)
        # ========================================
        hidden_states = self.merger(hidden_states)
        # (total_merged_tokens, out_hidden_size)
        # ※ total_merged_tokens = total_patches // 4

        # ========================================
        # Step 7: 元の順序に復元
        # ========================================
        reverse_index = torch.argsort(window_index)
        hidden_states = hidden_states[reverse_index]

        return hidden_states


# ============================================
# 使用例
# ============================================

def example_vision_encoder():
    """
    Qwen3-Omni Vision Encoder の使用例

    実際にモジュールをインスタンス化し、ダミー入力で
    フォワードパスを実行して各ステージの形状を確認する。

    SigLIP2-So400m ベースのため hidden_size=1152 だが、
    ここでは小さいサイズで高速にテストする。
    """

    # --- 縮小版で初期化 ---
    # 実モデル: hidden_size=1152, depth=27, num_heads=16
    hidden_size = 256
    depth = 4
    num_heads = 8
    out_hidden_size = 512  # Thinker LLM 入力次元に合わせる

    encoder = VisionEncoder(
        hidden_size=hidden_size,
        depth=depth,
        num_heads=num_heads,
        patch_size=14,
        temporal_patch_size=2,
        spatial_merge_size=2,
        window_size=112,
        fullatt_block_indexes=[3],  # 最終層のみフルアテンション
        out_hidden_size=out_hidden_size,
    )
    encoder.eval()

    patch_dim = 3 * 2 * 14 * 14  # = 1176

    # ========================================
    # 例1: 単一画像 (504×504)
    # ========================================
    H, W = 504, 504
    H_patches = H // 14  # 36
    W_patches = W // 14  # 36
    T_patches = 1         # 画像: T=1

    N_patches = T_patches * H_patches * W_patches  # 1296
    pixel_values = torch.randn(N_patches, patch_dim)
    grid_thw = torch.tensor([[T_patches, H_patches, W_patches]])

    # --- 各ステージの形状確認 ---

    # パッチ埋め込み
    with torch.no_grad():
        embed_out = encoder.patch_embed(pixel_values)
    assert embed_out.shape == (N_patches, hidden_size), \
        f"PatchEmbed: expected ({N_patches}, {hidden_size}), got {embed_out.shape}"

    # 2D RoPE
    rotary_pos_emb = encoder.rot_pos_emb(grid_thw)
    head_dim = hidden_size // num_heads
    assert rotary_pos_emb.shape == (N_patches, head_dim), \
        f"RoPE: expected ({N_patches}, {head_dim}), got {rotary_pos_emb.shape}"

    # ウィンドウインデックス
    window_index, cu_window_seqlens = encoder.get_window_index(grid_thw)
    N_merged_units = N_patches // (2 * 2)  # 324
    assert window_index.shape[0] == N_merged_units, \
        f"WindowIndex: expected {N_merged_units}, got {window_index.shape[0]}"

    # フルフォワードパス
    with torch.no_grad():
        output = encoder(pixel_values, grid_thw)
    N_merged = N_patches // 4  # 324
    assert output.shape == (N_merged, out_hidden_size), \
        f"Output: expected ({N_merged}, {out_hidden_size}), got {output.shape}"

    print("[Qwen3-Omni Vision Encoder 使用例]")
    print()
    print(f"  バックボーン: SigLIP2-So400m (実モデル hidden=1152, depth=27)")
    print(f"  テスト構成:   hidden={hidden_size}, depth={depth}, heads={num_heads}")
    print()
    print(f"  例1: 単一画像 ({H}x{W})")
    print(f"    入力:           pixel_values  {pixel_values.shape}   (N_patches, patch_dim)")
    print(f"                    grid_thw      {grid_thw.tolist()}")
    print(f"    PatchEmbed:     {embed_out.shape}         (N_patches, hidden_size)")
    print(f"    2D RoPE:        {rotary_pos_emb.shape}         (N_patches, head_dim)")
    print(f"    ViTブロック x{depth}: ({N_patches}, {hidden_size})")
    print(f"    PatchMerger:    {output.shape}          ({N_merged}, out_hidden_size)")
    print(f"    トークン削減:   {N_patches} → {N_merged} (1/4)")
    print()

    # ========================================
    # 例2: 動画 (8フレーム, 280×280)
    # ========================================
    T_frames = 8
    H_v, W_v = 280, 280
    T_patches_v = T_frames // 2   # 4 (temporal_patch_size=2)
    H_patches_v = H_v // 14      # 20
    W_patches_v = W_v // 14      # 20

    N_patches_v = T_patches_v * H_patches_v * W_patches_v  # 1600
    pixel_values_v = torch.randn(N_patches_v, patch_dim)
    grid_thw_v = torch.tensor([[T_patches_v, H_patches_v, W_patches_v]])

    with torch.no_grad():
        output_v = encoder(pixel_values_v, grid_thw_v)

    N_merged_v = N_patches_v // 4  # 400
    assert output_v.shape == (N_merged_v, out_hidden_size), \
        f"Video output: expected ({N_merged_v}, {out_hidden_size}), got {output_v.shape}"

    print(f"  例2: 動画 ({T_frames}フレーム, {H_v}x{W_v})")
    print(f"    入力:           pixel_values  {pixel_values_v.shape}")
    print(f"                    grid_thw      {grid_thw_v.tolist()}")
    print(f"    パッチ数:       T={T_patches_v} x H={H_patches_v} x W={W_patches_v} = {N_patches_v}")
    print(f"    PatchMerger後:  {output_v.shape}  ({N_merged_v} トークン)")
    print()

    # 動画の時間アライメント情報
    # 各時間パッチ = 2フレーム分。12.5Hz 音声トークンと対応。
    duration_sec = T_frames / 2.0  # 2 FPS なら 4秒 (例示)
    audio_tokens_per_sec = 12.5
    video_temporal_tokens = T_patches_v
    print(f"    [動画-音声 時間アライメント]")
    print(f"    temporal_patch_size=2 → {T_frames}フレーム → {T_patches_v}時間トークン")
    print(f"    各時間トークンの空間トークン: {H_patches_v // 2 * W_patches_v // 2} (PatchMerger後)")
    print(f"    音声は12.5Hz → 動的フレームレートで時間軸を合わせる")
    print()

    # ========================================
    # 例3: バッチ入力 (画像2枚)
    # ========================================
    N_a = 1 * 20 * 20   # 400 (280×280)
    N_b = 1 * 30 * 20   # 600 (420×280)
    N_total = N_a + N_b  # 1000

    pixel_values_batch = torch.randn(N_total, patch_dim)
    grid_thw_batch = torch.tensor([
        [1, 20, 20],  # 画像A: 280×280
        [1, 30, 20],  # 画像B: 420×280
    ])

    with torch.no_grad():
        output_batch = encoder(pixel_values_batch, grid_thw_batch)

    N_merged_batch = N_total // 4  # 250
    assert output_batch.shape == (N_merged_batch, out_hidden_size), \
        f"Batch output: expected ({N_merged_batch}, {out_hidden_size}), got {output_batch.shape}"

    print(f"  例3: バッチ入力 (画像2枚)")
    print(f"    画像A: 280x280 → {N_a} patches → {N_a // 4} merged tokens")
    print(f"    画像B: 420x280 → {N_b} patches → {N_b // 4} merged tokens")
    print(f"    結合入力:       pixel_values {pixel_values_batch.shape}")
    print(f"    結合出力:       {output_batch.shape}  (total={N_merged_batch})")
    print()

    # ウィンドウアテンションの詳細
    window_pixel = 112
    vit_merger_window = window_pixel // 2 // 14  # = 4
    print(f"  [ウィンドウアテンション]")
    print(f"    ウィンドウサイズ: {window_pixel}px → {vit_merger_window}x{vit_merger_window} LLMグリッド")
    print(f"    ウィンドウ内トークン: {vit_merger_window ** 2 * 4} (merge_unit含む)")
    print(f"    フルアテンション層: {sorted(encoder.fullatt_block_indexes)}")
    print(f"    ウィンドウアテンション層: 残り{depth - len(encoder.fullatt_block_indexes)}層")
    print()

    # ========================================
    # vs Qwen2.5-Omni まとめ
    # ========================================
    print(f"  [vs Qwen2.5-Omni 差分まとめ]")
    print(f"    バックボーン:     独自 ViT       → SigLIP2-So400m")
    print(f"    hidden_size:     1024           → 1152")
    print(f"    depth:           24             → 27")
    print(f"    パラメータ数:     ~675M          → ~540M")
    print(f"    初期化:          Qwen2.5-VL     → SigLIP2 事前学習 (Qwen3-VL)")
    print(f"    PatchMerger:     同一 (2x2→1, 1/4削減)")
    print(f"    patch_size:      同一 (14)")
    print(f"    temporal_patch:  同一 (2)")
    print(f"    ウィンドウ+フル:  同一パターン")


if __name__ == "__main__":
    example_vision_encoder()
