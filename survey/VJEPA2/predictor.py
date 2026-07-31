"""
V-JEPA 2.1 Predictor - 簡略化疑似コード
==========================================

Predictorは軽量なViTで、Encoderの出力(コンテキストトークン)から
マスクされたトークンの表現を予測する。

V-JEPA 2.1での拡張:
  - return_all_tokens=True: マスクトークンだけでなくコンテキストトークンも出力
    (Dense Prediction Loss の L_context 計算に使用)
  - Hierarchical入力: エンコーダの複数中間層出力を連結したものを入力として受け取る
    (Deep Self-Supervision)
  - Modality Embedding: 画像/動画識別トークン

対応する公式実装:
  - src/models/predictor.py
  - app/vjepa_2_1/models/predictor.py
"""

import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F


def apply_masks(x: torch.Tensor, masks: list, concat: bool = True):
    """
    マスクインデックスに対応するトークンを取り出す。

    入力:
        x:      (B, N, D)        全パッチトークン
        masks:  list of (B, K)   保持するパッチのインデックス
        concat: Trueの場合 (B*len(masks), K, D) で返す

    出力:
        concat=True:  (B*len(masks), K, D)
        concat=False: list of (B, K, D)
    """
    all_x = []
    for m in masks:
        idx = m.unsqueeze(-1).expand(-1, -1, x.size(-1))  # (B, K, D)
        all_x.append(torch.gather(x, dim=1, index=idx))    # (B, K, D)
    if concat:
        return torch.cat(all_x, dim=0)  # (B*len(masks), K, D)
    return all_x  # list of (B, K, D)


def repeat_interleave_batch(x: torch.Tensor, B: int, repeat: int) -> torch.Tensor:
    """
    バッチ次元でrepeat_interleaveを行う。

    例: B=2, repeat=3, x: (6, N, D)
        出力: (6, N, D) → 各サンプルを3回繰り返した形に並べ替え

    用途: masks_xが複数(len>1)の場合にターゲットトークンを複製して対応させる
    """
    N, D = x.shape[1], x.shape[2]
    x = x.reshape(-1, repeat, N, D)     # (B, repeat, N, D)
    x = x.permute(1, 0, 2, 3)           # (repeat, B, N, D)
    x = x.reshape(-1, N, D)             # (repeat*B, N, D)
    return x


# ============================================================
# Transformer ブロック (Predictor用、形状解説)
# ============================================================

class MultiHeadSelfAttention(nn.Module):
    """
    Multi-Head Self-Attention (Predictor用)

    入力:
        x:    (B, N_total, D_pred)  コンテキスト+マスクトークン
    出力:
        x:    (B, N_total, D_pred)
    """

    def __init__(self, dim: int, num_heads: int, qkv_bias: bool = True,
                 attn_drop: float = 0.0, proj_drop: float = 0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, mask=None, attn_mask=None, **kwargs) -> torch.Tensor:
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, heads, N, N)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, D)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class TransformerBlock(nn.Module):
    """
    Transformer Block (Predictor用)

    入力・出力とも: (B, N_total, D_pred)
    """

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0,
                 qkv_bias: bool = True, drop: float = 0.0, attn_drop: float = 0.0,
                 drop_path: float = 0.0, use_silu: bool = False):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadSelfAttention(dim, num_heads, qkv_bias, attn_drop, drop)
        self.norm2 = nn.LayerNorm(dim)
        hidden_dim = int(dim * mlp_ratio)
        act = nn.SiLU if use_silu else nn.GELU
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim), act(),
            nn.Dropout(drop),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(drop),
        )
        self.drop_path = nn.Identity() if drop_path == 0 else _DropPath(drop_path)

    def forward(self, x: torch.Tensor, mask=None, attn_mask=None, **kwargs) -> torch.Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x), mask=mask, attn_mask=attn_mask))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class _DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob == 0.0:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        rand_tensor = torch.rand(shape, dtype=x.dtype, device=x.device)
        rand_tensor = torch.floor(rand_tensor + keep_prob)
        return x.div(keep_prob) * rand_tensor


# ============================================================
# Multi-level MLP Fusion (V-JEPA 2.1: Deep Self-Supervision用)
# ============================================================

class MultiLevelMLP(nn.Module):
    """
    エンコーダの複数中間層出力を融合するMLP。

    Deep Self-Supervisionでは中間層K個の出力を channel方向に連結し、
    このMLPで元の次元に縮約してからPredictorへ送る。

    入力:
        x: (B, N, D * K)   K個の中間層出力を連結したもの
            D: エンコーダの embed_dim
            K: 出力層数 (通常4)

    出力:
        x: (B, N, D)        次元削減後
    """

    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = None):
        super().__init__()
        hidden_dim = hidden_dim or in_dim
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, D*K)
        x = self.fc1(x)   # (B, N, hidden_dim)
        x = self.act(x)
        x = self.fc2(x)   # (B, N, D)
        return x


# ============================================================
# Vision Transformer Predictor
# ============================================================

class VisionTransformerPredictor(nn.Module):
    """
    V-JEPA 2.1 Predictor

    エンコーダの出力(コンテキストトークン)を受け取り、
    マスクされた位置のトークン表現を予測する軽量ViT。

    V-JEPA 2.1での拡張:
    - return_all_tokens=True: コンテキストトークンの出力も返す (L_context用)
    - Multi-level入力: K層の中間出力を連結→MLP融合後に受け取る

    フォワード処理の流れ:
      1. エンコーダ出力を predictor_embed_dim に線形変換
      2. 可視トークン (context) に位置埋め込みを加算
      3. 学習可能なマスクトークンを生成し位置埋め込みを加算
      4. コンテキスト + マスクトークンを連結
      5. 元の時空間順序にソートし直す
      6. Transformer Blocks で処理
      7. マスクトークン位置のみ(またはすべて)を出力次元に投影

    入力:
        x:        (B*len(masks_x), N_ctx, D_enc)  エンコーダのコンテキスト出力
        masks_x:  list of (B, N_ctx)   コンテキストパッチのインデックス
        masks_y:  list of (B, N_pred)  ターゲットパッチのインデックス

    出力 (return_all_tokens=False):
        (B*len(masks_x), N_pred, D_out)  マスクトークン予測のみ

    出力 (return_all_tokens=True):
        z_pred:    (B*len(masks_x), N_pred, D_out)  マスクトークン予測
        z_context: (B*len(masks_x), N_ctx, D_out)   コンテキストトークン予測
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        num_frames: int = 1,
        tubelet_size: int = 2,
        embed_dim: int = 768,            # エンコーダの出力次元
        predictor_embed_dim: int = 384,  # Predictor内部次元
        out_embed_dim: int = None,       # 出力次元 (省略時 = embed_dim)
        depth: int = 6,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        norm_layer=nn.LayerNorm,
        use_mask_tokens: bool = False,   # 複数種のマスクトークンを使用するか
        num_mask_tokens: int = 2,        # マスクトークンの種類数
        zero_init_mask_tokens: bool = True,
        return_all_tokens: bool = False, # V-JEPA 2.1: コンテキストも出力
        use_rope: bool = False,
        use_silu: bool = False,
        modality_embedding: bool = False,
        # V-JEPA 2.1 Deep Self-Supervision: 複数レベル入力を融合するMLP
        levels_encoder: int = 1,  # エンコーダから来る中間層数 (1=単層, 4=4層連結)
    ):
        super().__init__()
        self.return_all_tokens = return_all_tokens
        self.embed_dim = embed_dim
        self.levels_encoder = levels_encoder

        if out_embed_dim is None:
            out_embed_dim = embed_dim

        # ========================================
        # 入力次元変換
        # Deep Self-Supervisionの場合: D_enc * K → predictor_embed_dim
        # 通常の場合:                  D_enc     → predictor_embed_dim
        # ========================================
        actual_input_dim = embed_dim * levels_encoder
        if levels_encoder > 1:
            # 複数レベル入力を融合するMLP (V-JEPA 2.1)
            self.multi_level_fusion = MultiLevelMLP(
                in_dim=actual_input_dim,
                out_dim=embed_dim,
            )
            self.predictor_embed = nn.Linear(embed_dim, predictor_embed_dim)
        else:
            self.multi_level_fusion = None
            self.predictor_embed = nn.Linear(embed_dim, predictor_embed_dim)

        # ========================================
        # マスクトークン (学習可能)
        # マスクされた位置のプレースホルダートークン
        # ========================================
        self.num_mask_tokens = 0
        self.mask_tokens = None
        if use_mask_tokens:
            self.num_mask_tokens = num_mask_tokens
            # 各マスクタイプ用の別々のトークン
            self.mask_tokens = nn.ParameterList([
                nn.Parameter(torch.zeros(1, 1, predictor_embed_dim))
                for _ in range(num_mask_tokens)
            ])
            if not zero_init_mask_tokens:
                for mt in self.mask_tokens:
                    nn.init.trunc_normal_(mt, std=0.02)
        else:
            # use_mask_tokens=False の場合はゼロ初期化の単一マスクトークン
            self.mask_tokens = nn.ParameterList([
                nn.Parameter(torch.zeros(1, 1, predictor_embed_dim))
            ])
            self.num_mask_tokens = 1

        # ========================================
        # 位置埋め込み (Predictor用)
        # エンコーダと同じ正弦波位置埋め込みを使用
        # ========================================
        if type(img_size) is int:
            img_size = (img_size, img_size)
        self.is_video = num_frames > 1
        self.num_frames = num_frames
        self.tubelet_size = tubelet_size
        self.patch_size = patch_size
        self.use_rope = use_rope

        if self.is_video:
            self.num_patches = (num_frames // tubelet_size) * \
                               (img_size[0] // patch_size) * (img_size[1] // patch_size)
        else:
            self.num_patches = (img_size[0] // patch_size) * (img_size[1] // patch_size)

        if not use_rope:
            self.predictor_pos_embed = nn.Parameter(
                torch.zeros(1, self.num_patches, predictor_embed_dim),
                requires_grad=False,
            )
            # sin-cos pos embedで初期化 (ここでは簡略化のためゼロのまま)
        else:
            self.predictor_pos_embed = None

        # ========================================
        # Modality Embedding (V-JEPA 2.1)
        # ========================================
        self.modality_embedding = modality_embedding
        if modality_embedding:
            self.img_embed = nn.Parameter(torch.zeros(1, 1, predictor_embed_dim))
            self.vid_embed = nn.Parameter(torch.zeros(1, 1, predictor_embed_dim))

        # ========================================
        # Transformer Blocks
        # ========================================
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.predictor_blocks = nn.ModuleList([
            TransformerBlock(
                dim=predictor_embed_dim, num_heads=num_heads,
                mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[i], use_silu=use_silu,
            )
            for i in range(depth)
        ])

        # ========================================
        # 出力投影
        # ========================================
        self.predictor_norm = norm_layer(predictor_embed_dim)
        self.predictor_proj = nn.Linear(predictor_embed_dim, out_embed_dim)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.zeros_(m.bias)
                nn.init.ones_(m.weight)
        for layer_id, layer in enumerate(self.predictor_blocks):
            layer.attn.proj.weight.data.div_(math.sqrt(2.0 * (layer_id + 1)))
            layer.mlp[-3].weight.data.div_(math.sqrt(2.0 * (layer_id + 1)))

    def forward(self, x: torch.Tensor, masks_x: list, masks_y: list,
                mask_index: int = 0, mod: str = "video"):
        """
        入力:
            x:        (B*len(masks_x), N_ctx, D_enc) または
                      (B*len(masks_x), N_ctx, D_enc*K) [K層連結時]
            masks_x:  list of (B, N_ctx)   コンテキストパッチインデックス
            masks_y:  list of (B, N_pred)  ターゲットパッチインデックス
            mask_index: 使用するマスクトークンのインデックス
            mod:      "video" or "image"

        出力 (return_all_tokens=False):
            z_pred: (B*len(masks_x), N_pred, D_out)

        出力 (return_all_tokens=True):
            z_pred:    (B*len(masks_x), N_pred, D_out)
            z_context: (B*len(masks_x), N_ctx, D_out)
        """
        if not isinstance(masks_x, list):
            masks_x = [masks_x]
        if not isinstance(masks_y, list):
            masks_y = [masks_y]

        B = len(x) // len(masks_x)  # バッチサイズ

        # ========================================
        # Step 1: 多レベル入力の融合 (V-JEPA 2.1)
        # D_enc*K → D_enc → predictor_embed_dim
        # ========================================
        if self.multi_level_fusion is not None:
            x = self.multi_level_fusion(x)  # (B*len(masks_x), N_ctx, D_enc)

        # ========================================
        # Step 2: エンコーダ次元 → Predictor次元 に変換
        # ========================================
        x = self.predictor_embed(x)         # (B*len(masks_x), N_ctx, D_pred)
        _, N_ctx, D_pred = x.shape

        # ========================================
        # Step 3: コンテキストトークンに位置埋め込みを加算
        # ========================================
        if not self.use_rope and self.predictor_pos_embed is not None:
            # 各バッチにpos_embedを適用: コンテキスト位置の埋め込みのみ抽出
            x_pos_embed = self.predictor_pos_embed.expand(B, -1, -1)  # (B, N_total, D_pred)
            x += apply_masks(x_pos_embed, masks_x)                     # (B*len(masks_x), N_ctx, D_pred)

        # ========================================
        # Step 4: ターゲット位置のマスクトークンを生成
        # マスクトークン = 学習可能なベクトル + 位置埋め込み
        # ========================================
        mask_idx = mask_index % self.num_mask_tokens
        # (1, 1, D_pred) → (B, N_total, D_pred) → マスク位置のみ抽出
        pred_tokens = self.mask_tokens[mask_idx].expand(B, self.num_patches, -1)
        pred_tokens = apply_masks(pred_tokens, masks_y)  # (B*len(masks_y), N_pred, D_pred)

        if not self.use_rope and self.predictor_pos_embed is not None:
            # マスクトークンにも位置埋め込みを加算
            pos_embs = self.predictor_pos_embed.expand(B, -1, -1)
            pos_embs = apply_masks(pos_embs, masks_y)             # (B*len(masks_y), N_pred, D_pred)
            pos_embs = repeat_interleave_batch(pos_embs, B, repeat=len(masks_x))
            pred_tokens = pred_tokens + pos_embs

        # ========================================
        # Step 5: コンテキストトークンを len(masks_x) 回複製し
        #         マスクトークンと連結
        # ========================================
        # masks_xが複数の場合 (多様なマスク戦略) に対応
        x = x.repeat(len(masks_x), 1, 1)               # (B*len(masks_x)^2, N_ctx, D_pred)
        x = torch.cat([x, pred_tokens], dim=1)          # (B*..., N_ctx+N_pred, D_pred)

        # ========================================
        # Step 6: 元の時空間順序にソートし直す
        # コンテキスト+ターゲットのインデックスを合わせてソート
        # RoPEでの正しい位置符号化のために必要
        # ========================================
        masks_x_cat = torch.cat(masks_x, dim=0)         # (B*len(masks_x), N_ctx)
        masks_y_cat = torch.cat(masks_y, dim=0)         # (B*len(masks_y), N_pred)
        masks_all = torch.cat([masks_x_cat, masks_y_cat], dim=1)  # (B*..., N_ctx+N_pred)

        # インデックス値でソート
        argsort = torch.argsort(masks_all, dim=1)         # (B*..., N_ctx+N_pred)
        masks_sorted = torch.stack(
            [masks_all[i, row] for i, row in enumerate(argsort)], dim=0
        )
        x = torch.stack([x[i, row] for i, row in enumerate(argsort)], dim=0)
        # x: (B*..., N_ctx+N_pred, D_pred)  ← 元の時空間順に並んだ完全シーケンス

        # ========================================
        # Step 7: Modality Embedding (V-JEPA 2.1)
        # ========================================
        if self.modality_embedding:
            if mod == "image":
                x = x + self.img_embed
            else:
                x = x + self.vid_embed

        # ========================================
        # Step 8: Transformer Blocks で処理
        # ========================================
        for blk in self.predictor_blocks:
            x = blk(x, mask=masks_sorted)  # (B*..., N_ctx+N_pred, D_pred)

        x = self.predictor_norm(x)  # (B*..., N_ctx+N_pred, D_pred)

        # ========================================
        # Step 9: 出力の抽出
        # ========================================
        if self.return_all_tokens:
            # V-JEPA 2.1: コンテキストとマスクの両方を返す
            # ソート前の順序に戻す
            reverse_argsort = torch.argsort(argsort, dim=1)
            x_all = torch.stack([x[i, row] for i, row in enumerate(reverse_argsort)], dim=0)
            # x_all: (B*..., N_ctx+N_pred, D_pred)

            # コンテキストとターゲットに分割 (元のソート前位置を使って判定)
            # [0:N_ctx] がコンテキスト, [N_ctx:] がターゲット (ソート前の順序)
            z_context_raw = x_all[:, :N_ctx, :]   # (B*..., N_ctx, D_pred)
            z_pred_raw    = x_all[:, N_ctx:, :]   # (B*..., N_pred, D_pred)

            z_context = self.predictor_proj(z_context_raw)  # (B*..., N_ctx, D_out)
            z_pred    = self.predictor_proj(z_pred_raw)     # (B*..., N_pred, D_out)
            return z_pred, z_context

        else:
            # V-JEPA 2 (元): マスクトークンのみ返す
            reverse_argsort = torch.argsort(argsort, dim=1)
            x_all = torch.stack([x[i, row] for i, row in enumerate(reverse_argsort)], dim=0)
            z_pred = x_all[:, N_ctx:, :]                     # (B*..., N_pred, D_pred)
            z_pred = self.predictor_proj(z_pred)             # (B*..., N_pred, D_out)
            return z_pred


# ============================================================
# 動作確認 example
# ============================================================

if __name__ == "__main__":
    import torch

    print("=" * 60)
    print("V-JEPA 2.1 Predictor 動作確認")
    print("=" * 60)

    # ----------------------------------------
    # 基本設定
    # ----------------------------------------
    B = 2
    N_total = 2048  # 総パッチ数 (8*16*16)
    D_enc = 1024    # エンコーダ次元 (ViT-L)
    D_pred = 384    # Predictor内部次元
    N_ctx = 700     # コンテキストパッチ数
    N_pred = 1300   # ターゲットパッチ数 (N_total - N_ctx = 1348 以下)

    # ----------------------------------------
    # V-JEPA 2 スタイル (masked tokensのみ予測)
    # ----------------------------------------
    print("\n[1] V-JEPA 2: masked tokensのみ予測")
    predictor_v2 = VisionTransformerPredictor(
        img_size=256,
        patch_size=16,
        num_frames=16,
        tubelet_size=2,
        embed_dim=D_enc,
        predictor_embed_dim=D_pred,
        out_embed_dim=D_enc,
        depth=6,
        num_heads=8,
        use_mask_tokens=True,
        num_mask_tokens=1,
        return_all_tokens=False,
    )
    predictor_v2.eval()

    # エンコーダ出力 (コンテキストトークン)
    z_enc = torch.randn(B, N_ctx, D_enc)

    # マスクインデックス
    perm = torch.randperm(N_total)
    ctx_idx = perm[:N_ctx]
    pred_idx = perm[N_ctx:N_ctx + N_pred]
    masks_x = [ctx_idx.unsqueeze(0).expand(B, -1)]    # list of (B, N_ctx)
    masks_y = [pred_idx.unsqueeze(0).expand(B, -1)]   # list of (B, N_pred)

    z_pred = predictor_v2(z_enc, masks_x, masks_y)
    print(f"  エンコーダ出力: {z_enc.shape}")
    print(f"  masks_x[0]:   {masks_x[0].shape}")
    print(f"  masks_y[0]:   {masks_y[0].shape}")
    print(f"  z_pred:        {z_pred.shape}")
    assert z_pred.shape == (B, N_pred, D_enc), f"Expected ({B}, {N_pred}, {D_enc})"

    # ----------------------------------------
    # V-JEPA 2.1 スタイル (コンテキストも予測、Dense Loss)
    # ----------------------------------------
    print("\n[2] V-JEPA 2.1: コンテキスト+マスクトークン両方を予測")
    predictor_v21 = VisionTransformerPredictor(
        img_size=256,
        patch_size=16,
        num_frames=16,
        tubelet_size=2,
        embed_dim=D_enc,
        predictor_embed_dim=D_pred,
        out_embed_dim=D_enc,
        depth=6,
        num_heads=8,
        use_mask_tokens=True,
        num_mask_tokens=1,
        return_all_tokens=True,   # ← V-JEPA 2.1 の重要フラグ
    )
    predictor_v21.eval()

    z_pred_21, z_context_21 = predictor_v21(z_enc, masks_x, masks_y)
    print(f"  エンコーダ出力: {z_enc.shape}")
    print(f"  z_pred:         {z_pred_21.shape}     ← マスクトークン予測")
    print(f"  z_context:      {z_context_21.shape}  ← コンテキストトークン予測")
    assert z_pred_21.shape == (B, N_pred, D_enc)
    assert z_context_21.shape == (B, N_ctx, D_enc)

    # ----------------------------------------
    # V-JEPA 2.1: Deep Self-Supervision (K=4層連結入力)
    # ----------------------------------------
    print("\n[3] V-JEPA 2.1: Deep Self-Supervision (4レベル連結入力)")
    K = 4  # 中間層数
    predictor_deep = VisionTransformerPredictor(
        img_size=256,
        patch_size=16,
        num_frames=16,
        tubelet_size=2,
        embed_dim=D_enc,
        predictor_embed_dim=D_pred,
        out_embed_dim=D_enc,
        depth=6,
        num_heads=8,
        use_mask_tokens=True,
        num_mask_tokens=1,
        return_all_tokens=True,
        levels_encoder=K,  # ← 4層分の連結入力
    )
    predictor_deep.eval()

    # エンコーダの4層分出力を連結: (B, N_ctx, D*K)
    z_enc_multilevel = torch.randn(B, N_ctx, D_enc * K)
    z_pred_d, z_ctx_d = predictor_deep(z_enc_multilevel, masks_x, masks_y)
    print(f"  多レベルエンコーダ出力: {z_enc_multilevel.shape}")
    print(f"  z_pred:                 {z_pred_d.shape}")
    print(f"  z_context:              {z_ctx_d.shape}")
    assert z_pred_d.shape == (B, N_pred, D_enc)
    assert z_ctx_d.shape == (B, N_ctx, D_enc)

    # ----------------------------------------
    # 複数マスク (masks_x = 2種類) のケース
    # ----------------------------------------
    print("\n[4] 複数マスク (len(masks_x)=2)")
    predictor_multi = VisionTransformerPredictor(
        img_size=256, patch_size=16, num_frames=16, tubelet_size=2,
        embed_dim=D_enc, predictor_embed_dim=D_pred, out_embed_dim=D_enc,
        depth=4, num_heads=8,
        use_mask_tokens=True, num_mask_tokens=2, return_all_tokens=False,
    )
    predictor_multi.eval()

    # マスク2種類
    perm2 = torch.randperm(N_total)
    masks_x2 = [
        perm[:N_ctx].unsqueeze(0).expand(B, -1),
        perm2[:N_ctx].unsqueeze(0).expand(B, -1),
    ]
    masks_y2 = [
        perm[N_ctx:N_ctx + N_pred].unsqueeze(0).expand(B, -1),
        perm2[N_ctx:N_ctx + N_pred].unsqueeze(0).expand(B, -1),
    ]
    # エンコーダ出力も2バッチ分: B * len(masks_x) = B * 2
    z_enc2 = torch.randn(B * 2, N_ctx, D_enc)

    z_pred_multi = predictor_multi(z_enc2, masks_x2, masks_y2, mask_index=0)
    # 出力: (B*len(masks_x)^2, N_pred, D_out) → (B*2*2, ...) となりうる
    print(f"  エンコーダ出力(2×B): {z_enc2.shape}")
    print(f"  z_pred:              {z_pred_multi.shape}")

    print("\n全テスト通過!")
