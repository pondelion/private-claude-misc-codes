"""
InternVL3.5 モデルアーキテクチャ
==================================

このファイルは InternVL3.5 の核心部分である以下3コンポーネントを実装しています:
  1. InternViT (Vision Encoder) - InternViT-300M / InternViT-6B
  2. MLP Projector (mlp1) - ViT特徴をLLM次元に射影
  3. InternVLChatModel - ViT + MLP + LLM を統合したマルチモーダルモデル

公式実装:
  internvl_chat/internvl/model/internvl_chat/modeling_intern_vit.py
  internvl_chat/internvl/model/internvl_chat/modeling_internvl_chat.py

============================================================
テンソル形状記法
============================================================
  B    : バッチサイズ
  P    : パッチ数 (1枚の画像をタイルに分割したもの, Dynamic High Resolution)
  N    : 系列長 (テキスト + 画像トークン)
  S    : ViT系列長 = (image_size/patch_size)^2 + 1 (CLSトークン含む)
  S'   : CLS除去後の系列長 = S-1 = 1024 (448px画像 / 14px patch)
  D_v  : ViT hidden size (InternViT-6B: 3200, InternViT-300M: 1024)
  D_l  : LLM hidden size (モデルによって異なる, 例: Qwen3-8B: 4096)
  H    : アテンションヘッド数
  V    : 語彙サイズ
"""

import math
import warnings
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import (BaseModelOutput,
                                           BaseModelOutputWithPooling,
                                           CausalLMOutputWithPast)


# ============================================================
# 1. 正規化レイヤー
# ============================================================

class InternRMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization

    通常の LayerNorm と異なり平均を引かず、RMS スケーリングのみを行う。
    InternViT では QK Normalization に使用し、学習安定性を向上させる。

    入力形状: (*, hidden_size)
    出力形状: (*, hidden_size)  ← 同形状
    """
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        # float32 で計算 (数値安定性のため)
        hidden_states = hidden_states.to(torch.float32)
        # 各トークンの RMS を計算
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        # 正規化
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


# ============================================================
# 2. ViT Embedding (パッチ埋め込み)
# ============================================================

class InternVisionEmbeddings(nn.Module):
    """
    画像をパッチに分割してトークン埋め込みに変換する。

    処理フロー:
      pixel_values (B*P, 3, 448, 448)
        ↓ Conv2d(in=3, out=D_v, kernel=14, stride=14)
      patch_embeds (B*P, D_v, 32, 32)
        ↓ flatten + transpose
      patch_embeds (B*P, 1024, D_v)
        ↓ CLS token結合
      embeddings (B*P, 1025, D_v)
        ↓ 位置埋め込み加算 (bicubic補間で任意サイズに対応)
      embeddings (B*P, 1025, D_v)
    """
    def __init__(self, config):
        super().__init__()
        self.embed_dim = config.hidden_size    # D_v
        self.image_size = config.image_size    # 448
        self.patch_size = config.patch_size    # 14

        # [CLS] トークン: (1, 1, D_v)
        self.class_embedding = nn.Parameter(
            torch.randn(1, 1, self.embed_dim)
        )

        # パッチ埋め込み: Conv2d で分割＆線形変換を同時実行
        # (B*P, 3, 448, 448) → (B*P, D_v, 32, 32)
        self.patch_embedding = nn.Conv2d(
            in_channels=3,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )

        # num_patches = (448/14)^2 = 1024
        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches + 1  # +1 for [CLS]

        # 学習可能な位置埋め込み: (1, 1025, D_v)
        self.position_embedding = nn.Parameter(
            torch.randn(1, self.num_positions, self.embed_dim)
        )

    def _get_pos_embed(self, pos_embed: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """
        入力サイズに合わせて位置埋め込みを bicubic 補間でリサイズ。

        pos_embed: (1, S_base^2, D_v)  S_base = image_size / patch_size = 32
        H, W     : 現在の高さ・幅 (パッチ単位)
        返値     : (1, H*W, D_v)
        """
        target_dtype = pos_embed.dtype
        # (1, D_v, S_base, S_base) に変換
        S_base = self.image_size // self.patch_size
        pos_embed = pos_embed.float().reshape(1, S_base, S_base, -1).permute(0, 3, 1, 2)
        # bicubic 補間で (H, W) にリサイズ
        pos_embed = F.interpolate(
            pos_embed, size=(H, W), mode='bicubic', align_corners=False
        )
        # (1, H*W, D_v) に変換して返す
        pos_embed = pos_embed.reshape(1, -1, H * W).permute(0, 2, 1).to(target_dtype)
        return pos_embed

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        入力: pixel_values  (B*P, 3, H_img, W_img)  ※通常 H_img=W_img=448
        出力: embeddings    (B*P, S_v, D_v)
              S_v = (H_img/patch_size) * (W_img/patch_size) + 1
                  = 32*32 + 1 = 1025  (448px の場合)
        """
        target_dtype = self.patch_embedding.weight.dtype

        # ステップ1: パッチ埋め込み
        # (B*P, 3, 448, 448) → (B*P, D_v, 32, 32)
        patch_embeds = self.patch_embedding(pixel_values.to(target_dtype))
        batch_size, _, height, width = patch_embeds.shape  # height=width=32

        # (B*P, D_v, 32, 32) → (B*P, 1024, D_v)
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)

        # ステップ2: CLS トークン結合
        # (1, 1, D_v) → (B*P, 1, D_v)
        class_embeds = self.class_embedding.expand(batch_size, 1, -1).to(target_dtype)
        # (B*P, 1025, D_v)
        embeddings = torch.cat([class_embeds, patch_embeds], dim=1)

        # ステップ3: 位置埋め込み加算 (CLS: そのまま / パッチ: bicubic補間)
        position_embedding = torch.cat([
            self.position_embedding[:, :1, :],                        # CLS 位置埋め込み
            self._get_pos_embed(self.position_embedding[:, 1:, :], height, width)  # パッチ位置埋め込み
        ], dim=1)  # (1, 1025, D_v)

        embeddings = embeddings + position_embedding.to(target_dtype)
        return embeddings  # (B*P, 1025, D_v)


# ============================================================
# 3. InternAttention (Multi-Head Self-Attention with QK Norm)
# ============================================================

class InternAttention(nn.Module):
    """
    QK Normalization 付き Multi-Head Self-Attention。

    QK Normalization: Q と K に RMSNorm を適用することで、
    大規模 ViT 学習時の attention score 発散を防ぎ安定性を高める。

    入力形状: hidden_states (B*P, S_v, D_v)
    出力形状:               (B*P, S_v, D_v)
    """
    def __init__(self, config):
        super().__init__()
        self.embed_dim = config.hidden_size     # D_v
        self.num_heads = config.num_attention_heads  # H_v (e.g. 25 for 6B)
        self.head_dim = self.embed_dim // self.num_heads  # D_v / H_v
        self.scale = self.head_dim ** -0.5

        # QKV を一括で計算
        self.qkv = nn.Linear(self.embed_dim, 3 * self.embed_dim, bias=config.qkv_bias)
        self.attn_drop = nn.Dropout(config.attention_dropout)
        self.proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.proj_drop = nn.Dropout(config.dropout)

        self.qk_normalization = config.qk_normalization
        if self.qk_normalization:
            # Q/K に別々の RMSNorm を適用 (head次元に展開された状態で)
            self.q_norm = InternRMSNorm(self.embed_dim, eps=config.layer_norm_eps)
            self.k_norm = InternRMSNorm(self.embed_dim, eps=config.layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        入力: hidden_states  (B*P, S_v, D_v)
        出力:                (B*P, S_v, D_v)
        """
        B, N, C = hidden_states.shape  # B=バッチ*パッチ数, N=S_v, C=D_v

        # ステップ1: QKV 計算
        # (B*P, S_v, D_v) → (B*P, S_v, 3*D_v) → (B*P, S_v, 3, H_v, D_head)
        qkv = self.qkv(hidden_states).reshape(B, N, 3, self.num_heads, self.head_dim)
        # → (3, B*P, H_v, S_v, D_head)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # それぞれ (B*P, H_v, S_v, D_head)

        # ステップ2: QK Normalization (オプション)
        if self.qk_normalization:
            B_, H_, N_, D_ = q.shape
            # (B*P, H_v, S_v, D_head) → (B*P, S_v, D_v) → RMSNorm → 元形状に戻す
            q = self.q_norm(q.transpose(1, 2).flatten(-2, -1)).view(B_, N_, H_, D_).transpose(1, 2)
            k = self.k_norm(k.transpose(1, 2).flatten(-2, -1)).view(B_, N_, H_, D_).transpose(1, 2)

        # ステップ3: Scaled Dot-Product Attention
        # (B*P, H_v, S_v, D_head) × (B*P, H_v, D_head, S_v) → (B*P, H_v, S_v, S_v)
        attn = (q * self.scale) @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # (B*P, H_v, S_v, S_v) × (B*P, H_v, S_v, D_head) → (B*P, H_v, S_v, D_head)
        # → (B*P, S_v, H_v, D_head) → (B*P, S_v, D_v)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)

        # ステップ4: 出力射影
        x = self.proj(x)      # (B*P, S_v, D_v)
        x = self.proj_drop(x)
        return x              # (B*P, S_v, D_v)


# ============================================================
# 4. InternMLP (Feed-Forward Network)
# ============================================================

class InternMLP(nn.Module):
    """
    ViT 内の Feed-Forward Network (FFN)。
    2 層 Linear + GELU 活性化。

    入力形状: (B*P, S_v, D_v)
    出力形状: (B*P, S_v, D_v)
    """
    def __init__(self, config):
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.act = nn.GELU()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # (B*P, S_v, D_v) → (B*P, S_v, D_int) → (B*P, S_v, D_v)
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


# ============================================================
# 5. InternVisionEncoderLayer (1 Transformer Block)
# ============================================================

class InternVisionEncoderLayer(nn.Module):
    """
    ViT の 1 層 Transformer ブロック。

    Pre-Norm 構造 + Layer Scaling (ls1, ls2) + Stochastic Depth。

    Layer Scaling: 学習初期に各残差接続をほぼゼロに近づけ、
                   深いネットワークの学習安定性を向上させる技術。

    入力形状: (B*P, S_v, D_v)
    出力形状: (B*P, S_v, D_v)
    """
    def __init__(self, config, drop_path_rate: float = 0.0):
        super().__init__()
        self.embed_dim = config.hidden_size

        # Pre-Norm 正規化レイヤー
        norm_cls = InternRMSNorm if config.norm_type == 'rms_norm' else nn.LayerNorm
        self.norm1 = norm_cls(self.embed_dim, eps=config.layer_norm_eps)
        self.norm2 = norm_cls(self.embed_dim, eps=config.layer_norm_eps)

        # メインブロック
        self.attn = InternAttention(config)
        self.mlp = InternMLP(config)

        # Layer Scaling: 学習可能なスカラー (初期値: config.initializer_factor ≈ 0.1)
        self.ls1 = nn.Parameter(config.initializer_factor * torch.ones(self.embed_dim))
        self.ls2 = nn.Parameter(config.initializer_factor * torch.ones(self.embed_dim))

        # Stochastic Depth (DropPath)
        from timm.models.layers import DropPath
        self.drop_path1 = DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()
        self.drop_path2 = DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        入力: (B*P, S_v, D_v)
        出力: (B*P, S_v, D_v)
        """
        # 残差接続 + Layer Scaling + Stochastic Depth
        # hidden_states = hidden_states + drop_path(attn(norm1(hidden_states)) * ls1)
        hidden_states = hidden_states + self.drop_path1(
            self.attn(self.norm1(hidden_states).to(hidden_states.dtype)) * self.ls1
        )
        # hidden_states = hidden_states + drop_path(mlp(norm2(hidden_states)) * ls2)
        hidden_states = hidden_states + self.drop_path2(
            self.mlp(self.norm2(hidden_states).to(hidden_states.dtype)) * self.ls2
        )
        return hidden_states  # (B*P, S_v, D_v)


# ============================================================
# 6. InternVisionEncoder (Transformer スタック)
# ============================================================

class InternVisionEncoder(nn.Module):
    """
    InternVisionEncoderLayer を num_hidden_layers 層積み重ねた ViT エンコーダー。
    Stochastic Depth を線形スケジュールで適用。

    入力形状: inputs_embeds  (B*P, S_v, D_v)
    出力形状: last_hidden_state (B*P, S_v, D_v)
    """
    def __init__(self, config):
        super().__init__()
        # Stochastic Depth の確率を層ごとに線形増加させる
        dpr = [x.item() for x in torch.linspace(0, config.drop_path_rate, config.num_hidden_layers)]
        self.layers = nn.ModuleList([
            InternVisionEncoderLayer(config, drop_path_rate=dpr[i])
            for i in range(config.num_hidden_layers)
        ])
        self.gradient_checkpointing = True  # 勾配チェックポイントで VRAM 削減

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        output_hidden_states: bool = False,
    ) -> BaseModelOutput:
        """
        入力: inputs_embeds  (B*P, S_v, D_v)
        出力: BaseModelOutput
              .last_hidden_state (B*P, S_v, D_v)
              .hidden_states     タプル (各層の出力, output_hidden_states=True の場合)
        """
        hidden_states = inputs_embeds
        all_hidden_states = () if output_hidden_states else None

        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            if self.gradient_checkpointing and self.training:
                hidden_states = torch.utils.checkpoint.checkpoint(layer, hidden_states)
            else:
                hidden_states = layer(hidden_states)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
        )


# ============================================================
# 7. InternVisionModel (完全な ViT)
# ============================================================

class InternVisionModel(nn.Module):
    """
    InternViT-6B / InternViT-300M の完全な Vision Transformer モデル。

    InternViT-6B 主要ハイパーパラメータ:
      image_size=448, patch_size=14
      hidden_size=3200, num_attention_heads=25
      intermediate_size=12800, num_hidden_layers=48
      qk_normalization=True, norm_type='rms_norm'

    入力形状:
      pixel_values  (B*P, 3, H_img, W_img)   ※ H_img=W_img=448
    出力形状:
      last_hidden_state (B*P, S_v, D_v)   S_v = 1025 (CLS含む)
      pooler_output     (B*P, D_v)         CLS トークン
    """
    def __init__(self, config):
        super().__init__()
        self.embeddings = InternVisionEmbeddings(config)
        self.encoder = InternVisionEncoder(config)

    def forward(
        self,
        pixel_values: torch.Tensor,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ) -> BaseModelOutputWithPooling:
        """
        入力: pixel_values  (B*P, 3, 448, 448)
        出力: .last_hidden_state  (B*P, S_v=1025, D_v)
              .pooler_output      (B*P, D_v)           ← CLS token
        """
        # (B*P, 3, 448, 448) → (B*P, 1025, D_v)
        hidden_states = self.embeddings(pixel_values)

        # 全 Transformer 層を通過
        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            output_hidden_states=output_hidden_states,
        )

        last_hidden_state = encoder_outputs.last_hidden_state  # (B*P, 1025, D_v)
        pooled_output = last_hidden_state[:, 0, :]             # (B*P, D_v) ← CLS

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
        )


# ============================================================
# 8. MLP Projector (ViT → LLM 次元変換)
# ============================================================

def build_mlp_projector(vit_hidden_size: int, llm_hidden_size: int, downsample_ratio: float = 0.5) -> nn.Sequential:
    """
    Pixel Shuffle 後の ViT 特徴を LLM 次元に射影する MLP。

    Pixel Shuffle により空間方向を (1/downsample_ratio)^2 倍に圧縮するため、
    チャンネル次元は (1/downsample_ratio)^2 = 4 倍になる (downsample_ratio=0.5の場合)。

    入力形状:
      (B*P*256, D_v * (1/downsample_ratio)^2)
        = (B*P*256, 3200*4) = (B*P*256, 12800)  ← InternViT-6B の場合
    出力形状:
      (B*P*256, D_l)
    """
    compress_factor = int(1 / downsample_ratio)  # = 2
    in_dim = vit_hidden_size * compress_factor ** 2  # 3200 * 4 = 12800

    return nn.Sequential(
        nn.LayerNorm(in_dim),
        nn.Linear(in_dim, llm_hidden_size),
        nn.GELU(),
        nn.Linear(llm_hidden_size, llm_hidden_size),
    )


# ============================================================
# 9. InternVLChatModel (統合マルチモーダルモデル)
# ============================================================

class InternVLChatModel(nn.Module):
    """
    InternVL3.5 の統合モデル。

    ViT-MLP-LLM パラダイム:
      1. InternViT で画像特徴を抽出
      2. Pixel Shuffle でトークン数を圧縮
      3. MLP Projector で LLM 次元に変換
      4. テキスト列の <IMG_CONTEXT> トークンを視覚特徴で置換
      5. LLM に入力して次トークンを予測

    IMG_CONTEXT_TOKEN の役割:
      テキスト列中の <IMG_CONTEXT> (id: img_context_token_id) は
      視覚特徴を挿入するプレースホルダー。
      forward() 内でこのトークンの埋め込みを vit_embeds で上書きする。

    入力形状:
      pixel_values  (B*P, 3, 448, 448)  ※ B*P は全パッチ数
      input_ids     (B, N)              ※ N: テキスト+IMG_CONTEXTの合計長
      attention_mask(B, N)
      image_flags   (B*P, 1)            ※ 1=有効な画像, 0=パディング
      labels        (B, N)              ※ 学習時のみ

    出力形状 (学習時):
      loss   スカラー
      logits (B, N, V)
    """
    def __init__(
        self,
        vision_config,
        llm_config,
        downsample_ratio: float = 0.5,
        ps_version: str = 'v2',
        select_layer: int = -1,
        language_model: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.downsample_ratio = downsample_ratio
        self.ps_version = ps_version
        self.select_layer = select_layer

        # Vision Encoder
        self.vision_model = InternVisionModel(vision_config)

        # MLP Projector
        vit_hidden_size = vision_config.hidden_size
        llm_hidden_size = llm_config.hidden_size
        self.mlp1 = build_mlp_projector(vit_hidden_size, llm_hidden_size, downsample_ratio)

        # Language Model (任意の因果LM)
        self.language_model = language_model  # 実際は Qwen2ForCausalLM 等

        # 1パッチあたりのビジュアルトークン数
        image_size = vision_config.image_size  # 448
        patch_size = vision_config.patch_size  # 14
        self.num_image_token = int((image_size // patch_size) ** 2 * (downsample_ratio ** 2))
        # = (448/14)^2 * 0.5^2 = 1024 * 0.25 = 256

        self.img_context_token_id: Optional[int] = None

    def pixel_shuffle(self, x: torch.Tensor, scale_factor: float = 0.5) -> torch.Tensor:
        """
        ViT 特徴を空間方向で再配置して圧縮するトークン圧縮手法。
        PixelShuffle の逆操作 (Unshuffle) と同様の概念。

        入力形状: (B*P, H_t, W_t, D_v)
                  ※ H_t=W_t=32 (448px/14px), D_v=3200

        変換過程:
          1. (B*P, W_t=32, H_t=32, D_v=3200)  ← view で H と C を変換
          2. (B*P, 32, 16, 6400)               ← H*scale=16, C/scale=6400
          3. (B*P, 16, 32, 6400)               ← permute (H, W 入れ替え)
          4. (B*P, 16, 16, 12800)              ← view で W と C を変換
          5. (B*P, 16, 16, 12800)              ← v2: permute (H, W 入れ替えを戻す)

        出力形状: (B*P, H_t*scale, W_t*scale, D_v/(scale^2))
                = (B*P, 16, 16, 12800)
                → reshape で (B*P, 256, 12800)

        scale_factor=0.5 の場合: トークン数 1024 → 256 に削減 (4倍圧縮)
        """
        n, w, h, c = x.size()
        # ステップ1: H 次元を縮め C 次元を拡大
        x = x.view(n, w, int(h * scale_factor), int(c / scale_factor))
        # ステップ2: H と W を入れ替え
        x = x.permute(0, 2, 1, 3).contiguous()
        # ステップ3: W 次元を縮め C 次元をさらに拡大
        x = x.view(n, int(h * scale_factor), int(w * scale_factor), int(c / (scale_factor ** 2)))
        # ステップ4 (v2 のみ): H と W を入れ替えて正しい向きに戻す
        if self.ps_version == 'v2':
            x = x.permute(0, 2, 1, 3).contiguous()
        return x

    def extract_feature(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        画像テンソルから LLM に渡せる視覚特徴を抽出する。

        入力: pixel_values  (B*P, 3, 448, 448)
        出力: vit_embeds    (B*P, 256, D_l)
              ※ 256 = num_image_token (1パッチあたりの視覚トークン数)
        """
        # ステップ1: ViT による特徴抽出
        # (B*P, 3, 448, 448) → (B*P, 1025, D_v)
        if self.select_layer == -1:
            vit_output = self.vision_model(
                pixel_values=pixel_values,
                output_hidden_states=False,
                return_dict=True,
            )
            vit_embeds = vit_output.last_hidden_state  # (B*P, 1025, D_v)
        else:
            # 中間層を使用する場合 (通常は -1 = 最終層)
            vit_output = self.vision_model(
                pixel_values=pixel_values,
                output_hidden_states=True,
                return_dict=True,
            )
            vit_embeds = vit_output.hidden_states[self.select_layer]  # (B*P, 1025, D_v)

        # ステップ2: CLS トークンを除去
        # (B*P, 1025, D_v) → (B*P, 1024, D_v)
        vit_embeds = vit_embeds[:, 1:, :]

        # ステップ3: 1D 系列を 2D 空間に戻す
        h = w = int(vit_embeds.shape[1] ** 0.5)  # h=w=32
        # (B*P, 1024, D_v) → (B*P, 32, 32, D_v)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], h, w, -1)

        # ステップ4: Pixel Shuffle でトークン圧縮
        # (B*P, 32, 32, D_v=3200) → (B*P, 16, 16, 12800)
        vit_embeds = self.pixel_shuffle(vit_embeds, scale_factor=self.downsample_ratio)

        # ステップ5: 2D 空間から 1D 系列に戻す
        # (B*P, 16, 16, 12800) → (B*P, 256, 12800)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], -1, vit_embeds.shape[-1])

        # ステップ6: MLP Projector で LLM 次元に変換
        # (B*P, 256, 12800) → (B*P, 256, D_l)
        vit_embeds = self.mlp1(vit_embeds)

        return vit_embeds  # (B*P, 256, D_l)

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        image_flags: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        loss_weight: Optional[List[float]] = None,
    ) -> CausalLMOutputWithPast:
        """
        統合フォワードパス。

        入力:
          pixel_values  (B*P, 3, 448, 448)  ※ 全サンプルの全パッチを結合
          input_ids     (B, N)              ※ テキスト+IMG_CONTEXT
          attention_mask(B, N)
          image_flags   (B*P, 1)            ※ 1=有効, 0=パディング画像
          labels        (B, N)              ※ 学習時: -100でマスク
          loss_weight   (B, N)              ※ オプション: トークンごとの重み

        出力 (CausalLMOutputWithPast):
          .loss   スカラー (labels が None でない場合)
          .logits (B, N, V)
        """
        # image_flags: (B*P, 1) → (B*P,)
        image_flags = image_flags.squeeze(-1)

        # ステップ1: テキスト埋め込みを取得
        # (B, N) → (B, N, D_l)
        input_embeds = self.language_model.get_input_embeddings()(input_ids).clone()

        # ステップ2: 画像特徴を抽出
        # (B*P, 3, 448, 448) → (B*P, 256, D_l)
        vit_embeds = self.extract_feature(pixel_values)

        # パディング画像 (image_flags==0) を除外
        # (B*P_valid, 256, D_l)
        vit_embeds = vit_embeds[image_flags == 1]

        # ステップ3: IMG_CONTEXT トークン位置に視覚特徴を挿入
        B, N, C = input_embeds.shape
        input_embeds_flat = input_embeds.reshape(B * N, C)
        input_ids_flat = input_ids.reshape(B * N)

        # IMG_CONTEXT トークンの位置マスク
        selected = (input_ids_flat == self.img_context_token_id)
        # 選択位置の埋め込みを視覚特徴で上書き
        # vit_embeds.reshape(-1, C): (B*P_valid*256, D_l)
        input_embeds_flat[selected] = input_embeds_flat[selected] * 0.0 + vit_embeds.reshape(-1, C)

        # (B, N, D_l) に戻す
        input_embeds = input_embeds_flat.reshape(B, N, C)

        # ステップ4: LLM に入力 (embeddings として渡す)
        outputs = self.language_model(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            return_dict=True,
        )
        logits = outputs.logits  # (B, N, V)

        # ステップ5: 損失計算
        loss = None
        if labels is not None:
            # 次トークン予測のためにシフト
            # logits:  (B, N-1, V)  ← 最後を除く
            # labels:  (B, N-1)     ← 最初を除く
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            shift_logits = shift_logits.view(-1, shift_logits.size(-1))  # (B*(N-1), V)
            shift_labels = shift_labels.view(-1).to(shift_logits.device)  # (B*(N-1),)

            if loss_weight is not None:
                # トークンごとの重み付き損失 (Square Averaging に対応)
                loss_weight_tensor = torch.tensor(
                    loss_weight, dtype=torch.float32, device=labels.device
                )
                shift_weights = loss_weight_tensor[..., 1:].contiguous().view(-1)
                loss_fct = CrossEntropyLoss(reduction='none')
                loss = loss_fct(shift_logits, shift_labels)
                loss = (loss * shift_weights).sum() / shift_weights.sum()
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(shift_logits, shift_labels)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    @torch.no_grad()
    def generate(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        max_new_tokens: int = 512,
        **generate_kwargs,
    ) -> torch.Tensor:
        """
        推論時のテキスト生成。

        入力:
          pixel_values  (B*P, 3, 448, 448)  ※ 画像がある場合
          input_ids     (B, N_prefix)
          attention_mask(B, N_prefix)
        出力:
          generated_ids (B, N_generated)
        """
        assert self.img_context_token_id is not None

        if pixel_values is not None:
            # 画像特徴を抽出
            vit_embeds = self.extract_feature(pixel_values)  # (B*P, 256, D_l)

            # テキスト埋め込みを取得
            input_embeds = self.language_model.get_input_embeddings()(input_ids)  # (B, N, D_l)
            B, N, C = input_embeds.shape
            input_embeds_flat = input_embeds.reshape(B * N, C)
            input_ids_flat = input_ids.reshape(B * N)

            # IMG_CONTEXT 位置に視覚特徴を挿入
            selected = (input_ids_flat == self.img_context_token_id)
            input_embeds_flat[selected] = vit_embeds.reshape(-1, C).to(input_embeds.device)
            input_embeds = input_embeds_flat.reshape(B, N, C)
        else:
            input_embeds = self.language_model.get_input_embeddings()(input_ids)

        # LLM で自己回帰生成
        outputs = self.language_model.generate(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            **generate_kwargs,
        )
        return outputs  # (B, N_generated)


# ============================================================
# 使用例
# ============================================================

if __name__ == '__main__':
    import torch

    print("=" * 60)
    print("InternVL3.5 モデルアーキテクチャ 動作確認")
    print("=" * 60)

    # --- 設定 ---
    class VisionConfig:
        hidden_size = 1024          # InternViT-300M (小さくして動作確認)
        num_attention_heads = 16
        intermediate_size = 4096
        num_hidden_layers = 4       # 実際は24層(300M) or 48層(6B)
        image_size = 448
        patch_size = 14
        qk_normalization = True
        norm_type = 'rms_norm'
        layer_norm_eps = 1e-6
        qkv_bias = True
        attention_dropout = 0.0
        dropout = 0.0
        drop_path_rate = 0.1
        initializer_factor = 0.1
        output_hidden_states = False
        use_return_dict = True

    class LLMConfig:
        hidden_size = 2048

    vision_cfg = VisionConfig()
    llm_cfg = LLMConfig()

    # --- InternVisionEmbeddings テスト ---
    print("\n[1] InternVisionEmbeddings テスト")
    embed_layer = InternVisionEmbeddings(vision_cfg)
    dummy_images = torch.randn(3, 3, 448, 448)   # B*P=3枚
    embed_out = embed_layer(dummy_images)
    print(f"  入力: pixel_values     {dummy_images.shape}")
    print(f"  出力: embeddings       {embed_out.shape}")
    assert embed_out.shape == (3, 1025, 1024), f"期待: (3, 1025, 1024), 実際: {embed_out.shape}"
    print("  OK: (B*P=3, S_v=1025, D_v=1024)")

    # --- InternAttention テスト ---
    print("\n[2] InternAttention テスト")
    attn_layer = InternAttention(vision_cfg)
    hidden = torch.randn(3, 1025, 1024)
    attn_out = attn_layer(hidden)
    print(f"  入力: hidden_states    {hidden.shape}")
    print(f"  出力:                  {attn_out.shape}")
    assert attn_out.shape == (3, 1025, 1024)
    print("  OK: (B*P=3, S_v=1025, D_v=1024)")

    # --- InternVisionModel テスト ---
    print("\n[3] InternVisionModel テスト")
    vit = InternVisionModel(vision_cfg)
    pixel_values = torch.randn(3, 3, 448, 448)   # 3パッチ
    vit_out = vit(pixel_values)
    print(f"  入力: pixel_values            {pixel_values.shape}")
    print(f"  出力: last_hidden_state       {vit_out.last_hidden_state.shape}")
    print(f"  出力: pooler_output (CLS)     {vit_out.pooler_output.shape}")
    assert vit_out.last_hidden_state.shape == (3, 1025, 1024)
    print("  OK: last_hidden_state = (B*P=3, S_v=1025, D_v=1024)")

    # --- Pixel Shuffle テスト ---
    print("\n[4] Pixel Shuffle テスト")

    class _MockModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.downsample_ratio = 0.5
            self.ps_version = 'v2'

        pixel_shuffle = InternVLChatModel.pixel_shuffle

    mock = _MockModel()
    # CLS を除去して (3, 1024, 1024) → (3, 32, 32, 1024)
    feat = vit_out.last_hidden_state[:, 1:, :]   # (3, 1024, 1024)
    h = w = int(feat.shape[1] ** 0.5)           # 32
    feat_2d = feat.reshape(3, h, w, 1024)       # (3, 32, 32, 1024)
    shuffled = mock.pixel_shuffle(mock, feat_2d, scale_factor=0.5)
    print(f"  入力: (B*P, H_t, W_t, D_v)   {feat_2d.shape}")
    print(f"  出力: (B*P, H_t/2, W_t/2, D_v*4)  {shuffled.shape}")
    shuffled_flat = shuffled.reshape(3, -1, shuffled.shape[-1])
    print(f"  flatten後:                    {shuffled_flat.shape}")
    assert shuffled_flat.shape == (3, 256, 1024 * 4)
    print("  OK: (B*P=3, 256, 4096) ← 4倍圧縮, 4倍チャンネル増加")

    # --- MLP Projector テスト ---
    print("\n[5] MLP Projector テスト")
    mlp1 = build_mlp_projector(vit_hidden_size=1024, llm_hidden_size=2048, downsample_ratio=0.5)
    mlp_in = shuffled_flat   # (3, 256, 4096)
    mlp_out = mlp1(mlp_in)
    print(f"  入力: (B*P, num_tokens, D_v*4)  {mlp_in.shape}")
    print(f"  出力: (B*P, num_tokens, D_l)    {mlp_out.shape}")
    assert mlp_out.shape == (3, 256, 2048)
    print("  OK: (B*P=3, 256, D_l=2048)")

    print("\n全テスト完了!")
    print(f"  InternViT-300M (簡略版4層): {sum(p.numel() for p in vit.parameters()):,} パラメータ")
    print(f"  MLP Projector:              {sum(p.numel() for p in mlp1.parameters()):,} パラメータ")
