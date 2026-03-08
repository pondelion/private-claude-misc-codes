"""
Qwen3VL - DeepStack Vision
===========================

Qwen3-VL の DeepStack 機構を疑似コードで示します。

DeepStackとは:
  ViT の中間レイヤーの特徴量を抽出し、LLM の各トランスフォーマー層に注入する機構。
  標準的なViT-LLM接続は最終層の特徴量のみを使用するが、
  DeepStackは複数の抽象度の特徴量をLLMの異なる深さに対応させて注入する。

主要コンポーネント:
1. DeepStack ViT 特徴量抽出 (Vision Encoder内)
2. DeepStack 注入 (LLM Transformer内)
3. 注入スケジュール (どのViT層をどのLLM層に注入するか)
4. 注入方式 (クロスアテンション or 加算)

論文: Qwen3-VL Technical Report (2025)

============================================================
Shape Convention
============================================================
N_patches: ViT入力パッチ数 (全画像の合計)
N_v: LLM入力視覚トークン数 = N_patches / 4 (merge後)
D_v: Vision Encoder隠れ次元 = 1152
D_llm: LLM隠れ次元 = 3584 (7Bモデル)
L_v: Vision Encoderのレイヤー数 = 32
L_llm: LLMのレイヤー数 = 28 (7Bモデル)
deepstack_interval: 中間特徴量を抽出するViT層間隔 = 4
B: バッチサイズ
T_seq: LLMシーケンス長
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple


# ============================================================
# DeepStack の設計原理
# ============================================================
#
# 【問題】標準的な Vision Encoder → LLM 接続
#
#   ViT layer 1 →
#   ViT layer 2 →
#   ...
#   ViT layer 32 → 最終特徴量 → MLP Merger → LLM embedding input
#                                              LLM layer 1
#                                              LLM layer 2
#                                              ...
#                                              LLM layer 28
#
#   問題: ViTの低レベル特徴(テクスチャ, エッジ)が失われ、
#         最終層の高レベル特徴(意味的情報)のみがLLMに渡される
#
# 【解決】DeepStack: 中間特徴量をLLMの各層に注入
#
#   ViT layer 1 →
#   ViT layer 4 → 中間特徴量 → 射影 → LLM layer 3 へ注入
#   ViT layer 8 → 中間特徴量 → 射影 → LLM layer 7 へ注入
#   ...
#   ViT layer 32 → 最終特徴量 → MLP Merger → LLM embedding input
#                                              LLM layer 0
#                                              LLM layer 1
#                                              ...
#                                              LLM layer 28
#
#   効果: ViTの異なる抽象度の特徴量がLLMの適切な深さに対応して注入される
#         浅いViT層 → 低レベル特徴 → LLM浅層
#         深いViT層 → 高レベル特徴 → LLM深層


# ============================================================
# DeepStack 設定
# ============================================================

DEEPSTACK_CONFIG = {
    "vision_num_layers": 32,       # L_v: ViTレイヤー数
    "llm_num_layers": 28,          # L_llm: LLMレイヤー数 (7Bモデル)
    "deepstack_interval": 4,       # ViTの何レイヤーごとに中間特徴を抽出するか
    "vision_embed_dim": 1152,      # D_v: ViT隠れ次元
    "llm_hidden_size": 3584,       # D_llm: LLM隠れ次元 (7Bモデル)
    # deepstack_num_features = L_v // deepstack_interval = 32 // 4 = 8
    # injection_llm_layers: 8個の特徴量を28層のLLMに均等分配
}

# 注入スケジュール計算の例 (L_v=32, L_llm=28, deepstack_interval=4)
# ViT中間特徴量の抽出層: [3, 7, 11, 15, 19, 23, 27, 31] (0-indexed)
#   (deepstack_interval=4 で割り切れる層インデックス - 1)
# LLMへの注入層: 均等分配
#   例: [2, 5, 8, 12, 15, 18, 21, 25] (L_llm=28を8分割)


# ============================================================
# DeepStack: ViT 中間特徴量の抽出 (Vision Encoder 内)
# ============================================================

class Qwen3VisionTransformerWithDeepStack(nn.Module):
    """
    DeepStack付き Vision Encoder

    Vision Encoder 内で deepstack_interval ごとに
    中間特徴量を保存するように拡張した ViT

    ========================================
    Shape
    ========================================
    入力:
        pixel_values: (N_patches, C×P²) = (N_patches, 588)
        grid_thw: (num_images, 3)

    出力:
        visual_tokens: (N_v, D_llm) = (N_patches/4, 3584)
        intermediate_features: List[(N_patches, D_v)]
            - 長さ: L_v // deepstack_interval = 32 // 4 = 8
            - 各要素: (N_patches, 1152)
    """

    def __init__(self, config: dict):
        super().__init__()
        self.num_layers = config.get("vision_num_layers", 32)
        self.deepstack_interval = config.get("deepstack_interval", 4)
        self.embed_dim = config.get("vision_embed_dim", 1152)

        # ViT ブロック (vision_encoder.py の Qwen3VisionBlock と同じ)
        # self.blocks = nn.ModuleList([...]) # 省略

    def forward(
        self,
        pixel_values: torch.Tensor,
        grid_thw: torch.LongTensor,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        ========================================
        処理フロー (DeepStack対応)
        ========================================
        """
        # 1. パッチ埋め込み
        # hidden_states: (N_patches, D_v)
        hidden_states = self.patch_embed(pixel_values)

        # 2. 2D RoPE 計算
        rotary_pos_emb = self.get_rotary_pos_emb(grid_thw)
        cu_seqlens = self._compute_cu_seqlens(grid_thw)

        # 3. ViT Blocks + DeepStack中間特徴量の収集
        intermediate_features = []

        for block_idx in range(self.num_layers):
            hidden_states = self.blocks[block_idx](hidden_states, cu_seqlens, rotary_pos_emb)
            # hidden_states: (N_patches, D_v) = (N_patches, 1152)

            # deepstack_interval ごとに中間特徴量を保存
            if (block_idx + 1) % self.deepstack_interval == 0:
                # 例: block_idx = 3, 7, 11, 15, 19, 23, 27, 31
                intermediate_features.append(hidden_states.clone())
                # 各要素: (N_patches, D_v) = (N_patches, 1152)

        # intermediate_features の長さ: L_v // deepstack_interval = 8

        # 4. 最終正規化
        hidden_states = self.norm(hidden_states)

        # 5. MLP Merger (2×2空間圧縮)
        visual_tokens = self.merger(hidden_states, grid_thw)
        # visual_tokens: (N_v, D_llm) = (N_patches/4, 3584)

        return visual_tokens, intermediate_features


# ============================================================
# DeepStack: 中間特徴量射影モジュール
# ============================================================

class DeepStackProjector(nn.Module):
    """
    ViT 中間特徴量を LLM 次元に射影するモジュール

    各中間特徴量セット (deepstack_intervalごと) に対して1つのProjectorがある

    ========================================
    Shape
    ========================================
    入力:
        vit_features: (N_patches, D_v) = (N_patches, 1152)
        grid_thw: (num_images, 3)

    出力:
        projected: (N_v, D_llm) = (N_patches/4, 3584)

    ========================================
    処理詳細
    ========================================
    1. MLP Merger (2×2空間圧縮): (N_patches, D_v) → (N_v, D_v×4)
    2. Linear 射影: (N_v, D_v×4) → (N_v, D_llm)

    注: 各中間特徴量セットに対して独立した PatchMerger + Linear が用意されることもある
        実装によっては共有重みを使う場合もある
    """

    def __init__(
        self,
        in_dim: int = 1152,
        out_dim: int = 3584,
        spatial_merge_size: int = 2,
    ):
        super().__init__()
        self.spatial_merge_size = spatial_merge_size

        # 2×2空間圧縮 + 射影
        self.merge_and_proj = nn.Sequential(
            nn.Linear(in_dim * spatial_merge_size ** 2, out_dim),  # (4608, 3584)
            nn.GELU(),
            nn.Linear(out_dim, out_dim),  # (3584, 3584)
        )

    def forward(
        self,
        vit_features: torch.Tensor,
        grid_thw: torch.LongTensor,
    ) -> torch.Tensor:
        """
        入力:
            vit_features: (N_patches, D_v) = (N_patches, 1152)
            grid_thw: (num_images, 3)

        出力:
            projected: (N_v, D_llm) = (N_patches/4, 3584)
        """
        m = self.spatial_merge_size  # = 2
        merged_list = []
        patch_idx = 0

        for thw in grid_thw:
            T, H, W = thw[0].item(), thw[1].item(), thw[2].item()
            N_i = T * H * W

            patches_i = vit_features[patch_idx : patch_idx + N_i]
            # patches_i: (T×H×W, D_v)

            # 2×2空間圧縮
            patches_i = patches_i.view(T, H // m, m, W // m, m, -1)
            patches_i = patches_i.permute(0, 1, 3, 2, 4, 5)
            patches_i = patches_i.reshape(T * (H // m) * (W // m), m * m * patches_i.shape[-1])
            # patches_i: (T×H/m×W/m, D_v×m²) = (N_i/4, 4608)

            merged_i = self.merge_and_proj(patches_i)
            # merged_i: (N_i/4, D_llm) = (N_i/4, 3584)

            merged_list.append(merged_i)
            patch_idx += N_i

        projected = torch.cat(merged_list, dim=0)
        # projected: (N_v, D_llm) = (N_patches/4, 3584)

        return projected


# ============================================================
# DeepStack: LLM 層への注入
# ============================================================

class DeepStackInjectionLayer(nn.Module):
    """
    DeepStack 注入層: LLM の各トランスフォーマー層に組み込む

    注入方式 (論文によりアテンションか加算かは実装依存):
    1. 加算方式: hidden_states += gate × projected_features
    2. クロスアテンション方式:
       hidden_states += CrossAttention(query=hidden_states, kv=projected_features)

    ========================================
    Shape
    ========================================
    入力:
        hidden_states: (B, T_seq, D_llm) - LLM の隠れ状態
        vit_features: (N_v, D_llm) - ViT 中間特徴量 (射影済み)
        visual_token_mask: (B, T_seq) bool - 視覚トークンの位置

    出力:
        hidden_states: (B, T_seq, D_llm) - 注入後の隠れ状態

    ========================================
    加算方式の処理詳細
    ========================================
    1. 視覚トークン位置の hidden_states を取得
    2. gate (学習可能スカラー) で重み付け
    3. vit_features を加算

    hidden_states[visual_mask] += gate × vit_features
    """

    def __init__(self, hidden_size: int = 3584, injection_mode: str = "add"):
        super().__init__()
        self.injection_mode = injection_mode

        if injection_mode == "add":
            # 加算方式: 学習可能なゲート (初期値=0で安定した学習)
            self.gate = nn.Parameter(torch.zeros(1))
            # gate: scalar, 初期値0 → 学習で徐々に開く

        elif injection_mode == "cross_attention":
            # クロスアテンション方式
            self.cross_attn = nn.MultiheadAttention(
                embed_dim=hidden_size,
                num_heads=8,
                batch_first=True,
            )
            self.norm = nn.RMSNorm(hidden_size)
            self.gate = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        hidden_states: torch.Tensor,
        vit_features: torch.Tensor,
        visual_token_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        ========================================
        加算方式の Shape
        ========================================
        入力:
            hidden_states: (B, T_seq, D_llm) = (B, T_seq, 3584)
            vit_features: (N_v, D_llm) = (N_patches/4, 3584)
            visual_token_mask: (B, T_seq) bool

        出力:
            hidden_states: (B, T_seq, D_llm)

        ========================================
        クロスアテンション方式の Shape
        ========================================
        入力:
            hidden_states: (B, T_seq, D_llm)
            vit_features: (N_v, D_llm) - 全視覚トークン

        内部:
            vit_features_batch: (B, N_v, D_llm) - バッチに対応するよう変換

        出力:
            hidden_states: (B, T_seq, D_llm)
        """
        if self.injection_mode == "add":
            # ==================================================
            # 加算方式
            # ==================================================
            if visual_token_mask is not None:
                # 視覚トークン位置のみに加算
                # visual_token_mask: (B, T_seq) bool
                # hidden_states[visual_token_mask]: (N_v_batch, D_llm) = (B×N_v, D_llm)
                # vit_features: (N_v_total, D_llm)

                # vit_features を flatten して対応位置に加算
                vit_flat = vit_features.view(-1, vit_features.shape[-1])
                # vit_flat: (N_v_total, D_llm)

                hidden_states[visual_token_mask] = (
                    hidden_states[visual_token_mask]
                    + torch.tanh(self.gate) * vit_flat
                )
                # 注: torch.tanh(gate) で [-1, 1] に制限、初期値=0で安定
            else:
                # visual_token_maskが提供されない場合、視覚トークン位置を特定できないため注入をスキップする
                # 通常の使用ではvisual_token_maskは必ず提供される（Qwen3VLのforward()参照）
                return hidden_states

        elif self.injection_mode == "cross_attention":
            # ==================================================
            # クロスアテンション方式
            # ==================================================
            B, T_seq, D_llm = hidden_states.shape
            N_v_total = vit_features.shape[0]

            # vit_features をバッチに展開
            # 注: バッチ内の各サンプルに対応する視覚トークンを特定する必要がある
            # ここでは簡略化のためバッチ内全視覚トークンを使用
            vit_batch = vit_features.unsqueeze(0).expand(B, -1, -1)
            # vit_batch: (B, N_v_total, D_llm)

            # クロスアテンション: hidden_states → query, vit_features → key/value
            residual = hidden_states
            hidden_states = self.norm(hidden_states)

            attn_output, _ = self.cross_attn(
                query=hidden_states,   # (B, T_seq, D_llm) as Q
                key=vit_batch,         # (B, N_v_total, D_llm) as K
                value=vit_batch,       # (B, N_v_total, D_llm) as V
            )
            # attn_output: (B, T_seq, D_llm)

            hidden_states = residual + torch.tanh(self.gate) * attn_output

        return hidden_states
        # hidden_states: (B, T_seq, D_llm)


# ============================================================
# DeepStack: 注入スケジュールの計算
# ============================================================

def compute_injection_schedule(
    vit_num_layers: int = 32,
    llm_num_layers: int = 28,
    deepstack_interval: int = 4,
) -> Dict:
    """
    DeepStack の注入スケジュール計算

    ========================================
    出力
    ========================================
    dict:
        "vit_extraction_layers": List[int]
            - 中間特徴量を抽出する ViT 層のインデックス
        "llm_injection_layers": List[int]
            - 各 ViT 特徴量を注入する LLM 層のインデックス
        "schedule": List[Tuple[int, int]]
            - [(vit_layer, llm_layer), ...]

    ========================================
    例 (L_v=32, L_llm=28, interval=4)
    ========================================
    ViT抽出層:    [3,  7, 11, 15, 19, 23, 27, 31]  (8個)
    LLM注入層:    [3,  6,  9, 12, 15, 18, 21, 24]  (8個, 均等分配)

    スケジュール:
    ViT layer 3  → LLM layer 3
    ViT layer 7  → LLM layer 6
    ViT layer 11 → LLM layer 9
    ViT layer 15 → LLM layer 12
    ViT layer 19 → LLM layer 15
    ViT layer 23 → LLM layer 18
    ViT layer 27 → LLM layer 21
    ViT layer 31 → LLM layer 24
    """
    import math

    # ViT 中間特徴量の抽出層
    vit_extraction_layers = [
        i for i in range(vit_num_layers - 1)
        if (i + 1) % deepstack_interval == 0
    ]
    # = [3, 7, 11, 15, 19, 23, 27, 31] for L_v=32, interval=4

    num_features = len(vit_extraction_layers)  # = 8

    # LLM への注入層 (均等分配)
    llm_injection_layers = [
        math.floor(i * llm_num_layers / num_features)
        for i in range(num_features)
    ]
    # = [0, 3, 7, 10, 14, 17, 21, 24] for L_llm=28, 8特徴量

    schedule = list(zip(vit_extraction_layers, llm_injection_layers))

    print("DeepStack 注入スケジュール:")
    print(f"  ViT抽出層:    {vit_extraction_layers}")
    print(f"  LLM注入層:    {llm_injection_layers}")
    print()
    print("  (ViT layer) → (LLM layer)")
    for vit_l, llm_l in schedule:
        print(f"    ViT {vit_l:2d}    →   LLM {llm_l:2d}")

    return {
        "vit_extraction_layers": vit_extraction_layers,
        "llm_injection_layers": llm_injection_layers,
        "schedule": schedule,
    }


# ============================================================
# DeepStack 統合: LLM フォワード内での使用
# ============================================================

class Qwen3LLMWithDeepStack(nn.Module):
    """
    DeepStack 注入付き LLM フォワードパス (疑似コード)

    ========================================
    Shape
    ========================================
    入力:
        inputs_embeds: (B, T_seq, D_llm)
        position_ids: (3, B, T_seq)
        intermediate_vit_features: List[(N_patches, D_v)] or List[(N_v, D_llm)]
            - DeepStack用のViT中間特徴量 (射影済みを想定)
        visual_token_mask: (B, T_seq) bool
            - 視覚トークンの位置

    出力:
        hidden_states: (B, T_seq, D_llm)

    ========================================
    処理フロー
    ========================================
    1. Word Embedding または inputs_embeds を入力
    2. LLM 層を順に適用
    3. 指定層で DeepStack 注入を適用
    """

    def __init__(self, config: dict):
        super().__init__()
        llm_num_layers = config.get("llm_num_layers", 28)
        vit_num_layers = config.get("vision_num_layers", 32)
        deepstack_interval = config.get("deepstack_interval", 4)
        hidden_size = config.get("llm_hidden_size", 3584)

        # 注入スケジュール計算
        schedule_info = compute_injection_schedule(
            vit_num_layers=vit_num_layers,
            llm_num_layers=llm_num_layers,
            deepstack_interval=deepstack_interval,
        )
        self.injection_schedule = dict(schedule_info["schedule"])
        # {vit_layer_idx: llm_layer_idx, ...}
        # 逆引き: {llm_layer: vit_feature_idx, ...}
        self.llm_to_vit_map = {v: i for i, (k, v) in enumerate(schedule_info["schedule"])}
        # {llm_layer_idx: feature_list_index, ...}

        # LLM トランスフォーマー層
        # self.layers = nn.ModuleList([...]) # 省略

        # DeepStack 注入層 (各注入ポイントに対して1つ)
        num_injections = len(schedule_info["schedule"])
        self.deepstack_injectors = nn.ModuleList([
            DeepStackInjectionLayer(hidden_size=hidden_size, injection_mode="add")
            for _ in range(num_injections)
        ])

        # ViT中間特徴量の射影器 (D_v → D_llm)
        vision_embed_dim = config.get("vision_embed_dim", 1152)
        self.deepstack_projectors = nn.ModuleList([
            DeepStackProjector(
                in_dim=vision_embed_dim,
                out_dim=hidden_size,
                spatial_merge_size=2,
            )
            for _ in range(num_injections)
        ])

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        position_ids: torch.Tensor,
        intermediate_vit_features: Optional[List[torch.Tensor]] = None,
        grid_thw: Optional[torch.LongTensor] = None,
        visual_token_mask: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        入力:
            inputs_embeds: (B, T_seq, D_llm)
            position_ids: (3, B, T_seq)
            intermediate_vit_features: List[(N_patches, D_v)] - ViT中間特徴量
                長さ: L_v // deepstack_interval = 8 (例)
            grid_thw: (num_images, 3) - Merger用グリッドサイズ
            visual_token_mask: (B, T_seq) bool - 視覚トークン位置

        出力:
            hidden_states: (B, T_seq, D_llm)
        """
        hidden_states = inputs_embeds
        # hidden_states: (B, T_seq, D_llm)

        for layer_idx, layer in enumerate(self.layers):

            # ========================================
            # DeepStack 注入 (指定層の前に実行)
            # ========================================
            if (intermediate_vit_features is not None
                    and layer_idx in self.llm_to_vit_map):

                feat_idx = self.llm_to_vit_map[layer_idx]
                # feat_idx: この注入ポイントに対応するViT特徴量のインデックス

                vit_feat = intermediate_vit_features[feat_idx]
                # vit_feat: (N_patches, D_v) = (N_patches, 1152)

                # ViT特徴量を射影: (N_patches, D_v) → (N_v, D_llm)
                projected_feat = self.deepstack_projectors[feat_idx](vit_feat, grid_thw)
                # projected_feat: (N_v, D_llm) = (N_patches/4, 3584)

                # LLM 隠れ状態に注入
                hidden_states = self.deepstack_injectors[feat_idx](
                    hidden_states=hidden_states,
                    vit_features=projected_feat,
                    visual_token_mask=visual_token_mask,
                )
                # hidden_states: (B, T_seq, D_llm) (注入後)

            # ========================================
            # 通常の LLM 層を適用
            # ========================================
            hidden_states = layer(
                hidden_states=hidden_states,
                position_ids=position_ids,
                attention_mask=attention_mask,
            )
            # hidden_states: (B, T_seq, D_llm)

        return hidden_states


# ============================================================
# 使用例
# ============================================================

def example_usage():
    """
    DeepStack の使用例

    ========================================
    Shape Summary
    ========================================
    Vision Encoder (DeepStack出力):
        visual_tokens:          (256, 3584)  # 448×448画像の例
        intermediate_features:  List × 8 of (1024, 1152)

    LLM Forward (DeepStack注入):
        Layer  0: 通常処理
        Layer  3: feat[0] = intermediate[0] (ViT layer 3) を注入
        Layer  6: feat[1] = intermediate[1] (ViT layer 7) を注入
        ...
        Layer 24: feat[7] = intermediate[7] (ViT layer 31) を注入
        Layer 25-27: 通常処理
    """
    print("=== DeepStack Example ===\n")

    print("【注入スケジュール (L_v=32, L_llm=28, interval=4)】")
    schedule = compute_injection_schedule(
        vit_num_layers=32,
        llm_num_layers=28,
        deepstack_interval=4,
    )

    print()

    # --- DeepStackProjector テスト ---
    # N_patches = 32×32 = 1024 (448×448 画像)
    N_patches = 1024
    N_v = 256         # N_patches / 4 (merge後)
    D_v = 1152
    D_llm = 3584
    B = 1
    T_seq = N_v + 10  # 視覚トークン + テキストトークン = 266

    projector = DeepStackProjector(in_dim=D_v, out_dim=D_llm, spatial_merge_size=2)
    vit_features = torch.randn(N_patches, D_v)         # (1024, 1152)
    grid_thw = torch.tensor([[1, 32, 32]], dtype=torch.long)  # T=1, H=32, W=32

    projected = projector(vit_features, grid_thw)
    print(f"[DeepStackProjector]")
    print(f"  vit_features:   {vit_features.shape}")   # (1024, 1152)
    print(f"  grid_thw:       {grid_thw.shape}")        # (1, 3)
    print(f"  projected:      {projected.shape}")       # (256, 3584)
    print()

    # --- DeepStackInjectionLayer テスト (加算方式) ---
    injection_layer = DeepStackInjectionLayer(hidden_size=D_llm, injection_mode="add")
    hidden_states = torch.randn(B, T_seq, D_llm)       # (1, 266, 3584)

    # visual_token_mask: 最初の N_v トークンが視覚トークン
    visual_mask = torch.zeros(B, T_seq, dtype=torch.bool)
    visual_mask[0, :N_v] = True                         # (1, 266) bool

    out = injection_layer(hidden_states, projected, visual_mask)
    print(f"[DeepStackInjectionLayer (add)]")
    print(f"  hidden_states:  {hidden_states.shape}")   # (1, 266, 3584)
    print(f"  vit_features:   {projected.shape}")       # (256, 3584)
    print(f"  visual_mask:    {visual_mask.shape}")     # (1, 266)
    print(f"  output:         {out.shape}")              # (1, 266, 3584)
    print()

    # --- DeepStackInjectionLayer テスト (visual_token_mask なし → early return) ---
    out_no_mask = injection_layer(hidden_states, projected, visual_token_mask=None)
    print(f"[DeepStackInjectionLayer (visual_mask=None → skip)]")
    print(f"  output:         {out_no_mask.shape}")     # (1, 266, 3584)
    print()

    print(f"【Shape Summary (448×448 画像)】")
    print(f"  N_patches = 32×32 = {N_patches} (ViT入力)")
    print(f"  N_v = {N_v} (MLP Merger後, 2×2圧縮)")
    print()
    print(f"  intermediate_features: 8 × {vit_features.shape}  [ViT中間特徴量, D_v={D_v}]")
    print(f"  各ViT特徴量は DeepStackProjector で {projected.shape} に変換後、LLMに注入")
    print()
    print(f"  加算方式: hidden_states[visual_mask] += tanh(gate) × projected_feat")
    print(f"  gate は学習可能スカラー (初期値=0 で安定した学習開始)")


if __name__ == "__main__":
    example_usage()
