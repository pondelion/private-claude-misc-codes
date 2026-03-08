"""
Qwen3VL - Interleaved MRoPE (Multimodal Rotary Position Embedding)
===================================================================

Qwen3-VL の位置エンコーディングを疑似コードで示します。

Qwen3-VL の主要イノベーション:
  Interleaved MRoPE - Qwen2.5-VL の Chunked MRoPE に対する改良

主要コンポーネント:
1. get_rope_index_3(): Qwen3-VL用のposition_ids計算
2. apply_multimodal_rotary_pos_emb(): 3D MRoPEの適用
3. 各モダリティ(テキスト/画像/動画)ごとのposition_ids計算の詳細

論文: Qwen3-VL Technical Report (2025)
実装参照: qwen-vl-finetune/qwenvl/data/rope2d.py

============================================================
Shape Convention
============================================================
B: バッチサイズ
T_seq: LLMシーケンス長 (テキスト + 視覚トークン)
D_llm: LLM隠れ次元 = 3584 (7Bモデル)
H_llm: LLMアテンションヘッド数 = 28 (7Bモデル)
head_dim_llm: D_llm // H_llm = 128 (7Bモデル)
merge_size: MLP Mergerの空間圧縮倍率 = 2
"""

import torch
from typing import Optional, List, Tuple


# ============================================================
# 特殊トークンID
# ============================================================

IMAGE_TOKEN_ID = 151655         # <image> プレースホルダー
VIDEO_TOKEN_ID = 151656         # <video> プレースホルダー
VISION_START_TOKEN_ID = 152652  # <|vision_start|>


# ============================================================
# Qwen3-VL vs Qwen2.5-VL: Interleaved MRoPE の違い
# ============================================================
#
# ============================================================
# Qwen2.5-VL (Chunked MRoPE) の動画position_ids:
# ============================================================
# video_grid_thw = (1, 3)  # 3フレーム × 2高さ × 2幅の例
# T=3, H=2, W=2 → 3×2×2=12 視覚トークン
#
# temporal position_ids:
#   [0, 0, 0, 0,   ← frame 0: 2×2 tokens
#    1, 1, 1, 1,   ← frame 1
#    2, 2, 2, 2]   ← frame 2
#
# ============================================================
# Qwen3-VL (Interleaved MRoPE) の動画position_ids:
# ============================================================
# 各フレームが独立したビジョン入力として分割される:
# video_grid_thw = (3, 3) ではなく、[[1,2,2],[1,2,2],[1,2,2]] (T=1に分割)
#
# 入力トークン列:
#   "<t0.00>" "<|vision_start|>" [frame0_patches×4] "<|vision_end|>"
#   "<t0.50>" "<|vision_start|>" [frame1_patches×4] "<|vision_end|>"
#   "<t1.00>" "<|vision_start|>" [frame2_patches×4] "<|vision_end|>"
#
# 各フレームのt position_ids = 0 (常に0! 時間情報はテキストタイムスタンプで)
# h position_ids = [0, 0, 1, 1]  (2×2グリッド)
# w position_ids = [0, 1, 0, 1]
#
# テキストタイムスタンプトークンは通常のテキストと同じ1D position_ids


# ============================================================
# get_rope_index_3: Qwen3-VL 用 position_ids 計算
# ============================================================

def get_rope_index_3(
    spatial_merge_size: int = 2,
    input_ids: Optional[torch.LongTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    second_per_grid_ts: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Qwen3-VL 用の Interleaved MRoPE position_ids 計算

    Qwen2.5-VL との違い:
    - 動画の video_grid_thw を T=1 に分割する前処理
    - 各フレームのt position_ids は常に0 (時間情報はタイムスタンプテキストで表現)

    ========================================
    Shape
    ========================================
    入力:
        spatial_merge_size: int = 2
        input_ids: (B, T_seq)
            - IMAGE_TOKEN_ID (151655) が視覚トークンの位置
            - VIDEO_TOKEN_ID (151656) が動画トークンの位置
        image_grid_thw: (num_images, 3) or None
            - 各画像の [T, H_patches_in_llm, W_patches_in_llm]
            - H_patches_in_llm = H_patches / spatial_merge_size
        video_grid_thw: (num_videos, 3) or None
        attention_mask: (B, T_seq) or None

    出力:
        position_ids: (3, B, T_seq)
            - 3次元: [temporal(t), height(h), width(w)]
            - テキストトークン: t=h=w=連続インデックス (1D MRoPE)
            - 画像トークン: t=0, h=[0,H_grid), w=[0,W_grid)
            - 動画トークン: t=0 (常に!), h=[0,H_grid), w=[0,W_grid)
        mrope_position_deltas: (B, 1)
            - シーケンス延長時の位置ID調整量

    ========================================
    position_ids の具体的な中身
    ========================================
    position_ids は shape (3, B, T_seq)。
    T_seq 軸の各トークン位置に対して [temporal, height, width] の3値が入る。

    例: テキスト3トークン → 画像2×2パッチ(merge後=4token) → テキスト1トークン
        T_seq = 3 + 4 + 1 = 8

        T_seq位置:   0     1     2     3          4          5          6       7
        モダリティ: text  text  text  vision     vision     vision     vision  text
                                     (r=0,c=0)  (r=0,c=1)  (r=1,c=0)  (r=1,c=1)

        temporal行:  0     1     2     0          0          0          0       4
        height行:    0     1     2     0          0          1          1       4
        width行:     0     1     2     0          1          0          1       4

    【テキストトークン (位置0,1,2,7)】
        temporal = height = width = 同じ値 (シーケンス上の通し番号)
        3軸が全て同値 → RoPE内積は cos(p_i - p_j) が3軸分かかるだけ
        = 通常の1D RoPEと等価。textにh/wという概念はないが、
          3軸全部同じ値にすることで「h/wを無効化」している。

    【visionトークン (位置3〜6)】
        temporal = 0 固定 (時間情報はテキストタイムスタンプ側で表現)
        height   = パッチの行インデックス (0〜H_grid-1)
        width    = パッチの列インデックス (0〜W_grid-1)
        → RoPE内積で「行が何行離れているか」「列が何列離れているか」が反映される
        = 2D空間RoPEとして機能

    【テキスト位置7の値が4な理由】
        直前の視覚ブロックのmax position_id + 1 から連続して採番される (st_idx)
        視覚ブロックのmax = max(t=0,h=1,w=1) = 1 → st_idx = 2+1 = 3 (text末尾) + 1 = 4
        ※ 視覚トークンが4個あっても「その先のテキストは4から始まる」という設計

    ========================================
    アルゴリズム
    ========================================
    Step 1: video_grid_thw を T=1 に分割 (Qwen3-VL特有)
        例: [[3, 16, 16]] → [[1,16,16], [1,16,16], [1,16,16]]

    Step 2: 各バッチサンプルを処理
        テキスト部分: 連続1D インデックス
        視覚部分: 2D/3D グリッドインデックス (t=0に固定)

    Step 3: position_ids をアセンブル
        shape: (3, B, T_seq)
    """

    # ========================================
    # Step 1: Qwen3-VL特有の前処理
    # video_grid_thw を T=1 に分割
    # ========================================
    if video_grid_thw is not None:
        # 例: video_grid_thw = [[3, 16, 16]] (T=3, H=16, W=16)
        # → 各フレームのTを繰り返してT=1に展開
        video_grid_thw = torch.repeat_interleave(video_grid_thw, video_grid_thw[:, 0], dim=0)
        # 例: [[3,16,16]] → [[3,16,16],[3,16,16],[3,16,16]] (3回繰り返し)

        video_grid_thw[:, 0] = 1
        # 例: → [[1,16,16],[1,16,16],[1,16,16]]
        # これにより各フレームが独立したt=0の視覚入力として扱われる

    # ========================================
    # Step 2: 視覚入力がない場合 (テキストのみ)
    # ========================================
    if input_ids is None or (image_grid_thw is None and video_grid_thw is None):
        if attention_mask is not None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            # position_ids: (B, T_seq) - 0から始まる連続インデックス
            position_ids.masked_fill_(attention_mask == 0, 1)
            position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)
            # position_ids: (3, B, T_seq) - t=h=w が全て同一

            max_position_ids = position_ids.max(0)[0].max(-1, keepdim=True)[0]
            mrope_position_deltas = max_position_ids + 1 - attention_mask.shape[-1]
            # mrope_position_deltas: (B, 1)
        else:
            T_seq = input_ids.shape[1]
            position_ids = torch.arange(T_seq).view(1, 1, -1).expand(3, input_ids.shape[0], -1)
            # position_ids: (3, B, T_seq)
            mrope_position_deltas = torch.zeros(input_ids.shape[0], 1)
            # mrope_position_deltas: (B, 1)

        return position_ids, mrope_position_deltas

    # ========================================
    # Step 3: 視覚+テキスト混合の position_ids 計算
    # ========================================
    B, T_seq = input_ids.shape

    # 初期化 (全て1で初期化)
    position_ids = torch.ones(3, B, T_seq, dtype=input_ids.dtype, device=input_ids.device)
    # position_ids: (3, B, T_seq)

    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)

    mrope_position_deltas = []
    image_index = 0
    video_index = 0

    for i in range(B):  # 各バッチサンプルを処理
        # パディングを除いた有効なトークン
        valid_input_ids = input_ids[i][attention_mask[i] == 1]
        # valid_input_ids: (T_valid,)

        # 画像・動画の数を数える
        vision_start_indices = (valid_input_ids == VISION_START_TOKEN_ID).nonzero(as_tuple=False).squeeze(1)
        vision_tokens = valid_input_ids[vision_start_indices + 1]
        image_nums = (vision_tokens == IMAGE_TOKEN_ID).sum().item()
        video_nums = (vision_tokens == VIDEO_TOKEN_ID).sum().item()

        input_tokens = valid_input_ids.tolist()
        llm_pos_ids_list = []  # 位置IDのセグメントリスト
        st = 0  # 現在の処理位置
        remain_images = image_nums
        remain_videos = video_nums

        # 各視覚入力を処理
        for _ in range(image_nums + video_nums):
            # 次の IMAGE または VIDEO トークンの位置を特定
            if IMAGE_TOKEN_ID in input_tokens and remain_images > 0:
                ed_image = input_tokens.index(IMAGE_TOKEN_ID, st)
            else:
                ed_image = len(input_tokens) + 1

            if VIDEO_TOKEN_ID in input_tokens and remain_videos > 0:
                ed_video = input_tokens.index(VIDEO_TOKEN_ID, st)
            else:
                ed_video = len(input_tokens) + 1

            if ed_image < ed_video:
                # ===== 画像の処理 =====
                t, h, w = (
                    image_grid_thw[image_index][0],  # T (画像は常に1)
                    image_grid_thw[image_index][1],  # H_grid = H_patches / merge_size
                    image_grid_thw[image_index][2],  # W_grid = W_patches / merge_size
                )
                image_index += 1
                remain_images -= 1
                ed = ed_image
            else:
                # ===== 動画フレームの処理 (T=1に分割済み) =====
                t, h, w = (
                    video_grid_thw[video_index][0],  # 常に1 (Step 1で分割済み)
                    video_grid_thw[video_index][1],  # H_grid
                    video_grid_thw[video_index][2],  # W_grid
                )
                video_index += 1
                remain_videos -= 1
                ed = ed_video

            # LLMグリッドサイズ (merge_size=2 で割って圧縮後のサイズ)
            llm_grid_t = t.item()                            # = 1 (常に!)
            llm_grid_h = h.item() // spatial_merge_size     # H_patches / 2
            llm_grid_w = w.item() // spatial_merge_size     # W_patches / 2
            text_len = ed - st  # 視覚トークン前のテキスト長

            # ----- テキスト部分の position_ids -----
            st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
            # テキスト: t=h=w が全て同じ1D インデックス
            llm_pos_ids_list.append(
                torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx
            )
            # shape: (3, text_len)

            # ----- 視覚部分の position_ids -----
            # Qwen3-VL: t_index は常に 0 (タイムスタンプはテキストで表現)
            t_index = torch.arange(llm_grid_t).view(-1, 1).expand(-1, llm_grid_h * llm_grid_w).flatten()
            # t_index: (llm_grid_t × llm_grid_h × llm_grid_w,) = 0, 0, 0, ... (全て0)

            h_index = torch.arange(llm_grid_h).view(1, -1, 1).expand(llm_grid_t, -1, llm_grid_w).flatten()
            # h_index: (llm_grid_t × llm_grid_h × llm_grid_w,)
            # 例 (H=4): [0,0,0,0, 1,1,1,1, 2,2,2,2, 3,3,3,3] (W=4の場合)

            w_index = torch.arange(llm_grid_w).view(1, 1, -1).expand(llm_grid_t, llm_grid_h, -1).flatten()
            # w_index: (llm_grid_t × llm_grid_h × llm_grid_w,)
            # 例 (W=4): [0,1,2,3, 0,1,2,3, 0,1,2,3, 0,1,2,3] (H=4の場合)

            llm_pos_ids_list.append(
                torch.stack([t_index, h_index, w_index]) + text_len + st_idx
            )
            # shape: (3, llm_grid_t×llm_grid_h×llm_grid_w)

            st = ed + llm_grid_t * llm_grid_h * llm_grid_w

        # 末尾のテキスト
        if st < len(input_tokens):
            st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
            text_len = len(input_tokens) - st
            llm_pos_ids_list.append(
                torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx
            )
            # shape: (3, text_len)

        # 全セグメントを結合
        llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
        # llm_positions: (3, T_valid)

        position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(position_ids.device)
        mrope_position_deltas.append(llm_positions.max() + 1 - T_seq)

    mrope_position_deltas = torch.tensor(mrope_position_deltas).unsqueeze(1)
    # mrope_position_deltas: (B, 1)

    return position_ids, mrope_position_deltas
    # position_ids: (3, B, T_seq)


# ============================================================
# 3D MRoPE の適用
# ============================================================

def apply_multimodal_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor,
    mrope_section: List[int],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    LLM のアテンション内で 3D MRoPE を適用

    Qwen3-VL の MRoPE は head_dim を 3 分割して
    各部分に t, h, w の異なる周波数成分を適用する

    ========================================
    Shape
    ========================================
    入力:
        q: (B, H_llm, T_seq, head_dim) = (B, 28, T_seq, 128)
        k: (B, H_llm_kv, T_seq, head_dim) = (B, 4, T_seq, 128) (GQAのKV heads)
        cos: (max_seq_len, head_dim // 2)
        sin: (max_seq_len, head_dim // 2)
        position_ids: (3, B, T_seq)
        mrope_section: List[int] = [16, 24, 24]
            - 各次元の head_dim の分割数 (合計 = head_dim // 2 = 64)
            - t: 16, h: 24, w: 24

    出力:
        q_embed: (B, H_llm, T_seq, head_dim)
        k_embed: (B, H_llm_kv, T_seq, head_dim)

    ========================================
    処理詳細
    ========================================
    head_dim = 128
    head_dim // 2 = 64  (RoPE は半分に適用)
    mrope_section = [16, 24, 24]  (合計 = 64)

    cos, sin の分割:
        cos_t: (B, T_seq, 16)   ← t 次元の周波数
        cos_h: (B, T_seq, 24)   ← h 次元の周波数
        cos_w: (B, T_seq, 24)   ← w 次元の周波数

    position_ids による indexing:
        t_ids: (B, T_seq) → cos_t[t_ids]
        h_ids: (B, T_seq) → cos_h[h_ids]
        w_ids: (B, T_seq) → cos_w[w_ids]

    結合:
        cos_embed = cat([cos_t[t_ids], cos_h[h_ids], cos_w[w_ids]], dim=-1)
        # (B, T_seq, 64)
    """
    # mrope_section の例: [16, 24, 24] → 合計 64 = head_dim // 2
    cos_sections = torch.split(cos, mrope_section, dim=-1)
    sin_sections = torch.split(sin, mrope_section, dim=-1)
    # cos_sections: [(max_len, 16), (max_len, 24), (max_len, 24)]

    # 3次元の position_ids で各周波数成分を indexing
    # position_ids[0]: (B, T_seq) - temporal
    # position_ids[1]: (B, T_seq) - height
    # position_ids[2]: (B, T_seq) - width
    cos_t = cos_sections[0][position_ids[0]]   # (B, T_seq, 16)
    cos_h = cos_sections[1][position_ids[1]]   # (B, T_seq, 24)
    cos_w = cos_sections[2][position_ids[2]]   # (B, T_seq, 24)

    sin_t = sin_sections[0][position_ids[0]]   # (B, T_seq, 16)
    sin_h = sin_sections[1][position_ids[1]]   # (B, T_seq, 24)
    sin_w = sin_sections[2][position_ids[2]]   # (B, T_seq, 24)

    cos_embed = torch.cat([cos_t, cos_h, cos_w], dim=-1)
    # cos_embed: (B, T_seq, 64) = (B, T_seq, head_dim//2)

    sin_embed = torch.cat([sin_t, sin_h, sin_w], dim=-1)
    # sin_embed: (B, T_seq, 64)

    # head次元に展開してbroadcast
    cos_embed = cos_embed.unsqueeze(1)  # (B, 1, T_seq, head_dim//2)
    sin_embed = sin_embed.unsqueeze(1)  # (B, 1, T_seq, head_dim//2)

    # RoPE 適用
    q_embed = rotate_half_and_apply(q, cos_embed, sin_embed)
    # q_embed: (B, H_llm, T_seq, head_dim)

    k_embed = rotate_half_and_apply(k, cos_embed, sin_embed)
    # k_embed: (B, H_llm_kv, T_seq, head_dim)

    return q_embed, k_embed


def rotate_half_and_apply(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """
    RoPE (Rotary Position Embedding) の適用

    ========================================
    Shape
    ========================================
    入力:
        x: (B, H, T_seq, head_dim)
        cos: (B, 1, T_seq, head_dim//2)
        sin: (B, 1, T_seq, head_dim//2)

    出力:
        x_rotated: (B, H, T_seq, head_dim)

    ========================================
    数式
    ========================================
    head_dim = 128 の場合:
        x = [x_0, x_1, ..., x_63,  x_64, x_65, ..., x_127]
              ↑ "実部" (前半 head_dim//2)   ↑ "虚部" (後半 head_dim//2)

    rotate_half(x):
        x_rotated = [-x_64, -x_65, ..., -x_127, x_0, x_1, ..., x_63]

    RoPE:
        x_embed = x * cos + rotate_half(x) * sin

    テキストトークン (t=h=w=同値):
        実効的に 1D RoPE と等価 (t, h, w が全て同じインデックスを使用)

    画像トークン (t=0, h/wが2D):
        t次元は全て0の周波数成分 → 無効化
        h, w 次元が 2D 空間位置を表現
    """
    # x の前半と後半を取得
    x_half = x.shape[-1] // 2
    x1 = x[..., :x_half]    # (B, H, T_seq, head_dim//2) - "実部"
    x2 = x[..., x_half:]    # (B, H, T_seq, head_dim//2) - "虚部"

    # rotate_half(x) = [-x2, x1]
    x_rotated = torch.cat([-x2, x1], dim=-1)
    # x_rotated: (B, H, T_seq, head_dim)

    # cos, sin を head_dim 全体に拡張
    # cos: (B, 1, T_seq, head_dim//2) → (B, 1, T_seq, head_dim) by repeat
    cos_full = cos.repeat(1, 1, 1, 2)  # (B, 1, T_seq, head_dim)
    sin_full = sin.repeat(1, 1, 1, 2)  # (B, 1, T_seq, head_dim)

    # RoPE 適用
    x_embed = x * cos_full + x_rotated * sin_full
    # x_embed: (B, H, T_seq, head_dim)

    return x_embed


# ============================================================
# 具体的な数値例
# ============================================================

def concrete_example():
    """
    Interleaved MRoPE の具体的な数値例

    ========================================
    シナリオ
    ========================================
    入力: テキスト2トークン + 画像(2×2グリッド=4トークン) + テキスト2トークン
    T_seq = 2 + 4 + 2 = 8
    merge_size = 2 → llm_grid_h=1, llm_grid_w=1 (2×2 patches → 1×1 tokens)

    実際の例: 448×448画像 (32×32 patches → 16×16=256 visual tokens)
    ここでは説明のため 2×2 パッチ (4 tokens) の例を使用

    ========================================
    入力トークン列のイメージ
    ========================================
    入力:
        [T0, T1, <|vision_start|>, <image>, <image>, <image>, <image>, <|vision_end|>, T2, T3]
         ↑   ↑                                                                          ↑   ↑
         テキスト                    視覚トークン × 4                                   テキスト

    image_grid_thw = [[1, 2, 2]]  # T=1, H_patches=2, W_patches=2
    spatial_merge_size = 2
    → llm_grid_t=1, llm_grid_h=1, llm_grid_w=1
    この例では4 IMAGE_TOKENが実際には4つの視覚トークンに対応

    ========================================
    注: 実際のimage_grid_thwは T=1, H=H_patches (ViT出力のH)
        llm_grid_h = H_patches // merge_size (Merger後の高さ)
    ========================================

    より現実的な例 (448×448 画像, merge_size=2):
        image_grid_thw = [[1, 32, 32]]
        llm_grid_h = 32 // 2 = 16
        llm_grid_w = 32 // 2 = 16
        視覚トークン数 = 1 × 16 × 16 = 256

    ========================================
    position_ids の例 (簡略版: 2×2グリッド)
    ========================================
    """
    print("=== Interleaved MRoPE 具体例 ===\n")

    print("【シナリオ】")
    print("  入力: [TEXT×2] [IMG_VISION × 4 tokens] [TEXT×2]")
    print("  image_grid_thw = [[1, 2, 2]] (T=1, H=2, W=2)")
    print("  spatial_merge_size = 2")
    print("  llm_grid: T=1, H=1, W=1 (merge後)")
    print()

    # position_ids の構築 (手動)
    # テキスト部分: t=h=w が全て同じ連続インデックス
    text_pre = torch.arange(2).view(1, -1).expand(3, -1)
    # text_pre: (3, 2) = [[0,1],[0,1],[0,1]]

    # 視覚部分 (画像 2×2 グリッド → 1×1 LLM グリッド)
    # llm_grid_t=1, llm_grid_h=1, llm_grid_w=1
    t_idx = torch.tensor([0])          # t=0 (常に0)
    h_idx = torch.tensor([0])          # h=0 (1×1グリッドなので)
    w_idx = torch.tensor([0])          # w=0

    vision_pos = torch.stack([t_idx, h_idx, w_idx]) + 2
    # vision_pos: (3, 1) = [[2],[2],[2]] → まずst_idx=2でオフセット

    # 実際には4トークン (元の2×2パッチ数と同じ)
    # 説明のため1トークンで示す

    # テキスト後半
    text_post_start = max(text_pre.max().item(), vision_pos.max().item()) + 1
    text_post = torch.arange(2).view(1, -1).expand(3, -1) + text_post_start

    print("【position_ids の例】")
    print(f"  テキスト前半 [0,1]:  t={text_pre[0].tolist()}, h={text_pre[1].tolist()}, w={text_pre[2].tolist()}")
    print(f"  視覚トークン [2]:    t={vision_pos[0].tolist()}, h={vision_pos[1].tolist()}, w={vision_pos[2].tolist()}")
    print(f"  テキスト後半 [3,4]:  t={text_post[0].tolist()}, h={text_post[1].tolist()}, w={text_post[2].tolist()}")
    print()

    print("【画像 vs 動画の違い】")
    print()
    print("  画像 (T=1, H=4, W=4 の例):")
    print("    t_index: [0,0,0,...,0]  ← 全て0 (画像なので時間なし)")
    print("    h_index: [0,0,0,0, 1,1,1,1, 2,2,2,2, 3,3,3,3]  ← 高さ方向グリッド")
    print("    w_index: [0,1,2,3, 0,1,2,3, 0,1,2,3, 0,1,2,3]  ← 幅方向グリッド")
    print()

    print("  動画 (Qwen3-VL, T=3フレーム, H=2, W=2 の例):")
    print("  入力トークン列:")
    print("    <t0.00> <vision_start> [frame0×4 tokens] <vision_end>")
    print("    <t0.50> <vision_start> [frame1×4 tokens] <vision_end>")
    print("    <t1.00> <vision_start> [frame2×4 tokens] <vision_end>")
    print()
    print("  video_grid_thw (分割後): [[1,2,2],[1,2,2],[1,2,2]]")
    print()
    print("  frame0の position_ids:")
    print("    t_index: [0,0,0,0]  ← 常に0 (時間情報はタイムスタンプテキスト)")
    print("    h_index: [0,0,1,1]")
    print("    w_index: [0,1,0,1]")
    print()
    print("  frame1の position_ids:")
    print("    t_index: [0,0,0,0]  ← frame0と同じ0 (インデックスは<t0.50>の後続から)")
    print("    h_index: [0,0,1,1]  ← frame0と同じ (ただしst_idxでオフセット)")
    print("    w_index: [0,1,0,1]")
    print()
    print("  ポイント: t軸は常に0 → 時間的情報はテキストタイムスタンプ<t0.50>等で表現")

    print()
    print("【Qwen2.5-VL (Chunked) との比較】")
    print()
    print("  Qwen2.5-VL の動画 position_ids (T=3, H=2, W=2):")
    print("    t_index: [0,0,0,0, 50,50,50,50, 100,100,100,100]")
    print("    (second_per_grid_t × 2 = 25 × 2 = 50 のスケーリング)")
    print()
    print("  Qwen3-VL の動画 position_ids:")
    print("    t_index: [0,0,0,0, 0,0,0,0, 0,0,0,0]  ← 全フレームで0")
    print("    (時間情報はテキストタイムスタンプ '<t0.00>', '<t0.50>', '<t1.00>' で表現)")


if __name__ == "__main__":
    # 具体的な数値例を表示
    concrete_example()

    print("\n\n=== get_rope_index_3 のテスト ===\n")

    # 簡単なテスト
    B, T_seq = 1, 10
    input_ids = torch.zeros(B, T_seq, dtype=torch.long)
    # 位置5にIMAGE_TOKEN_ID (151655) を設定
    input_ids[0, 3] = IMAGE_TOKEN_ID  # 視覚トークン位置

    image_grid_thw = torch.tensor([[1, 4, 4]])  # T=1, H=4, W=4
    # llm_grid: T=1, H=2, W=2 (spatial_merge_size=2)
    # 視覚トークン数 = 1×2×2 = 4

    attention_mask = torch.ones(B, T_seq, dtype=torch.long)

    try:
        position_ids, mrope_deltas = get_rope_index_3(
            spatial_merge_size=2,
            input_ids=input_ids,
            image_grid_thw=image_grid_thw,
            attention_mask=attention_mask,
        )
        print(f"position_ids shape: {position_ids.shape}")
        # (3, 1, 10)
        print(f"mrope_deltas shape: {mrope_deltas.shape}")
        # (1, 1)
        print(f"t position_ids: {position_ids[0, 0].tolist()}")
        print(f"h position_ids: {position_ids[1, 0].tolist()}")
        print(f"w position_ids: {position_ids[2, 0].tolist()}")
    except Exception as e:
        print(f"テスト実行エラー (正常: 実際の実装では特殊トークンIDが必要): {e}")
