"""
Qwen3VL - メインフロー
======================

Qwen3-VLの全体処理フロー（画像+テキスト入力 → logits出力）を疑似コードで示します。

論文: Qwen3-VL Technical Report (2025)
公式実装: https://github.com/QwenLM/Qwen3-VL

処理の流れ:
1. 前処理: 画像/動画のスマートリサイズ → パッチ化
2. Vision Encoder (SigLIP-2 ViT): パッチ埋め込み → ViT layers → DeepStack特徴抽出
3. MLP Merger: 2×2空間圧縮 → LLM次元に射影
4. Interleaved MRoPE: position_ids 計算
5. LLM (Qwen3): 視覚+テキスト混合トークン → logits

============================================================
Shape Convention
============================================================
B: バッチサイズ
T_seq: LLMシーケンス長 (視覚トークン + テキストトークン)
N_patches: ViT入力パッチ数 (全バッチ・全画像の合計)
N_v: LLM入力視覚トークン数 = N_patches / (merge_size²)
P: パッチサイズ = 14 (ピクセル)
C: 入力チャンネル数 = 3 (RGB)
D_v: Vision Encoder隠れ次元 = 1152
D_llm: LLM隠れ次元 = 3584 (7Bモデル)
vocab_size: 語彙サイズ = 151936
merge_size: MLP Mergerの空間圧縮倍率 = 2
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
from PIL import Image
import types


# ============================================================
# 設定
# ============================================================

MODEL_CONFIGS = {
    "Qwen3-VL-2B": {
        "vision_config": {
            "embed_dim": 1152,      # D_v: Vision Encoder隠れ次元
            "num_heads": 16,        # H_v: Vision Encoderのアテンションヘッド数
            "num_layers": 32,       # L_v: Vision Encoderのレイヤー数
            "patch_size": 14,       # P: パッチサイズ (px)
            "spatial_merge_size": 2, # merge_size: MLP Mergerの空間圧縮倍率
        },
        "llm_config": {
            "hidden_size": 1536,    # D_llm: LLM隠れ次元 (2Bモデル)
            "num_layers": 28,       # L_llm: LLMレイヤー数
            "num_heads": 12,        # H_llm: LLMアテンションヘッド数
            "vocab_size": 151936,   # 語彙サイズ
        },
    },
    "Qwen3-VL-7B": {
        "vision_config": {
            "embed_dim": 1152,
            "num_heads": 16,
            "num_layers": 32,
            "patch_size": 14,
            "spatial_merge_size": 2,
        },
        "llm_config": {
            "hidden_size": 3584,    # D_llm: LLM隠れ次元 (7Bモデル)
            "num_layers": 28,
            "num_heads": 28,
            "vocab_size": 151936,
        },
    },
}

# 特殊トークンID
IMAGE_TOKEN_ID = 151655       # <image>プレースホルダー
VIDEO_TOKEN_ID = 151656       # <video>プレースホルダー
VISION_START_TOKEN_ID = 151652  # <|vision_start|>
VISION_END_TOKEN_ID = 151653    # <|vision_end|>


# ============================================================
# 前処理: スマートリサイズ
# ============================================================

def smart_resize(
    height: int,
    width: int,
    factor: int,
    min_pixels: Optional[int] = None,
    max_pixels: Optional[int] = None,
) -> Tuple[int, int]:
    """
    アスペクト比を維持しつつ、パッチ数が[min_pixels, max_pixels]に収まるようリサイズ

    ========================================
    Shape
    ========================================
    入力:
        height: int - 画像の高さ (px)
        width: int - 画像の幅 (px)
        factor: int - factor=patch_size × merge_size = 14 × 2 = 28
        min_pixels: int - 最小ピクセル数 (= IMAGE_MIN_TOKEN_NUM × factor²)
        max_pixels: int - 最大ピクセル数 (= IMAGE_MAX_TOKEN_NUM × factor²)

    出力:
        (h_bar, w_bar): (int, int) - リサイズ後の高さ・幅 (factorで割り切れる)

    ========================================
    処理詳細
    ========================================
    1. h_bar, w_bar を factor で丸める
    2. h_bar × w_bar > max_pixels: βで縮小
       β = sqrt((h × w) / max_pixels)
       h_bar = floor(h / β / factor) × factor
    3. h_bar × w_bar < min_pixels: βで拡大
       β = sqrt(min_pixels / (h × w))
       h_bar = ceil(h × β / factor) × factor

    例:
        元画像: 1000×500px, factor=28, max_pixels=16384×28²=12,845,056
        → h_bar=1000, w_bar=504 (500を28の倍数に丸め)
        → 1000×504 > max? → β計算してリサイズ
    """
    # 注: 実装の詳細はvision_process.pyのsmart_resize()参照
    import math

    MAX_RATIO = 200
    IMAGE_MIN_TOKEN_NUM = 4
    IMAGE_MAX_TOKEN_NUM = 16384

    max_pixels = max_pixels or (IMAGE_MAX_TOKEN_NUM * factor ** 2)
    min_pixels = min_pixels or (IMAGE_MIN_TOKEN_NUM * factor ** 2)

    if max(height, width) / min(height, width) > MAX_RATIO:
        raise ValueError(f"アスペクト比が大きすぎます: {max(height, width) / min(height, width):.1f}")

    h_bar = max(factor, round(height / factor) * factor)
    w_bar = max(factor, round(width / factor) * factor)

    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = math.floor(height / beta / factor) * factor
        w_bar = math.floor(width / beta / factor) * factor
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor

    return h_bar, w_bar


# ============================================================
# 前処理: 画像をパッチに分割
# ============================================================

def preprocess_image(
    image: Image.Image,
    patch_size: int = 14,
    merge_size: int = 2,
    min_pixels: Optional[int] = None,
    max_pixels: Optional[int] = None,
) -> Tuple[torch.Tensor, Tuple[int, int, int]]:
    """
    PIL画像を ViT 入力パッチに変換

    ========================================
    Shape
    ========================================
    入力:
        image: PIL.Image (H_orig, W_orig, 3)
    出力:
        patches: (N_patches, C×P²)
            N_patches = H_patches × W_patches
            H_patches = H_resized / P
            W_patches = W_resized / P
            C×P² = 3 × 14² = 588
        grid_thw: (3,) = [T=1, H_patches, W_patches]

    ========================================
    処理詳細 (Processor内部で実行)
    ========================================
    1. RGB変換
    2. smart_resize() でリサイズ
       factor = patch_size × merge_size = 14 × 2 = 28
    3. ToTensor + Normalize (ImageNet統計量)
       mean = [0.485, 0.456, 0.406]
       std  = [0.229, 0.224, 0.225]
    4. パッチ分割:
       (C, H_resized, W_resized) → (N_patches, C×P²)

    例:
        入力: 448×672 PIL Image
        factor=28 → リサイズ後: 448×672 (そのまま)
        H_patches = 448 / 14 = 32
        W_patches = 672 / 14 = 48
        N_patches = 32 × 48 = 1536
        patches: (1536, 588)
        grid_thw: [1, 32, 48]
    """
    W_orig, H_orig = image.size
    factor = patch_size * merge_size  # = 28

    h_resized, w_resized = smart_resize(H_orig, W_orig, factor=factor,
                                         min_pixels=min_pixels, max_pixels=max_pixels)

    image_resized = image.resize((w_resized, h_resized))

    # (H, W, C) → (C, H, W)
    img_tensor = torch.from_numpy(
        __import__('numpy').array(image_resized).transpose(2, 0, 1)
    ).float() / 255.0

    # ImageNet正規化
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img_tensor = (img_tensor - mean) / std
    # img_tensor: (3, H_resized, W_resized)

    H_patches = h_resized // patch_size   # 例: 448//14 = 32
    W_patches = w_resized // patch_size   # 例: 672//14 = 48
    N_patches = H_patches * W_patches     # 例: 1536

    # パッチ分割: (3, H, W) → (N_patches, 3×P²)
    patches = img_tensor.unfold(1, patch_size, patch_size) \
                         .unfold(2, patch_size, patch_size)
    # patches: (3, H_patches, W_patches, P, P)
    patches = patches.permute(1, 2, 0, 3, 4).reshape(N_patches, 3 * patch_size * patch_size)
    # patches: (N_patches, 588)

    grid_thw = (1, H_patches, W_patches)
    return patches, grid_thw


# ============================================================
# Qwen3VLForConditionalGeneration - メインクラス
# ============================================================

class Qwen3VLForConditionalGeneration(nn.Module):
    """
    Qwen3-VL フルモデル

    ========================================
    入力 Shape
    ========================================
    pixel_values: (N_patches, C×P²) = (N_patches, 588)
        - 全バッチ・全画像のパッチを1次元に並べたもの
    image_grid_thw: (num_images, 3)
        - 各画像の [T, H_patches, W_patches]
        - 例: [[1, 32, 32], [1, 48, 24]]
    input_ids: (B, T_seq)
        - テキストトークン + IMAGE_TOKEN プレースホルダー
        - IMAGE_TOKEN_ID (151655) が視覚トークンの位置
    attention_mask: (B, T_seq)
        - 有効トークンに1、パディングに0
    position_ids: (3, B, T_seq)
        - Interleaved MRoPEのための [temporal, height, width] 位置ID
    labels: (B, T_seq) or None
        - 訓練時: -100 (損失除外) or トークンID
        - 推論時: None

    ========================================
    出力 Shape
    ========================================
    loss: scalar (訓練時のみ)
    logits: (B, T_seq, vocab_size) = (B, T_seq, 151936)

    ========================================
    内部次元 (7Bモデル)
    ========================================
    D_v = 1152 (Vision Encoder)
    D_llm = 3584 (LLM)
    merge_size = 2 (MLP Merger空間圧縮)
    """

    def __init__(self, config):
        super().__init__()

        # ========================================
        # 1. Vision Encoder (SigLIP-2 ViT)
        # ========================================
        # 詳細は vision_encoder.py 参照
        self.visual = Qwen3VisionTransformerPretrainedModel(config.vision_config)
        # 入力: (N_patches, C×P²) = (N_patches, 588)
        # 出力: (N_patches, D_v) = (N_patches, 1152)

        # ========================================
        # 2. LLM (Qwen3)
        # ========================================
        self.model = Qwen3Model(config)
        # 入力: inputs_embeds (B, T_seq, D_llm)
        # 出力: hidden_states (B, T_seq, D_llm)

        # ========================================
        # 3. Language Model Head
        # ========================================
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # (B, T_seq, D_llm) → (B, T_seq, vocab_size)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        **kwargs,
    ) -> Dict:
        """
        Qwen3VL フォワードパス

        ========================================
        処理フロー & Shape変化
        ========================================
        Step 1: Vision Encoding
            pixel_values: (N_patches, 588)
            → visual_features: (N_v, D_llm)

        Step 2: 入力埋め込み構築
            input_ids: (B, T_seq)
            → inputs_embeds: (B, T_seq, D_llm)  # 視覚埋め込みで置換

        Step 3: MRoPE position_ids
            (3, B, T_seq)

        Step 4: LLM Forward
            inputs_embeds: (B, T_seq, D_llm)
            → hidden_states: (B, T_seq, D_llm)

        Step 5: LM Head
            → logits: (B, T_seq, vocab_size)

        Step 6: Loss (訓練時のみ)
            → loss: scalar
        """
        # ========================================
        # Step 1: Vision Encoding
        # ========================================
        if pixel_values is not None:
            # pixel_values: (N_patches, C×P²) = (N_patches, 588)
            # image_grid_thw: (num_images, 3)

            # Vision Encoder
            visual_features = self.visual(pixel_values, grid_thw=image_grid_thw)
            # visual_features: (N_v, D_llm)
            # N_v = sum(T×H×W / merge_size² for all images)
            # = N_patches / merge_size² = N_patches / 4

        if pixel_values_videos is not None:
            # 動画も同様に処理
            video_features = self.visual(pixel_values_videos, grid_thw=video_grid_thw)
            # video_features: (N_v_video, D_llm)

        # ========================================
        # Step 2: 入力埋め込みの構築
        # ========================================
        # input_ids の IMAGE_TOKEN_ID 位置に visual_features を埋め込む
        if inputs_embeds is None:
            inputs_embeds = self._get_model_inputs_embeds(
                input_ids=input_ids,
                pixel_values=pixel_values,
                pixel_values_videos=pixel_values_videos,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
            )
        # inputs_embeds: (B, T_seq, D_llm)
        # - テキスト位置: word_embedding(input_ids)
        # - 視覚トークン位置: visual_features の対応スライス

        # ========================================
        # Step 3: MRoPE Position IDs の取得
        # ========================================
        # position_ids は外部から渡されるか、ここで計算
        if position_ids is None:
            position_ids, mrope_position_deltas = get_rope_index(
                spatial_merge_size=self.visual.spatial_merge_size,
                input_ids=input_ids,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                attention_mask=attention_mask,
            )
        # position_ids: (3, B, T_seq)
        # 3次元: [temporal, height, width]

        # ========================================
        # Step 4: LLM Forward
        # ========================================
        # inputs_embeds: (B, T_seq, D_llm)
        # position_ids: (3, B, T_seq) - Interleaved MRoPE用
        outputs = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            **kwargs,
        )
        hidden_states = outputs.last_hidden_state
        # hidden_states: (B, T_seq, D_llm)

        # ========================================
        # Step 5: Language Model Head
        # ========================================
        logits = self.lm_head(hidden_states)
        # logits: (B, T_seq, vocab_size) = (B, T_seq, 151936)

        # ========================================
        # Step 6: Loss 計算 (訓練時のみ)
        # ========================================
        loss = None
        if labels is not None:
            loss = self._compute_loss(logits, labels)
            # loss: scalar

        return {
            "loss": loss,
            "logits": logits,
            "hidden_states": outputs.hidden_states,
        }

    def _get_model_inputs_embeds(
        self,
        input_ids: torch.LongTensor,
        pixel_values: Optional[torch.Tensor],
        pixel_values_videos: Optional[torch.Tensor],
        image_grid_thw: Optional[torch.LongTensor],
        video_grid_thw: Optional[torch.LongTensor],
    ) -> torch.FloatTensor:
        """
        input_ids のIMAGE/VIDEO_TOKEN位置に視覚埋め込みを挿入して inputs_embeds を構築

        ========================================
        Shape
        ========================================
        入力:
            input_ids: (B, T_seq)
                - IMAGE_TOKEN_ID (151655) が視覚トークンの位置
                - VIDEO_TOKEN_ID (151656) が動画トークンの位置
            pixel_values: (N_patches, 588) or None
            image_grid_thw: (num_images, 3) or None

        出力:
            inputs_embeds: (B, T_seq, D_llm)
                - テキスト位置: word_embedding(input_ids)
                - IMAGE_TOKEN位置: visual_features の連続したスライス
                - VIDEO_TOKEN位置: video_features の連続したスライス

        ========================================
        処理詳細
        ========================================
        1. テキスト埋め込みを初期化: word_embedding(input_ids)
        2. 画像トークン: input_ids == IMAGE_TOKEN_ID のマスクで位置特定
           → visual_features を対応位置に代入
        3. 動画トークン: input_ids == VIDEO_TOKEN_ID のマスクで位置特定
           → video_features を対応位置に代入
        """
        # テキスト埋め込み
        inputs_embeds = self.model.embed_tokens(input_ids)
        # inputs_embeds: (B, T_seq, D_llm)

        if pixel_values is not None:
            # Vision Encoder で視覚特徴量を計算
            visual_features = self.visual(pixel_values, grid_thw=image_grid_thw)
            # visual_features: (N_v, D_llm)
            # N_v = sum(1 × H_patches/merge_size × W_patches/merge_size for all images)

            # IMAGE_TOKEN_ID の位置に視覚特徴量を代入
            image_mask = (input_ids == IMAGE_TOKEN_ID)
            # image_mask: (B, T_seq) bool

            # 注: visual_features は全バッチの視覚トークンを1次元に並べた形
            # image_mask.flatten() で対応するインデックスを特定して代入
            inputs_embeds[image_mask] = visual_features.to(inputs_embeds.dtype)

        if pixel_values_videos is not None:
            video_features = self.visual(pixel_values_videos, grid_thw=video_grid_thw)
            # video_features: (N_v_video, D_llm)

            video_mask = (input_ids == VIDEO_TOKEN_ID)
            inputs_embeds[video_mask] = video_features.to(inputs_embeds.dtype)

        return inputs_embeds
        # inputs_embeds: (B, T_seq, D_llm)

    def _compute_loss(
        self,
        logits: torch.FloatTensor,
        labels: torch.LongTensor,
    ) -> torch.FloatTensor:
        """
        Cross-Entropy Loss 計算 (テキストトークンのみ)

        詳細は loss_computation.py 参照

        ========================================
        Shape
        ========================================
        入力:
            logits: (B, T_seq, vocab_size)
            labels: (B, T_seq)
                - -100: 視覚トークン・プレフィックス (損失除外)
                - 正値: テキストトークンID (損失計算対象)

        出力:
            loss: scalar
        """
        # 1トークンずらして次トークン予測
        shift_logits = logits[..., :-1, :].contiguous()
        # shift_logits: (B, T_seq-1, vocab_size)

        shift_labels = labels[..., 1:].contiguous()
        # shift_labels: (B, T_seq-1)

        # Cross-Entropy (-100 は自動的に無視)
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            # (B×(T_seq-1), vocab_size)
            shift_labels.view(-1),
            # (B×(T_seq-1),)
            ignore_index=-100,
        )
        # loss: scalar
        return loss


# ============================================================
# 使用例
# ============================================================

def example_usage():
    """
    Qwen3-VL の推論サンプル

    ========================================
    Shape Summary
    ========================================
    Input (448×448 画像1枚):
        pixel_values:   (N_patches=1024, 588)
        image_grid_thw: (1, 3) = [[1, 32, 32]]
        input_ids:      (B=1, T_seq) - テキスト + 256 IMAGE_TOKEN (merge後)

    Output:
        logits: (B=1, T_seq, 151936)
    """
    print("=== Qwen3-VL Example Usage ===\n")

    # 448×448 画像の例
    # N_patches = (448/14) × (448/14) = 32 × 32 = 1024
    # N_v = N_patches / 4 = 256 (MLP Mergerの2×2圧縮後)
    # T_text = 10 (テキストトークン)
    # T_seq = T_text + N_v = 10 + 256 = 266

    B = 1
    N_patches = 1024    # 32×32 パッチ (448×448 画像)
    C_times_P2 = 588    # C×P² = 3 × 14² = 588
    N_v = 256           # N_patches / 4 (merge後の視覚トークン数)
    T_text = 10         # テキストトークン数 (デモ用)
    T_seq = T_text + N_v  # = 266
    vocab_size = 151936
    hidden_size = 3584

    # config オブジェクト (SimpleNamespace で属性アクセス可能に)
    config_obj = types.SimpleNamespace(
        vision_config={
            "embed_dim": 1152,
            "num_heads": 16,
            "num_layers": 32,
            "patch_size": 14,
            "spatial_merge_size": 2,
            "d_llm": hidden_size,
        },
        hidden_size=hidden_size,
        vocab_size=vocab_size,
    )

    # モデル初期化
    model = Qwen3VLForConditionalGeneration(config_obj)
    model.eval()

    # ダミー入力テンソル
    # pixel_values: (N_patches, C×P²) = (1024, 588)
    pixel_values = torch.randn(N_patches, C_times_P2)

    # image_grid_thw: (num_images, 3) = [[T=1, H_patches=32, W_patches=32]]
    image_grid_thw = torch.tensor([[1, 32, 32]], dtype=torch.long)

    # input_ids: (B, T_seq) = (1, 266)
    # 位置 T_text..T_seq-1 に IMAGE_TOKEN_ID (151655) を設定
    input_ids = torch.zeros(B, T_seq, dtype=torch.long)
    input_ids[0, T_text:] = IMAGE_TOKEN_ID  # 256個の視覚トークンプレースホルダー

    # attention_mask: (B, T_seq) = (1, 266)
    attention_mask = torch.ones(B, T_seq, dtype=torch.long)

    print(f"Input shapes:")
    print(f"  pixel_values:   {pixel_values.shape}")     # (1024, 588)
    print(f"  image_grid_thw: {image_grid_thw.shape}")   # (1, 3)
    print(f"  input_ids:      {input_ids.shape}")        # (1, 266)
    print(f"  attention_mask: {attention_mask.shape}")   # (1, 266)
    print()

    with torch.no_grad():
        output = model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            attention_mask=attention_mask,
        )

    print(f"Output shapes:")
    print(f"  logits: {output['logits'].shape}")         # (1, 266, 151936)
    print()
    print(f"注: N_v = {N_v} = {N_patches} / 4 (MLP Mergerの2×2圧縮後)")


# ============================================================
# ダミークラス（実装は他ファイル参照）
# ============================================================

class Qwen3VisionTransformerPretrainedModel(nn.Module):
    """
    SigLIP-2 ベース Vision Encoder (+ MLP Merger)
    詳細は vision_encoder.py 参照

    ========================================
    Shape
    ========================================
    入力:
        pixel_values: (N_patches, C×P²) = (N_patches, 588)
        grid_thw: (num_images, 3) = [[T, H_patches, W_patches], ...]

    出力:
        visual_tokens: (N_v, D_llm)
            N_v = sum(T × H_patches/merge_size × W_patches/merge_size)
            = N_patches / (merge_size²) = N_patches / 4
    """
    def __init__(self, config):
        super().__init__()
        self.spatial_merge_size = config.get("spatial_merge_size", 2)
        self.d_llm = config.get("d_llm", 3584)
        # 注: 完全実装はvision_encoder.pyのQwen3VisionTransformerPretrainedModel参照。ここではexample()実行のためshapeのみ正しいダミーを返す

    def forward(self, pixel_values, grid_thw):
        # pixel_values: (N_patches, 588)
        # → (N_v, D_llm)
        # 注: 完全実装はvision_encoder.pyのQwen3VisionTransformerPretrainedModel参照。ここではexample()実行のためshapeのみ正しいダミーを返す
        N_v = pixel_values.shape[0] // (self.spatial_merge_size ** 2)
        return torch.zeros(N_v, self.d_llm)


class Qwen3Model(nn.Module):
    """
    Qwen3 LLM (Decoder-only Transformer)

    ========================================
    Shape
    ========================================
    入力:
        inputs_embeds: (B, T_seq, D_llm)
        attention_mask: (B, T_seq)
        position_ids: (3, B, T_seq) - Interleaved MRoPE

    出力:
        last_hidden_state: (B, T_seq, D_llm)
    """
    def __init__(self, config):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        # 注: 実際のLLM実装は省略

    def forward(self, inputs_embeds, attention_mask, position_ids, **kwargs):
        # inputs_embeds: (B, T_seq, D_llm)
        # → last_hidden_state: (B, T_seq, D_llm)
        # 注: Qwen3 LLMバックボーンは28層×数万行のため省略。transformersライブラリの実装参照。example()実行のためshapeのみ正しいダミーを返す
        return types.SimpleNamespace(
            last_hidden_state=torch.zeros_like(inputs_embeds),
            hidden_states=None,
        )


def get_rope_index(
    spatial_merge_size,
    input_ids,
    image_grid_thw,
    video_grid_thw,
    attention_mask,
):
    """
    Interleaved MRoPE の position_ids 計算
    詳細は rope_and_position.py 参照

    ========================================
    Shape
    ========================================
    出力:
        position_ids: (3, B, T_seq)
        mrope_position_deltas: (B, 1)
    """
    # 注: 完全実装はrope_and_position.pyのget_rope_index_3()参照。example()実行のためshapeのみ正しいダミーを返す
    B, T_seq = input_ids.shape
    return torch.zeros(3, B, T_seq, dtype=torch.long), torch.zeros(B, 1, dtype=torch.long)


if __name__ == "__main__":
    example_usage()
