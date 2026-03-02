"""
Qwen3-Omni メインフロー - 全推論パイプライン簡略化疑似コード
============================================================

テキスト/音声/画像/動画の入力から、テキスト+音声の出力までの
6段階パイプラインを網羅する。

主な差分 (vs Qwen2.5-Omni):
    - Audio Encoder: Whisper-large-v3 (640M) → AuT (650M, スクラッチ学習)
    - Vision Encoder: ViT → SigLIP2-So400m (540M)
    - Thinker: Dense 7B → MoE 30B-A3B (活性パラメータ3B)
    - Talker: Dense → MoE 3B-A0.3B + マルチコードブック自己回帰
    - Token2Wav: DiT+BigVGAN → MTP (80M) + Code2Wav (200M, 因果ConvNet)
    - 位置符号化: TMRoPE → TM-RoPE (同一方式、MoE対応)
    - 出力音声レート: 12.5Hz コードトークン → 24kHz 波形

6段階パイプライン:
    Stage 1: 入力前処理
        - テキスト: Qwen トークナイザ (BPE, vocab=151,643)
        - 音声: 16kHz → 128ch メルスペクトログラム (25ms窓, 10msホップ)
        - 画像: patch_size=14, temporal_patch_size=2, 動的解像度
        - 動画: 動的フレームレート、音声と12.5Hzで同期

    Stage 2: エンコーダ
        - AuT Encoder (650M): メル → 12.5Hz トークン
            3× Conv2D (8倍ダウン) + 32× Self-Attention
        - SigLIP2-So400m (540M): パッチ → マージトークン
            ViT + PatchMerger (2×2→1)

    Stage 3: MoE Thinker (30B-A3B)
        - TM-RoPE によるマルチモーダル特徴統合
        - テキストトークン生成 (自己回帰)
        - 中間層隠れ状態を Talker に渡す
        - マルチモーダル特徴も Talker に直接渡す

    Stage 4: MoE Talker (3B-A0.3B)
        - 入力: Thinker中間層隠れ状態 + マルチモーダル特徴 + ストリームテキスト
        - マルチコードブック自己回帰: バックボーンが線形ヘッドで第0コードブック予測

    Stage 5: MTP Module (80M)
        - 固定ステップ自己回帰の密なTransformer
        - 現フレームの残余コードブックを予測

    Stage 6: Code2Wav (200M)
        - 軽量因果ConvNet
        - マルチコードブック RVQ → 24kHz 波形
        - ストリーミング 80ms フレーム単位

ストリーミング/並行処理:
    - チャンク化プリフィリング: Thinkerが現チャンクをprefillする間に
      Talkerが前チャンクから音声生成
    - Thinker と Talker の非同期動作
    - MoE によるロングシーケンスの KV キャッシュ削減
    - 初回パケット遅延: 234ms (並行度1)

HuggingFace 公式使用パターン:
    from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor
    from qwen_omni_utils import process_mm_info

    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        MODEL_PATH, dtype="auto", device_map="auto"
    )
    processor = Qwen3OmniMoeProcessor.from_pretrained(MODEL_PATH)

    text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    audios, images, videos = process_mm_info(conversation, use_audio_in_video=True)
    inputs = processor(text=text, audio=audios, images=images, videos=videos,
                       return_tensors="pt", padding=True)

    text_ids, audio = model.generate(
        **inputs, speaker="Ethan",
        thinker_return_dict_in_generate=True,
    )

API差分 (vs Qwen2.5-Omni):
    - クラス名: Qwen2_5OmniForConditionalGeneration → Qwen3OmniMoeForConditionalGeneration
    - プロセッサ: Qwen2_5OmniProcessor → Qwen3OmniMoeProcessor
    - model.disable_talker() で ~10GB VRAM 節約
    - generate() が (text_ids, audio) タプルを返す
    - speaker: "Chelsie" 等 → "Ethan", "Chelsie", "Aiden"
    - thinker_return_dict_in_generate, thinker_max_new_tokens, thinker_do_sample パラメータ
    - use_audio_in_video パラメータ (動画の音声トラック利用)
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Optional, Tuple


# ============================================
# 定数
# ============================================

SAMPLE_RATE = 16000           # 入力音声サンプリングレート (16kHz)
OUTPUT_SAMPLE_RATE = 24000    # 出力音声サンプリングレート (24kHz)
MEL_BINS = 128                # メルスペクトログラムのビン数
MEL_WINDOW_MS = 25            # メルスペクトログラムの窓サイズ (25ms)
MEL_HOP_MS = 10               # メルスペクトログラムのホップサイズ (10ms)
PATCH_SIZE = 14               # SigLIP2 パッチサイズ
TEMPORAL_PATCH_SIZE = 2       # 時間方向パッチサイズ
SPATIAL_MERGE_SIZE = 2        # PatchMerger 統合サイズ (2×2→1)
IMAGE_FACTOR = 28             # 画像サイズの倍数制約 (PATCH_SIZE * SPATIAL_MERGE_SIZE)
VOCAB_SIZE = 151643           # テキストボキャブラリサイズ (BPE)
AUDIO_TOKEN_RATE = 12.5       # 音声トークンレート (Hz) ※Qwen2.5-Omniの25Hzから半減
AUDIO_TOKEN_MS = 80           # 1トークンあたりの音声長 (ms)
NUM_RVQ_CODEBOOKS = 4         # RVQコードブック数 (推定)
CODE2WAV_FRAME_MS = 80        # Code2Wav のフレーム長 (ms)
SPEAKER_OPTIONS = ["Ethan", "Chelsie", "Aiden"]  # 利用可能話者


# ============================================
# 全体パイプラインクラス
# ============================================

class Qwen3OmniFullPipeline(nn.Module):
    """
    Qwen3-Omni 全体パイプライン

    6段階構成:
        Stage 1: 入力前処理 (テキスト / 音声 / 画像 / 動画)
        Stage 2: エンコーダ (AuT 650M + SigLIP2 540M)
        Stage 3: MoE Thinker 30B-A3B (マルチモーダル統合 + テキスト生成)
        Stage 4: MoE Talker 3B-A0.3B (第0コードブック予測)
        Stage 5: MTP Module 80M (残余コードブック予測)
        Stage 6: Code2Wav 200M (マルチコードブック RVQ → 24kHz 波形)

    Qwen2.5-Omni との主要差分:
        - Dense → MoE (Thinker 30B-A3B, Talker 3B-A0.3B)
        - Whisper → AuT (スクラッチ学習, 12.5Hz)
        - ViT → SigLIP2-So400m
        - DiT+BigVGAN → MTP + Code2Wav (因果ConvNet)
        - 出力がタプル (text_ids, audio)
    """

    def __init__(self):
        super().__init__()

        # ========================================
        # コンポーネント初期化
        # ========================================

        # Stage 2a: AuT Encoder (650M)
        # 詳細: audio_encoder.py
        # Whisper-large-v3 → AuT へ完全置換
        # 20M時間の教師あり音声データでスクラッチ学習
        self.audio_tower = AuTEncoder(
            num_mel_bins=128,           # メルスペクトログラムビン数
            d_model=768,                # AuT内部隠れ次元
            encoder_layers=32,          # Self-Attentionレイヤー数
            encoder_attention_heads=12, # アテンションヘッド数
            encoder_ffn_dim=3072,       # FFN中間次元
            output_dim=1024,            # 出力特徴次元 (Thinkerへの入力)
        )

        # Stage 2b: SigLIP2-So400m Vision Encoder (540M)
        # ViT → SigLIP2 へ変更。構造は ViT + PatchMerger 同等
        self.visual = SigLIP2Encoder(
            hidden_size=1024,           # SigLIP2 隠れ次元
            depth=24,                   # Transformerレイヤー数
            num_heads=16,               # アテンションヘッド数
            patch_size=14,              # パッチサイズ
            temporal_patch_size=2,      # 時間パッチサイズ
            spatial_merge_size=2,       # PatchMerger統合サイズ (2×2→1)
        )

        # Stage 3: MoE Thinker (30B-A3B)
        # Dense 7B → MoE 30B (活性パラメータ 3B)
        self.thinker = MoEThinker(
            hidden_size=4096,           # LLM隠れ次元
            num_layers=32,              # Transformerレイヤー数
            num_heads=32,               # アテンションヘッド数
            num_experts=16,             # エキスパート数 (推定)
            num_active_experts=2,       # 活性エキスパート数 (推定)
            intermediate_size=14336,    # FFN中間次元 (各エキスパート)
            vocab_size=VOCAB_SIZE,      # ボキャブラリサイズ
            talker_tap_layer=16,        # Talkerに渡す中間層のインデックス
        )

        # Stage 4: MoE Talker (3B-A0.3B)
        # Dense Talker → MoE (活性パラメータ 0.3B)
        # マルチコードブック自己回帰: 第0コードブックのみ予測
        self.talker = MoETalker(
            hidden_size=1024,           # Talker隠れ次元
            num_layers=8,               # Transformerレイヤー数
            num_experts=8,              # エキスパート数 (推定)
            num_active_experts=2,       # 活性エキスパート数 (推定)
            codebook_size=8295,         # 第0コードブックサイズ
            thinker_hidden_size=4096,   # Thinker隠れ次元 (射影用)
        )

        # Stage 5: MTP Module (80M)
        # 固定ステップ自己回帰の密なTransformer
        # 第0コードブック以外の残余コードブックを予測
        self.mtp = MTPModule(
            hidden_size=512,            # MTP隠れ次元
            num_layers=4,               # Transformerレイヤー数
            num_codebooks=NUM_RVQ_CODEBOOKS,  # 全コードブック数
            codebook_size=8295,         # 各コードブックのサイズ
        )

        # Stage 6: Code2Wav (200M)
        # DiT+BigVGAN → 軽量因果ConvNet
        # マルチコードブック RVQ → 24kHz 波形
        self.code2wav = Code2Wav(
            num_codebooks=NUM_RVQ_CODEBOOKS,
            codebook_size=8295,
            output_sample_rate=OUTPUT_SAMPLE_RATE,
        )


    def forward(
        self,
        text: str,
        audio: Optional[np.ndarray] = None,        # 16kHz モノラル
        image: Optional[torch.Tensor] = None,       # (3, H, W)
        video: Optional[torch.Tensor] = None,       # (T, 3, H, W)
        video_audio: Optional[np.ndarray] = None,   # 動画付随音声 (16kHz)
        return_audio: bool = False,
        speaker: str = "Ethan",
        use_audio_in_video: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Qwen3-Omni 6段階フォワードパス

        入力:
            text:  str            - ユーザーテキスト
            audio: (N_samples,)   - 16kHz モノラル音声 (optional)
            image: (3, H, W)      - RGB画像 (optional)
            video: (T, 3, H, W)   - 動画フレーム (optional)
            video_audio: (N_samples_v,) - 動画付随音声 (optional)
            return_audio: bool    - 音声出力を生成するか
            speaker: str          - 話者名 ("Ethan", "Chelsie", "Aiden")
            use_audio_in_video: bool - 動画音声トラックを利用するか

        出力:
            Dict {
                'text_ids': (1, L_gen)           - 生成テキストトークンID
                'text':     str                  - 生成テキスト
                'audio':    (1, N_samples) or None - 24kHz 音声波形
            }

        ※ HuggingFace の model.generate() は (text_ids, audio) タプルを返す
        """

        # ========================================
        # Stage 1: 入力前処理
        # ========================================

        # テキストのトークン化 (BPE, vocab=151,643)
        input_ids = self.tokenize(text)
        # input_ids: (1, L_text) - テキストトークンID

        # 音声の前処理 (メルスペクトログラム変換)
        audio_features = None
        audio_feature_lengths = None
        if audio is not None:
            audio_features, audio_feature_lengths = self.preprocess_audio(audio)
            # audio_features:        (1, 128, T_mel) - 128ch メルスペクトログラム
            # audio_feature_lengths: (1,)            - 有効フレーム数

        # 動画付随音声の前処理 (use_audio_in_video=True の場合)
        video_audio_features = None
        video_audio_feature_lengths = None
        if video_audio is not None and use_audio_in_video:
            video_audio_features, video_audio_feature_lengths = self.preprocess_audio(video_audio)
            # video_audio_features:        (1, 128, T_mel_v)
            # video_audio_feature_lengths: (1,)

        # 画像の前処理 (動的解像度)
        pixel_values = None
        image_grid_thw = None
        if image is not None:
            pixel_values, image_grid_thw = self.preprocess_image(image)
            # pixel_values:   (N_patches, C_patch) - パッチ化された画像
            # image_grid_thw: (1, 3)               - [T=2, H_patches, W_patches]

        # 動画の前処理 (動的フレームレート、12.5Hzで音声と同期)
        pixel_values_videos = None
        video_grid_thw = None
        if video is not None:
            pixel_values_videos, video_grid_thw = self.preprocess_video(video)
            # pixel_values_videos: (N_total_patches, C_patch) - パッチ化された動画
            # video_grid_thw:      (1, 3) - [T_patches, H_patches, W_patches]


        # ========================================
        # Stage 2: エンコーダ処理
        # ========================================

        # AuT Encoder (650M): メル → 12.5Hz トークン
        # ※ Qwen2.5-Omni の Whisper (25Hz) から半減
        audio_embeds = None
        if audio_features is not None:
            audio_embeds = self.audio_tower(audio_features, audio_feature_lengths)
            # audio_embeds: (1, T_mel//8, 1024)
            # 軸: (batch, 音声トークン数, 特徴次元)
            # T_mel//8: 3×Conv2D で8倍ダウンサンプリング
            # 1トークン = 80ms (12.5Hz) ※Qwen2.5-Omniは40ms (25Hz)

        # AuT Encoder (動画付随音声)
        video_audio_embeds = None
        if video_audio_features is not None:
            video_audio_embeds = self.audio_tower(
                video_audio_features, video_audio_feature_lengths
            )
            # video_audio_embeds: (1, T_mel_v//8, 1024)

        # SigLIP2-So400m (540M): パッチ → マージトークン
        # ※ Qwen2.5-Omni の ViT から SigLIP2 へ変更
        image_embeds = None
        if pixel_values is not None:
            image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
            # image_embeds: (N_merged, 1024)
            # N_merged = N_patches // (spatial_merge_size^2)
            # PatchMerger: 2×2=4パッチ → 1トークン

        # SigLIP2 (動画)
        video_embeds = None
        if pixel_values_videos is not None:
            video_embeds = self.visual(pixel_values_videos, grid_thw=video_grid_thw)
            # video_embeds: (N_merged_video, 1024)


        # ========================================
        # Stage 3: MoE Thinker (30B-A3B)
        # ========================================

        # テキスト埋め込み
        text_embeds = self.thinker.embed_tokens(input_ids)
        # text_embeds: (1, L_text, 4096)

        # マルチモーダル特徴の統合 (masked_scatter)
        # input_ids の特殊トークン位置にエンコーダ出力を埋め込む
        merged_embeds = self.merge_multimodal_features(
            text_embeds=text_embeds,
            input_ids=input_ids,
            audio_embeds=audio_embeds,              # <AUDIO> 位置に配置
            image_embeds=image_embeds,              # <IMAGE> 位置に配置
            video_embeds=video_embeds,              # <VIDEO> 位置に配置
            video_audio_embeds=video_audio_embeds,  # 動画音声も統合
        )
        # merged_embeds: (1, L_total, 4096)
        # L_total = L_text + L_audio_tokens + L_image_tokens + L_video_tokens + ...

        # TM-RoPE 位置ID計算 (3軸: temporal, height, width)
        position_ids = self.thinker.get_rope_index(
            input_ids=input_ids,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            audio_feature_lengths=audio_feature_lengths,
        )
        # position_ids: (3, 1, L_total) - [temporal, height, width]
        # テキスト: 3軸同一 (標準1D-RoPEと等価)
        # 音声:   3軸同一、1 ID = 80ms (12.5Hz)
        # 画像:   temporal固定、height/widthは空間位置
        # 動画:   temporalが時間増分

        # テキスト生成 (自己回帰) + 中間層隠れ状態の抽出
        generated_ids, hidden_states, mid_hidden_states, multimodal_features = \
            self.thinker.generate(
                inputs_embeds=merged_embeds,
                position_ids=position_ids,
                max_new_tokens=1024,
            )
        # generated_ids:       (1, L_gen)        - 生成テキストトークンID
        # hidden_states:       (1, L_gen, 4096)  - 最終層隠れ状態
        # mid_hidden_states:   (1, L_gen, 4096)  - 中間層 (talker_tap_layer) 隠れ状態
        # multimodal_features: (1, L_mm, 4096)   - マルチモーダル特徴 (Talkerに直接渡す)

        # テキストデコード
        generated_text = self.decode_tokens(generated_ids)


        # ========================================
        # Stage 4: MoE Talker (3B-A0.3B) [optional]
        # ========================================

        audio_waveform = None
        if return_audio:
            # Thinkerの中間層隠れ状態 + マルチモーダル特徴 + テキストを受けて
            # 第0コードブックを自己回帰予測
            codebook_0_tokens = self.talker.generate(
                thinker_mid_hidden=mid_hidden_states,      # (1, L_gen, 4096)
                multimodal_features=multimodal_features,   # (1, L_mm, 4096)
                text_token_ids=generated_ids,               # (1, L_gen)
                speaker=speaker,
            )
            # codebook_0_tokens: (1, L_speech) - 第0コードブックのトークン列
            # L_speech: 音声フレーム数 (12.5Hz)
            # バックボーンMoEが線形ヘッドで第0コードブックを予測


            # ========================================
            # Stage 5: MTP Module (80M)
            # ========================================

            # 固定ステップ自己回帰で残余コードブック (1, 2, ..., K-1) を予測
            all_codebook_tokens = self.mtp(
                codebook_0=codebook_0_tokens,  # (1, L_speech)
            )
            # all_codebook_tokens: (1, L_speech, NUM_RVQ_CODEBOOKS)
            # 各フレームに対して全コードブックのトークンが揃う
            # MTP: 密なTransformerで残余コードブックを固定ステップ予測


            # ========================================
            # Stage 6: Code2Wav (200M)
            # ========================================

            # マルチコードブック RVQ → 24kHz 波形
            # 軽量因果ConvNet、ストリーミング 80ms フレーム単位
            audio_waveform = self.code2wav(
                codebook_tokens=all_codebook_tokens,  # (1, L_speech, NUM_RVQ_CODEBOOKS)
            )
            # audio_waveform: (1, N_samples) at 24kHz
            # N_samples = L_speech * (OUTPUT_SAMPLE_RATE / AUDIO_TOKEN_RATE)
            #           = L_speech * 1920  (80ms × 24kHz = 1920 samples/frame)


        return {
            'text_ids': generated_ids,
            'text': generated_text,
            'audio': audio_waveform,
        }


    # ============================================
    # Stage 1: 前処理ヘルパーメソッド
    # ============================================

    def tokenize(self, text: str) -> torch.Tensor:
        """
        テキストのトークン化 (ChatMLフォーマット, BPE vocab=151,643)

        入力:
            text: str - ユーザーテキスト

        出力:
            input_ids: (1, L_text) - トークンID

        ChatMLフォーマット例:
            <|im_start|>system
            You are a helpful assistant.<|im_end|>
            <|im_start|>user
            <|audio_bos|><|AUDIO|><|audio_eos|>Describe this audio.<|im_end|>
            <|im_start|>assistant

        ※ 実際には Qwen3OmniMoeProcessor.apply_chat_template() を使用
        """
        # 概念的な処理のみ記載
        input_ids = torch.randint(0, VOCAB_SIZE, (1, len(text.split()) + 20))
        return input_ids


    def preprocess_audio(self, audio: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        音声の前処理: 16kHz波形 → 128chメルスペクトログラム

        入力:
            audio: (N_samples,) np.ndarray - 16kHz モノラル音声

        出力:
            mel_features:   (1, 128, T_mel) - メルスペクトログラム
                128:   メルビン数
                T_mel: フレーム数 = N_samples / 160 + 1 (10msホップ)
            feature_lengths: (1,) - 有効フレーム数

        メルスペクトログラム計算:
            - 窓サイズ: 25ms (400サンプル @ 16kHz)
            - ホップサイズ: 10ms (160サンプル @ 16kHz)
            - メルビン: 128チャネル
            - 結果のフレームレート: 100Hz (10ms間隔)
            - AuT で 8倍ダウンサンプリング後: 12.5Hz
        """
        T_mel = len(audio) // 160 + 1
        mel_features = torch.randn(1, 128, T_mel)
        feature_lengths = torch.tensor([T_mel])
        return mel_features, feature_lengths


    def preprocess_image(self, image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        画像の前処理: 動的解像度 + パッチ化

        入力:
            image: (3, H, W) - RGB画像

        出力:
            pixel_values: (N_patches, C_patch) - パッチ化された画像
                N_patches = T × H_patches × W_patches
                C_patch   = 3 × patch_size^2 × temporal_patch_size
                          = 3 × 14 × 14 × 2 = 1176
                ※ T=2: 各画像は2同一フレームとして扱う (temporal_patch_size=2)
            grid_thw: (1, 3) - [T=2, H_patches, W_patches]

        動的解像度:
            - 画像サイズを IMAGE_FACTOR=28 の倍数に調整
            - IMAGE_FACTOR = PATCH_SIZE(14) × SPATIAL_MERGE_SIZE(2)
        """
        C, H, W = image.shape
        # smart_resize: 28の倍数にリサイズ
        H_new = (H // IMAGE_FACTOR) * IMAGE_FACTOR
        W_new = (W // IMAGE_FACTOR) * IMAGE_FACTOR
        H_patches = H_new // PATCH_SIZE  # 空間パッチ数 (高さ)
        W_patches = W_new // PATCH_SIZE  # 空間パッチ数 (幅)
        T = 2  # 画像は2同一フレームとして扱う

        N_patches = T * H_patches * W_patches
        C_patch = 3 * PATCH_SIZE * PATCH_SIZE * TEMPORAL_PATCH_SIZE  # 1176
        pixel_values = torch.randn(N_patches, C_patch)
        # pixel_values: (N_patches, 1176)

        grid_thw = torch.tensor([[T, H_patches, W_patches]])
        # grid_thw: (1, 3)
        return pixel_values, grid_thw


    def preprocess_video(
        self,
        video: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        動画の前処理: 動的フレームレート + パッチ化

        入力:
            video: (T_frames, 3, H, W) - 動画フレーム

        出力:
            pixel_values: (N_total_patches, C_patch) - パッチ化された動画
                N_total_patches = T_patches × H_patches × W_patches
                T_patches = T_frames // temporal_patch_size
                C_patch   = 3 × 14 × 14 × 2 = 1176
            grid_thw: (1, 3) - [T_patches, H_patches, W_patches]

        動画と音声の同期:
            - 動画フレームは12.5Hzで音声トークンと同期
            - use_audio_in_video=True で動画の音声トラックも利用可能
        """
        T_frames, C, H, W = video.shape
        H_new = (H // IMAGE_FACTOR) * IMAGE_FACTOR
        W_new = (W // IMAGE_FACTOR) * IMAGE_FACTOR
        H_patches = H_new // PATCH_SIZE
        W_patches = W_new // PATCH_SIZE
        T_patches = T_frames // TEMPORAL_PATCH_SIZE

        N_total_patches = T_patches * H_patches * W_patches
        C_patch = 3 * PATCH_SIZE * PATCH_SIZE * TEMPORAL_PATCH_SIZE
        pixel_values = torch.randn(N_total_patches, C_patch)
        # pixel_values: (N_total_patches, 1176)

        grid_thw = torch.tensor([[T_patches, H_patches, W_patches]])
        # grid_thw: (1, 3)
        return pixel_values, grid_thw


    def merge_multimodal_features(
        self,
        text_embeds: torch.Tensor,
        input_ids: torch.Tensor,
        audio_embeds: Optional[torch.Tensor],
        image_embeds: Optional[torch.Tensor],
        video_embeds: Optional[torch.Tensor],
        video_audio_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        マルチモーダル特徴をテキスト埋め込みに統合

        入力:
            text_embeds:       (B, L_text, 4096)        - テキスト埋め込み
            input_ids:         (B, L_text)               - トークンID (特殊トークン位置特定用)
            audio_embeds:      (B, L_audio, 1024) or None - AuT 音声特徴
            image_embeds:      (N_merged, 1024) or None   - SigLIP2 画像特徴
            video_embeds:      (N_merged_v, 1024) or None - SigLIP2 動画特徴
            video_audio_embeds: (B, L_va, 1024) or None   - 動画付随音声特徴

        出力:
            merged: (B, L_total, 4096) - 統合後の埋め込み

        処理:
            1. input_ids から特殊トークン位置を特定
               <AUDIO>: 音声トークン位置
               <IMAGE>: 画像トークン位置
               <VIDEO>: 動画トークン位置
            2. エンコーダ出力を線形射影 (1024 → 4096)
            3. masked_scatter で対応位置に配置
        """
        B, L, D = text_embeds.shape  # (B, L_text, 4096)

        # 音声特徴の配置
        if audio_embeds is not None:
            # audio_embeds: (B, L_audio, 1024) → 線形射影 → (B, L_audio, 4096)
            # input_ids の <AUDIO> トークン位置に scatter
            pass

        # 画像特徴の配置
        if image_embeds is not None:
            # image_embeds: (N_merged, 1024) → 線形射影 → (N_merged, 4096)
            # input_ids の <IMAGE> トークン位置に scatter
            pass

        # 動画特徴の配置
        if video_embeds is not None:
            # video_embeds: (N_merged_v, 1024) → 線形射影 → (N_merged_v, 4096)
            # input_ids の <VIDEO> トークン位置に scatter
            pass

        # 動画付随音声の配置
        if video_audio_embeds is not None:
            # video_audio_embeds: (B, L_va, 1024) → 線形射影 → (B, L_va, 4096)
            # 動画音声用の特殊トークン位置に scatter
            pass

        return text_embeds  # 簡略化: 実際には scatter 後のテンソル


    def decode_tokens(self, token_ids: torch.Tensor) -> str:
        """トークンIDをテキストにデコード (BPE vocab=151,643)"""
        return "Generated text response"


# ============================================
# サブモジュール (詳細は個別ファイルを参照)
# ============================================

class AuTEncoder(nn.Module):
    """
    AuT (Audio Transformer) Encoder の簡略版

    詳細: audio_encoder.py

    3× Conv2D (8倍ダウン) + 32× Self-Attention → 12.5Hz トークン

    主な差分 (vs Qwen2.5-Omni Whisper):
        - ダウンサンプリング: Conv1D ×2 (4倍) → Conv2D ×3 (8倍)
        - トークンレート: 25Hz → 12.5Hz
        - エンコーダ層: 12 → 32
        - ウィンドウ: 固定2秒 → 動的1-8秒

    パラメータ: ~650M
    """

    def __init__(self, num_mel_bins, d_model, encoder_layers,
                 encoder_attention_heads, encoder_ffn_dim, output_dim):
        super().__init__()
        self.output_dim = output_dim
        # 概念的な構成のみ (実装は audio_encoder.py)
        self.proj = nn.Linear(d_model, output_dim)

    def forward(self, mel_features, feature_lengths):
        """
        入力:
            mel_features:   (B, 128, T_mel) - メルスペクトログラム
            feature_lengths: (B,)            - 有効フレーム数

        出力:
            audio_embeds: (B, T_mel//8, output_dim) - 12.5Hz トークン列

        処理フロー:
            (B, 128, T_mel)
              → 3× Conv2D 8倍ダウン: (B, T_mel//8, d_model)
              → 32× Self-Attention:   (B, T_mel//8, d_model)
              → LayerNorm + Linear:   (B, T_mel//8, output_dim)
        """
        B = mel_features.shape[0]
        T_mel = mel_features.shape[2]
        T_tokens = T_mel // 8  # 8倍ダウンサンプリング
        return torch.randn(B, T_tokens, self.output_dim)


class SigLIP2Encoder(nn.Module):
    """
    SigLIP2-So400m Vision Encoder の簡略版

    ViT + PatchMerger (2×2→1) でパッチを統合

    主な差分 (vs Qwen2.5-Omni ViT):
        - ViT → SigLIP2-So400m (540M)
        - シグモイド損失による対比学習で事前学習
        - PatchMerger は同一 (2×2→1)

    パラメータ: ~540M
    """

    def __init__(self, hidden_size, depth, num_heads, patch_size,
                 temporal_patch_size, spatial_merge_size):
        super().__init__()
        self.spatial_merge_size = spatial_merge_size
        self.hidden_size = hidden_size

    def forward(self, pixel_values, grid_thw):
        """
        入力:
            pixel_values: (N_patches, C_patch) - パッチ化された画像/動画
                C_patch = 3 × 14 × 14 × 2 = 1176
            grid_thw:     (num_images, 3) - [T, H_patches, W_patches]

        出力:
            features: (N_merged, hidden_size) - マージ後トークン列
                N_merged = N_patches // (spatial_merge_size^2)
                         = N_patches // 4

        処理フロー:
            (N_patches, 1176)
              → パッチ埋め込み: (N_patches, hidden_size)
              → ViT Transformerレイヤー: (N_patches, hidden_size)
              → PatchMerger 2×2→1: (N_merged, hidden_size)
        """
        N_patches = pixel_values.shape[0]
        N_merged = N_patches // (self.spatial_merge_size ** 2)
        return torch.randn(N_merged, self.hidden_size)


class MoEThinker(nn.Module):
    """
    MoE Thinker LLM (30B-A3B) の簡略版

    Dense 7B (Qwen2.5-Omni) → MoE 30B (活性パラメータ 3B) に置換

    特徴:
        - Mixture of Experts: 全パラメータ30Bだが推論時は3Bのみ活性化
        - TM-RoPE: マルチモーダル特徴の位置符号化 (3軸)
        - 中間層隠れ状態を Talker に渡す (talker_tap_layer)
        - マルチモーダル特徴も Talker に直接渡す
        - MoE により長シーケンスの KV キャッシュ効率化

    パラメータ: 30B (活性 3B)
    """

    def __init__(self, hidden_size, num_layers, num_heads,
                 num_experts, num_active_experts,
                 intermediate_size, vocab_size, talker_tap_layer):
        super().__init__()
        self.hidden_size = hidden_size
        self.talker_tap_layer = talker_tap_layer
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def embed_tokens(self, input_ids):
        """
        テキスト埋め込み

        入力: input_ids (B, L_text)
        出力: embeddings (B, L_text, hidden_size=4096)
        """
        return self.embed(input_ids)

    def get_rope_index(self, **kwargs):
        """
        TM-RoPE 位置IDの計算

        入力:
            input_ids:             (B, L_text)
            image_grid_thw:        (num_images, 3) or None
            video_grid_thw:        (num_videos, 3) or None
            audio_feature_lengths: (B,) or None

        出力:
            position_ids: (3, B, L_total) - [temporal, height, width]

        各モダリティの位置割り当て:
            テキスト: 3軸同一値 (標準1D-RoPEと等価)
            音声:   3軸同一、1 ID = 80ms (12.5Hz) ※ Qwen2.5-Omniは40ms
            画像:   temporal固定、height/widthは空間位置
            動画:   temporalが時間方向に増分
        """
        return torch.zeros(3, 1, 100)

    def generate(self, inputs_embeds, position_ids, max_new_tokens):
        """
        テキスト自己回帰生成 + 中間層隠れ状態の抽出

        入力:
            inputs_embeds: (B, L_total, 4096) - 統合埋め込み
            position_ids:  (3, B, L_total)    - TM-RoPE 位置ID
            max_new_tokens: int               - 最大生成トークン数

        出力:
            generated_ids:       (B, L_gen)       - 生成テキストトークンID
            hidden_states:       (B, L_gen, 4096) - 最終層隠れ状態
            mid_hidden_states:   (B, L_gen, 4096) - 中間層隠れ状態 (→Talker)
            multimodal_features: (B, L_mm, 4096)  - マルチモーダル特徴 (→Talker)

        ※ thinker_return_dict_in_generate=True で隠れ状態を取得
        ※ チャンク化プリフィリング: 現チャンクprefill中にTalkerが前チャンク処理
        """
        B, L, D = inputs_embeds.shape
        gen_ids = torch.randint(0, VOCAB_SIZE, (B, max_new_tokens))
        hidden = torch.randn(B, max_new_tokens, D)
        mid_hidden = torch.randn(B, max_new_tokens, D)
        mm_features = torch.randn(B, L, D)  # プリフィル部分のマルチモーダル特徴
        return gen_ids, hidden, mid_hidden, mm_features


class MoETalker(nn.Module):
    """
    MoE Talker (3B-A0.3B) の簡略版

    Dense Talker (Qwen2.5-Omni) → MoE 3B (活性パラメータ 0.3B) に置換

    特徴:
        - Thinker中間層隠れ状態を受信
        - マルチモーダル特徴も直接受信
        - ストリームテキストを入力
        - マルチコードブック自己回帰: バックボーンが線形ヘッドで第0コードブック予測
        - 残余コードブックの予測は MTP Module に委譲

    パラメータ: 3B (活性 0.3B)
    """

    def __init__(self, hidden_size, num_layers, num_experts,
                 num_active_experts, codebook_size, thinker_hidden_size):
        super().__init__()
        self.codebook_size = codebook_size
        # Thinker隠れ次元 → Talker隠れ次元への射影
        self.thinker_proj = nn.Linear(thinker_hidden_size, hidden_size)

    def generate(
        self,
        thinker_mid_hidden: torch.Tensor,
        multimodal_features: torch.Tensor,
        text_token_ids: torch.Tensor,
        speaker: str = "Ethan",
    ) -> torch.Tensor:
        """
        第0コードブックの自己回帰予測

        入力:
            thinker_mid_hidden:  (B, L_gen, 4096) - Thinker中間層隠れ状態
            multimodal_features: (B, L_mm, 4096)  - マルチモーダル特徴
            text_token_ids:      (B, L_gen)        - 生成テキストトークンID
            speaker:             str               - 話者名 ("Ethan", "Chelsie", "Aiden")

        出力:
            codebook_0: (B, L_speech) - 第0コードブックのトークン列
                L_speech: 音声フレーム数 (12.5Hz)

        処理:
            1. thinker_mid_hidden を射影 (4096 → 1024)
            2. multimodal_features と text_token_ids を条件として統合
            3. MoE バックボーンで自己回帰生成
            4. 線形ヘッドで第0コードブックを予測

        ※ Thinker と非同期動作: Thinkerが次チャンクをprefillする間に生成
        """
        B = thinker_mid_hidden.shape[0]
        L_gen = thinker_mid_hidden.shape[1]
        # 概算: テキストトークン1つあたり約2音声フレーム
        L_speech = L_gen * 2
        return torch.randint(0, self.codebook_size, (B, L_speech))


class MTPModule(nn.Module):
    """
    MTP (Multi-Token Prediction) Module (80M)

    固定ステップ自己回帰の密なTransformer
    第0コードブック (Talker出力) から残余コードブック (1, 2, ..., K-1) を予測

    Qwen2.5-Omni では DiT が音声コード→波形を一括で変換していたが、
    Qwen3-Omni では MTP + Code2Wav の2段階に分離

    パラメータ: ~80M
    """

    def __init__(self, hidden_size, num_layers, num_codebooks, codebook_size):
        super().__init__()
        self.num_codebooks = num_codebooks
        self.codebook_size = codebook_size
        # 各コードブック用の埋め込みと予測ヘッド
        self.codebook_embeds = nn.ModuleList([
            nn.Embedding(codebook_size, hidden_size)
            for _ in range(num_codebooks)
        ])
        self.codebook_heads = nn.ModuleList([
            nn.Linear(hidden_size, codebook_size)
            for _ in range(num_codebooks)
        ])

    def forward(self, codebook_0: torch.Tensor) -> torch.Tensor:
        """
        残余コードブックの予測

        入力:
            codebook_0: (B, L_speech) - 第0コードブックトークン列 (Talker出力)

        出力:
            all_codebooks: (B, L_speech, num_codebooks) - 全コードブックトークン
                all_codebooks[:, :, 0] = codebook_0 (入力そのまま)
                all_codebooks[:, :, 1:] = 予測された残余コードブック

        処理:
            1. 第0コードブックを埋め込み: (B, L_speech, hidden_size)
            2. 固定ステップ自己回帰: k=1,...,K-1 の順に予測
               各ステップで前コードブックの情報を条件として使用
            3. 各コードブックヘッドで予測: (B, L_speech, codebook_size)
               → argmax → (B, L_speech)
        """
        B, L_speech = codebook_0.shape
        all_codebooks = torch.zeros(B, L_speech, self.num_codebooks, dtype=torch.long)
        all_codebooks[:, :, 0] = codebook_0

        # 残余コードブックを固定ステップで順に予測
        for k in range(1, self.num_codebooks):
            # 概念的: 前コードブックを条件として k 番目を予測
            all_codebooks[:, :, k] = torch.randint(
                0, self.codebook_size, (B, L_speech)
            )

        # all_codebooks: (B, L_speech, NUM_RVQ_CODEBOOKS)
        return all_codebooks


class Code2Wav(nn.Module):
    """
    Code2Wav (200M) - 軽量因果ConvNet

    マルチコードブック RVQ → 24kHz 波形変換
    ストリーミング対応: 80ms フレーム単位で逐次出力

    Qwen2.5-Omni の Token2Wav (DiT + BigVGAN) との差分:
        - DiT + BigVGAN → 軽量因果ConvNet
        - 拡散モデル → 直接変換 (低レイテンシ)
        - ストリーミング: 80ms フレーム単位

    パラメータ: ~200M
    """

    def __init__(self, num_codebooks, codebook_size, output_sample_rate):
        super().__init__()
        self.num_codebooks = num_codebooks
        self.codebook_size = codebook_size
        self.output_sample_rate = output_sample_rate
        # 1フレーム (80ms) あたりの出力サンプル数
        self.samples_per_frame = int(output_sample_rate * CODE2WAV_FRAME_MS / 1000)
        # = 24000 * 0.08 = 1920 samples/frame

    def forward(self, codebook_tokens: torch.Tensor) -> torch.Tensor:
        """
        マルチコードブック RVQ → 24kHz 波形

        入力:
            codebook_tokens: (B, L_speech, num_codebooks) - 全コードブックトークン

        出力:
            waveform: (B, N_samples) - 24kHz 音声波形
                N_samples = L_speech × samples_per_frame
                          = L_speech × 1920

        処理:
            1. 各コードブックを埋め込み + RVQ 復号 (加算)
               → (B, L_speech, hidden_size)
            2. 因果ConvNet でアップサンプリング
               → (B, N_samples)
            3. ストリーミング: 80ms フレーム単位で逐次生成可能

        ※ ストリーミングでは1フレーム (80ms=1920samples) ずつ出力
        """
        B, L_speech, K = codebook_tokens.shape
        N_samples = L_speech * self.samples_per_frame
        # N_samples = L_speech × 1920
        waveform = torch.randn(B, N_samples)
        return waveform


# ============================================
# 推論パイプラインの使用例
# ============================================

def example_inference():
    """
    Qwen3-Omni 全体パイプラインの使用例

    Qwen3OmniFullPipeline を実際にインスタンス化し、
    6段階パイプラインの各ステージのフォワードパスを実行して形状を確認する。

    ※ 縮小版モデルでの実行。実モデルの各パラメータはコメントに記載。
    """

    pipeline = Qwen3OmniFullPipeline()
    pipeline.eval()

    print("=" * 70)
    print("Qwen3-Omni 全体パイプライン 使用例")
    print("=" * 70)

    # ========================================
    # 例1: テキスト + 音声入力 (Stage 1-3)
    # ========================================
    print()
    print("-" * 50)
    print("例1: テキスト + 音声入力")
    print("-" * 50)

    audio = np.random.randn(80000).astype(np.float32)  # 5秒 @ 16kHz
    audio_features, audio_feature_lengths = pipeline.preprocess_audio(audio)
    # audio_features: (1, 128, 501)

    T_mel = audio_features.shape[2]
    T_audio_tokens = T_mel // 8  # 8倍ダウンサンプリング

    # AuT Encoder
    audio_embeds = pipeline.audio_tower(audio_features, audio_feature_lengths)
    # audio_embeds: (1, T_mel//8, 1024)

    print(f"  [Stage 1: 入力前処理]")
    print(f"    音声入力: {audio.shape} ({len(audio)/SAMPLE_RATE:.1f}秒 @ {SAMPLE_RATE}Hz)")
    print(f"    メルスペクトログラム: {audio_features.shape} (B, 128ch, T_mel={T_mel})")
    print(f"    有効フレーム数: {audio_feature_lengths.tolist()}")
    print()
    print(f"  [Stage 2: AuT Encoder (650M)]")
    print(f"    入力: {audio_features.shape} → 3xConv2D 8倍ダウン → 32xSelfAttn")
    print(f"    出力: {audio_embeds.shape} (B, T_mel//8={T_audio_tokens}, 1024)")
    print(f"    トークンレート: 12.5Hz (80ms/token)")
    print(f"    ダウンサンプリング: {T_mel} → {T_audio_tokens} ({T_mel/T_audio_tokens:.1f}倍)")
    print(f"    [vs Qwen2.5-Omni] 25Hz(40ms) → 12.5Hz(80ms), トークン数半減")

    # ========================================
    # 例2: テキスト + 画像入力 (Stage 1-3)
    # ========================================
    print()
    print("-" * 50)
    print("例2: テキスト + 画像入力")
    print("-" * 50)

    image = torch.randn(3, 504, 504)  # RGB画像
    pixel_values, image_grid_thw = pipeline.preprocess_image(image)
    # pixel_values: (N_patches, 1176)

    H_new = (504 // IMAGE_FACTOR) * IMAGE_FACTOR  # 504 → 504
    W_new = (504 // IMAGE_FACTOR) * IMAGE_FACTOR
    H_patches = H_new // PATCH_SIZE  # 36
    W_patches = W_new // PATCH_SIZE  # 36
    T_img = 2
    N_patches = T_img * H_patches * W_patches  # 2 * 36 * 36 = 2592

    # SigLIP2 Encoder
    image_embeds = pipeline.visual(pixel_values, grid_thw=image_grid_thw)
    N_merged = N_patches // (SPATIAL_MERGE_SIZE ** 2)  # 2592 // 4 = 648

    print(f"  [Stage 1: 入力前処理]")
    print(f"    画像入力: {image.shape} (C, H, W)")
    print(f"    動的解像度: {504}x{504} → {H_new}x{W_new} (28の倍数)")
    print(f"    パッチ化: {pixel_values.shape} (N_patches={N_patches}, C_patch=1176)")
    print(f"    grid_thw: {image_grid_thw.tolist()} [T={T_img}, H={H_patches}, W={W_patches}]")
    print()
    print(f"  [Stage 2: SigLIP2-So400m (540M)]")
    print(f"    入力: {pixel_values.shape} → ViT → PatchMerger 2x2→1")
    print(f"    出力: {image_embeds.shape} (N_merged={N_merged}, 1024)")
    print(f"    パッチマージ: {N_patches} → {N_merged} ({N_patches/N_merged:.0f}倍圧縮)")
    print(f"    [vs Qwen2.5-Omni] ViT → SigLIP2-So400m")

    # テキスト処理 + Thinker
    input_ids = pipeline.tokenize("What is shown in this image?")
    text_embeds = pipeline.thinker.embed_tokens(input_ids)
    position_ids = pipeline.thinker.get_rope_index(
        input_ids=input_ids, image_grid_thw=image_grid_thw,
    )

    generated_ids, hidden_states, mid_hidden, mm_features = pipeline.thinker.generate(
        inputs_embeds=text_embeds,
        position_ids=position_ids,
        max_new_tokens=50,
    )

    print()
    print(f"  [Stage 3: MoE Thinker (30B-A3B)]")
    print(f"    テキスト埋め込み: {text_embeds.shape} (B, L_text, 4096)")
    print(f"    TM-RoPE position_ids: {position_ids.shape} [temporal, height, width]")
    print(f"    生成テキストID: {generated_ids.shape} (B, L_gen)")
    print(f"    最終層隠れ状態: {hidden_states.shape} (B, L_gen, 4096)")
    print(f"    中間層隠れ状態: {mid_hidden.shape} (→ Talker)")
    print(f"    マルチモーダル特徴: {mm_features.shape} (→ Talker)")
    print(f"    [vs Qwen2.5-Omni] Dense 7B → MoE 30B-A3B")

    # ========================================
    # 例3: 動画 + 音声入力 (Stage 1-3)
    # ========================================
    print()
    print("-" * 50)
    print("例3: 動画 + 音声入力 (use_audio_in_video=True)")
    print("-" * 50)

    video = torch.randn(16, 3, 280, 280)  # 16フレーム
    pixel_values_v, video_grid_thw = pipeline.preprocess_video(video)
    video_embeds = pipeline.visual(pixel_values_v, grid_thw=video_grid_thw)

    # 動画付随音声 (16フレーム分 ≈ 概算)
    video_audio = np.random.randn(32000).astype(np.float32)  # 2秒 @ 16kHz
    va_features, va_lengths = pipeline.preprocess_audio(video_audio)
    va_embeds = pipeline.audio_tower(va_features, va_lengths)

    H_v = (280 // IMAGE_FACTOR) * IMAGE_FACTOR  # 280
    H_p_v = H_v // PATCH_SIZE  # 20
    W_p_v = H_p_v  # 20
    T_p_v = 16 // TEMPORAL_PATCH_SIZE  # 8
    N_patches_v = T_p_v * H_p_v * W_p_v
    N_merged_v = N_patches_v // (SPATIAL_MERGE_SIZE ** 2)

    print(f"  [Stage 1: 入力前処理]")
    print(f"    動画入力: {video.shape} (T, C, H, W)")
    print(f"    パッチ化: {pixel_values_v.shape} (N_total_patches, C_patch)")
    print(f"    grid_thw: {video_grid_thw.tolist()} [T={T_p_v}, H={H_p_v}, W={W_p_v}]")
    print(f"    動画音声: {video_audio.shape} → メル: {va_features.shape}")
    print()
    print(f"  [Stage 2: SigLIP2 + AuT]")
    print(f"    動画: {pixel_values_v.shape} → SigLIP2 → {video_embeds.shape}")
    print(f"    音声: {va_features.shape} → AuT → {va_embeds.shape}")
    print(f"    動画と音声の同期: 12.5Hz")

    # ========================================
    # 例4: Talker + MTP + Code2Wav (Stage 4-6)
    # ========================================
    print()
    print("-" * 50)
    print("例4: 音声生成パイプライン (Stage 4-6)")
    print("-" * 50)

    # Stage 4: MoE Talker → 第0コードブック
    codebook_0 = pipeline.talker.generate(
        thinker_mid_hidden=mid_hidden,       # (1, 50, 4096)
        multimodal_features=mm_features,     # (1, L_mm, 4096)
        text_token_ids=generated_ids,        # (1, 50)
        speaker="Ethan",
    )
    # codebook_0: (1, L_speech)

    # Stage 5: MTP → 全コードブック
    all_codebooks = pipeline.mtp(codebook_0=codebook_0)
    # all_codebooks: (1, L_speech, NUM_RVQ_CODEBOOKS)

    # Stage 6: Code2Wav → 24kHz 波形
    waveform = pipeline.code2wav(codebook_tokens=all_codebooks)
    # waveform: (1, N_samples)

    L_speech = codebook_0.shape[1]
    N_samples = waveform.shape[1]
    duration_sec = N_samples / OUTPUT_SAMPLE_RATE

    print(f"  [Stage 4: MoE Talker (3B-A0.3B)]")
    print(f"    入力: mid_hidden {mid_hidden.shape} + mm_features {mm_features.shape}")
    print(f"    入力: text_token_ids {generated_ids.shape}, speaker='Ethan'")
    print(f"    出力: codebook_0 {codebook_0.shape} (第0コードブックのみ)")
    print(f"    [vs Qwen2.5-Omni] Dense Talker → MoE 3B-A0.3B")
    print()
    print(f"  [Stage 5: MTP Module (80M)]")
    print(f"    入力: codebook_0 {codebook_0.shape}")
    print(f"    出力: all_codebooks {all_codebooks.shape} ({NUM_RVQ_CODEBOOKS}コードブック)")
    print(f"    固定ステップ自己回帰で残余コードブック予測")
    print()
    print(f"  [Stage 6: Code2Wav (200M)]")
    print(f"    入力: all_codebooks {all_codebooks.shape}")
    print(f"    出力: waveform {waveform.shape} ({duration_sec:.2f}秒 @ {OUTPUT_SAMPLE_RATE}Hz)")
    print(f"    1フレーム = {CODE2WAV_FRAME_MS}ms = {pipeline.code2wav.samples_per_frame} samples")
    print(f"    ストリーミング: {CODE2WAV_FRAME_MS}ms フレーム単位で逐次出力")
    print(f"    [vs Qwen2.5-Omni] DiT+BigVGAN → 因果ConvNet (低レイテンシ)")

    # ========================================
    # 例5: 全体フォワードパス (6段階一括)
    # ========================================
    print()
    print("-" * 50)
    print("例5: 全体フォワードパス (6段階一括)")
    print("-" * 50)

    with torch.no_grad():
        result = pipeline(
            text="What is shown in this image?",
            image=image,
            return_audio=True,
            speaker="Ethan",
        )

    print(f"  入力: text + 画像 (504x504)")
    print(f"  return_audio=True, speaker='Ethan'")
    print()
    print(f"  出力:")
    print(f"    text_ids: {result['text_ids'].shape}")
    print(f"    text: '{result['text']}'")
    if result['audio'] is not None:
        audio_dur = result['audio'].shape[1] / OUTPUT_SAMPLE_RATE
        print(f"    audio: {result['audio'].shape} ({audio_dur:.2f}秒 @ {OUTPUT_SAMPLE_RATE}Hz)")
    else:
        print(f"    audio: None")

    # ========================================
    # 例6: HuggingFace 公式APIとの対応
    # ========================================
    print()
    print("-" * 50)
    print("例6: HuggingFace 公式API対応表")
    print("-" * 50)

    print(f"""
  [HuggingFace 公式コード]
    from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor
    from qwen_omni_utils import process_mm_info

    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(MODEL_PATH, ...)
    processor = Qwen3OmniMoeProcessor.from_pretrained(MODEL_PATH)

    text = processor.apply_chat_template(conversation, ...)  # → Stage 1 (tokenize)
    audios, images, videos = process_mm_info(conversation)   # → Stage 1 (前処理)
    inputs = processor(text=text, audio=audios, ...)          # → Stage 1 (テンソル化)

    text_ids, audio = model.generate(**inputs, speaker="Ethan")
    # text_ids → Stage 3 (Thinker生成)
    # audio   → Stage 4-6 (Talker→MTP→Code2Wav)

  [対応関係]
    Stage 1: processor.apply_chat_template() + process_mm_info() + processor()
    Stage 2: model内部 (AuT + SigLIP2)
    Stage 3: model.generate() → text_ids
    Stage 4: model.generate() → 内部 Talker → codebook_0
    Stage 5: model.generate() → 内部 MTP → all_codebooks
    Stage 6: model.generate() → audio (24kHz waveform)

  [API差分 vs Qwen2.5-Omni]
    クラス名:     Qwen2_5OmniForConditionalGeneration → Qwen3OmniMoeForConditionalGeneration
    プロセッサ:   Qwen2_5OmniProcessor → Qwen3OmniMoeProcessor
    generate戻り値: テキストのみ → (text_ids, audio) タプル
    VRAM節約:     なし → model.disable_talker() で ~10GB 節約
    話者:         "Chelsie" 等 → "Ethan", "Chelsie", "Aiden"
    新パラメータ: thinker_return_dict_in_generate, thinker_max_new_tokens,
                  thinker_do_sample, use_audio_in_video
""")

    # ========================================
    # パイプライン全体のレイテンシ情報
    # ========================================
    print("-" * 50)
    print("パイプライン全体情報")
    print("-" * 50)
    print(f"""
  [パラメータ数]
    AuT Encoder:      ~650M
    SigLIP2-So400m:   ~540M
    MoE Thinker:      ~30B (活性 3B)
    MoE Talker:       ~3B  (活性 0.3B)
    MTP Module:       ~80M
    Code2Wav:         ~200M
    合計:             ~34.5B (活性 ~4.8B)

  [ストリーミング/並行処理]
    - チャンク化プリフィリング:
        Thinker が現チャンクを prefill する間に
        Talker が前チャンクから音声生成 (非同期)
    - MoE により長シーケンスの KV キャッシュ削減
    - 初回パケット遅延: 234ms (並行度1)

  [音声トークンレート比較]
    Qwen2.5-Omni: 25Hz  (40ms/token, Whisper 4倍ダウン)
    Qwen3-Omni:   12.5Hz (80ms/token, AuT 8倍ダウン) → トークン数半減
""")


if __name__ == "__main__":
    example_inference()
