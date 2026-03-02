"""
Qwen3-ASR - ForcedAligner (タイムスタンプ予測) 詳細
====================================================

このファイルはQwen3-ForcedAligner-0.6Bの
詳細な処理フローを理解するための疑似コードです。

論文: https://arxiv.org/abs/2601.21337 (Section 3)
関連論文: https://arxiv.org/abs/2601.18220 (LLM-ForcedAligner)
公式実装: qwen_asr/inference/qwen3_forced_aligner.py

============================================================
概要
============================================================
Qwen3-ForcedAligner-0.6Bは非自己回帰 (NAR) のタイムスタンプ予測モデル。
音声とテキストのペアから、各単語/文字の開始・終了タイムスタンプを推定する。

キーアイデア: Slot-Filling方式
- テキスト中に [time] 特殊トークンを挿入
- LLMが全 [time] スロットに同時にタイムスタンプインデックスを予測
- インデックス × 80ms = 実際のタイムスタンプ

============================================================
Shape Convention
============================================================
B:           バッチサイズ
T_mel:       メルスペクトログラムフレーム数
T_audio:     Audio Encoder出力フレーム数 (= T_mel // 8)
T_text:      テキストトークン数 ([time]トークン含む)
T_combined:  Audio + Text の統合トークン数
D_hidden:    LLM隠れ次元 (1536 for 0.6B)
N_classes:   タイムスタンプクラス数 (最大3750 = 300s / 80ms)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass


# ============================================================
# データクラス
# ============================================================

@dataclass
class TimestampResult:
    """タイムスタンプ結果"""
    text: str          # 単語/文字
    start_sec: float   # 開始時刻 (秒)
    end_sec: float     # 終了時刻 (秒)


# ============================================================
# 1. ForcedAligner 全体
# ============================================================

class Qwen3ForcedAligner:
    """
    Qwen3-ForcedAligner-0.6B

    ========================================
    アーキテクチャ
    ========================================
    - AuT Encoder: 180M params (Qwen3-ASR-0.6Bと共有)
    - Qwen3-0.6B LLM: テキスト+音声の統合理解
    - Timestamp Prediction Layer: Linear(D_hidden → N_classes)
      - N_classes = 3750 (= 300s / 80ms)

    ========================================
    通常ASRとの違い
    ========================================
    - 自己回帰ではなく、NAR (非自己回帰) 推論
    - LM Headの代わりに Timestamp Prediction Layer
    - 出力シーケンスと入力シーケンスのshiftなし (causal training)
    - [time] トークン位置のみに損失を計算

    ========================================
    対応言語 (11言語)
    ========================================
    Chinese, English, Cantonese, French, German, Italian,
    Japanese, Korean, Portuguese, Russian, Spanish
    """

    def __init__(self, model, processor):
        self.model = model          # ForcedAligner用モデル
        self.processor = processor  # Qwen3ForceAlignProcessor

        # タイムスタンプ関連定数
        self.frame_duration_ms = 80   # AuT Encoderの1フレーム = 80ms
        self.max_classes = 3750       # 最大300秒対応 (300000ms / 80ms)

    def align(
        self,
        audio: np.ndarray,              # (num_samples,) float32 @ 16kHz
        text: str,                       # 認識テキスト (例: "Hello world")
        language: str = "English",       # 言語
        granularity: str = "word",       # "word" or "char"
    ) -> List[TimestampResult]:
        """
        音声-テキスト ペアの強制アライメント

        ========================================
        Shape
        ========================================
        入力:
            audio: (num_samples,) float32 @ 16kHz
            text: str - 認識テキスト

        処理中:
            mel_features:   (1, T_mel, 128) - メルスペクトログラム
            audio_features: (1, T_audio, D_aut_out) - Audio Encoder出力
            input_ids:      (1, T_text) - [time]トークン含むトークンID
            hidden_states:  (1, T_combined, D_hidden=1536)
            timestamp_logits: (1, T_combined, N_classes=3750)

        出力:
            List[TimestampResult]: [(text, start_sec, end_sec), ...]
        """

        # ========================================
        # ステップ1: テキストのトークン化 ([time]挿入)
        # ========================================
        # 言語に応じた分かち書き + [time]トークン挿入
        #
        # 例 (英語, word-level):
        #   入力: "Hello world"
        #   → 分かち書き: ["Hello", "world"]
        #   → [time]挿入: "Hello [time][time] world [time][time]"
        #                         ↑start ↑end         ↑start ↑end
        #
        # 例 (中国語, char-level):
        #   入力: "你好世界"
        #   → 分かち書き: ["你", "好", "世", "界"]
        #   → [time]挿入: "你 [time][time] 好 [time][time] 世 [time][time] 界 [time][time]"
        #
        # 例 (日本語, char-level):
        #   入力: "こんにちは"
        #   → 分かち書き: ["こ", "ん", "に", "ち", "は"]
        #   → [time]挿入: 各文字の後に [time][time]
        tokenized_text, word_boundaries = self._tokenize_with_time_slots(
            text, language, granularity
        )
        # tokenized_text: "[time]挿入済みテキスト"
        # word_boundaries: [(word, time_slot_indices), ...]

        # ========================================
        # ステップ2: 入力準備 (Processor)
        # ========================================
        # Chat Template + Audio + Text → model入力
        messages = [
            {"role": "system", "content": ""},
            {"role": "user", "content": [
                {"type": "audio", "audio": audio},
                {"type": "text", "text": tokenized_text},
            ]},
        ]

        inputs = self.processor(
            messages=messages,
            return_tensors="pt",
        )
        # inputs:
        # {
        #   "input_ids":             (1, T_text) int64
        #   "attention_mask":        (1, T_text)
        #   "input_features":        (1, T_mel, 128)
        #   "feature_attention_mask": (1, T_mel)
        # }

        # ========================================
        # ステップ3: モデル推論 (NAR, 非自己回帰)
        # ========================================
        # 通常のASRと異なり、generate() ではなく forward() で一括推論
        with torch.no_grad():
            outputs = self.model(**inputs)
        # outputs["timestamp_logits"]: (1, T_combined, N_classes=3750)

        # ========================================
        # ステップ4: [time]スロットのタイムスタンプ抽出
        # ========================================
        timestamp_logits = outputs["timestamp_logits"]
        # timestamp_logits: (1, T_combined, 3750)

        # [time]トークン位置のみ抽出
        time_token_id = self.processor.time_token_id
        time_mask = (inputs["input_ids"][0] == time_token_id)  # (T_text,)
        time_positions = time_mask.nonzero(as_tuple=True)[0]   # (num_time_slots,)

        # 各[time]位置でargmax → タイムスタンプインデックス
        time_logits = timestamp_logits[0, time_positions]  # (num_time_slots, 3750)
        time_indices = time_logits.argmax(dim=-1)          # (num_time_slots,)

        # ========================================
        # ステップ5: インデックス → 秒数変換
        # ========================================
        # index × 80ms = タイムスタンプ (ms)
        timestamps_ms = time_indices.float() * self.frame_duration_ms
        # timestamps_ms: (num_time_slots,) float

        # ========================================
        # ステップ6: タイムスタンプ修正 (LISベース)
        # ========================================
        # 予測されたタイムスタンプが単調増加でない場合、
        # LIS (Longest Increasing Subsequence) ベースで修正
        timestamps_ms = self._fix_timestamps_monotonicity(timestamps_ms)

        # ========================================
        # ステップ7: 結果構築
        # ========================================
        # [time][time] ペアから (start, end) を抽出
        results = []
        for word, (start_slot_idx, end_slot_idx) in word_boundaries:
            start_sec = timestamps_ms[start_slot_idx].item() / 1000.0
            end_sec = timestamps_ms[end_slot_idx].item() / 1000.0
            results.append(TimestampResult(
                text=word,
                start_sec=start_sec,
                end_sec=end_sec,
            ))

        return results

    def _tokenize_with_time_slots(
        self,
        text: str,
        language: str,
        granularity: str,
    ) -> Tuple[str, List[Tuple[str, Tuple[int, int]]]]:
        """
        言語に応じた分かち書きと[time]スロット挿入

        ========================================
        言語別の分かち書き戦略
        ========================================
        1. スペース区切り言語 (English, French, German等):
           - スペースで単語分割
           - 各単語の後に [time][time] を挿入

        2. 中国語 (Chinese, Cantonese):
           - 文字単位で分割
           - 各文字の後に [time][time] を挿入

        3. 日本語 (Japanese):
           - 文字単位で分割 (漢字/ひらがな/カタカナ)
           - 各文字の後に [time][time] を挿入

        4. 韓国語 (Korean):
           - jieba辞書ベースの分かち書き
           - 各単位の後に [time][time] を挿入

        ========================================
        出力例 (English, word-level)
        ========================================
        入力: "Hello world"
        出力:
            text: "Hello [time][time] world [time][time]"
            boundaries: [
                ("Hello", (0, 1)),   # slot index 0=start, 1=end
                ("world", (2, 3)),   # slot index 2=start, 3=end
            ]
        """
        pass

    def _fix_timestamps_monotonicity(
        self,
        timestamps_ms: torch.Tensor,
    ) -> torch.Tensor:
        """
        タイムスタンプの単調性修正 (LISベース)

        ========================================
        問題
        ========================================
        モデルの予測は必ずしも単調増加にならない場合がある。
        例: [100, 200, 150, 300, 250, 400]
                        ↑ 逆転     ↑ 逆転

        ========================================
        解決: LIS (Longest Increasing Subsequence)
        ========================================
        1. 最長増加部分列を求める
           [100, 200, 150, 300, 250, 400]
           → LIS: [100, 200, 300, 400] (インデックス: 0, 1, 3, 5)

        2. LISに含まれないインデックスのタイムスタンプを線形補間
           index 2 (150): 200と300の間で補間
           index 4 (250): 300と400の間で補間

        ========================================
        Shape
        ========================================
        入力/出力: (num_time_slots,) float
        """
        pass


# ============================================================
# 2. ForcedAligner モデル内部
# ============================================================

class Qwen3ForcedAlignerModel(nn.Module):
    """
    ForcedAligner の内部モデル

    ========================================
    通常ASRモデルとの差分
    ========================================
    1. LM Head → Timestamp Prediction Layer に置換
    2. 出力シーケンスのshift なし (causal training)
    3. [time]トークン位置のみにCE損失を計算
    """

    def __init__(self, config):
        super().__init__()
        # Audio Encoder (ASRと同じ)
        self.audio_tower = None  # Qwen3ASRAudioEncoder

        # Text Model (Qwen3-0.6B)
        self.model = None  # Qwen3ASRThinkerTextModel

        # Timestamp Prediction Layer (LM Headの代わり)
        # D_hidden → N_classes (classify_num)
        self.timestamp_head = nn.Linear(
            config.hidden_size,  # 1536 (0.6B)
            config.classify_num,  # 3750 (= 300s / 80ms)
            bias=True,
        )

    def forward(
        self,
        input_ids: torch.Tensor,               # (B, T_text)
        attention_mask: torch.Tensor,           # (B, T_text)
        input_features: torch.Tensor,           # (B, T_mel, 128)
        feature_attention_mask: torch.Tensor,   # (B, T_mel)
    ) -> Dict[str, torch.Tensor]:
        """
        ========================================
        Shape
        ========================================
        入力:
            input_ids:              (B, T_text) int64
            input_features:         (B, T_mel, 128) float
            feature_attention_mask: (B, T_mel) int64

        中間:
            audio_features:    (B, T_audio, D_aut_out)
            inputs_embeds:     (B, T_combined, D_hidden=1536)
            hidden_states:     (B, T_combined, D_hidden=1536)

        出力:
            timestamp_logits: (B, T_combined, N_classes=3750)
        """
        # 1. Audio Encoding (ASRと同じ)
        audio_features = self._encode_audio(input_features, feature_attention_mask)

        # 2. Token Embedding + Audio Scatter (ASRと同じ)
        inputs_embeds = self._merge_audio_text(input_ids, audio_features)

        # 3. LLM Forward
        hidden_states = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
        )
        # hidden_states: (B, T_combined, 1536)

        # 4. Timestamp Prediction (LM Headの代わり)
        # ★ 通常ASRとの重要な違い:
        #    ASR: lm_head(hidden_states) → logits (B, T, 151936)
        #    FA:  timestamp_head(hidden_states) → logits (B, T, 3750)
        timestamp_logits = self.timestamp_head(hidden_states)
        # timestamp_logits: (B, T_combined, 3750)

        return {"timestamp_logits": timestamp_logits}

    def _encode_audio(self, input_features, feature_attention_mask):
        """Audio Encoder (audio_encoder.py参照)"""
        pass

    def _merge_audio_text(self, input_ids, audio_features):
        """Audio-Text Embedding統合 (main_flow.py参照)"""
        pass


# ============================================================
# 3. ForcedAligner 学習時の特殊処理
# ============================================================

class ForcedAlignerTraining:
    """
    ForcedAligner 学習の特殊処理 (疑似コード)

    ========================================
    通常のLLM学習との3つの違い
    ========================================

    1. Causal Training (シフトなし)
       通常LLM: output[:-1] vs labels[1:]  (1トークンシフト)
       FA:      output vs labels  (シフトなし、同期)

    2. [time]位置のみにCE損失
       通常LLM: 全トークンにCE損失
       FA:      [time]トークン位置のみにCE損失

    3. 動的スロット挿入
       学習時に各単語/文字の後にランダムに[time]を挿入/省略
       → 推論時にどの粒度でも対応可能に
    """

    @staticmethod
    def compute_loss(
        timestamp_logits: torch.Tensor,   # (B, T_combined, N_classes=3750)
        timestamp_labels: torch.Tensor,   # (B, T_combined) int64
        time_slot_mask: torch.Tensor,     # (B, T_combined) bool
    ) -> torch.Tensor:
        """
        ForcedAligner 損失計算

        ========================================
        Shape
        ========================================
        入力:
            timestamp_logits: (B, T_combined, 3750) - モデル出力
            timestamp_labels: (B, T_combined) int64 - 正解インデックス
                - [time]位置: 0〜3749 のタイムスタンプインデックス
                - 非[time]位置: -100 (無視)
            time_slot_mask: (B, T_combined) bool
                - True: [time]トークン位置

        出力:
            loss: scalar - Cross Entropy Loss (CE)

        ========================================
        処理詳細
        ========================================
        """
        # ★ 重要: シフトなし (causal training)
        # 通常LLM: shift_logits = logits[..., :-1, :], shift_labels = labels[..., 1:]
        # FA:      logits と labels をそのまま使用

        # [time]位置のみでCE損失を計算
        loss = F.cross_entropy(
            timestamp_logits.view(-1, timestamp_logits.size(-1)),  # (B*T, 3750)
            timestamp_labels.view(-1),                              # (B*T,)
            ignore_index=-100,  # [time]以外の位置は無視
        )

        return loss

    @staticmethod
    def dynamic_slot_insertion(
        text: str,
        timestamps: List[Tuple[str, float, float]],
        frame_duration_ms: int = 80,
    ) -> Tuple[str, List[int]]:
        """
        動的スロット挿入 (学習時のデータ拡張)

        ========================================
        処理
        ========================================
        各単語/文字について、ランダムに:
        - [time][time] を挿入 (start + end)
        - [time] のみ挿入 (start or end)
        - 何も挿入しない

        これにより、推論時に任意の粒度 (word/char/sentence)
        でタイムスタンプを要求できるモデルになる。

        ========================================
        タイムスタンプインデックス変換
        ========================================
        timestamp_ms / frame_duration_ms = index
        例: 1600ms / 80ms = 20 → index=20
        """
        pass


# ============================================================
# 入出力shape一覧表
# ============================================================
"""
========================================
ForcedAligner Shape遷移 (10秒音声, "Hello world" の例)
========================================

T_mel = 1000 (100Hz × 10秒)
T_audio = 125 (12.5Hz × 10秒)

テキスト: "Hello [time][time] world [time][time]"
→ T_text ≈ 10 (トークン数、tokenizer依存)

| 段階                    | テンソル名          | Shape                 | 説明                         |
|------------------------|--------------------|-----------------------|------------------------------|
| メルスペクトログラム     | input_features     | (1, 1000, 128)        | 100Hz, 128 mel bins          |
| Audio Encoder出力      | audio_features     | (1, 125, D_aut_out)   | 12.5Hz音声表現               |
| トークンID             | input_ids          | (1, T_text)           | [time]含むトークン列          |
| Audio+Text統合         | inputs_embeds      | (1, T_combined, 1536) | 統合埋め込み                  |
| LLM hidden states     | hidden_states      | (1, T_combined, 1536) | Qwen3-0.6B出力               |
| Timestamp logits       | timestamp_logits   | (1, T_combined, 3750) | 全位置の予測                  |
| [time]位置抽出         | time_logits        | (1, 4, 3750)          | 4スロット (2単語×2)           |
| argmax                 | time_indices       | (1, 4)                | タイムスタンプインデックス     |
| ×80ms                  | timestamps_ms      | (1, 4)                | ミリ秒単位のタイムスタンプ     |
| 結果                   | results            | List[TimestampResult] | [(word, start, end), ...]    |

========================================
タイムスタンプ分解能と最大長
========================================
フレーム時間:   80ms (= AuT Encoder出力の1フレーム)
最大クラス数:   3750
最大音声長:     3750 × 80ms = 300,000ms = 300秒 (5分)
分解能:         80ms (0.08秒)

========================================
ASRモデルとForcedAlignerの出力層の比較
========================================
ASR:
  lm_head: Linear(4096, 151936, bias=False)
  → 語彙分布 (next token prediction)
  → 自己回帰デコーディング

ForcedAligner:
  timestamp_head: Linear(1536, 3750, bias=True)
  → タイムスタンプ分布 (slot filling)
  → 非自己回帰、全スロット同時予測
"""
