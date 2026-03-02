"""
Qwen2.5-Omni ロス計算 - 簡略化疑似コード
==========================================

Pre-training (3段階) と Post-training (Thinker + Talker) のロス計算

公式論文: Section 4 (Pre-training), Section 5 (Post-training)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import math


# ============================================
# Pre-training ロス
# ============================================

class PretrainingLoss(nn.Module):
    """
    Qwen2.5-Omni Pre-training のロス計算

    3段階のPre-training:

    Stage 1: エンコーダ学習 (LLM凍結)
        - LLMパラメータを凍結
        - Audio Encoder + Vision Encoder のアダプタのみ学習
        - 音声-テキスト & 画像-テキスト ペアで学習
        - Vision Encoder: Qwen2.5-VL から初期化
        - Audio Encoder: Whisper-large-v3 から初期化
        - ロス: Cross-Entropy (テキスト生成)

    Stage 2: 全パラメータ学習
        - 全パラメータを解凍
        - データ構成:
            - 800B トークン: 画像/動画関連
            - 300B トークン: 音声関連
            - 100B トークン: 動画+音声関連
            - テキストデータも含む (言語能力維持)
        - max_length: 8,192 トークン
        - ロス: Cross-Entropy (テキスト生成)

    Stage 3: 長系列学習
        - 長音声・長動画データを追加
        - max_length: 32,768 トークン
        - ロス: Cross-Entropy (テキスト生成)
    """

    def __init__(self, vocab_size: int = 151643, ignore_index: int = -100):
        super().__init__()
        self.vocab_size = vocab_size
        self.ignore_index = ignore_index

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Pre-training のロス計算

        入力:
            logits: (B, L, vocab_size) - モデルの出力ロジット
                B: バッチサイズ
                L: シーケンス長
                vocab_size: 151643

            labels: (B, L) - 教師ラベル
                -100: 無視 (入力部分、パディング)
                0-151642: 正解トークンID

        出力:
            Dict {
                'loss': scalar - Cross-Entropy Loss
                'num_tokens': scalar - 有効トークン数
            }

        ★ Next-Token Prediction:
            logits[t] → labels[t+1] を予測
            入力部分 (labels=-100) はロスに含めない
        """

        # シフト: logits[:-1] → labels[1:]
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # shift_logits: (B, L-1, vocab_size)
        # shift_labels: (B, L-1)

        # Cross-Entropy Loss
        loss = F.cross_entropy(
            shift_logits.view(-1, self.vocab_size),  # (B*(L-1), vocab_size)
            shift_labels.view(-1),                    # (B*(L-1),)
            ignore_index=self.ignore_index,
            reduction='mean',
        )

        # 有効トークン数 (ignore_index でない数)
        num_tokens = (shift_labels != self.ignore_index).sum()

        return {
            'loss': loss,
            'num_tokens': num_tokens,
        }


# ============================================
# Thinker Post-training ロス
# ============================================

class ThinkerPosttrainingLoss(nn.Module):
    """
    Thinker Post-training のロス計算

    ChatML形式の指示追従データで学習

    データ形式:
        <|im_start|>system
        You are a helpful assistant.<|im_end|>
        <|im_start|>user
        <|vision_bos|><|IMAGE|><|vision_eos|>What is in this image?<|im_end|>
        <|im_start|>assistant
        A cat sitting on a windowsill.<|im_end|>

    ロス計算:
        assistant の応答部分のみでロスを計算
        system, user, 特殊トークンは labels=-100 で無視

    データ構成:
        - 純テキスト対話データ
        - 画像モダリティ会話データ
        - 音声モダリティ会話データ
        - 混合モダリティ会話データ
    """

    def __init__(self, vocab_size: int = 151643, ignore_index: int = -100):
        super().__init__()
        self.vocab_size = vocab_size
        self.ignore_index = ignore_index

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        入力:
            logits: (B, L, vocab_size) - Thinker出力
            labels: (B, L) - 教師ラベル
                assistant応答部分のみ有効なトークンID
                それ以外は -100

        出力:
            Dict { 'loss': scalar }
        """
        # Pre-trainingと同一のCross-Entropy
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        loss = F.cross_entropy(
            shift_logits.view(-1, self.vocab_size),
            shift_labels.view(-1),
            ignore_index=self.ignore_index,
            reduction='mean',
        )

        return {'loss': loss}

    @staticmethod
    def create_labels_from_chatml(
        input_ids: torch.Tensor,
        assistant_start_token_id: int,
        assistant_end_token_id: int,
        ignore_index: int = -100,
    ) -> torch.Tensor:
        """
        ChatML形式のinput_idsからlabelsを作成

        入力:
            input_ids: (B, L) - 全トークンID
            assistant_start_token_id: <|im_start|>assistant の次のトークンID
            assistant_end_token_id: <|im_end|> のトークンID

        出力:
            labels: (B, L) - assistant応答部分のみ有効

        例:
            input_ids:  [sys, ..., sys_end, user, ..., user_end, asst, A, cat, ..., asst_end]
            labels:     [-100, ..., -100,   -100, ..., -100,    -100, A, cat, ..., asst_end]
        """
        B, L = input_ids.shape
        labels = torch.full_like(input_ids, ignore_index)

        for b in range(B):
            in_assistant = False
            for i in range(L):
                if input_ids[b, i] == assistant_start_token_id:
                    in_assistant = True
                    continue
                if in_assistant:
                    labels[b, i] = input_ids[b, i]
                if input_ids[b, i] == assistant_end_token_id:
                    in_assistant = False

        return labels


# ============================================
# Talker Post-training ロス (3段階)
# ============================================

class TalkerICLLoss(nn.Module):
    """
    Talker Post-training Stage 1: In-Context Learning (ICL)

    音声継続タスクによるNext-Token Prediction
    対話コンテキスト + 音声応答のデータで学習

    目的:
        - セマンティック表現から音声への単調マッピングを学習
        - 韻律、感情、アクセントの多様な表現を獲得
        - 音色の分離 (timbre disentanglement): 特定の音声パターンと
          稀なテキストパターンの関連付けを防止

    データ:
        - マルチモーダルコンテキスト + 音声応答の対話
        - テキスト監督 (Thinkerと同様) も併用
    """

    def __init__(self, codebook_size: int = 8295, ignore_index: int = -100):
        super().__init__()
        self.codebook_size = codebook_size
        self.ignore_index = ignore_index

    def forward(
        self,
        codec_logits: torch.Tensor,
        codec_labels: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        入力:
            codec_logits: (B, L_codec, codebook_size) - Talker出力
                B: バッチサイズ
                L_codec: 音声コードシーケンス長
                codebook_size: 8295

            codec_labels: (B, L_codec) - 正解コードトークン
                -100: 無視 (コンテキスト部分)
                0-8294: 正解コードID

        出力:
            Dict { 'loss': scalar - Codec Next-Token Prediction Loss }
        """
        shift_logits = codec_logits[..., :-1, :].contiguous()
        shift_labels = codec_labels[..., 1:].contiguous()

        loss = F.cross_entropy(
            shift_logits.view(-1, self.codebook_size),
            shift_labels.view(-1),
            ignore_index=self.ignore_index,
            reduction='mean',
        )

        return {'loss': loss}


class TalkerDPOLoss(nn.Module):
    """
    Talker Post-training Stage 2: DPO (Direct Preference Optimization)

    音声生成の安定性と品質を向上させるためのDPO学習

    DPO Loss:
        L_DPO(P_θ; P_ref) = -E_{(x, y_w, y_l) ~ D} [
            log σ(β * (log P_θ(y_w|x) / P_ref(y_w|x) - log P_θ(y_l|x) / P_ref(y_l|x)))
        ]

    変数:
        x: 入力シーケンス (テキスト)
        y_w: 好ましい音声 (低WER, 自然な韻律)
        y_l: 好ましくない音声 (高WER, 不自然な韻律)
        P_θ: 現在のモデル
        P_ref: 参照モデル (Stage 1終了時のスナップショット)
        β: 温度パラメータ

    ランキング基準:
        - WER (Word Error Rate): 音声からテキストへの書き起こし精度
        - 句読点停止エラー率: 適切な箇所で停止するか
    """

    def __init__(self, beta: float = 0.1):
        """
        パラメータ:
            beta: DPOの温度パラメータ
                小さい値 → 参照モデルに近い
                大きい値 → 報酬の差を強く反映
        """
        super().__init__()
        self.beta = beta

    def forward(
        self,
        policy_chosen_logps: torch.Tensor,
        policy_rejected_logps: torch.Tensor,
        reference_chosen_logps: torch.Tensor,
        reference_rejected_logps: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        DPO Loss の計算

        入力:
            policy_chosen_logps: (B,) - P_θ(y_w|x) の対数確率
                B: バッチサイズ

            policy_rejected_logps: (B,) - P_θ(y_l|x) の対数確率

            reference_chosen_logps: (B,) - P_ref(y_w|x) の対数確率

            reference_rejected_logps: (B,) - P_ref(y_l|x) の対数確率

        出力:
            Dict {
                'loss': scalar - DPO Loss
                'chosen_rewards': (B,) - 選好された音声の暗黙的報酬
                'rejected_rewards': (B,) - 拒否された音声の暗黙的報酬
            }

        計算の流れ:
            1. 対数確率比の計算:
               log_ratio_w = log P_θ(y_w|x) - log P_ref(y_w|x)
               log_ratio_l = log P_θ(y_l|x) - log P_ref(y_l|x)

            2. DPO Loss:
               loss = -log σ(β * (log_ratio_w - log_ratio_l))
        """

        # 対数確率比
        log_ratio_chosen = policy_chosen_logps - reference_chosen_logps
        # log_ratio_chosen: (B,) - 選好音声の対数確率比

        log_ratio_rejected = policy_rejected_logps - reference_rejected_logps
        # log_ratio_rejected: (B,) - 拒否音声の対数確率比

        # DPO Loss
        logits = self.beta * (log_ratio_chosen - log_ratio_rejected)
        # logits: (B,) - β * (報酬差)

        loss = -F.logsigmoid(logits).mean()
        # loss: scalar

        # 暗黙的報酬 (モニタリング用)
        chosen_rewards = self.beta * log_ratio_chosen.detach()
        rejected_rewards = self.beta * log_ratio_rejected.detach()

        return {
            'loss': loss,
            'chosen_rewards': chosen_rewards,
            'rejected_rewards': rejected_rewards,
        }

    @staticmethod
    def compute_sequence_logps(
        logits: torch.Tensor,
        labels: torch.Tensor,
        ignore_index: int = -100,
    ) -> torch.Tensor:
        """
        シーケンスの対数確率を計算

        入力:
            logits: (B, L, vocab_size) - モデル出力
            labels: (B, L) - 正解ラベル

        出力:
            log_probs: (B,) - 各サンプルの対数確率

        計算:
            log P(y|x) = Σ_{t: labels[t]≠ignore} log P(y_t | y_{<t}, x)
        """
        B, L, V = logits.shape

        # シフト
        shift_logits = logits[:, :-1, :]
        shift_labels = labels[:, 1:]

        # トークンごとの対数確率
        log_probs = F.log_softmax(shift_logits, dim=-1)
        # log_probs: (B, L-1, vocab_size)

        # 正解トークンの対数確率を取得
        per_token_logps = torch.gather(
            log_probs, 2, shift_labels.unsqueeze(-1)
        ).squeeze(-1)
        # per_token_logps: (B, L-1)

        # 無視トークンをマスク
        mask = (shift_labels != ignore_index).float()
        per_token_logps = per_token_logps * mask

        # シーケンス全体の対数確率
        sequence_logps = per_token_logps.sum(dim=-1)
        # sequence_logps: (B,)

        return sequence_logps


class TalkerSpeakerFinetuneLoss(nn.Module):
    """
    Talker Post-training Stage 3: Speaker Fine-tuning

    特定の話者に適応するためのファインチューニング

    目的:
        - Talkerが特定の音声を採用できるようにする
        - 自然さと制御可能性の向上
        - マルチスピーカー対応

    データ:
        - 複数話者の指示追従データ
        - 話者ごとの音声特徴を学習
    """

    def __init__(self, codebook_size: int = 8295, ignore_index: int = -100):
        super().__init__()
        self.codebook_size = codebook_size
        self.ignore_index = ignore_index

    def forward(
        self,
        codec_logits: torch.Tensor,
        codec_labels: torch.Tensor,
        speaker_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        入力:
            codec_logits: (B, L, codebook_size) - Talker出力
            codec_labels: (B, L) - 正解コードトークン
            speaker_ids: (B,) - 話者ID (optional)

        出力:
            Dict { 'loss': scalar }
        """
        shift_logits = codec_logits[..., :-1, :].contiguous()
        shift_labels = codec_labels[..., 1:].contiguous()

        loss = F.cross_entropy(
            shift_logits.view(-1, self.codebook_size),
            shift_labels.view(-1),
            ignore_index=self.ignore_index,
            reduction='mean',
        )

        return {'loss': loss}


# ============================================
# 使用例
# ============================================

def example_loss_computation():
    """
    ロス計算の使用例

    各ロスクラスを実際にインスタンス化し、ダミーデータで
    フォワードパスを実行してロス値と形状を確認する
    """

    # ========================================
    # 1. Pre-training ロス (Cross-Entropy)
    # ========================================
    # Pre-training 3段階:
    #   Stage 1: エンコーダ学習 (LLM凍結), データ: 音声-テキスト, 画像-テキスト
    #   Stage 2: 全パラメータ学習, 800B(画像/動画)+300B(音声)+100B(動画+音声)
    #   Stage 3: 長系列学習, max_length=32768

    B, L, V = 2, 100, 151643
    logits = torch.randn(B, L, V)
    labels = torch.randint(0, V, (B, L))
    labels[:, :30] = -100  # 入力部分は無視 (system/user部分)

    pretrain_loss = PretrainingLoss(vocab_size=V)
    result = pretrain_loss(logits, labels)

    assert result['loss'].dim() == 0  # scalar
    assert result['num_tokens'].item() == (labels != -100).sum().item() - B  # shift分

    # ========================================
    # 2. Thinker Post-training ロス (ChatML形式)
    # ========================================
    # ChatML: assistant応答部分のみでロスを計算

    thinker_pt_loss = ThinkerPosttrainingLoss(vocab_size=V)

    # create_labels_from_chatml でラベル生成
    # 簡易版: assistant_start/end トークンIDを指定
    input_ids_chatml = torch.randint(0, V, (B, L))
    ast_start = 50  # ダミーのassistant開始トークンID
    ast_end = 51    # ダミーのassistant終了トークンID
    # 一部にassistant区間を設定
    input_ids_chatml[0, 40] = ast_start
    input_ids_chatml[0, 60] = ast_end
    input_ids_chatml[1, 50] = ast_start
    input_ids_chatml[1, 80] = ast_end

    labels_chatml = ThinkerPosttrainingLoss.create_labels_from_chatml(
        input_ids_chatml, ast_start, ast_end
    )
    # assistant区間のみ有効、他は-100
    assert (labels_chatml[0, :41] == -100).all()  # start以前は無視
    assert (labels_chatml[0, 41:61] != -100).any()  # assistant応答部分は有効

    thinker_result = thinker_pt_loss(logits, labels_chatml)
    assert thinker_result['loss'].dim() == 0

    # ========================================
    # 3. Talker ICL ロス (Codec Next-Token Prediction)
    # ========================================
    # Stage 1: 音声継続タスクでcodecトークンのNTP
    codebook_size = 8295
    L_codec = 200

    codec_logits = torch.randn(B, L_codec, codebook_size)
    codec_labels = torch.randint(0, codebook_size, (B, L_codec))
    codec_labels[:, :50] = -100  # コンテキスト部分は無視

    icl_loss = TalkerICLLoss(codebook_size=codebook_size)
    icl_result = icl_loss(codec_logits, codec_labels)
    assert icl_result['loss'].dim() == 0

    # ========================================
    # 4. Talker DPO ロス
    # ========================================
    # Stage 2: 好ましい音声 vs 好ましくない音声 の選好学習

    B_dpo = 4
    dpo_loss = TalkerDPOLoss(beta=0.1)

    # compute_sequence_logps で対数確率を計算
    dpo_logits = torch.randn(B_dpo, 80, codebook_size)
    dpo_labels = torch.randint(0, codebook_size, (B_dpo, 80))
    seq_logps = TalkerDPOLoss.compute_sequence_logps(dpo_logits, dpo_labels)
    assert seq_logps.shape == (B_dpo,)

    # DPOロス計算
    policy_chosen = torch.randn(B_dpo)
    policy_rejected = policy_chosen - 0.5  # 選好音声の方が高確率
    ref_chosen = torch.randn(B_dpo)
    ref_rejected = torch.randn(B_dpo)

    dpo_result = dpo_loss(policy_chosen, policy_rejected, ref_chosen, ref_rejected)
    assert dpo_result['loss'].dim() == 0
    assert dpo_result['chosen_rewards'].shape == (B_dpo,)
    assert dpo_result['rejected_rewards'].shape == (B_dpo,)

    # ========================================
    # 5. Talker Speaker Fine-tune ロス
    # ========================================
    # Stage 3: 話者適応のCross-Entropy
    speaker_loss = TalkerSpeakerFinetuneLoss(codebook_size=codebook_size)
    speaker_result = speaker_loss(codec_logits, codec_labels)
    assert speaker_result['loss'].dim() == 0

    # --- 結果表示 ---
    print(f"[ロス計算 使用例]")
    print()
    print(f"  1. Pre-training Loss:")
    print(f"     入力: logits {logits.shape}, labels {labels.shape}")
    print(f"     loss={result['loss'].item():.4f}, 有効トークン={result['num_tokens'].item()}")
    print()
    print(f"  2. Thinker Post-training Loss (ChatML):")
    print(f"     labels_chatml: assistant区間のみ有効 (他は-100)")
    print(f"     loss={thinker_result['loss'].item():.4f}")
    print()
    print(f"  3. Talker ICL Loss (Codec NTP):")
    print(f"     入力: codec_logits {codec_logits.shape}")
    print(f"     loss={icl_result['loss'].item():.4f}")
    print()
    print(f"  4. Talker DPO Loss (beta=0.1):")
    print(f"     seq_logps: {seq_logps.shape}  (バッチごとの対数確率)")
    print(f"     loss={dpo_result['loss'].item():.4f}")
    print(f"     chosen_rewards:  {dpo_result['chosen_rewards'].tolist()}")
    print(f"     rejected_rewards: {dpo_result['rejected_rewards'].tolist()}")
    print()
    print(f"  5. Talker Speaker FT Loss:")
    print(f"     loss={speaker_result['loss'].item():.4f}")


if __name__ == "__main__":
    example_loss_computation()
