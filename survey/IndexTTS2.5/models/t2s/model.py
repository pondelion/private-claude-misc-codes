"""CAMPPlus条件のローカルT2Sを学習・推論APIへ接続する。

本文token列、話者・感情の埋め込み、言語ID、過去の音声semantic tokenを入力し、
次の音声semantic tokenを予測する。本文の続きを生成するモデルではない。
GPT-2本体とsemantic tokenの分類層はローカル実装を使う。
構成は公式gpt設定から読み込み、学習では音声semantic tokenのCEを計算する。

B=発話数、L=本文token数、U=GT semantic ID数、D=GPTの隠れ状態の次元数、V=音声semantic tokenの語彙数。
同じ入力・重み・設定での比較にはlogitsを使う。sampling結果は乱数にも依存する。
"""
from __future__ import annotations

from typing import Any

from torch import nn
from torch.nn import functional as F
import torch

from models.config import resolve_config


class T2SConfig:
    """公式gpt設定全体を保持する。容量変更はvariantで明示する。"""

    def __init__(self, **spec: Any) -> None:
        """公式設定を解決する。

        独自の既定小型モデルへfallbackしない。

        Args:
            **spec: 公式設定の参照先、または展開済み設定と容量variant。

        Returns:
            None。層・設定を登録する。ここでは学習・音声生成を行わない。
        """
        self.core = resolve_config("gpt", spec)

    @property
    def dim(self) -> int:
        """公式model_dim、すなわちGPT特徴次元数を返す。

        Returns:
            GPTの特徴次元数。
        """
        return self.core["model_dim"]

    @property
    def semantic_vocab(self) -> int:
        """audio開始ID未満を占めるsemantic codeの語彙数を返す。

        Returns:
            音声開始tokenより前に割り当てたsemantic codeの語彙数。
        """
        return self.core.get("start_mel_token", 8192)


class TextToSemantic(nn.Module):
    """ローカルUnifiedVoiceを保持し、モデル本体をrandom初期化する。"""

    def __init__(self, config: T2SConfig) -> None:
        """公式configでCAMPPlus版を構築し、checkpointは読み込まない。

        Args:
            config: モデルの層数・特徴次元数・条件付け等を保持した設定。

        Returns:
            None。層・設定を登録する。ここでは学習・音声生成を行わない。
        """
        super().__init__()
        from models.t2s.voice import UnifiedVoice

        self.config = config
        self.core = UnifiedVoice(**config.core, spk_cond_mode="campplus")

    def conditioning(
        self,
        speaker: torch.Tensor,
        emotion: torch.Tensor,
    ) -> torch.Tensor:
        """話者[B,192]と抽出済み感情[B,1024]を公式射影で条件[B,3,D]にする。

        Args:
            speaker: 話者埋め込み[B,192]。Bは発話数、192はCAMPPlusの出力の特徴次元数。
            emotion: 感情埋め込み[B,1024]。Bは発話数、1024は感情encoderの出力の特徴次元数。

        Returns:
            条件特徴[B,3,D]。Bは発話数、3は条件slot数、DはGPTの隠れ状態の次元数。
            話者・感情を統合したslotの後にゼロslotを2個置く。
        """
        # 1. 公式の射影層を使用。感情frontend出力→2段Linear、話者→Linear。
        style = self.core.spk_emb_proj(speaker)[:, None]
        emotion_projected = self.core.emo_layer(
            self.core.emovec_layer(emotion), )
        # 2. 公開CAMPPlus推論と同じ条件1個+ゼロslot2個を配置する。
        zeros = torch.zeros(style.size(0),
                            2,
                            style.size(2),
                            device=style.device)
        return torch.cat((style + emotion_projected[:, None], zeros), dim=1)

    def token_logits(
        self,
        text_ids: torch.Tensor,
        speaker: torch.Tensor,
        emotion: torch.Tensor,
        language: torch.Tensor,
        audio_inputs: torch.Tensor,
    ) -> torch.Tensor:
        """本文と参照音声の特徴を条件に、次の音声semantic tokenを予測する。

        GPTには、話者・感情の条件、本文の埋め込み、音声semantic tokenの
        埋め込みを、この順に連結して入力する。言語条件は本文の埋め込みへ加算する。
        音声列が[BOS, s1, s2]なら、各位置で[s1, s2, s3]の分類logitsを返す。
        s1等は音声codecの離散ID。本文tokenや波形サンプルは予測しない。

        Args:
            text_ids: 本文ID[B,L]。Bは発話数、Lはpadding込みの本文token数。
            speaker: 話者埋め込み[B,192]。Bは発話数、192はCAMPPlusの出力の特徴次元数。
            emotion: 感情埋め込み[B,1024]。Bは発話数、1024は感情encoderの出力の特徴次元数。
            language: モデル用言語ID[B]。Bは発話数。本文tokenと同じ発話順。
            audio_inputs: 開始トークンに過去の音声semantic tokenを続けたID列[B,A]。Bは発話数、
                Aは開始トークンを含むトークン数。

        Returns:
            次の音声semantic tokenの分類logits[B,A,V]。Bは発話数、
            Aは開始トークンを含む音声semantic token列の長さ、Vは音声semantic tokenの語彙数。
        """
        # 1. 話者・感情条件と本文を埋め込み列[B,P,D]にする。
        # Bは発話数、Pは条件と本文の合計位置数、DはGPTの特徴次元数。
        # mask[B,P+1]の末尾1位置は音声開始トークン用。
        condition = self.conditioning(speaker, emotion)
        _, prefix, mask = self.core.prepare_gpt_inputs(
            condition,
            text_ids,
            language,
        )
        # 2. [BOS, 過去の音声semantic token]のID列[B,A]を埋め込む。
        # 音声tokenの埋め込み[B,A,D]へ、音声列内の位置埋め込みを加算する。
        audio = self.core.mel_embedding(audio_inputs)
        audio = audio + self.core.mel_pos_embedding(audio)
        attention = torch.cat((
            mask[:, :-1],
            mask.new_ones(audio_inputs.shape),
        ),
                              dim=1)
        # 3. 条件・本文[B,P,D]と音声[B,A,D]を連結し、GPTへ入力する。
        # 全位置の隠れ状態[B,P+A,D]から、音声位置[B,A,D]だけを取り出す。
        hidden = self.core.gpt(
            inputs_embeds=torch.cat((prefix, audio), dim=1),
            attention_mask=attention,
            return_dict=True,
        ).last_hidden_state[:, prefix.size(1):]
        # 4. 各音声位置から次の音声semantic tokenを分類する。[B,A,D]→[B,A,V]。
        # Vは終了トークン等を含む音声semantic tokenの語彙数。
        return self.core.mel_head(self.core.final_norm(hidden))

    def forward(
        self,
        text_ids: torch.Tensor,
        text_lengths: torch.Tensor,
        codes: torch.Tensor,
        code_lengths: torch.Tensor,
        speaker: torch.Tensor,
        emotion: torch.Tensor,
        language: torch.Tensor,
        text_loss_weight: float = 0.0,
    ) -> dict[str, torch.Tensor]:
        """本文・話者・感情・言語を条件に、正解音声の次token予測のCEを計算する。

        正解音声をcodecで離散化したID列を[s1,...,sU]とする。
        音声列の入力は[BOS,s1,...,sU]、予測の正解は[s1,...,sU,EOS]。
        本文tokenの予測損失は計算しない。

        発話ごとにpaddingを除き、音声semantic tokenとEOSの分類CEを計算して平均する。

        Args:
            text_ids: 本文ID[B,L]。Bは発話数、Lはpadding込みの本文token数。
            text_lengths: 本文の有効token数[B]。Bは発話数。
            codes: 離散semantic ID[B,U]。Bは発話数、Uはsemantic token数。
            code_lengths: 音声semantic列の有効token数[B]。Bは発話数。
            speaker: 話者埋め込み[B,192]。Bは発話数、192はCAMPPlusの出力の特徴次元数。
            emotion: 感情埋め込み[B,1024]。Bは発話数、1024は感情encoderの出力の特徴次元数。
            language: モデル用言語ID[B]。Bは発話数。本文tokenと同じ発話順。
            text_loss_weight: 本文損失の重み。現在の音声CE用adapterでは0のみ対応。

        Returns:
            lossとaudio_lossを持つ辞書。両値は同じ平均cross-entropy損失[]。[]はscalar。
        """
        if text_loss_weight != 0:
            raise ValueError("This training adapter uses audio CE only")
        losses = []
        for index in range(text_ids.size(0)):
            length, frames = int(text_lengths[index]), int(code_lengths[index])
            if length < 1 or frames < 1:
                raise ValueError("Positive text/code lengths required")
            body = codes[index:index + 1, :frames]
            inputs = F.pad(body, (1, 0), value=self.core.start_mel_token)
            targets = F.pad(body, (0, 1), value=self.core.stop_mel_token)
            # 入力/GT:[1,U+1]、logits:[1,U+1,V]→CEの語彙軸を中央へ。
            logits = self.token_logits(
                text_ids[index:index + 1, :length],
                speaker[index:index + 1],
                emotion[index:index + 1],
                language[index:index + 1],
                inputs,
            )
            losses.append(F.cross_entropy(logits.transpose(1, 2), targets))
        loss = torch.stack(losses).mean()
        return {"loss": loss, "audio_loss": loss}

    @torch.no_grad()
    def generate(
        self,
        text_ids: torch.Tensor,
        speaker: torch.Tensor,
        emotion: torch.Tensor,
        language: torch.Tensor,
        reference_features: torch.Tensor,
        max_tokens: int = 1500,
        temperature: float = 0.8,
        top_p: float = 0.8,
        top_k: int = 30,
        num_beams: int = 3,
        repetition_penalty: float = 10.0,
    ) -> torch.Tensor:
        """ローカルの自己回帰生成を実行し、EOSを除いたID[1,U]を返す。

        text_ids:[1,L]、話者[1,192]、感情[1,1024]、言語[1]、
        reference_features:[1,Tr50,1024]。Tr50=話者参照の50Hz frame数。
        HF sampling/beam/penaltyも公式推論の既定に合わせる。

        Args:
            text_ids: 本文ID[B,L]。Bは発話数、Lはpadding込みの本文token数。
            speaker: 話者埋め込み[B,192]。Bは発話数、192はCAMPPlusの出力の特徴次元数。
            emotion: 感情埋め込み[B,1024]。Bは発話数、1024は感情encoderの出力の特徴次元数。
            language: モデル用言語ID[B]。Bは発話数。本文tokenと同じ発話順。
            reference_features: 話者参照特徴[1,Tr50,1024]。1は単一発話、
                Tr50は参照の約50Hzフレーム数。
            max_tokens: 生成するsemantic token数の上限。
            temperature: samplingまたは初期ノイズの温度係数。
            top_p: 累積確率による候補語彙の打ち切り閾値。
            top_k: 候補語彙をscore上位何個へ制限するか。0では無効。
            num_beams: beam数。
            repetition_penalty: 既出tokenのscoreへ掛ける繰り返し抑制係数。

        Returns:
            semantic ID[B,U]。Bは発話数、Uは生成長。終了token以降を切り詰める。
        """
        if self.training or text_ids.size(0) != 1:
            raise ValueError("Call eval() and use one utterance")
        # 1. 感情埋め込み[1,1024]を2段の射影でGPTの隠れ状態の次元数[1,D]へ変換する。
        projected = self.core.emo_layer(self.core.emovec_layer(emotion))
        # 2. 本文・話者・感情からsemantic ID[1,A]を自己回帰生成する。
        # AはEOSを含み得る生成長。KV cacheとsamplingはローカル実装を使う。
        codes, _ = self.core.inference_speech(
            reference_features,
            text_ids,
            langs=language,
            emo_vec=projected,
            campplus_embedding=speaker,
            do_sample=True,
            top_p=top_p,
            top_k=top_k,
            temperature=temperature,
            num_beams=num_beams,
            num_return_sequences=1,
            length_penalty=0.0,
            repetition_penalty=repetition_penalty,
            max_generate_length=max_tokens,
        )
        # 3. 公開infer_v2_5と同様、最初のEOS以後をcodecへ渡さない。
        positions = (codes[0] == self.core.stop_mel_token).nonzero()
        length = int(positions[0, 0]) if positions.numel() else codes.size(1)
        return codes[:, :length]
