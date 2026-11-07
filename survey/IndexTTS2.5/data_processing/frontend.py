"""本文・参照音声から、学習と合成で共通のIndexTTS条件を抽出する。

感情embeddingを利用者が用意する必要はない。感情参照音声が未指定なら、
話者参照音声のsemantic特徴から感情embeddingを抽出する。
学習用prepare_data.pyもこのクラスを使い、抽出方法と本文正規化を共有する。

固定する公式資産:
    W2v-BERT、正規化統計、CAMPPlus、感情Conformer/Perceiver、本文tokenizer、
    言語別normalizer、Mel設定。構築済みの公式FrontendAssetsから取り出して登録する。
    T2S本体の学習済み重みは使わず、感情frontendの重みだけを使用する。
    このfrontendは推論用で、特徴抽出中に勾配を計算しない。

shape:
    入力waveform[C,S]は音声channel数Cとsample数S。mono化とresample後に
    features[1,T50,1024]、speaker[1,192]、emotion[1,1024]、Mel[1,F,R]を得る。
    1=発話batch数、T50=約50Hz frame数、F=Mel bin数、R=参照Mel長。
    元のsample rateは必ず実音声の値を渡し、16kHzだと仮定しない。
"""
from __future__ import annotations

from typing import TYPE_CHECKING
import re

from torch import nn
import torch
import torchaudio

from models.conditioning import ReferenceConditioner, SemanticFeatureExtractor

if TYPE_CHECKING:
    from data_processing.resources import FrontendAssets

LANGUAGES = ("ZH", "EN", "JA", "ES", "AR")


def language_id(language: str) -> int:
    """公式lang_to_tokenを使用し、独自compact言語IDへ置き換えない。

    Args:
        language: 読み上げ言語コードまたはモデル用言語ID。

    Returns:
        公式の言語埋め込み表に対応する整数ID。
    """
    from data_processing.tokenizer import lang_to_token

    if language.upper() not in LANGUAGES:
        raise ValueError("Unsupported language")
    return lang_to_token(language.lower())


class IndexTTSFrontend(nn.Module):
    """実音声から条件を計算する、学習前処理と推論で共通のfrontend。"""

    def __init__(self, assets: FrontendAssets) -> None:
        """構築済み公式モデルから必要な抽出器・統計・前処理設定を登録する。

        Args:
            assets: 構築済みの固定特徴抽出器・tokenizer・vocoderと設定。

        Returns:
            None。層・設定を登録する。ここでは学習・音声生成を行わない。
        """
        super().__init__()
        self.semantic = SemanticFeatureExtractor(
            assets.semantic_model,
            assets.semantic_mean,
            assets.semantic_std.square(),
        )
        self.reference = ReferenceConditioner(
            assets.campplus_model,
            assets.emotion_encoder.emo_conditioning_encoder,
            assets.emotion_encoder.emo_perceiver_encoder,
        )
        self.processor = assets.extract_features
        self.tokenizer = assets.tokenizer
        self.text_normalizer = assets.text_process
        self.japanese_processor = assets.ja_text_process
        self.mel_fn = assets.mel_fn
        self.mel_sample_rate = int(assets.cfg.s2mel["preprocess_params"]["sr"])

    def mono_resample(self, waveform: torch.Tensor, sample_rate: int,
                      target_rate: int) -> torch.Tensor:
        """音声[C,S]をmono[1,Starget]へ変換する。

        Starget=変換後sample数。

        Args:
            waveform: 波形[C,S]。Cは音声チャネル数、Sはサンプル数。
            sample_rate: 入力音声のサンプリング周波数、単位はHz。
            target_rate: 変換先のサンプリング周波数、単位はHz。

        Returns:
            mono波形[1,Starget]。1は音声チャネル数、Stargetは変換後サンプル数。
        """
        if (waveform.ndim != 2 or waveform.size(0) < 1 or waveform.size(1) == 0
                or not waveform.is_floating_point()
                or not torch.isfinite(waveform).all() or sample_rate <= 0
                or target_rate <= 0):
            raise ValueError(
                "Expected finite floating waveform [channels,samples]")
        # channel軸だけ平均する。[C,S]→[1,S]。音声の切り詰めは行わない。
        mono = waveform.detach().cpu().float().mean(dim=0, keepdim=True)
        return torchaudio.functional.resample(mono, sample_rate, target_rate)

    @torch.no_grad()
    def extract_semantic(self, waveform: torch.Tensor,
                         sample_rate: int) -> torch.Tensor:
        """音声[C,S]から正規化semantic特徴[1,T50,1024]を抽出する。

        Args:
            waveform: 波形[C,S]。Cは音声チャネル数、Sはサンプル数。
            sample_rate: 入力音声のサンプリング周波数、単位はHz。

        Returns:
            正規化semantic特徴[1,T50,1024]。1は単一発話、T50は約50Hzフレーム数。
        """
        # 1. frontend入力用に16kHz mono[1,S16]へ変換する。
        audio16 = self.mono_resample(waveform, sample_rate, 16000)
        # 2. 公式processorで音響特徴[1,Tin,Fin]と有効mask[1,Tin]を作る。
        # Tinはfrontendのフレーム数、Finはfrontendの特徴次元数。
        inputs = self.processor(audio16,
                                sampling_rate=16000,
                                return_tensors="pt")
        device = self.semantic.mean.device
        # 3. W2v-BERT hidden[17]を固定統計で正規化する。
        return self.semantic(inputs["input_features"].to(device),
                             inputs["attention_mask"].to(device))

    @torch.no_grad()
    def extract_embeddings(
        self,
        waveform: torch.Tensor,
        sample_rate: int,
        emotion_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """話者音声と感情特徴からspeaker[1,192]、emotion[1,1024]を返す。

        waveform[C,S]は話者参照。emotion_features[1,Te50,1024]は感情参照の正規化特徴で、
        Te50は約50Hz frame数。感情参照省略時は話者特徴を渡す。

        Args:
            waveform: 波形[C,S]。Cは音声チャネル数、Sはサンプル数。
            sample_rate: 入力音声のサンプリング周波数、単位はHz。
            emotion_features: 感情参照特徴[B,Te,1024]。Bは発話数、Teは約50Hzの感情参照フレーム数。

        Returns:
            話者埋め込み[1,192]と感情埋め込み[1,1024]の組。1は単一発話。
        """
        # 1. 話者音声を16kHzのmono[1,S16]へ変換し、fbank[Tbank,80]を計算する。S16は波形長、Tbankはフレーム
        # 数。
        # Tbank=fbank frame数。時間平均の除去はReferenceConditionerで行う。
        audio16 = self.mono_resample(waveform, sample_rate, 16000)
        device = self.semantic.mean.device
        fbank = torchaudio.compliance.kaldi.fbank(
            audio16.to(device),
            num_mel_bins=80,
            dither=0,
            sample_frequency=16000,
        )
        # 2. batch軸追加[1,Tbank,80]と感情有効長[1]を抽出器へ渡す。
        lengths = torch.tensor([emotion_features.size(1)], device=device)
        return self.reference(fbank[None], emotion_features, lengths)

    @torch.no_grad()
    def extract_mel(self, waveform: torch.Tensor,
                    sample_rate: int) -> torch.Tensor:
        """音声[C,S]から対応vocoder仕様の参照Mel[1,F,R]を計算する。

        Args:
            waveform: 波形[C,S]。Cは音声チャネル数、Sはサンプル数。
            sample_rate: 入力音声のサンプリング周波数、単位はHz。

        Returns:
            参照Mel[1,F,R]。1は単一発話、FはMel bin数、Rは参照フレーム数。
        """
        mono = self.mono_resample(waveform, sample_rate, self.mel_sample_rate)
        return self.mel_fn(mono.to(self.semantic.mean.device))

    def normalize_text(self, text: str, language: str) -> str:
        """公式推論と同じ文字正規化・言語別処理・日本語G2Pを適用する。

        Args:
            text: 読み上げる本文。
            language: 読み上げ言語コードまたはモデル用言語ID。

        Returns:
            言語別の正規化と発音注釈の処理を適用した本文。
        """
        from data_processing.text import apply_pronunciation_annotations
        from data_processing.text import normalize_other_language

        def replace_character(match: re.Match[str]) -> str:
            """一致した引用符等を本文の統一表記へ置き換える。

            Args:
                match: 正規表現の一致オブジェクト。

            Returns:
                統一表記へ変換した文字列。
            """
            return self.text_normalizer.char_rep_map[match.group()]

        text = self.text_normalizer.clean_pattern.sub(replace_character, text)
        if language in {"ZH", "EN"}:
            text = self.text_normalizer.normalize(text)
        elif language in {"JA", "ES"}:
            text = normalize_other_language(text, language.lower(),
                                            self.text_normalizer.assets_dir)
        if language == "ES":
            text = text.upper()
        elif language in {"ZH", "EN", "JA"}:
            text = text.lower()
        text = apply_pronunciation_annotations(text)
        if language == "JA":
            text = self.japanese_processor.process_ja_text(text)
        return re.sub(r"<\|([^|]+)\|>", self._uppercase_special, text)

    @staticmethod
    def _uppercase_special(match: re.Match[str]) -> str:
        """本文中のspecial token名を公式と同じ大文字にする。

        Args:
            match: 正規表現の一致オブジェクト。

        Returns:
            special token名を大文字にした文字列。
        """
        return f"<|{match.group(1).upper()}|>"

    def tokenize(self, text: str, language: str) -> torch.Tensor:
        """本文を正規化し、言語prefix付き本文ID[1,L]を作る。

        L=token数。

        Args:
            text: 読み上げる本文。
            language: 読み上げ言語コードまたはモデル用言語ID。

        Returns:
            言語prefix付き本文ID[1,L]。1は単一発話、Lは本文token数。
        """
        if language not in LANGUAGES or not text.strip():
            raise ValueError("Nonempty text and supported language required")
        normalized = self.normalize_text(text, language)
        ids = self.tokenizer.encode(f"<|{language.lower()}|> " + normalized,
                                    allowed_special="all")
        if not ids:
            raise ValueError("Tokenizer returned no tokens")
        # BOS/EOSはT2S側が追加する。ここでは本文IDへbatch軸だけを付ける。
        return torch.tensor(ids,
                            dtype=torch.long,
                            device=self.semantic.mean.device)[None]

    @torch.no_grad()
    def forward(
        self,
        waveform: torch.Tensor,
        sample_rate: int,
        emotion_waveform: torch.Tensor | None = None,
        emotion_sample_rate: int | None = None,
    ) -> dict[str, torch.Tensor]:
        """話者音声[C,S]と任意感情音声[Ce,Se]から合成条件を自動抽出する。

        C/Ce=channel数、S/Se=各音声sample数。感情音声とsample rateは両方指定。
        戻り値はreference_features[1,T50,1024]、reference_mel[1,F,R]、speaker[1,192]、
        emotion[1,1024]。感情省略時は話者音声を兼用する。

        Args:
            waveform: 波形[C,S]。Cは音声チャネル数、Sはサンプル数。
            sample_rate: 入力音声のサンプリング周波数、単位はHz。
            emotion_waveform: 任意の感情参照波形[Ce,Se]。Ceは音声チャネル数、Seはサンプル数。
            emotion_sample_rate: 感情参照音声のサンプリング周波数。音声省略時はNone。

        Returns:
            reference_features[1,T50,1024]、reference_mel[1,F,R]、speaker[1,192]、
            emotion[1,1024]を格納した辞書。1は単一発話、T50は約50Hz参照長、FはMel bin数、Rは参照Mel長。
        """
        if (emotion_waveform is None) != (emotion_sample_rate is None):
            raise ValueError("Provide both emotion waveform and sample rate")
        # 1. 話者参照音声のsemantic特徴とMelを作る。
        features = self.extract_semantic(waveform, sample_rate)
        mel = self.extract_mel(waveform, sample_rate)
        # 2. 未指定なら同じ特徴を感情抽出にも使う。別音声の指定時だけ再抽出する。
        emotion_features = features
        if emotion_waveform is not None and emotion_sample_rate is not None:
            emotion_features = self.extract_semantic(emotion_waveform,
                                                     emotion_sample_rate)
        # 3. 音声から話者/感情embeddingを計算する。外部の感情ラベルは不要。
        speaker, emotion = self.extract_embeddings(waveform, sample_rate,
                                                   emotion_features)
        return {
            "reference_features": features,
            "reference_mel": mel,
            "speaker": speaker,
            "emotion": emotion
        }
