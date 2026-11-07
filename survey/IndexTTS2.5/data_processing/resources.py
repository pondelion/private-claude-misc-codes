"""必要な固定pretrained重みだけを読み、ローカルfrontendへ接続する。

IndexTTSの完成pipeline、生成GPT、codec、S2Melを外部からロードしない。
W2v-BERTだけは許可された外部pretrainedモデルをTransformersで実行する。
CAMPPlus/BigVGAN/感情frontendはローカルに定義したネットワークへ重みを読む。
Tensorのdtype/shapeが一致しない重みを黙ってrandom値で補完しない。
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from torch import nn
import torch

from data_processing.mel import MelSpectrogram
from data_processing.text import TextNormalizer
from data_processing.tokenizer import TextTokenizer
from models.t2s.voice import EmotionEncoder
from pretrained.bigvgan.model import AttrDict, BigVGAN
from pretrained.campplus.model import CAMPPlus


class JapaneseTextProcessor:
    """公開推論のg2p_ratio=0と同じ、本文を保持する日本語処理。"""

    def process_ja_text(self, text: str) -> str:
        """ratio=0では漢字の確率的な読み置換を行わず、文字列を保持する。

        Args:
            text: 読み上げる本文。

        Returns:
            入力と同じ本文。公式のg2p_ratio=0に対応する。
        """
        return text


@dataclass
class FrontendAssets:
    """固定抽出器・vocoder・前処理設定を保持する。生成モデルは含まない。

    Attributes:
        extract_features: W2v-BERT入力の特徴と有効maskを作る固定processor。
        cfg: 公式config.yamlから読んだ階層設定。
        device: 特徴抽出と合成に用いるデバイス指定。
        semantic_model: 固定重みのW2v-BERT特徴抽出器。
        semantic_mean: semantic特徴の正規化平均[1024]。1024は特徴次元数。
        semantic_std: semantic特徴の正規化標準偏差[1024]。1024は特徴次元数。
        campplus_model: 固定重みのCAMPPlus話者特徴ネットワーク。
        emotion_encoder: 固定重みの感情ConformerとPerceiver。
        bigvgan: 固定重みの波形生成ネットワーク。
        tokenizer: 本文をtoken IDへ変換する固定語彙のtokenizer。
        text_process: 中国語・英語の読み展開と注釈の保護処理。
        ja_text_process: 日本語本文処理。公開推論のg2p_ratio=0に対応する。
        mel_fn: vocoderと同じ設定の波形→Mel変換。
    """

    cfg: Any
    device: str
    semantic_model: nn.Module
    extract_features: Any
    semantic_mean: torch.Tensor
    semantic_std: torch.Tensor
    campplus_model: nn.Module
    emotion_encoder: EmotionEncoder
    bigvgan: nn.Module
    tokenizer: TextTokenizer
    text_process: TextNormalizer
    ja_text_process: JapaneseTextProcessor
    mel_fn: MelSpectrogram


def load_assets(model_dir: Path, device: str) -> FrontendAssets:
    """ローカル資産をstrictにロードする。

    自動download/compileは行わない。

    Args:
        model_dir: 事前配置した公開重みと設定のディレクトリ。
        device: 処理先。cpu、cuda:0等のPyTorchデバイス指定。

    Returns:
        固定特徴抽出器、vocoder、前処理設定を保持するFrontendAssets。
    """
    from omegaconf import OmegaConf
    from transformers import SeamlessM4TFeatureExtractor, Wav2Vec2BertModel

    config = OmegaConf.load(model_dir / "config.yaml")
    # 1. 許可された外部pretrained semanticモデルとprocessorを読む。
    w2v_dir = model_dir / "hf_cache/w2v-bert-2.0"
    processor = SeamlessM4TFeatureExtractor.from_pretrained(
        str(w2v_dir), local_files_only=True)
    semantic = Wav2Vec2BertModel.from_pretrained(
        str(w2v_dir), local_files_only=True).to(device).eval()
    stats = torch.load(model_dir / str(config.w2v_stat),
                       map_location=device,
                       weights_only=True)
    # 2. CAMPPlusはローカル定義へ外部pretrained重みをロードする。
    campplus = CAMPPlus(feat_dim=80, embedding_size=192)
    campplus.load_state_dict(torch.load(model_dir /
                                        "hf_cache/campplus_cn_common.bin",
                                        map_location="cpu",
                                        weights_only=True),
                             strict=True)
    # 3. IndexTTS GPT配布checkpointから感情frontend部分だけを取り出す。
    # 生成GPTのpretrained重みをscratchモデルへ流用しない。
    emotion = EmotionEncoder(
        OmegaConf.to_container(config.gpt.emo_condition_module))
    checkpoint = torch.load(model_dir / str(config.gpt_checkpoint),
                            map_location="cpu",
                            weights_only=True)
    checkpoint = checkpoint.get("model", checkpoint)
    state = {
        name: value
        for name, value in checkpoint.items()
        if name.startswith(("emo_conditioning_encoder.",
                            "emo_perceiver_encoder."))
    }
    emotion.load_state_dict(state, strict=True)
    # 4. BigVGANもローカル定義へ外部pretrained重みをロードする。
    import json

    directory = model_dir / "hf_cache/bigvgan"
    vocoder = BigVGAN(AttrDict(
        json.loads((directory / "config.json").read_text())),
                      use_cuda_kernel=False)
    state = torch.load(directory / "bigvgan_generator.pt",
                       map_location="cpu",
                       weights_only=True)
    vocoder.load_state_dict(state["generator"], strict=True)
    vocoder.remove_weight_norm()
    # 5. Mel前処理を実vocoderと同じ設定で構築する。
    preprocess = config.s2mel.preprocess_params
    spect = preprocess.spect_params
    mel = MelSpectrogram(int(preprocess.sr),
                         int(spect.n_fft),
                         int(spect.win_length),
                         int(spect.hop_length),
                         int(spect.n_mels),
                         float(spect.get("fmin", 0)),
                         None if spect.get("fmax", "None") == "None" else 8000,
                         center=False)
    return FrontendAssets(
        config,
        device,
        semantic,
        processor,
        stats["mean"].to(device),
        stats["var"].sqrt().to(device),
        campplus.to(device).eval(),
        emotion.to(device).eval(),
        vocoder.to(device).eval(),
        TextTokenizer(model_dir),
        TextNormalizer(enable_glossary=True,
                       assets_dir=model_dir / "normalization"),
        JapaneseTextProcessor(),
        mel.to(device),
    )
