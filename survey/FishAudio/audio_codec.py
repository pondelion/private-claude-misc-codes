"""公式DAC/DownsampleRVQ本体をそのまま使うencoder/decoder境界。

codecはcausal CNN、ConvNeXt、sliding-window Transformer、RVQ、波形decoder
からなる独立した大規模モデル。約1000行の本体を小さな近似CNNで代用しない。
yamlからrandom初期化し、勾配を保つreconstruct()でcodec自体も学習できる。

Dual-ARとの関係:
    学習前に音声をencodeしてGT IDを保存し、生成後にIDをdecodeして波形へ
    戻す。両段で同じcodec/codebookを使う。IDの意味はcodecの重みで決まる。
    build_official_codecは構造の構築だけであり、pretrained重みは読み込まない。

使い方:
    公式fish_speechをimport可能にし、PyTorch、Hydra、OmegaConfと公式codecの
    依存を用意する。build_official_codec(yaml_path)で本体を作り、必要な
    checkpointをロードしてAudioCodec本体へ渡す。公式pretrained推論の
    読み込み例はpretrained_inference.pyを参照する。
    encodeへ44.1kHz波形[B,S]と実sample長[B]を渡すと、IDs[B,N,T]と
    code長[B]を返す。decodeへIDsを渡すと波形[B,1,Swav]を返す。
    B=発話数、N=book数（通常10）、T=約21.53Hz長、S/Swav=sample数。

codec自体を学習するとき:
    reconstructはno_gradを使わず、予測波形とquantizerのVQResultを返す。
    MultiResolutionSTFTLossは予測/GT波形を複数窓長のスペクトルで比較する。
    これにVQ損失等を合わせる必要があり、この部品だけを公開GAN学習recipeと
    同一とは扱わない。Dual-ARのtraining_example.pyはcodec更新を行わない。
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import torch
from torch import nn
from torch.nn import functional as F

if TYPE_CHECKING:
    from fish_speech.models.dac.rvq import VQResult


def build_official_codec(yaml_path: str | Path) -> nn.Module:
    """公式YAMLの実codecをrandom構築する。

    重み取得・installはしない。yaml_pathは公式modded_dac_vq.yaml等のpath。戻り値は実DAC module。

    Args:
        yaml_path: codecの構成を記述したYAMLのパス。

    Returns:
        指定YAMLで構築した音声codecのnn.Module。
    """
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    return instantiate(OmegaConf.load(yaml_path))


class AudioCodec(nn.Module):
    """実DACのID抽出・波形復元と、学習用再構成経路を提供するadapter。"""

    def __init__(self, official_codec: nn.Module) -> None:
        """構築済み公式DAC本体を登録する。

        codecの重みを近似モデルへ置換しない。

        Args:
            official_codec: 構築済みのDAC音声codec。

        Returns:
            None。層・設定を登録する。ここでは学習・音声生成を行わない。
        """
        super().__init__()
        self.codec = official_codec

    @torch.no_grad()
    def encode(
        self,
        waveform: torch.Tensor,
        lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """44.1kHz波形[B,S]からIDs[B,N,T]と有効code長[B]を返す。

        B=発話数、S=padding込みsample数、N=book数（通常10）、T=codec時刻数。
        lengths[B]は各波形の有効sample数。Tのrateは約21.53Hz。

        Args:
            waveform: 波形[B,S]。Bは発話数、Sは44.1kHzのサンプル数。
            lengths: 各波形の有効サンプル数[B]。Bは発話数。

        Returns:
            codec ID[B,N,T]と有効code長[B]の組。Bは発話数、Nはbook数、Tは音声フレーム数。
        """
        return self.codec.encode(waveform, audio_lengths=lengths)

    @torch.no_grad()
    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """codec IDs[B,N,T]を波形[B,1,S]へ変換する。

        B=発話数、N=book数（通常10）、T=codec frame数、Sは概ねT*2048。channel数1はmono。
        入出力sample rateは公式codec設定（通常44.1kHz）。

        Args:
            codes: codec ID[B,N,T]。Bは発話数、Nはbook数、Tは音声フレーム数。

        Returns:
            復元波形[B,1,S]。Bは発話数、1はmonoチャネル、Sは44.1kHzのサンプル数。
        """
        # ID -> quantizerの本当のlookup/post-Transformer/up -> 波形decoder。
        # codesは[B,10,T21]。DAC.decodeは連続latent用なのでIDを直接渡さない。
        return self.codec.from_indices(codes)

    def reconstruct(
        self,
        waveform: torch.Tensor,
    ) -> tuple[torch.Tensor, VQResult]:
        """[B,1,S] -> reconstructed[B,1,S], VQResult（勾配あり）。

        手元のDAC.forwardはencode()の戻り値IDsをdecoderへ渡すため、その便利関数を通らず、
        公開encoder→quantizer.z→decoderを直接接続する。モデル本体をdummyに変更する処理ではない。B=発話数、
        1=mono channel、S=44.1kHzの入力sample数。VQResultのz:[B,1024,T512]、
        codes:[B,N,T21]、各VQ lossはscalarまたはbatch集約前tensor。T512=512倍圧縮後、
        T21=2048倍圧縮後の時刻数、N=book数。

        Args:
            waveform: 波形[B,1,S]。Bは発話数、1はmonoチャネル、Sはサンプル数。

        Returns:
            復元波形[B,1,S]とVQResultの組。Bは発話数、1はmonoチャネル、Sは波形サンプル数。
            VQResultの各shapeは本文に記す。
        """
        length = waveform.size(-1)
        frame_length = int(self.codec.frame_length)
        # Spad=frame_lengthの倍数。[B,1,S]->[B,1,Spad]、実sampleは切らない。
        padded = F.pad(waveform, (0, (-length) % frame_length))
        encoded = self.codec.encoder(padded)  # [B,1,Spad]->[B,1024,T512]。
        result = self.codec.quantizer(
            encoded)  # .codes:[B,10,S/2048]、.zは復元latent。
        reconstructed = self.codec.decoder(
            result.z)  # [B,1024,T512]->[B,1,Spad]。
        # 右paddingだけを除去する。[B,1,Spad] -> [B,1,S]。
        return reconstructed[..., :length], result


class MultiResolutionSTFTLoss(nn.Module):
    """スクラッチcodec用の実STFT損失部品。非公開GAN学習recipeの代用ではない。"""

    def __init__(self, fft_sizes: tuple[int, ...] = (512, 1024, 2048)) -> None:
        """複数のFFT窓長を指定し、各Hann窓をbufferとして保持する。

        Args:
            fft_sizes: 損失を計算する複数のSTFT窓長。波形サンプル単位の整数列。

        Returns:
            None。層・設定を登録する。ここでは学習・音声生成を行わない。
        """
        super().__init__()
        self.fft_sizes = fft_sizes
        for size in fft_sizes:
            self.register_buffer(f"window_{size}", torch.hann_window(size))

    def forward(self, prediction: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        """予測/GT波形[B,1,S]からscalarの複数解像度スペクトル損失を返す。

        B=発話数、1=mono channel、S=波形sample数。窓長NfftごとにF=Nfft/2+1周波数bin、
        T=STFT時刻数の振幅[B,F,T]を比較する。

        Args:
            prediction: 予測波形[B,1,S]。Bは発話数、1はmonoチャネル、Sはサンプル数。
            target: GT波形[B,1,S]。Bは発話数、1はmonoチャネル、Sはサンプル数。

        Returns:
            複数解像度で平均したSTFT損失[]。[]はscalar。
        """
        # mono channel軸だけを除く。[B,1,S]->[B,S]。
        prediction, target = prediction[:, 0].float(), target[:, 0].float()
        total = prediction.new_zeros(())
        for size in self.fft_sizes:
            window = getattr(self, f"window_{size}")
            # 波形[B,S]->複素STFT[B,F,T]->振幅[B,F,T]。各解像度でF/Tは異なる。
            pred = torch.stft(prediction,
                              size,
                              hop_length=size // 4,
                              window=window,
                              return_complex=True).abs().clamp_min(1e-7)
            truth = torch.stft(target,
                               size,
                               hop_length=size // 4,
                               window=window,
                               return_complex=True).abs().clamp_min(1e-7)
            # [B,F,T]全体をscalarへ集約するlog振幅誤差と相対spectral誤差。
            log_error = F.l1_loss(pred.log(), truth.log())
            spectral_error = (torch.linalg.vector_norm(pred - truth) /
                              torch.linalg.vector_norm(truth).clamp_min(1e-7))
            total = total + log_error + spectral_error
        return total / len(self.fft_sizes)
