"""公式AudioVAE V2と、VoxCPM2の連続patch表現を接続するadapter。

役割:
    backboneは波形ではなくlatentを予測するため、学習データの事前抽出と
    合成後の波形復元に同じAudioVAEを使う。このクラスはVAE本体を再定義せず、
    構築・重み読込済みの実moduleを受け取り、paddingと軸変換を担当する。

使い方とshape:
    adapter = AudioVAEAdapter(official_audio_vae, patch_size=4)
    adapter.encode(waveform)は16kHz mono波形[B,S16]から[B,N,4,64]を返す。
    B=発話数、S16=入力sample数、N=patch数、4=patch内frame数、64=特徴次元数。
    入力を1patch分のsample数の倍数へ右paddingし、25Hz latentをまとめる。
    adapter.decode(patches)は48kHz波形[B,1,S48]を返す。
    encode入力とdecode出力はsample rateが違い、sample数も3倍相当になる。

注意する契約:
    resample/mono化は呼び出し前に行う。encode時のpadding分までdecodeされる
    ため、元の実長が必要なら別途保存して対応する48kHz長でcropする。
    両メソッドはno_gradで、固定VAEを使うbackbone学習向け。VAEそのものの
    scratch学習には公式VAEの勾配付き経路と対応lossを別途使う。
    PyTorch/einopsと実VAEが必要。接続例はmain_flow.py。
"""
import torch
from einops import rearrange
from torch import nn
from torch.nn import functional as F


class AudioVAEAdapter(nn.Module):
    """公式AudioVAEV2本体を注入。外部重みのdownloadは行わない。

    VAEのcausal conv/高帯域decoder/streaming stateは独立した大きなsubsystem
    のため公式本体を参照する。ランダム初期化も公式config constructorで可能。
    codec事前学習のGAN全体はbackboneのCFM objectiveとは別の学習である。
    """

    def __init__(self, audio_vae: nn.Module, patch_size: int = 4) -> None:
        """公式AudioVAE実体と、1patchへまとめるlatent frame数を登録する。

        Args:
            audio_vae: 公開重みを読み込んだ音声VAE。波形と連続latentを相互変換する。
            patch_size: 1個のpatchへまとめる音声latentフレーム数P。

        Returns:
            None。層・設定を登録する。ここでは学習・音声生成を行わない。
        """
        super().__init__()
        self.audio_vae, self.patch_size = audio_vae, patch_size

    @torch.no_grad()
    def encode(self, waveform: torch.Tensor) -> torch.Tensor:
        """16kHz波形[B,S16]から連続patch[B,N,P,64]を抽出する。

        B=発話数、S16=波形sample数、P=patch_size、N=ceil(S16/(640*P))。
        latentの1frameは640sampleに対応し、frame rateは25Hz。

        Args:
            waveform: 16kHz波形[B,S16]。Bは発話数、S16はサンプル数。

        Returns:
            連続patch[B,N,P,64]。Bは発話数、Nはpatch数、Pはpatch内フレーム数、64はVAEのlatentの特徴次元数。
        """
        # 1patch分のsample数640*Pの倍数まで、波形末尾をzero paddingする。
        hop = 640 * self.patch_size
        waveform = F.pad(waveform, (0, (-waveform.size(-1)) % hop))
        latent = self.audio_vae.encode(waveform, 16000)  # [B,64,N*P]。
        # 時刻をpatch数Nとpatch内frame数Pへ分け、64次元の特徴軸を末尾へ移す。
        return rearrange(latent, "b c (n p) -> b n p c", p=self.patch_size)

    @torch.no_grad()
    def decode(self, patches: torch.Tensor) -> torch.Tensor:
        """patch[B,N,P,64]から48kHz波形[B,1,S48]を実VAEで生成する。

        B=発話数、N=patch数、P=patch内frame数、64=VAEのlatentの特徴次元数、1=mono channel。
        S48は概ねN*P*1920。encode時のpadding分も含むので必要なら元長でcropする。

        Args:
            patches: 連続patch[B,N,P,64]。Bは発話数、Nはpatch数、Pはpatch内フレーム数、
                64はVAEのlatentの特徴次元数。

        Returns:
            48kHz波形[B,1,S48]。Bは発話数、1はmonoチャネル、S48は波形サンプル数。
        """
        # [B,N,P,64]->[B,64,N*P]としてVAEが期待する連続frame列へ戻す。
        latent = rearrange(patches, "b n p c -> b c (n p)")
        return self.audio_vae.decode(latent)  # [B,1,S48k]、sample数≈N*4*1920。
