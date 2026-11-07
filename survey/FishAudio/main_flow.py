"""学習済みDual-ARと実codecを接続して、promptから音声を合成する。

準備:
    DualARと対応AudioCodecに学習済み重みを読み、FishAudioPipelineへ渡す。
    pipeline.eval()とdevice移動を行う。data.SequenceBuilder等でsystem、
    任意reference、user本文、assistant headerまでのpromptを作る。
    生成対象のGT audioをpromptへ入れてはいけない。

呼び出し:
    result = pipeline(prompt_main_ids, prompt_codes, max_frames=2000)
    main IDs:[1,S]、codes:[1,S,N]。S=混在prompt長、N=book数。
    mainはglobal tokenizer ID、codesはbook内IDという別の表現である。
    Dual-ARが新規IDs[1,T,N]を作り、book-first[1,N,T]へ転置してcodecへ渡す。
    T=生成frame数。戻り値dictのwaveformは44.1kHz mono[1,1,Swav]で、
    Swav=生成sample数。参照音声部分は含まない。保存は呼び出し側で行う。

関連ファイル:
    学習データと更新処理はtraining_example.py。公式checkpointを使って
    text/pathからWAVまで生成するCLIはpretrained_inference.py。
    本ファイルはscratch側moduleの接続用で、自動重み取得やCLIは持たない。
"""
import torch
from torch import nn

from audio_codec import AudioCodec
from dual_ar import DualAR


class FishAudioPipeline(nn.Module):
    """混在prompt、Dual-ARによるID生成、実codecの波形復元をつなぐ。"""

    def __init__(self, dual_ar: DualAR, codec: AudioCodec) -> None:
        """同じcodebook定義を使うDual-ARと実codec adapterを登録する。

        Args:
            dual_ar: 学習済みのSlow/Fast自己回帰生成モデル。
            codec: semantic IDと連続音声特徴を相互変換するモデル。

        Returns:
            None。層・設定を登録する。ここでは学習・音声生成を行わない。
        """
        super().__init__()
        self.dual_ar, self.codec = dual_ar, codec

    @torch.no_grad()
    def forward(
        self,
        prompt_main_ids: torch.Tensor,
        prompt_codes: torch.Tensor,
        max_frames: int = 2000,
        acoustic_vocab_size: int | None = None,
    ) -> dict[str, torch.Tensor]:
        """prompt_main_ids[1,S]とprompt_codes[1,S,N]から音声を生成する。

        S=system/reference/user/assistant headerを含むprompt位置数、N=book数。
        max_frames=生成frame上限、acoustic_vocab_size=acoustic bookの有効語彙数。
        戻り値codes:[1,T,N]、waveform:[1,1,Swav]。T=新規frame数、Swav=44.1kHz sample数、
        1channel=mono。codeの転置はcodec APIへの軸整列。

        Args:
            prompt_main_ids: 生成条件の共通語彙ID[1,S]。1は単一発話、Sはpromptの位置数。
            prompt_codes: 生成条件のcodec ID[1,S,N]。1は単一発話、Sはprompt長、
                Nはcodebook数。
            max_frames: 生成する音声codecフレーム数の上限。
            acoustic_vocab_size: 非semantic codebookでsamplingする有効な語彙数。

        Returns:
            codes[1,T,N]とwaveform[1,1,Swav]の辞書。Tは生成音声フレーム数、Nはbook数、
            Swavは波形サンプル数。
        """
        if self.training:
            raise ValueError("call eval() before synthesis")
        codes = self.dual_ar.generate(prompt_main_ids,
                                      prompt_codes,
                                      max_frames=max_frames,
                                      acoustic_vocab_size=acoustic_vocab_size)
        if codes.size(1) == 0:
            raise RuntimeError(
                "EOS was generated before the first audio frame")
        # [1,T,N]->[1,N,T]のbook-first表現を実codecへ渡す -> [1,1,Swav]。
        waveform = self.codec.decode(codes.transpose(1, 2))
        return {"codes": codes, "waveform": waveform}
