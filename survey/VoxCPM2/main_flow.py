"""VoxCPM2 backboneと実AudioVAEをつなぐ単一発話の合成pipeline。

準備:
    学習済みVoxCPM2と対応AudioVAEAdapterをVoxCPM2Pipelineへ渡し、
    pipeline.eval()にする。本文は同じtokenizerのint64 ID[L]にする。
    L=本文token数。参照音声は先にadapter.encodeでlatent[N,4,64]へ変換し、
    発話batch軸を除いて渡す。N=patch数、4=patch内frame数、64=VAEのlatentの特徴次元数。

2種類の音声条件:
    referenceは独立した話者の証拠としてprefixへ入れ、生成波形には含めない。
    continuationは生成直前の既知音声で、本文IDにはその書き起こしも含める。
    continuationを使うときは既知部分と生成部分を一緒にdecodeし、最後に
    既知部分の波形だけを切る。decoderに音響的な直前文脈を与えるためである。

呼び出しと戻り値:
    result = pipeline(text_ids, reference=ref, steps=10, cfg=2.0)
    result["patches"]:[1,Ng,4,64]、result["waveform"]:[1,1,S48]。
    Ng=新規patch数、S48=48kHz sample数。波形ファイル保存は呼び出し側で行う。
    学習はtraining_example.py、公式pretrainedでのpath/text入力とWAV保存は
    pretrained_inference.pyに分けている。このファイルにはCLIはない。
"""
import torch
from torch import nn

from audio_vae import AudioVAEAdapter
from data import pack_sequence
from model import VoxCPM2


class VoxCPM2Pipeline(nn.Module):
    """text/reference条件、連続patch AR、公式AudioVAEをつなぐ合成pipeline。"""

    def __init__(self, backbone: VoxCPM2,
                 audio_vae_adapter: AudioVAEAdapter) -> None:
        """学習済みbackboneと、対応する実AudioVAEのadapterを登録する。

        Args:
            backbone: 学習済みの連続音声patch生成モデル。
            audio_vae_adapter: 連続patchと波形を相互変換する音声VAEの接続モジュール。

        Returns:
            None。層・設定を登録する。ここでは学習・音声生成を行わない。
        """
        super().__init__()
        self.backbone, self.audio = backbone, audio_vae_adapter

    @torch.no_grad()
    def forward(
        self,
        text_ids: torch.Tensor,
        reference: torch.Tensor | None = None,
        continuation: torch.Tensor | None = None,
        **generation: int | float,
    ) -> dict[str, torch.Tensor]:
        """text_ids:[L]、reference/continuation:[N,4,64]（任意）。

        continuationのtext_idsはprompt transcriptとtarget本文の両方を含める。
        referenceは独立話者証拠、continuationは時間的に直前の観測音声。L=本文token数、N=参照/観測patch数、
        4=patch内frame数、64=VAEのlatentの特徴次元数。generationはbackbone.generateの
        max_patches/min_patches/steps/cfg。戻り値patches:[1,Ng,4,64]、
        waveform:[1,1,S48]。Ng=新規生成patch数、S48=48kHzの新規波形sample数、channel数1=mono。

        Args:
            text_ids: 本文ID[L]。Lは本文token数。
            reference: 任意の参照patch[Nr,P,C]。Nrは参照patch数、Pはpatch内フレーム数、
                CはVAEのlatentの特徴次元数。
            continuation: 任意の直前音声patch[Nc,P,C]。Ncは観測patch数、Pはpatch内フレーム数、
                CはVAEのlatentの特徴次元数。
            **generation: generateへ渡す生成設定。max_patches・min_patchesは生成長、
                stepsはODE分割数、cfgは条件強調係数。

        Returns:
            patches[1,Ng,P,C]とwaveform[1,1,S48]の辞書。Ngは生成patch数、Pはpatch内長、
            CはVAEのlatentの特徴次元数、S48は48kHzの波形長。
        """
        if self.training:
            raise ValueError("call eval() before synthesis")
        cfg = self.backbone.config
        device = next(self.backbone.parameters()).device
        # target=NoneでAUDIO_STARTまでのconditioningだけを構築する。
        packed = pack_sequence(
            text_ids.to(device), None, cfg,
            reference.to(device) if reference is not None else None)
        ids, patches, mask = (packed[key]
                              for key in ("text_ids", "patches", "audio_mask"))
        if continuation is not None:
            continuation = continuation.to(device)
            ids = torch.cat((ids, ids.new_zeros(continuation.size(0))))
            # S=prefix位置数、Nc=観測patch数。各trackを同じ位置軸へ連結する。
            patches = torch.cat((patches, continuation))  # [S+Nc,4,64]。
            mask = torch.cat((mask,
                              torch.ones(continuation.size(0),
                                         device=device,
                                         dtype=torch.bool)))
        # 発話batch軸1を追加。ids/mask:[1,S]、patches:[1,S,4,64]。
        generated = self.backbone.generate(ids[None], patches[None],
                                           mask[None], **generation)
        if continuation is not None:
            # contextもdecodeしてから、そのsample数だけ落とす。stateful decoder無しで
            # continuationの最初のgenerated patchをゼロ状態からdecodeしないため。
            full = torch.cat((continuation[None], generated),
                             dim=1)  # [1,Nc+Ng,4,64]。
            waveform = self.audio.decode(full)
            waveform = waveform[...,
                                continuation.size(0) * cfg.patch_size * 1920:]
        else:
            waveform = self.audio.decode(generated)
        return {"patches": generated, "waveform": waveform}
