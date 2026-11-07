"""VoxCPM2の学習・生成をまとめる本文条件付き連続latentモデル。

モデルの読み方:
    音声は離散codec IDではなく、AudioVAEが抽出した連続latentを使う。
    LocEncが1patch内の複数frameを1ベクトルへまとめ、本文embeddingと
    混在させてTSLMへ入れる。TSLM出力はFSQで情報量を制限し、音響特徴と
    融合して独立したRALMへ渡す。両LMの状態を条件にLocDiTが次patchを作る。
    stop headは次patchを生成した位置で終了すべきかを判定する。

学習の呼び出し:
    config_from_dictでJSONからVoxConfigを構築し、VoxCPM2(config)を作る。
    data.pyのcollate_sequencesが返すdictをmodel(**batch)へ渡す。
    patches:[B,S,P,C]、ID/mask:[B,S]。B=発話数、S=混在列長、
    P=patch内frame数（4）、C=VAE特徴次元数（64）。戻り値lossはscalarで、
    targetだけのvelocity MSEとcontinue/stop CEの和。GT patchを予測する
    条件は必ず1位置前から取り、予測対象そのものをLM条件に漏らさない。

生成の呼び出し:
    model.eval()後、data.pack_sequence(..., target=None)でprefixを作り、
    batch軸を追加してgenerateへ渡す。戻るのは新規patch[1,N,P,C]で、
    波形ではない。main_flow.pyが対応AudioVAEでdecodeする。
    学習例とデータ準備はtraining_example.py。PyTorchとeinopsが必要。
    defaultの特徴次元数/層数は計算資源に合わせて確認し、語彙・境界IDはtokenizerと揃える。
"""
from dataclasses import dataclass, field, replace
from typing import Any

import torch
from einops import rearrange
from torch import nn
from torch.nn import functional as F

from flow_matching import PatchFlowMatching
from local_modules import LocalDiT, LocalEncoder, ScalarQuantizer
from transformer import Transformer, TransformerConfig


def _default_local_config() -> TransformerConfig:
    """LocEnc/LocDiT用の既定Transformer構成を独立したinstanceで返す。

    Returns:
        独立したLocalEncoder・LocalDiT用のTransformerConfig。
    """
    return TransformerConfig(
        dim=1024,
        intermediate=4096,
        layers=12,
        heads=16,
        kv_heads=2,
        head_dim=64,
    )


@dataclass
class VoxConfig:
    """text/latentの語彙・次元とTSLM/局所Transformer構成を保持する。

    Attributes:
        vocab_size: 本文または共通tokenizerの語彙数。
        patch_size: 音声1patch内のlatentフレーム数P。
        latent_dim: 音声latentの特徴次元数C。
        fsq_dim: 有限スカラー量子化を行う内部特徴次元数Q。
        fsq_scale: FSQの刻み幅1/scaleを決める係数。
        base: 長期semantic LMのTransformer構成。
        local: patch内encoderとDiTのTransformer構成。
        residual_layers: 残差LMの層数。
        residual_no_rope: 残差LMでRoPEを無効にするか。
        mean_mode: 通常の瞬間速度でなくmean-flow objectiveを用いるか。
        audio_start: target音声区間の開始token ID。
        audio_end: target音声区間の終了token ID。
        ref_start: 話者参照音声区間の開始token ID。
        ref_end: 話者参照音声区間の終了token ID。
    """

    vocab_size: int = 73448  # 構築用既定。使用tokenizerの実語彙数で上書きする。
    patch_size: int = 4
    latent_dim: int = 64
    fsq_dim: int = 512
    fsq_scale: int = 9
    base: TransformerConfig = field(default_factory=TransformerConfig)
    local: TransformerConfig = field(default_factory=_default_local_config)
    residual_layers: int = 8
    residual_no_rope: bool = True
    mean_mode: bool = False
    audio_start: int = 101
    audio_end: int = 102
    ref_start: int = 103
    ref_end: int = 104


def config_from_dict(values: dict[str, Any]) -> VoxConfig:
    """JSON由来の辞書を、base/localも含めた型付きVoxConfigへ変換する。

    Args:
        values: JSONから読み込んだ構成設定の辞書。

    Returns:
        入れ子のTransformer設定も型付きにしたモデル設定。
    """
    values = dict(values)
    for name in ("base", "local"):
        if name in values:
            values[name] = TransformerConfig(**values[name])
    return VoxConfig(**values)


class VoxCPM2(nn.Module):
    """連続latent patchの階層AR生成と、CFM/stopの学習objectiveを実装する。"""

    def __init__(self, config: VoxConfig) -> None:
        """LocEnc、独立TSLM/RALM、FSQ、LocDiTとstop headを構築する。

        Args:
            config: モデルの層数・特徴次元数・条件付け等を保持した設定。

        Returns:
            None。層・設定を登録する。ここでは学習・音声生成を行わない。
        """
        super().__init__()
        self.config = config
        dim, local_dim = config.base.dim, config.local.dim
        self.text_embedding = nn.Embedding(config.vocab_size, dim)
        self.local_encoder = LocalEncoder(config.local, config.latent_dim)
        self.enc_to_lm = nn.Linear(local_dim, dim)
        self.tslm = Transformer(config.base)
        self.fsq = ScalarQuantizer(dim, config.fsq_dim, config.fsq_scale)
        self.fusion = nn.Linear(2 * dim, dim)
        self.ralm = Transformer(
            replace(config.base,
                    layers=config.residual_layers,
                    no_rope=config.residual_no_rope))
        self.semantic_to_dit = nn.Linear(dim, local_dim)
        self.residual_to_dit = nn.Linear(dim, local_dim)
        self.flow = PatchFlowMatching(LocalDiT(config.local,
                                               config.latent_dim),
                                      mean_mode=config.mean_mode)
        self.stop = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(),
                                  nn.Linear(dim, 2, bias=False))

    def encode_history(
        self,
        text_ids: torch.Tensor,
        patches: torch.Tensor,
        audio_mask: torch.Tensor,
        valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """text/audio文脈からsemantic/residual各[B,S,Dl]を計算する。

        text_ids:[B,S]、patches:[B,S,P,C]、audio_mask/valid:[B,S]。B=発話数、S=LM位置数、
        P=patch内frame数、C=VAE latentの特徴次元数、Dl=TSLM/RALMの隠れ状態の次元数。
        audio_maskは音声位置、validはpadding以外の位置。

        Args:
            text_ids: 本文trackのID[B,S]。Bは発話数、Sは本文と音声を含む全位置数。
            patches: 連続音声patch[B,S,P,C]。Bは発話数、SはLM列長、Pはpatch内フレーム数、
                CはVAEのlatentの特徴次元数。
            audio_mask: 音声patch位置のbool mask[B,S]。Bは発話数、Sは列長。
            valid: 有効な位置のbool mask[B,S]。Bは発話数、Sは列長。Falseはpadding。

        Returns:
            semantic隠れ状態とresidual隠れ状態の組。各[B,S,Dl]。Bは発話数、SはLM位置数、
            Dlは長期LMの隠れ状態の次元数。
        """
        # Textとaudioの2trackは同じS位置軸。audio_maskが1ならLocEnc埋込を使う。
        encoded = self.enc_to_lm(
            self.local_encoder(patches))  # [B,S,P,C]->[B,S,Dlm]
        scale = self.config.base.scale_emb if self.config.base.use_mup else 1.0
        text = self.text_embedding(text_ids) * scale  # [B,S,Dlm]
        # modality mask[B,S,1]で埋込[B,S,Dl]を選択。音声/textを足して混ぜない。
        combined = torch.where(audio_mask[..., None], encoded, text)
        semantic = self.tslm(combined, valid=valid)  # causal [B,S,Dlm]
        # Text位置は連続のまま。音声位置だけFSQボトルネックを通す。
        semantic = torch.where(audio_mask[..., None], self.fsq(semantic),
                               semantic)
        # semanticと観測音響を特徴軸で連結。[B,S,Dl]x2->[B,S,2Dl]->[B,S,Dl]。
        residual_input = self.fusion(
            torch.cat((semantic, encoded * audio_mask[..., None]), dim=-1))
        residual = self.ralm(residual_input, valid=valid)
        return semantic, residual

    def forward(
        self,
        text_ids: torch.Tensor,
        patches: torch.Tensor,
        audio_mask: torch.Tensor,
        valid: torch.Tensor,
        loss_mask: torch.Tensor,
        stop_labels: torch.Tensor,
        progress: float = 0.0,
    ) -> dict[str, torch.Tensor]:
        """text_ids[B,S], patches[B,S,P,C], 各mask/stop_labels[B,S]。

        同位置のclean patchを見て自分自身を予測しないよう、LM出力を必ず1位置右へshiftする。GT patchはshiftしない。
        このズレがteacher forcing。B=発話数、S=LM位置数、P=patch内frame数、C=VAE latentの特徴次元数。
        loss_maskはtarget patchのみ1、stop_labelsは最後のtarget位置のみ1。
        戻り値loss/flow_loss/stop_lossはscalar、stop_logitsは[B,S,2]。
        progress=学習進捗0..1。optional mean-flowの区間samplingに使う。

        Args:
            text_ids: 本文trackのID[B,S]。Bは発話数、Sは本文と音声を含む全位置数。
            patches: 連続音声patch[B,S,P,C]。Bは発話数、SはLM列長、Pはpatch内フレーム数、
                CはVAEのlatentの特徴次元数。
            audio_mask: 音声patch位置のbool mask[B,S]。Bは発話数、Sは列長。
            valid: 有効な位置のbool mask[B,S]。Bは発話数、Sは列長。Falseはpadding。
            loss_mask: 学習するtarget音声位置だけが真のmask[B,S]。Bは発話数、Sは列長。
            stop_labels: 終了判定のGT[B,S]。Bは発話数、Sは列長。最後のtarget patchだけが1。
            progress: 学習の進捗率。0から1でmean-flow区間の抽出比率を調整する。

        Returns:
            loss・flow_loss・stop_lossのscalar[]とstop_logits[B,S,2]を格納した辞書。Bは発話数、
            SはLM位置数、2は終了・継続の分類数。
        """
        semantic, residual = self.encode_history(text_ids, patches, audio_mask,
                                                 valid)
        # 各[B,S,Dl]を1位置ずらし、位置iのGTを見たhiddenでiを予測させない。
        previous_semantic = F.pad(semantic[:, :-1], (0, 0, 1, 0))
        previous_residual = F.pad(residual[:, :-1], (0, 0, 1, 0))
        # sem/resは足さずconcat。LocDiT内で2つの独立prefix tokenへ分かれる。
        mu = torch.cat((self.semantic_to_dit(previous_semantic),
                        self.residual_to_dit(previous_residual)),
                       dim=-1)
        # Dd=LocDiTの隠れ状態の次元数。mu:[B,S,2Dd]。直前clean patchも同じ1位置shiftを適用。
        previous_patch = torch.cat(
            (torch.zeros_like(patches[:, :1]), patches[:, :-1]), dim=1)
        # LM位置を独立patch batchへ畳む。[B,S,P,C]->[B*S,C,P]。
        target = rearrange(patches, "b s p c -> (b s) c p")
        condition = rearrange(previous_patch, "b s p c -> (b s) c p")
        # [B,S]->[B,S,P]。target patch内のP frame全部を教師対象にする。
        patch_mask = loss_mask[..., None].expand(-1, -1,
                                                 self.config.patch_size)
        flow_loss = self.flow(target, rearrange(mu,
                                                "b s d -> (b s) d"), condition,
                              rearrange(patch_mask, "b s p -> (b s) p"),
                              progress)
        stop_logits = self.stop(previous_semantic)  # [B,S,2]、同じshift位置でstop分類。
        # CEのclass軸を中央へ移す。[B,S,2]->[B,2,S]、CE結果は[B,S]。
        stop_errors = F.cross_entropy(rearrange(stop_logits.float(),
                                                "b s k -> b k s"),
                                      stop_labels,
                                      reduction="none")
        stop_loss = (stop_errors *
                     loss_mask).sum() / loss_mask.sum().clamp_min(1)
        return {
            "loss": flow_loss + stop_loss,
            "flow_loss": flow_loss,
            "stop_loss": stop_loss,
            "stop_logits": stop_logits
        }

    @torch.no_grad()
    def generate(
        self,
        text_ids: torch.Tensor,
        patches: torch.Tensor,
        audio_mask: torch.Tensor,
        max_patches: int = 1000,
        min_patches: int = 1,
        steps: int = 10,
        cfg: float = 2.0,
    ) -> torch.Tensor:
        """単一sampleのcompact prefixから連続patch [1,N,P,C]を生成。

        prefixはAUDIO_STARTで終わるか、continuation用の観測patchで終わる。
        reference-only prefixのLocDiT previous条件はゼロにする（隣接音声ではない）。
        入力text_ids/audio_mask:[1,S]、patches:[1,S,P,C]。S=conditioning位置数、
        N=生成patch数、P=patch内frame数、C=VAE特徴次元数。min/max_patchesは生成長制限、
        stepsはpatchごとのODE分割数、cfgはguidance。

        Args:
            text_ids: 本文trackのID[B,S]。Bは発話数、Sは本文と音声を含む全位置数。
            patches: 連続音声patch[B,S,P,C]。Bは発話数、SはLM列長、Pはpatch内フレーム数、
                CはVAEのlatentの特徴次元数。
            audio_mask: 音声patch位置のbool mask[B,S]。Bは発話数、Sは列長。
            max_patches: 生成する音声patch数の上限。
            min_patches: 終了判定を許可する前に生成する最小patch数。
            steps: Euler積分のステップ数。
            cfg: CFGで条件付き速度を強調する係数。

        Returns:
            生成patch[1,N,P,C]。1は単一発話、Nは生成patch数、Pはpatch内フレーム数、
            CはVAEのlatentの特徴次元数。
        """
        if self.training or text_ids.size(0) != 1:
            raise ValueError("use eval() and batch=1")
        generated = []
        previous = patches[:, -1] if bool(
            audio_mask[0, -1]) else torch.zeros_like(patches[:, -1])
        for index in range(max_patches):
            valid = torch.ones_like(audio_mask, dtype=torch.bool)
            semantic, residual = self.encode_history(text_ids, patches,
                                                     audio_mask, valid)
            mu = torch.cat((self.semantic_to_dit(
                semantic[:, -1]), self.residual_to_dit(residual[:, -1])),
                           dim=-1)
            patch = self.flow.sample(mu,
                                     rearrange(previous, "b p c -> b c p"),
                                     steps=steps,
                                     cfg=cfg)
            patch = rearrange(patch, "b c p -> b p c")  # [1,C,P]->[1,P,C]。
            generated.append(patch)
            # 最終patchも生成してからstopを見る。公式packerのlast-patch labelと整合。
            if index + 1 >= min_patches and int(
                    self.stop(semantic[:, -1]).argmax(-1)) == 1:
                break
            # 次の生成の観測historyへ1patchを加える。text/mask:[1,S]->[1,S+1]。
            text_ids = F.pad(text_ids, (0, 1), value=0)
            patches = torch.cat((patches, patch[:, None]),
                                dim=1)  # [1,S+1,P,C]。
            audio_mask = F.pad(audio_mask, (0, 1), value=True)
            previous = patch
        if not generated:
            raise ValueError("max_patches must be positive")
        return torch.stack(generated, dim=1)  # N個の[1,P,C]->[1,N,P,C]。
