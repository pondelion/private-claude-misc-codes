"""固定特徴抽出器とvocoderをT2S・S2Melへ接続する。

W2v-BERT、CAMPPlus、BigVGANは外部の学習済みモデルを利用する。
感情Conformer/PerceiverはIndexTTS自身の構成要素で、IndexTTSのGPT
checkpointから読み込む。独立した外部感情モデルの重みではない。
W2v-BERTは外部の学習済みモデルを読み込む。CAMPPlus、感情encoder、BigVGANは
ローカルのネットワーク定義へ固定重みを読み込む。資産の準備はprepare_assets.pyで行う。
この学習例では抽出器・感情frontend・vocoderを固定して再利用する。
T2S学習中にcodecを更新すると離散GTの意味が変わるため、codecも先に学習し固定する。

役割:
    生音声をT2S/codecが扱う特徴へ変える前段と、Melを波形へ戻す後段の境界。
    SemanticFeatureExtractorはW2v-BERT hidden layerと固定統計を使う。
    ReferenceConditionerはCAMPPlusで声質、Conformer/Perceiverで感情を
    取り出す。render_melは実vocoderにMelを渡す。

呼び出し前に行うこと:
    公式configで各nn.Moduleを構築し、対応重み・mean/varianceをロードする。
    音声のmono化、resample、SeamlessM4T frontend、80bin fbank抽出は
    呼び出し側で行う。このファイルは音声pathの読み込みや自動取得を行わない。
    train/evalとdeviceも注入する実moduleに設定する。

入出力:
    SemanticFeatureExtractor: frontend[B,Tin,Fin]→特徴[B,T50,1024]。
    ReferenceConditioner: fbank[B,Tbank,80]と特徴[B,T50,1024]→
        speaker[B,192]とemotion[B,1024]。
    render_mel: Mel[B,F,Tmel]→波形[B,1,S]。
    B=発話数、Tin/Tbank/T50/Tmel=各段のframe数、Fin=音声前処理の出力特徴次元数、
    F=Mel bin数、S=波形sample数。異なる時間軸を同じ長さと解釈しない。
    scratch学習用cacheへ保存する項目はtraining_example.pyにまとめている。
"""
from torch import nn
from torch.nn import functional as F
import torch


class SemanticFeatureExtractor(nn.Module):
    """公式W2v-BERTの中間特徴を固定統計で正規化する抽出器。"""

    def __init__(self, w2v_bert: nn.Module, mean: torch.Tensor,
                 variance: torch.Tensor) -> None:
        """実W2v-BERTと、各特徴の平均/分散[D]またはbroadcast互換shapeを受ける。

        Args:
            w2v_bert: 外部の学習済みW2v-BERT特徴抽出モデル。
            mean: 正規化平均[1024]。1024はW2v-BERT特徴次元数。
            variance: 正規化分散[1024]。1024はW2v-BERT特徴次元数。

        Returns:
            None。層・設定を登録する。ここでは学習・音声生成を行わない。
        """
        super().__init__()
        self.model = w2v_bert
        self.register_buffer("mean", mean)
        self.register_buffer("std", variance.sqrt())

    @torch.no_grad()
    def forward(self, input_features: torch.Tensor,
                attention_mask: torch.Tensor) -> torch.Tensor:
        """音響特徴[B,Tin,Fin]と有効mask[B,Tin]から[B,T50,1024]を得る。

        B=発話数、Tin=frontend時刻数、Fin=frontend特徴次元数、T50=約50Hz時刻数。
        入力は16kHz波形そのものではなく、SeamlessM4T frontendの出力。

        Args:
            input_features:音声frontendの特徴[B,Tin,Fin]。Bは発話数、
                Tinはfrontendフレーム数、Finはfrontend特徴次元数。
            attention_mask: 有効位置のmask[B,K]。Bは発話数、Kは参照可能な位置数。1が有効。

        Returns:
            正規化semantic特徴[B,T50,1024]。Bは発話数、T50は約50Hzフレーム数。
        """
        # SeamlessM4TFeatureExtractorで得た[B,Tin,Fin]。波形sample軸ではない。
        output = self.model(input_features=input_features,
                            attention_mask=attention_mask,
                            output_hidden_states=True)
        # 公式infer_v2_5.get_embはhidden_states[17]を使う。
        return (output.hidden_states[17] -
                # 正規化特徴[B,T50,1024]。Bは発話数、T50は約50Hzの特徴フレーム数。
                self.mean) / self.std


class ReferenceConditioner(nn.Module):
    """公式CAMPPlus/Conformer/Perceiver本体を注入して参照特徴を作る。

    cacheを使わずjoint trainingする場合も、同じ計算を勾配付きで呼べる。
    本クラス自体にno_gradは付けない。固定するかどうかは呼び出し側が決める。
    """

    def __init__(self, campplus: nn.Module, emotion_conformer: nn.Module,
                 emotion_perceiver: nn.Module) -> None:
        """話者抽出器と、感情Conformer/Perceiverの公式実体を登録する。

        Args:
            campplus: 学習済みの話者特徴抽出モジュール。
            emotion_conformer: 感情特徴抽出用Conformer。
            emotion_perceiver: 感情特徴を固定個数へ集約するPerceiver。

        Returns:
            None。層・設定を登録する。ここでは学習・音声生成を行わない。
        """
        super().__init__()
        self.campplus = campplus
        self.emotion_conformer = emotion_conformer
        self.emotion_perceiver = emotion_perceiver

    def forward(
        self,
        speaker_fbank: torch.Tensor,
        emotion_features: torch.Tensor,
        emotion_lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """参照特徴からspeaker[B,192]とemotion[B,1024]を返す。

        speaker_fbank:[B,Tbank,80]、emotion_features:[B,T50,1024]、
        emotion_lengths:[B]。B=発話数、Tbank=fbank時刻数、T50=約50Hz時刻数。
        speaker側はpaddingのない参照窓を使い、emotion側は有効長でmaskする。

        Args:
            speaker_fbank: 話者参照のfilterbank特徴[B,Tbank,80]。Bは発話数、Tbankはフレーム数。
            emotion_features: 感情参照特徴[B,Te,1024]。Bは発話数、Teは約50Hzの感情参照フレーム数。
            emotion_lengths: 各感情参照の有効フレーム数[B]。Bは発話数。

        Returns:
            話者埋め込み[B,192]と感情埋め込み[B,1024]の組。Bは発話数。
        """
        # 話者fbank[B,Tbank,80]と感情特徴[B,T50,1024]。Bは発話数、TbankとT50は各特徴の時間長。
        # 公式kaldi fbankと同じ時間方向平均除去。ここは非paddingの参照窓を渡す。
        speaker_fbank = speaker_fbank - speaker_fbank.mean(dim=1, keepdim=True)
        # 話者埋め込み[B,192]。Bは発話数、192はCAMPPlusの出力の特徴次元数。
        speaker = self.campplus(speaker_fbank)
        # Tenc=Conformer後の時刻数、Denc=Conformer出力の特徴次元数。
        # 感情参照[B,T50,1024]→隠れ状態[B,Tenc,Denc]と有効mask[B,1,Tenc]。Tencは圧縮後長、Dencは
        # encoder出力の特徴次元数。
        hidden, valid = self.emotion_conformer(emotion_features,
                                               emotion_lengths)
        # Perceiverはcontextの前へ1個のlearned queryをKVとしても連結する。
        # maskもquery分をTrueで追加。[B,1,Tenc] -> [B,1+Tenc]。
        mask = F.pad(valid.squeeze(1).bool(), (1, 0), value=True)
        # 1個のlearned queryで可変長参照を集約。[B,Tenc,Denc]->[B,1,1024]。
        emotion = self.emotion_perceiver(hidden, mask=mask).squeeze(1)
        # 話者[B,192]と感情[B,1024]を返す。Bは発話数、末尾軸は各埋め込みの特徴次元数。
        return speaker, emotion


@torch.no_grad()
def render_mel(vocoder: nn.Module, mel: torch.Tensor) -> torch.Tensor:
    """公式BigVGAN等の本体を注入。

    [B,F,Tmel] -> [B,1,S]。

    sample rate/hopはvocoder configに従う。値域・Mel正規化が異なるvocoderを名前だけで交換すると音質が破綻するため、
    公式対応checkpointを使う。 B=発話数、F=Mel bin数、Tmel=Melフレーム数、Sは波形サンプル数、1はmonoの音声チャネル。

    Args:
        vocoder: Mel[B,F,M]を波形[B,1,S]へ変換するモジュール。
        mel: Mel特徴[B,F,M]。Bは発話数、FはMel bin数、MはMelフレーム数。

    Returns:
        波形[B,1,S]。Bは発話数、1はmonoチャネル、Sは生成サンプル数。
    """
    return vocoder(mel)
