"""ローカルのsemantic-to-Melネットワークを学習・合成APIへ接続する。

DiT、長さ調整、Euler積分は本ディレクトリの実装を使う。公式s2mel設定から、
time/style token、MLP/WaveNet出力、weight norm、条件dropout、mask、
U-skip/long skip等の選択も公式と同じになる。
小型variantは演算構造を維持して特徴次元数・層数等だけを変更する。

B=発話数、T=参照と生成対象を合計したMel長、F=Mel bin数、
C=semantic条件の特徴次元数、Dsem=入力semantic特徴次元数（通常1024）。
"""
from __future__ import annotations

from typing import Any

from torch import nn
import torch

from models.config import resolve_config


class LengthRegulator(nn.Module):
    """公式InterpolateRegulatorの入出力を既存pipelineへ接続する。"""

    def __init__(self, network: nn.Module) -> None:
        """構築済みの長さ調整ネットワークを登録し、パラメータを共有する。

        Args:
            network: ローカルS2Mel本体が保持する構築済みモジュール。重みを共有する。

        Returns:
            None。層・設定を登録する。ここでは学習・音声生成を行わない。
        """
        super().__init__()
        self.core = network

    def forward(
        self,
        semantic: torch.Tensor,
        source_lengths: torch.Tensor,
        mel_lengths: torch.Tensor,
    ) -> torch.Tensor:
        """特徴[B,Ts,Dsem]を条件[B,max(mel_lengths),C]へ変換する。

        Ts=入力特徴長。source_lengths[B]は接続APIの有効長。公式処理はbatch全体の入力時間軸を最大Mel長へ補間し、
        出力をmaskする。 発話ごとに切り出して補間する独自処理は行わない。

        Args:
            semantic: Mel時刻に整列する連続条件[B,T,C]。Bは発話数、Tは条件時間長、Cは条件特徴次元数。
            source_lengths: semantic入力の有効長[B]。Bは発話数。
            mel_lengths: 整列先のMelフレーム数[B]。Bは発話数。

        Returns:
            Mel時刻へ整列したsemantic特徴[B,M,C]。Bは発話数、Mは指定Mel長、Cはregulator出力の特徴次元数。
        """
        if source_lengths.shape != mel_lengths.shape:
            raise ValueError("Expected matching length vectors [batch]")
        # 公式推論と同じ引数。n_quantizers=3は公開infer_v2_5の指定値。
        # 補間・content射影・Conv・GroupNorm・出力maskはローカルの長さ調整ネットワークで計算する。
        return self.core(semantic, ylens=mel_lengths, n_quantizers=3,
                         f0=None)[0]


class ConditionalFlowMatching(nn.Module):
    """ローカルflowモデルを学習と合成へ接続する。"""

    def __init__(self, network: nn.Module) -> None:
        """構築済みのflowネットワークを登録し、パラメータを共有する。

        Args:
            network: ローカルS2Mel本体が保持する構築済みモジュール。重みを共有する。

        Returns:
            None。層・設定を登録する。ここでは学習・音声生成を行わない。
        """
        super().__init__()
        self.core = network

    @property
    def estimator(self) -> nn.Module:
        """velocity推定器であるローカルDiTを返す。

        Returns:
            CFMが保持する速度予測ネットワーク。
        """
        return self.core.estimator

    def forward(
        self,
        mel: torch.Tensor,
        lengths: torch.Tensor,
        prompt_lengths: torch.Tensor,
        semantic: torch.Tensor,
        style: torch.Tensor,
    ) -> torch.Tensor:
        """Mel[B,F,T]、長さ/参照長[B]、条件[B,T,C]、話者[B,192]→scalar loss。

        noise、時刻sample、prompt処理、条件dropout、velocity予測はローカルのflowネットワークで計算する。
        学習APIでは公式戻り値tupleの第1要素であるlossだけを返す。

        Args:
            mel: Mel特徴[B,F,M]。Bは発話数、FはMel bin数、MはMelフレーム数。
            lengths: 各発話の有効時間長[B]。Bは発話数。
            prompt_lengths: 各発話の参照Melフレーム数[B]。Bは発話数。
            semantic: Mel時刻に整列する連続条件[B,T,C]。Bは発話数、Tは条件時間長、Cは条件特徴次元数。
            style: 話者条件[B,192]。Bは発話数。

        Returns:
            flow速度の回帰損失[]。[]はscalar。参照区間とpaddingを損失から除外する。
        """
        return self.core(mel, lengths, prompt_lengths, semantic, style)[0]

    @torch.no_grad()
    def sample(
        self,
        semantic: torch.Tensor,
        lengths: torch.Tensor,
        prompt: torch.Tensor,
        style: torch.Tensor,
        steps: int = 25,
        cfg: float = 0.7,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """条件[B,T,C]、長さ[B]、参照Mel[B,F,R]、話者[B,192]→Mel[B,F,T-R]。

        R=参照Mel長。初期noiseとEuler/CFGの計算はローカルのflow積分器で計算する。
        出力先頭R frameを除く操作も公開infer_v2_5と同じ位置で行う。

        Args:
            semantic: Mel時刻に整列する連続条件[B,T,C]。Bは発話数、Tは条件時間長、Cは条件特徴次元数。
            lengths: 各発話の有効時間長[B]。Bは発話数。
            prompt: 参照Mel[B,F,R]。Bは発話数、FはMel bin数、Rは参照長。
            style: 話者条件[B,192]。Bは発話数。
            steps: Euler積分のステップ数。
            cfg: CFGで条件付き速度を強調する係数。
            temperature: samplingまたは初期ノイズの温度係数。

        Returns:
            生成区間のMel[B,F,T-R]。Bは発話数、FはMel bin数、Tは全Mel長、Rは参照長。
        """
        if self.core.training or steps < 1:
            raise ValueError("Call eval() and use positive CFM steps")
        # 1. DiTのRoPE/mask等のcacheをローカルのDiTで準備する。KV cacheは使わない。
        self.estimator.setup_caches(max_batch_size=semantic.size(0),
                                    max_seq_length=semantic.size(1))
        # 2. 公式CFMは参照区間を含む[B,F,T]を返す。
        full = self.core.inference(
            semantic,
            lengths,
            prompt,
            style,
            None,
            steps,
            temperature=temperature,
            inference_cfg_rate=cfg,
        )
        # 3. 参照区間だけ除く。[B,F,T]→[B,F,T-R]。
        return full[:, :, prompt.size(-1):]


class S2Mel(nn.Module):
    """ローカルの音声特徴→Melネットワークを保持する学習用モジュール。"""

    def __init__(self, **spec: Any) -> None:
        """公式s2mel section全体でregulator/CFM/DiTをrandom構築する。

        Args:
            **spec: 公式設定の参照先、または展開済み設定と容量variant。

        Returns:
            None。層・設定を登録する。ここでは学習・音声生成を行わない。
        """
        super().__init__()
        from models.s2mel.network import SemanticToMelNetwork
        from omegaconf import OmegaConf

        self.configuration = resolve_config("s2mel", spec)
        self.core = SemanticToMelNetwork(OmegaConf.create(self.configuration))

    @property
    def regulator(self) -> LengthRegulator:
        """公式regulatorを接続APIへ包む。

        同じパラメータを参照し、再構築しない。

        Returns:
            本体と重みを共有するLengthRegulator。
        """
        return LengthRegulator(self.core.models["length_regulator"])

    @property
    def flow(self) -> ConditionalFlowMatching:
        """公式CFMを接続APIへ包む。

        重みはself.coreと共有する。

        Returns:
            本体と重みを共有するConditionalFlowMatching。
        """
        return ConditionalFlowMatching(self.core.models["cfm"])

    def forward(
        self,
        features: torch.Tensor,
        mel: torch.Tensor,
        speaker: torch.Tensor,
    ) -> torch.Tensor:
        """1発話の特徴[Ts,1024]、Mel[F,M]、話者[192]からscalar lossを返す。

        Ts=semantic長、F=Mel bin数、M=Mel長。prompt長sampleだけが学習例側の選択。
        regulator/DiTのforwardは公式の処理を変更しない。

        Args:
            features: 単一発話のsemantic特徴[Ts,1024]。Tsは特徴フレーム数。
            mel: 単一発話のGT Mel[F,M]。FはMel bin数、Mはフレーム数。
            speaker: 単一発話の話者埋め込み[192]。192はCAMPPlusの出力の特徴次元数。

        Returns:
            flow速度の回帰損失[]。[]はscalar。
        """
        # 1. 発話batch軸を追加し、公式regulatorでMel長に整列する。
        lengths = torch.tensor([mel.size(-1)], device=mel.device)
        condition = self.core.models["length_regulator"](
            features[None],
            ylens=lengths,
            n_quantizers=3,
            f0=None,
        )[0]
        # 2. DiTの位置情報・mask計算に必要なcacheを準備する。
        self.core.models["cfm"].estimator.setup_caches(
            max_batch_size=1,
            max_seq_length=mel.size(-1),
        )
        # 3. 観測prompt長を選び、noise/velocity処理は公式CFMへ渡す。
        prompt_length = torch.randint(
            0,
            max(1,
                int(mel.size(-1) * 0.3) + 1),
            (1, ),
            device=mel.device,
        )
        return self.core(
            mel[None],
            lengths,
            prompt_length,
            condition,
            speaker[None],
        )[0]
