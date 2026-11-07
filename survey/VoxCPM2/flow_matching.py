"""次の連続音声patchを、noiseから生成する条件付きFlow Matching。

学習時:
    targetはGT VAE patch[N,C,P]、previousは直前patch[N,C,P]、muは
    TSLM/RALM由来の条件[N,2D]。N=処理patch数、C=VAE特徴次元数、
    P=patch内frame数、D=LocDiTの隠れ状態の次元数。mask[N,P]は教師に含める位置を示す。
    tをsampleし、state=(1-t)*target+t*noiseを1回作る。LocDiTに
    noise-targetを予測させ、mask付きvelocity MSEで学ぶ。
    学習時に長い生成loop全体をbackpropagateするわけではない。

生成時:
    sample(mu, previous, ...)はnoiseからt=1→0へEuler更新し、
    新しいpatch[N,C,P]を返す。t=0がclean、t=1がnoiseで、IndexTTSと逆。
    CFGはmuを与える条件付き予測とmuを落とした予測を合成する。
    mean_mode=Trueは時刻区間を使う別経路で、JVPを含む。既定は通常のCFM。

組み込み:
    flow = PatchFlowMatching(LocalDiT(local_config, latent_dim=64))
    学習時はflow(target, mu, previous, mask)のscalarをbackwardする。
    model.pyがLM条件とGTのshift、batch/patchの並べ替えを担当する。
    PyTorchとeinopsを使用し、単独のCLIはない。
"""
import torch
from einops import rearrange
from torch import nn

from local_modules import LocalDiT


class PatchFlowMatching(nn.Module):
    """patch局所のCFM/mean-flow学習と、CFG付きODE生成を行う。"""

    def __init__(self,
                 estimator: LocalDiT,
                 cfg_drop: float = 0.1,
                 scheduler: str = "log-norm",
                 mean_mode: bool = False) -> None:
        """実LocDiT、LM条件dropout率、時刻分布、mean-flow利用有無を登録する。

        Args:
            estimator: 現在のlatent patchと時刻から速度を予測するLocalDiT。
            cfg_drop: 学習時にLM条件を無効化する確率。
            scheduler: flow学習で時刻を抽出する分布の種類。
            mean_mode: 区間の平均速度を学習するmean-flowを使うか。

        Returns:
            None。層・設定を登録する。ここでは学習・音声生成を行わない。
        """
        super().__init__()
        self.estimator = estimator
        self.cfg_drop = cfg_drop
        self.scheduler = scheduler
        self.mean_mode = mean_mode

    def forward(
        self,
        target: torch.Tensor,
        mu: torch.Tensor,
        previous: torch.Tensor,
        mask: torch.Tensor,
        progress: float = 0.0,
    ) -> torch.Tensor:
        """target/previous:[N,C,P], mu:[N,2D], mask:[N,P] -> scalar。

        MSEをC方向平均してから、target segmentのP位置で重み付き平均する。ordinary CFMが既定。
        公式のoptional mean-flowもJVPで実装する。N=処理patch数（通常B*S）、C=VAE特徴次元数、
        P=patch内frame数、D=LocDiTの隠れ状態の次元数。progressは学習進捗0..1で、
        mean-flow区間sample比率を調整する。

        Args:
            target: GT patch[N,C,P]。Nは処理patch数、CはVAEのlatentの特徴次元数、
                Pはpatch内フレーム数。
            mu: semantic・residual条件[N,2D]。Nは処理patch数、DはLocalDiTの内部特徴の次元数。
            previous: 直前の音声patch[N,C,P]。Nは処理patch数、CはVAEのlatentの特徴次元数、
                Pはpatch内フレーム数。
            mask: 損失を計算するpatch内フレームの重み[N,P]。Nは処理patch数、Pはpatch内フレーム数。
            progress: 学習の進捗率。0から1でmean-flow区間の抽出比率を調整する。

        Returns:
            有効フレームで平均した速度回帰損失[]。[]はscalar。
        """
        count = target.size(0)
        mu = mu * (torch.rand(count, device=target.device)
                   >= self.cfg_drop)[:, None]
        if self.scheduler == "log-norm":
            t = torch.sigmoid(torch.randn(count, device=target.device) - 0.4)
            r = torch.sigmoid(torch.randn(count, device=target.device) - 0.4)
        elif self.scheduler == "uniform":
            t, r = torch.rand(count, device=target.device), torch.rand(
                count, device=target.device)
        else:
            raise ValueError("unsupported time scheduler")
        if self.mean_mode:
            use_interval = torch.rand(
                count, device=target.device) < 0.25 + 0.5 * progress
            low, high = torch.minimum(r, t), torch.maximum(r, t)
            r, t = torch.where(use_interval, low,
                               t), torch.where(use_interval, high, t)
        else:
            r = t
        t, r = t.to(target.dtype), r.to(target.dtype)
        noise = torch.randn_like(target)
        # t[N]を[N,1,1]へ拡張し、同patch内の全C特徴/P位置へ共通のnoise率を使う。
        state = (1 - t[:, None, None]) * target + t[:, None, None] * noise
        velocity_target = noise - target  # tが増える向きのvelocity。

        def field(state_value: torch.Tensor, r_value: torch.Tensor,
                  t_value: torch.Tensor) -> torch.Tensor:
            """state[N,C,P]と区間端r/t[N]からvelocity[N,C,P]を得る。

            N=patch数、C=latentの特徴次元数、P=patch内frame数。mu/previousはclosure条件。

            Args:
                state_value: 積分途中のpatch状態[N,C,P]。Nは処理patch数、
                    CはVAEのlatentの特徴次元数、Pはpatch内フレーム数。
                r_value: 区間の開始時刻[N]。Nは処理patch数。
                t_value: 区間の終了時刻[N]。Nは処理patch数。

            Returns:
                予測したpatch速度[N,C,P]。Nは処理patch数、CはVAEのlatentの特徴次元数、Pはpatch内フレーム数。
            """
            return self.estimator(state_value, mu, t_value, previous,
                                  t_value - r_value)

        if self.mean_mode:
            from torch.func import jvp
            # forward-mode ADを使うため、flash SDPAではなくmath backendにする。
            from torch.nn.attention import SDPBackend, sdpa_kernel
            with sdpa_kernel(SDPBackend.MATH):
                predicted, derivative = jvp(
                    field, (state, r, t),
                    (velocity_target, torch.zeros_like(r), torch.ones_like(t)))
            velocity_target = velocity_target - (t - r)[:, None,
                                                        None] * derivative
        else:
            predicted = field(state, r, t)
        # latent特徴軸Cを平均。[N,C,P]->[N,P]。maskも[N,P]なのでframe数で正規化。
        errors = (predicted.float() -
                  velocity_target.detach().float()).square().mean(dim=1)
        return (errors * mask).sum() / mask.sum().clamp_min(1)

    @torch.no_grad()
    def sample(
        self,
        mu: torch.Tensor,
        previous: torch.Tensor,
        steps: int = 10,
        cfg: float = 2.0,
        sway: float = 1.0,
        zero_star: bool = True,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """mu[N,2D]・previous[N,C,P]を条件にlatent patch[N,C,P]を生成する。

        N=処理patch数、D=LocDiTの隠れ状態の次元数、C=VAE特徴次元数、P=patch内frame数。steps=ODE分割数、
        cfg=guidance scale、sway=時刻再配置係数、zero_star=初期ゼロfieldとunconditional再
        scale、temperature=初期noise倍率。

        Args:
            mu: semantic・residual条件[N,2D]。Nは処理patch数、DはLocalDiTの内部特徴の次元数。
            previous: 直前の音声patch[N,C,P]。Nは処理patch数、CはVAEのlatentの特徴次元数、
                Pはpatch内フレーム数。
            steps: Euler積分のステップ数。
            cfg: CFGで条件付き速度を強調する係数。
            sway: ODEの積分時刻の配置を調整する係数。
            zero_star: 初期速度をゼロにする処理と無条件速度の再scaleを使うか。
            temperature: samplingまたは初期ノイズの温度係数。

        Returns:
            生成patch[N,C,P]。Nは処理patch数、CはVAEのlatentの特徴次元数、Pはpatch内フレーム数。
        """
        if self.training or steps < 1:
            raise ValueError("sample requires eval() and steps>=1")
        state = torch.randn_like(previous) * temperature
        schedule = torch.linspace(1,
                                  0,
                                  steps + 1,
                                  device=mu.device,
                                  dtype=mu.dtype)
        schedule = schedule + sway * (torch.cos(torch.pi / 2 * schedule) - 1 +
                                      schedule)
        if not (schedule[:-1] >= schedule[1:]).all():
            raise ValueError("sway produced a non-monotonic time schedule")
        zero_steps = max(1, int((steps + 1) * 0.04))
        for index in range(steps):
            t = schedule[index].expand(mu.size(0))
            dt = schedule[index] - schedule[index + 1]
            if zero_star and index < zero_steps:
                velocity = torch.zeros_like(state)
            else:
                delta = dt.expand_as(
                    t) if self.mean_mode else torch.zeros_like(t)
                conditional = self.estimator(state, mu, t, previous, delta)
                unconditional = self.estimator(state, torch.zeros_like(mu), t,
                                               previous, delta)
                if zero_star:
                    # patchごとにfieldをflat化。[N,C,P]->[N,C*P]。
                    positive = rearrange(conditional, "n c p -> n (c p)")
                    negative = rearrange(unconditional, "n c p -> n (c p)")
                    # 内積/二乗normはpatchごとの1値[N]。C/P全体へbroadcastする。
                    scale = (positive * negative
                             ).sum(-1) / negative.square().sum(-1).add(1e-8)
                    unconditional = unconditional * scale[:, None, None]
                velocity = unconditional + cfg * (conditional - unconditional)
            state = state - dt * velocity  # 1->0なので減算する。
        return state  # [N,C,P]、連続latentでありtoken IDではない。
