"""CFMのnoise付与、velocity損失、Euler更新とCFGを局所実装。

参照: index-tts revision d9e41aac89fd00b3d71497fddb287b7f24613712。
各層・演算をこのファイルに実装し、外部のモデル本体はimportしない。
B=発話数、T=時間位置数、D=特徴次元数。
"""
from __future__ import annotations

from abc import ABC
from typing import Any

from tqdm import tqdm
import torch

from models.s2mel.dit import DiT


class BASECFM(torch.nn.Module, ABC):
    """GT Melとノイズを補間し、速度回帰の学習とEuler積分を行う。"""

    def __init__(self, args: Any) -> None:
        """GT Melとノイズを補間し、速度回帰の学習とEuler積分を行うための層と設定を構築する。

        args=公式configと同じ階層の構成値。

        Args:
            args: 対象コンストラクタへ渡す位置引数、または階層化したモデル設定。

        Returns:
            None。層・設定を登録する。ここでは学習・音声生成を行わない。
        """
        super().__init__()
        self.sigma_min = 1e-6

        self.estimator = None

        self.in_channels = args.DiT.in_channels

        self.criterion = torch.nn.MSELoss(
        ) if args.reg_loss_type == "l2" else torch.nn.L1Loss()

        if hasattr(args.DiT, 'zero_prompt_speech_token'):
            self.zero_prompt_speech_token = args.DiT.zero_prompt_speech_token
        else:
            self.zero_prompt_speech_token = False

    @torch.inference_mode()
    def inference(self,
                  mu: torch.Tensor,
                  x_lens: torch.Tensor,
                  prompt: Any,
                  style: torch.Tensor,
                  f0: torch.Tensor | None,
                  n_timesteps: Any,
                  temperature: Any = 1.0,
                  inference_cfg_rate: Any = 0.5) -> torch.Tensor:
        """初期ノイズを生成し、条件付きflowのEuler積分でMelへ変換する。

        Args:
            mu: Mel時刻に整列済みのsemantic条件[B,T,C]。Bは発話数、Tは全Mel長、Cは条件特徴次元数。
            x_lens: 各発話のMel有効フレーム数[B]。Bは発話数。
            prompt: 参照Mel[B,F,R]。Bは発話数、FはMel bin数、Rは参照長。
            style: 話者条件[B,192]。Bは発話数。
            f0: 任意の基本周波数[B,T]。Bは発話数、Tは条件時間長。Noneでは使用しない。
            n_timesteps: Euler積分のステップ数。
            temperature: samplingまたは初期ノイズの温度係数。
            inference_cfg_rate: CFGで条件付き速度を強調する係数。

        Returns:
            参照区間を含むMel[B,F,T]。Bは発話数、FはMel bin数、Tは全Mel長。
        """
        B, T = mu.size(0), mu.size(1)
        z = torch.randn([B, self.in_channels, T],
                        device=mu.device) * temperature
        t_span = torch.linspace(0, 1, n_timesteps + 1, device=mu.device)

        return self.solve_euler(z, x_lens, prompt, mu, style, f0, t_span,
                                inference_cfg_rate)

    def solve_euler(self,
                    x: torch.Tensor,
                    x_lens: torch.Tensor,
                    prompt: Any,
                    mu: torch.Tensor,
                    style: torch.Tensor,
                    f0: torch.Tensor | None,
                    t_span: torch.Tensor,
                    inference_cfg_rate: Any = 0.5) -> torch.Tensor:
        """時刻列に沿って速度を評価し、Mel状態をEuler法で更新する。

        Args:
            x: 積分途中のMel状態[B,F,T]。Bは発話数、FはMel bin数、Tは全Mel長。
            x_lens: 各発話のMel有効フレーム数[B]。Bは発話数。
            prompt: 参照Mel[B,F,R]。Bは発話数、FはMel bin数、Rは参照長。
            mu: Mel時刻に整列済みのsemantic条件[B,T,C]。Bは発話数、Tは全Mel長、Cは条件特徴次元数。
            style: 話者条件[B,192]。Bは発話数。
            f0: 任意の基本周波数[B,T]。Bは発話数、Tは条件時間長。Noneでは使用しない。
            t_span: 積分時刻の列[J+1]。JはEuler積分のステップ数。
            inference_cfg_rate: CFGで条件付き速度を強調する係数。

        Returns:
            最後の積分時刻のMel[B,F,T]。Bは発話数、FはMel bin数、Tは全Mel長。
        """
        t, _, _ = t_span[0], t_span[-1], t_span[1] - t_span[0]

        # 各積分時刻の状態を保持し、最後の状態を出力する。
        # 中間状態はこの関数の外部出力には含めない。
        sol = []
        # 参照区間を生成状態から除外し、固定の参照条件として別に渡す。
        prompt_len = prompt.size(-1)
        prompt_x = torch.zeros_like(x)
        prompt_x[..., :prompt_len] = prompt[..., :prompt_len]
        x[..., :prompt_len] = 0
        if self.zero_prompt_speech_token:
            mu[..., :prompt_len] = 0
        for step in tqdm(range(1, len(t_span))):
            dt = t_span[step] - t_span[step - 1]
            if inference_cfg_rate > 0:
                # 条件あり・条件なしの入力を発話軸で連結する。[B,...]→[2B,...]。
                stacked_prompt_x = torch.cat(
                    [prompt_x, torch.zeros_like(prompt_x)], dim=0)
                stacked_style = torch.cat(
                    [style, torch.zeros_like(style)], dim=0)
                stacked_mu = torch.cat([mu, torch.zeros_like(mu)], dim=0)
                stacked_x = torch.cat([x, x], dim=0)
                stacked_t = torch.cat([t.unsqueeze(0), t.unsqueeze(0)], dim=0)

                # 1回の速度予測で、条件あり・条件なしの両方を計算する。
                stacked_dphi_dt = self.estimator(
                    stacked_x,
                    stacked_prompt_x,
                    x_lens,
                    stacked_t,
                    stacked_style,
                    stacked_mu,
                )

                # 予測速度[2B,F,T]を条件あり・条件なしの各[B,F,T]へ分ける。
                dphi_dt, cfg_dphi_dt = stacked_dphi_dt.chunk(2, dim=0)

                # 条件あり速度を基準に、条件ありと条件なしの差を加えて強調する。
                dphi_dt = (1.0 + inference_cfg_rate
                           ) * dphi_dt - inference_cfg_rate * cfg_dphi_dt
            else:
                dphi_dt = self.estimator(x, prompt_x, x_lens, t.unsqueeze(0),
                                         style, mu)

            x = x + dt * dphi_dt
            t = t + dt
            sol.append(x)
            if step < len(t_span) - 1:
                dt = t_span[step + 1] - t
            x[:, :, :prompt_len] = 0

        return sol[-1]

    def forward(self, x1: torch.Tensor, x_lens: torch.Tensor,
                prompt_lens: torch.Tensor, mu: torch.Tensor,
                style: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """GT Melとノイズを補間し、速度回帰の学習とEuler積分を行う。

        Args:
            x1: GT Mel[B,F,T]。Bは発話数、FはMel bin数、Tは全フレーム数。
            x_lens: 各発話のMel有効フレーム数[B]。Bは発話数。
            prompt_lens: 各発話の参照Melフレーム数[B]。Bは発話数。
            mu: Mel時刻に整列済みのsemantic条件[B,T,C]。Bは発話数、Tは全Mel長、Cは条件特徴次元数。
            style: 話者条件[B,192]。Bは発話数。

        Returns:
            速度回帰損失[]と推定終点Mel[B,F,T]の組。Bは発話数、FはMel bin数、Tは全Mel長。[]はscalar。
        """
        b, _, t = x1.shape

        # 発話ごとに時刻tを一様分布から取り、broadcast用の[B,1,1]にする。
        t = torch.rand([b, 1, 1], device=mu.device, dtype=x1.dtype)
        # GTと同shapeの標準正規ノイズ[B,F,T]を生成する。
        z = torch.randn_like(x1)

        y = (1 - (1 - self.sigma_min) * t) * z + t * x1
        u = x1 - (1 - self.sigma_min) * z

        prompt = torch.zeros_like(x1)
        for bib in range(b):
            prompt[bib, :, :prompt_lens[bib]] = x1[bib, :, :prompt_lens[bib]]
            # 参照区間の生成状態をゼロにし、参照Melは別入力へ保持する。
            y[bib, :, :prompt_lens[bib]] = 0
            if self.zero_prompt_speech_token:
                mu[bib, :, :prompt_lens[bib]] = 0

        estimator_out = self.estimator(y, prompt, x_lens,
                                       t.squeeze(1).squeeze(1), style, mu,
                                       prompt_lens)
        loss = 0
        for bib in range(b):
            loss += self.criterion(
                estimator_out[bib, :, prompt_lens[bib]:x_lens[bib]],
                u[bib, :, prompt_lens[bib]:x_lens[bib]])
        loss /= b

        return loss, estimator_out + (1 - self.sigma_min) * z


class CFM(BASECFM):
    """DiTをvelocity推定器として条件付きFlow Matchingを構築する。
    """

    def __init__(self, args: Any) -> None:
        """DiTをvelocity推定器として条件付きFlow Matchingを構築するための層と設定を構築する。

        args=公式configと同じ階層の構成値。

        Args:
            args: 対象コンストラクタへ渡す位置引数、または階層化したモデル設定。

        Returns:
            None。層・設定を登録する。ここでは学習・音声生成を行わない。
        """
        super().__init__(args)
        if args.dit_type == "DiT":
            self.estimator = DiT(args)
        else:
            raise NotImplementedError(
                f"Unknown diffusion type {args.dit_type}")

    def enable_torch_compile(self) -> None:
        """速度推定器をtorch.compileの対象として登録する。

        Returns:
            None。推論用estimatorをcompile対象に置き換える。
        """
        if torch.distributed.is_initialized():
            torch._inductor.config.reorder_for_compute_comm_overlap = True
        self.estimator = torch.compile(
            self.estimator,
            fullgraph=True,
            dynamic=True,
        )
