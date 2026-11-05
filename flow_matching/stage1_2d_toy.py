"""段階1：2次元のおもちゃデータでFlow Matchingを学習・可視化する。

ノイズ(t=0)とデータ(t=1)を直線で結んだ点 x_t を入力に、その直線の方向 x1 - x0 を
MLPで回帰する。生成は学習した速度場をEuler法で t=0→1 に積分する。
時刻の向き(t=0がノイズ、t=1がデータ)はIndexTTS2.5の models/s2mel/flow.py と同じ。

使い方:
    python stage1_2d_toy.py                 # 二つの三日月(moons)
    python stage1_2d_toy.py --data 8gauss   # 円周上の8個のガウス分布
出力: outputs/stage1_<data>/ に図(jpg)とアニメーション(gif)。
"""
import argparse
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from matplotlib.animation import FuncAnimation, PillowWriter
from sklearn.datasets import make_moons

LIM = 3.5  # 全ての図で共通の表示範囲 [-LIM, LIM]


def sample_data(name: str, n: int) -> torch.Tensor:
    """目標分布(データ)から n 点をサンプルする。戻り値は[n,2]。"""
    if name == "moons":
        x, _ = make_moons(n_samples=n, noise=0.05)
        x = (x - [0.5, 0.25]) * 1.5  # 原点付近・ノイズと同程度の広がりに合わせる
        return torch.tensor(x, dtype=torch.float32)
    if name == "8gauss":
        angles = torch.randint(0, 8, (n,)) * (2 * np.pi / 8)
        centers = 2.5 * torch.stack([torch.cos(angles), torch.sin(angles)], dim=1)
        return centers + 0.15 * torch.randn(n, 2)
    raise ValueError(name)


class VelocityNet(nn.Module):
    """位置 x[B,2] と時刻 t[B,1] を受け取り、速度 v[B,2] を返すMLP。"""

    def __init__(self, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 2),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([x, t], dim=1))


def flow_matching_loss(model: nn.Module, x1: torch.Tensor) -> torch.Tensor:
    """Flow Matchingの損失。学習の本体はこの関数だけ。"""
    x0 = torch.randn_like(x1)                # ノイズ(始点)
    t = torch.rand(x1.shape[0], 1)           # 時刻 t ~ U(0,1)
    xt = (1 - t) * x0 + t * x1               # 始点と終点を結ぶ直線上の点
    target_v = x1 - x0                       # その直線の速度(tによらず一定)
    return ((model(xt, t) - target_v) ** 2).mean()


@torch.no_grad()
def euler_sample(model: nn.Module, x0: torch.Tensor, n_steps: int) -> torch.Tensor:
    """dx/dt = v(x,t) を t=0→1 までEuler法で積分する。
    戻り値は各ステップの状態を並べた[n_steps+1, B, 2](先頭がノイズ、末尾が生成結果)。"""
    xs = [x0]
    x = x0
    dt = 1.0 / n_steps
    for i in range(n_steps):
        t = torch.full((x.shape[0], 1), i * dt)
        x = x + model(x, t) * dt
        xs.append(x)
    return torch.stack(xs)


def setup_ax(ax, title: str = ""):
    ax.set_xlim(-LIM, LIM)
    ax.set_ylim(-LIM, LIM)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=10)


def plot_training_pairs(data: str, out: Path):
    """図1：学習で使う(ノイズ, データ)のペアと、それを結ぶ直線。"""
    x0 = torch.randn(2000, 2)
    x1 = sample_data(data, 2000)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    setup_ax(axes[0], "noise x0 ~ N(0, I)  (t=0)")
    axes[0].scatter(*x0.T, s=2, c="tab:blue")
    setup_ax(axes[1], "data x1  (t=1)")
    axes[1].scatter(*x1.T, s=2, c="tab:red")
    setup_ax(axes[2], "training pairs: straight lines x0 -> x1\n(pairs are random, so lines cross)")
    for a, b in zip(x0[:150], x1[:150]):
        axes[2].plot([a[0], b[0]], [a[1], b[1]], c="gray", lw=0.5, alpha=0.6)
    axes[2].scatter(*x0[:150].T, s=6, c="tab:blue", label="x0 (noise)", zorder=3)
    axes[2].scatter(*x1[:150].T, s=6, c="tab:red", label="x1 (data)", zorder=3)
    axes[2].legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "fig1_training_pairs.jpg", dpi=110)
    plt.close(fig)


def plot_loss(losses: list[float], out: Path):
    """図2：損失の推移(200stepの移動平均も重ねる)。"""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(losses, lw=0.5, alpha=0.4, label="per step")
    k = 200
    ma = np.convolve(losses, np.ones(k) / k, mode="valid")
    ax.plot(np.arange(k - 1, len(losses)), ma, lw=1.5, label=f"moving avg ({k})")
    ax.set_xlabel("step")
    ax.set_ylabel("MSE(v_pred, x1 - x0)")
    ax.set_title("training loss (does not go to 0; see README)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "fig2_loss.jpg", dpi=110)
    plt.close(fig)


@torch.no_grad()
def plot_marginal_vs_generated(model, data: str, out: Path):
    """図3：上段=学習の補間式で作った x_t の分布、下段=学習した速度場で積分した途中状態と速度場。
    両者が各 t で一致していれば、速度場が「ノイズ→データの分布の流れ」を正しく学べている。"""
    ts = [0.0, 0.25, 0.5, 0.75, 1.0]
    n_steps = 100
    x0 = torch.randn(2000, 2)
    traj = euler_sample(model, x0, n_steps)
    x1 = sample_data(data, 2000)
    x0_ref = torch.randn(2000, 2)
    g = torch.linspace(-LIM, LIM, 21)
    gx, gy = torch.meshgrid(g, g, indexing="xy")
    grid = torch.stack([gx.flatten(), gy.flatten()], dim=1)

    fig, axes = plt.subplots(2, len(ts), figsize=(3.2 * len(ts), 6.6))
    for j, t in enumerate(ts):
        xt = (1 - t) * x0_ref + t * x1
        setup_ax(axes[0, j], f"interpolation (1-t)x0 + t x1, t={t}")
        axes[0, j].scatter(*xt.T, s=2, c="tab:purple")
        v = model(grid, torch.full((grid.shape[0], 1), t))
        setup_ax(axes[1, j], f"ODE state + learned v(x,t), t={t}")
        axes[1, j].quiver(grid[:, 0], grid[:, 1], v[:, 0], v[:, 1], color="gray",
                          angles="xy", scale_units="xy", scale=6, width=0.004)
        axes[1, j].scatter(*traj[round(t * n_steps)].T, s=2, c="tab:green")
    axes[0, 0].set_ylabel("target path (training)")
    axes[1, 0].set_ylabel("generated (Euler, 100 steps)")
    fig.tight_layout()
    fig.savefig(out / "fig3_marginal_vs_generated.jpg", dpi=110)
    plt.close(fig)


@torch.no_grad()
def plot_steps(model, data: str, out: Path):
    """図4：Eulerのステップ数を変えたときの生成結果(同じ初期ノイズを使う)。"""
    steps = [1, 2, 4, 8, 32, 100]
    x0 = torch.randn(2000, 2)
    fig, axes = plt.subplots(1, len(steps) + 1, figsize=(3.0 * (len(steps) + 1), 3.4))
    setup_ax(axes[0], "real data")
    axes[0].scatter(*sample_data(data, 2000).T, s=2, c="tab:red")
    for ax, n in zip(axes[1:], steps):
        setup_ax(ax, f"Euler {n} step(s)")
        ax.scatter(*euler_sample(model, x0, n)[-1].T, s=2, c="tab:green")
    fig.tight_layout()
    fig.savefig(out / "fig4_euler_steps.jpg", dpi=110)
    plt.close(fig)


@torch.no_grad()
def plot_trajectories(model, data: str, out: Path):
    """図5：左=学習の直線(ペアごと)、右=学習した速度場で積分した実際の軌跡(曲がる)。"""
    n = 300
    x0 = torch.randn(n, 2)
    traj = euler_sample(model, x0, 100)  # [101, n, 2]
    x1 = sample_data(data, n)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5))
    setup_ax(axes[0], "training targets: straight lines x0 -> x1 (random pairs)")
    for a, b in zip(x0, x1):
        axes[0].plot([a[0], b[0]], [a[1], b[1]], c="gray", lw=0.4, alpha=0.5)
    setup_ax(axes[1], "learned ODE trajectories (no crossing, may curve)")
    for i in range(n):
        axes[1].plot(traj[:, i, 0], traj[:, i, 1], c="gray", lw=0.4, alpha=0.5)
    for ax, end in zip(axes, [x1, traj[-1]]):
        ax.scatter(*x0.T, s=5, c="tab:blue", zorder=3, label="start (noise)")
        ax.scatter(*end.T, s=5, c="tab:red", zorder=3, label="end")
        ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "fig5_trajectories.jpg", dpi=110)
    plt.close(fig)


@torch.no_grad()
def save_animation(model, out: Path):
    """flow.gif：ノイズの点が速度場に沿ってデータへ流れる様子。"""
    n_steps = 60
    traj = euler_sample(model, torch.randn(1500, 2), n_steps)
    g = torch.linspace(-LIM, LIM, 17)
    gx, gy = torch.meshgrid(g, g, indexing="xy")
    grid = torch.stack([gx.flatten(), gy.flatten()], dim=1)

    fig, ax = plt.subplots(figsize=(5, 5))
    setup_ax(ax)
    v0 = model(grid, torch.zeros(grid.shape[0], 1))
    quiv = ax.quiver(grid[:, 0], grid[:, 1], v0[:, 0], v0[:, 1], color="lightgray",
                     angles="xy", scale_units="xy", scale=6, width=0.004)
    scat = ax.scatter(*traj[0].T, s=3, c="tab:green")

    def update(i):
        t = i / n_steps
        v = model(grid, torch.full((grid.shape[0], 1), t))
        quiv.set_UVC(v[:, 0], v[:, 1])
        scat.set_offsets(traj[i].numpy())
        ax.set_title(f"t = {t:.2f}")
        return quiv, scat

    # 最後のフレーム(生成結果)で少し止める
    frames = list(range(n_steps + 1)) + [n_steps] * 15
    FuncAnimation(fig, update, frames=frames).save(out / "flow.gif", writer=PillowWriter(fps=15))
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", choices=["moons", "8gauss"], default="moons")
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(4)  # 小さいMLPなのでCPUで十分
    out = Path(__file__).parent / "outputs" / f"stage1_{args.data}"
    out.mkdir(parents=True, exist_ok=True)

    plot_training_pairs(args.data, out)

    model = VelocityNet()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    losses = []
    t0 = time.time()
    for step in range(1, args.steps + 1):
        loss = flow_matching_loss(model, sample_data(args.data, args.batch))
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
        if step % 1000 == 0:
            print(f"step {step:5d}  loss(last 1000 avg) {np.mean(losses[-1000:]):.4f}  "
                  f"{time.time() - t0:.1f}s", flush=True)

    model.eval()
    t0 = time.time()
    plot_loss(losses, out)
    plot_marginal_vs_generated(model, args.data, out)
    plot_steps(model, args.data, out)
    plot_trajectories(model, args.data, out)
    save_animation(model, out)
    print(f"plots saved to {out}  ({time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()
