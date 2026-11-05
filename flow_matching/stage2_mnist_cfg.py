"""段階2：MNISTで条件付きFlow MatchingとCFG(classifier-free guidance)を学習・可視化する。

段階1(stage1_2d_toy.py)からの差分:
- データ: 2次元の点 → MNIST画像[1,28,28](画素を[-1,1]に変換)
- 速度を出すNN: MLP → diffusers の UNet2DModel。数字ラベルをクラス埋め込みとして入れる
- 学習: 確率 p_uncond でラベルを「条件なし」用のラベル(10)に置き換える(CFG用)
- 生成: v = (1 + w) * v(条件あり) - w * v(条件なし) でEuler積分する(w = CFGの強さ)
- 評価: 別に学習した数字分類器で、生成画像が指定した数字に見えるか(正解率)と、
  同じ数字の生成画像どうしがどれだけ違うか(多様性)を測る

使い方:
    python stage2_mnist_cfg.py
出力: outputs/stage2_mnist/ に図(jpg)、アニメーション(gif)、評価値(metrics.json)。
"""
import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import UNet2DModel
from matplotlib.animation import FuncAnimation, PillowWriter
from torchvision.datasets import MNIST
from torchvision.utils import make_grid

NULL_LABEL = 10  # 「条件なし」を表すラベル。数字0〜9の次の番号を割り当てる
CFG_WS = [-1.0, 0.0, 0.5, 1.0, 2.0, 4.0]  # 評価するCFGの強さ。-1は条件なしの速度だけを使う
STEP_COUNTS = [1, 2, 4, 8, 16, 32, 100]   # 評価するEulerのステップ数
DEFAULT_W = 1.0     # ステップ数を比べるときのCFGの強さ
DEFAULT_STEPS = 32  # CFGの強さを比べるときのステップ数
CHUNK = 125         # 評価時に一度に生成する枚数(CFGで2倍になってもNNへ通すのは250枚。VRAMの使用量を抑えるため)


def load_mnist(root: Path, train: bool, device) -> tuple[torch.Tensor, torch.Tensor]:
    """MNISTを丸ごとGPUに載せる。戻り値は画像[N,1,28,28](値域[-1,1])とラベル[N]。"""
    ds = MNIST(root, train=train, download=True)
    x = ds.data.float().div(255).mul(2).sub(1).unsqueeze(1)
    return x.to(device), ds.targets.to(device)


def build_model() -> UNet2DModel:
    """28x28 → 14x14 → 7x7 と解像度を下げる小さいU-Net。クラス埋め込みは時刻埋め込みに足される。"""
    return UNet2DModel(
        sample_size=28, in_channels=1, out_channels=1,
        block_out_channels=(64, 128, 128), layers_per_block=2,
        down_block_types=("DownBlock2D", "AttnDownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "AttnUpBlock2D", "UpBlock2D"),
        num_class_embeds=11,  # 数字0〜9 + 条件なし
    )


def velocity(model, x, t, labels):
    """速度 v(x, t, ラベル)。UNet2DModelの時刻埋め込みは0〜1000程度の値を想定しているのでtを1000倍して渡す。"""
    return model(x, t * 1000, class_labels=labels).sample


def flow_matching_loss(model, x1, labels, p_uncond: float):
    """段階1と同じ損失。違いはラベルを渡すことと、一部のラベルを「条件なし」に置き換えること。"""
    labels = torch.where(torch.rand_like(labels, dtype=torch.float) < p_uncond,
                         torch.full_like(labels, NULL_LABEL), labels)
    x0 = torch.randn_like(x1)
    t = torch.rand(x1.shape[0], device=x1.device)
    tb = t.view(-1, 1, 1, 1)
    xt = (1 - tb) * x0 + tb * x1
    return F.mse_loss(velocity(model, xt, t, labels), x1 - x0)


@torch.no_grad()
def cfg_velocity(model, x, t, labels, w: float):
    """CFG付きの速度 (1 + w) * v(条件あり) - w * v(条件なし)。条件ありとなしを1つのバッチにまとめて計算する。"""
    if w == 0:
        return velocity(model, x, t, labels)
    null = torch.full_like(labels, NULL_LABEL)
    if w == -1:
        return velocity(model, x, t, null)
    v_cond, v_uncond = velocity(model, torch.cat([x, x]), torch.cat([t, t]),
                                torch.cat([labels, null])).chunk(2)
    return (1 + w) * v_cond - w * v_uncond


@torch.no_grad()
def euler_sample(model, x0, labels, w: float, n_steps: int, return_traj: bool = False):
    """t=0→1 のEuler積分。return_traj=Trueなら各ステップの状態xと、
    その時点の速度から推定した最終画像 x + (1 - t) * v も返す。"""
    x = x0
    dt = 1.0 / n_steps
    states, preds = [x0], []
    for i in range(n_steps):
        t = torch.full((x.shape[0],), i * dt, device=x.device)
        v = cfg_velocity(model, x, t, labels, w)
        if return_traj:
            preds.append(x + (1 - i * dt) * v)
        x = x + v * dt
        if return_traj:
            states.append(x)
    if return_traj:
        preds.append(x)
        return torch.stack(states), torch.stack(preds)
    return x


class DigitClassifier(nn.Module):
    """生成画像の評価用の小さいCNN分類器。"""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(), nn.Linear(64 * 7 * 7, 128), nn.ReLU(), nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.net(x)


def train_classifier(x_train, y_train, x_test, y_test) -> tuple[nn.Module, float]:
    """分類器を2epoch学習し、テスト正解率と一緒に返す。"""
    clf = DigitClassifier().to(x_train.device)
    opt = torch.optim.Adam(clf.parameters(), lr=1e-3)
    for _ in range(2):
        perm = torch.randperm(len(x_train), device=x_train.device)
        for i in range(0, len(perm), 256):
            idx = perm[i:i + 256]
            loss = F.cross_entropy(clf(x_train[idx]), y_train[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
    clf.eval()
    return clf, classify_acc(clf, x_test, y_test)


@torch.no_grad()
def classify_acc(clf, x, labels) -> float:
    pred = torch.cat([clf(xc).argmax(1) for xc in x.split(CHUNK)])
    return (pred == labels).float().mean().item()


def diversity(x: torch.Tensor, labels: torch.Tensor) -> float:
    """同じ数字の画像どうしの平均L2距離を、数字ごとに計算して平均する。"""
    ds = []
    for c in range(10):
        xc = x[labels == c].flatten(1)
        d = torch.cdist(xc, xc)
        n = len(xc)
        ds.append((d.sum() / (n * (n - 1))).item())
    return float(np.mean(ds))


def to_img(x: torch.Tensor, nrow: int) -> np.ndarray:
    """画像[N,1,28,28]を nrow 列のグリッド画像[H,W]にする。"""
    return make_grid(x.clamp(-1, 1).add(1).div(2).cpu(), nrow=nrow, padding=2, pad_value=1)[0].numpy()


def plot_training_path(x1: torch.Tensor, out: Path):
    """図1：学習で使う x_t = (1-t) x0 + t x1 と、正解の速度 x1 - x0。"""
    x1 = x1[:4]
    x0 = torch.randn_like(x1)
    ts = [0.0, 0.25, 0.5, 0.75, 1.0]
    cols = [(1 - t) * x0 + t * x1 for t in ts] + [(x1 - x0) / 2]  # 速度は値域が広いので半分にして表示
    grid = torch.stack(cols, dim=1).flatten(0, 1)
    fig, ax = plt.subplots(figsize=(8, 5.6))
    ax.imshow(to_img(grid, len(cols)), cmap="gray")
    ax.set_xticks([15 + 30 * i for i in range(len(cols))])
    ax.set_xticklabels([f"x_t, t={t}" for t in ts] + ["target v\n= x1 - x0"], fontsize=9)
    ax.set_yticks([])
    ax.set_title("training input x_t = (1-t) x0 + t x1  and target velocity")
    fig.tight_layout()
    fig.savefig(out / "fig1_training_path.jpg", dpi=110)
    plt.close(fig)


def plot_loss(losses: list[float], steps_per_epoch: int, out: Path):
    """図2：損失の推移(1epochごとの平均も重ねる)。"""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(losses, lw=0.5, alpha=0.4, label="per step")
    n_ep = len(losses) // steps_per_epoch
    ep_mean = np.array(losses[:n_ep * steps_per_epoch]).reshape(n_ep, -1).mean(1)
    ax.plot((np.arange(n_ep) + 0.5) * steps_per_epoch, ep_mean, "o-", label="epoch mean")
    ax.set_xlabel("step")
    ax.set_ylabel("MSE(v_pred, x1 - x0)")
    ax.set_ylim(0, max(ep_mean) * 1.5)
    ax.set_title("training loss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "fig2_loss.jpg", dpi=110)
    plt.close(fig)


def plot_grids(grids: list[np.ndarray], titles: list[str], suptitle: str, path: Path):
    """生成画像グリッドを横に並べる(行=数字0〜9)。"""
    fig, axes = plt.subplots(1, len(grids), figsize=(2.6 * len(grids), 4.6))
    for ax, g, title in zip(axes, grids, titles):
        ax.imshow(g, cmap="gray")
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    fig.suptitle(suptitle)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_generation_process(model, device, out: Path):
    """図5：生成の途中。上=その時点の状態 x_t、下=その時点の速度から推定した最終画像 x_t + (1-t) v。"""
    n_steps = 100
    show_ts = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    labels = torch.arange(10, device=device)
    states, preds = euler_sample(model, torch.randn(10, 1, 28, 28, device=device), labels,
                                 DEFAULT_W, n_steps, return_traj=True)
    fig, axes = plt.subplots(2, 1, figsize=(9, 17))
    for ax, traj, title in zip(axes, [states, preds],
                               ["state x_t during Euler integration",
                                "predicted final image x_t + (1-t) v(x_t, t)"]):
        grid = traj[show_ts].transpose(0, 1).flatten(0, 1)  # 行=数字、列=時刻
        ax.imshow(to_img(grid, len(show_ts)), cmap="gray")
        ax.set_xticks([15 + 30 * i for i in range(len(show_ts))])
        ax.set_xticklabels([f"{s / n_steps:.1f}" for s in show_ts])
        ax.set_xlabel("t")
        ax.set_yticks([])
        ax.set_title(title)
    fig.suptitle(f"generation process (w={DEFAULT_W}, {n_steps} Euler steps)")
    fig.tight_layout()
    fig.savefig(out / "fig5_generation_process.jpg", dpi=110)
    plt.close(fig)


def save_animation(model, device, out: Path):
    """flow.gif：数字0〜9(各4枚)がノイズから形になっていく様子。"""
    n_steps = 40
    labels = torch.arange(10, device=device).repeat_interleave(4)
    states, _ = euler_sample(model, torch.randn(40, 1, 28, 28, device=device), labels,
                             DEFAULT_W, n_steps, return_traj=True)
    frames = [to_img(s, 10) for s in states.view(n_steps + 1, 10, 4, 1, 28, 28)
              .transpose(1, 2).flatten(1, 2)]  # 列=数字、行=サンプル
    fig, ax = plt.subplots(figsize=(5, 2.3))
    ax.axis("off")
    im = ax.imshow(frames[0], cmap="gray", vmin=0, vmax=1)

    def update(i):
        im.set_data(frames[i])
        ax.set_title(f"t = {i / n_steps:.2f}", fontsize=9)
        return (im,)

    FuncAnimation(fig, update, frames=list(range(n_steps + 1)) + [n_steps] * 15).save(
        out / "flow.gif", writer=PillowWriter(fps=12))
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--p_uncond", type=float, default=0.1, help="学習時にラベルを条件なしにする確率")
    parser.add_argument("--n_eval_per_class", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda"
    root = Path(__file__).parent
    out = root / "outputs" / "stage2_mnist"
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    x_train, y_train = load_mnist(root / "data", True, device)
    x_test, y_test = load_mnist(root / "data", False, device)
    print(f"data loaded: train {tuple(x_train.shape)}  ({time.time() - t0:.1f}s)", flush=True)
    plot_training_path(x_train, out)

    model = build_model().to(device)
    print(f"UNet params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    steps_per_epoch = len(x_train) // args.batch
    losses = []
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        perm = torch.randperm(len(x_train), device=device)
        for i in range(steps_per_epoch):
            idx = perm[i * args.batch:(i + 1) * args.batch]
            loss = flow_matching_loss(model, x_train[idx], y_train[idx], args.p_uncond)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        print(f"epoch {epoch:2d}  loss {np.mean(losses[-steps_per_epoch:]):.4f}  "
              f"{time.time() - t0:.1f}s", flush=True)
    model.eval()
    plot_loss(losses, steps_per_epoch, out)

    t0 = time.time()
    clf, clf_acc = train_classifier(x_train, y_train, x_test, y_test)
    print(f"classifier test acc {clf_acc:.4f}  ({time.time() - t0:.1f}s)", flush=True)

    # 評価用の初期ノイズとラベル。全ての条件で同じものを使う
    t0 = time.time()
    labels = torch.arange(10, device=device).repeat_interleave(args.n_eval_per_class)
    noise = torch.randn(len(labels), 1, 28, 28, device=device)
    show = (torch.arange(10, device=device) * args.n_eval_per_class)[:, None] + torch.arange(6, device=device)
    show = show.flatten()  # 図に載せる分：各数字の最初の6枚

    def evaluate(w, n_steps):
        x = torch.cat([euler_sample(model, nc, lc, w, n_steps)
                       for nc, lc in zip(noise.split(CHUNK), labels.split(CHUNK))])
        return x, {"w": w, "steps": n_steps, "acc": classify_acc(clf, x, labels),
                   "diversity": diversity(x, labels)}

    real_idx = torch.cat([torch.nonzero(y_test == c)[:args.n_eval_per_class, 0] for c in range(10)])
    metrics = {"classifier_test_acc": clf_acc,
               "real_test_diversity": diversity(x_test[real_idx], y_test[real_idx]),
               "cfg": [], "steps": []}
    grids = []
    for w in CFG_WS:
        x, m = evaluate(w, DEFAULT_STEPS)
        metrics["cfg"].append(m)
        grids.append(to_img(x[show], 6))
        print(f"w={w:4.1f} steps={DEFAULT_STEPS}: acc {m['acc']:.3f}  diversity {m['diversity']:.2f}", flush=True)
    plot_grids(grids, [f"w={w}" + (" (uncond)" if w == -1 else " (no CFG)" if w == 0 else "")
                       for w in CFG_WS],
               f"CFG strength w (rows = digit 0-9, {DEFAULT_STEPS} Euler steps, same noise)",
               out / "fig3_cfg.jpg")
    grids = []
    for n in STEP_COUNTS:
        x, m = evaluate(DEFAULT_W, n)
        metrics["steps"].append(m)
        grids.append(to_img(x[show], 6))
        print(f"w={DEFAULT_W:4.1f} steps={n:3d}: acc {m['acc']:.3f}  diversity {m['diversity']:.2f}", flush=True)
    plot_grids(grids, [f"{n} step(s)" for n in STEP_COUNTS],
               f"number of Euler steps (rows = digit 0-9, w={DEFAULT_W}, same noise)",
               out / "fig4_euler_steps.jpg")
    print(f"real test diversity {metrics['real_test_diversity']:.2f}  eval {time.time() - t0:.1f}s", flush=True)

    plot_generation_process(model, device, out)
    save_animation(model, device, out)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"outputs saved to {out}", flush=True)


if __name__ == "__main__":
    main()
