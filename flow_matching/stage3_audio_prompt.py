"""段階3：AudioMNISTのMelで、参照Melを先頭に置いて後半を生成するFlow Matching。

段階2(stage2_mnist_cfg.py)からの差分:
- データ: MNIST画像 → 60人が数字を読み上げた音声(AudioMNIST)のlog-Mel[T, 80]。起動時に全wavから計算する
- 1サンプル = 同じ話者の2発話をつないだMel。前半(参照Mel)はそのまま与え、後半だけを生成する
- 条件: 数字ラベル → 参照Mel + フレームごとの条件(どちらの区間か、数字、区間内の進み具合)
- 速度を出すNN: U-Net → フレームを並べたTransformer(長さが発話ごとに違うのでpaddingのmaskを使う)
- 損失: 後半(生成する区間)のフレームだけで計算する
- CFGの条件なし: 参照Melとフレームごとの条件を全部0にした入力
- 評価: 生成Melを数字分類器にかけた正解率と、Griffin-Limで波形に戻してWavLM話者モデルで
  「参照の話者の声になっているか」を測る。話者は学習に使っていない10人で評価する

使い方:
    # data/AudioMNIST に https://github.com/soerenab/AudioMNIST を置いておく(data/ 以下だけでよい)
    python stage3_audio_prompt.py
出力: outputs/stage3_audiomnist/ に図(jpg)、音声(audio/*.wav)、評価値(metrics.json)。
"""
import argparse
import json
import random
import re
import time
from pathlib import Path

import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from diffusers.models.embeddings import TimestepEmbedding, Timesteps
from transformers import AutoFeatureExtractor, WavLMForXVector

ROOT = Path(__file__).parent
SR = 16000
N_FFT = 1024
HOP = 256
N_MELS = 80
MAX_PROMPT = 64   # 参照Melの最大フレーム数(長い場合は先頭から切る)
MAX_TARGET = 64   # 生成する区間の最大フレーム数
MAX_LEN = MAX_PROMPT + MAX_TARGET
COND_DIM = 12     # フレームごとの条件: 数字のone-hot(10) + 区間内の進み具合(1) + 生成区間かどうか(1)
CFG_WS = [0.0, 0.7, 2.0]  # 0.7はIndexTTS2.5 s2melのデフォルト(models/s2mel/model.py の cfg)
N_STEPS = 32


# ---------------------------------------------------------------- データ

def load_wave(path: str) -> np.ndarray:
    """wav(48kHz)を読み、16kHzに変換して、ピークが0.9になるよう音量をそろえる
    (AudioMNISTの録音はピークの中央値が-34dBFSと非常に小さく、ファイルごとのばらつきも大きいため)。"""
    y, sr = sf.read(path, dtype="float32")
    y = torchaudio.functional.resample(torch.from_numpy(y), sr, SR).numpy()
    return y / np.abs(y).max() * 0.9


def wave_to_logmel(y: np.ndarray) -> torch.Tensor:
    """波形からlog-Mel[80, T]を作る。Griffin-Limで戻すときと同じlibrosaのMelフィルタを使う。"""
    mel = librosa.feature.melspectrogram(y=y, sr=SR, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS, power=1.0)
    return torch.from_numpy(np.log(np.maximum(mel, 1e-5)))


class AudioMNISTMel:
    """AudioMNISTの全wavをlog-Melにしてメモリに置き、正規化して話者ごとに引けるようにする。
    元の録音は前後の無音がほぼ切られているので、無音カットはしない。"""

    def __init__(self, data_dir: Path):
        self.paths = sorted(str(p) for p in data_dir.glob("*/*.wav"))
        names = [re.match(r"(\d)_(\d+)_(\d+)\.wav", Path(p).name) for p in self.paths]
        self.digits = [int(m.group(1)) for m in names]
        self.speakers = [int(m.group(2)) for m in names]
        logmels = [wave_to_logmel(load_wave(p)) for p in self.paths]

        # 評価用(学習に使わない話者)の10人を、全体の男女比(男48:女12)に合わせて男8人・女2人選ぶ
        meta = json.loads((data_dir / "audioMNIST_meta.txt").read_text())
        self.gender = gender = {int(k): v["gender"] for k, v in meta.items()}
        all_speakers = sorted(set(self.speakers))
        self.test_speakers = sorted([s for s in all_speakers if gender[s] == "male"][-8:]
                                    + [s for s in all_speakers if gender[s] == "female"][-2:])
        self.train_speakers = [s for s in all_speakers if s not in self.test_speakers]

        # 学習用話者のフレームでMel binごとに平均0・標準偏差1に正規化し、[T, 80]にする
        train_frames = torch.cat([m for m, s in zip(logmels, self.speakers) if s in self.train_speakers], dim=1)
        self.mean, self.std = train_frames.mean(1), train_frames.std(1)
        self.mels = [((m - self.mean[:, None]) / self.std[:, None]).T for m in logmels]
        self.by_speaker = {}
        for i, s in enumerate(self.speakers):
            self.by_speaker.setdefault(s, []).append(i)

    def to_wave(self, mel: torch.Tensor) -> np.ndarray:
        """正規化Mel[T, 80]をGriffin-Limで波形に戻す。"""
        logmel = mel.float().cpu().T * self.std[:, None] + self.mean[:, None]
        return librosa.feature.inverse.mel_to_audio(
            logmel.exp().numpy(), sr=SR, n_fft=N_FFT, hop_length=HOP, power=1.0, n_iter=32)


def build_batch(ds: AudioMNISTMel, pairs: list[tuple[int, int]], device):
    """(参照の発話, 生成する発話)の組を、つないだMelとフレームごとの条件にする。
    戻り値:
        x1: つないだMel[B, L, 80](padding部分は0)
        cond: フレームごとの条件[B, L, 12]
        prompt_lens, total_lens: 参照区間の長さと全体の長さ[B]
    """
    B = len(pairs)
    x1 = torch.zeros(B, MAX_LEN, N_MELS)
    cond = torch.zeros(B, MAX_LEN, COND_DIM)
    prompt_lens = torch.zeros(B, dtype=torch.long)
    total_lens = torch.zeros(B, dtype=torch.long)
    for b, (ip, it) in enumerate(pairs):
        mp, mt = ds.mels[ip][:MAX_PROMPT], ds.mels[it][:MAX_TARGET]
        lp, lt = len(mp), len(mt)
        x1[b, :lp], x1[b, lp:lp + lt] = mp, mt
        for start, length, digit, is_target in [(0, lp, ds.digits[ip], 0), (lp, lt, ds.digits[it], 1)]:
            cond[b, start:start + length, digit] = 1
            cond[b, start:start + length, 10] = torch.linspace(0, 1, length)
            cond[b, start:start + length, 11] = is_target
        prompt_lens[b], total_lens[b] = lp, lp + lt
    return x1.to(device), cond.to(device), prompt_lens.to(device), total_lens.to(device)


def region_masks(prompt_lens, total_lens):
    """参照区間・生成区間・paddingのmask[B, L]を返す。"""
    pos = torch.arange(MAX_LEN, device=prompt_lens.device)[None]
    prompt_mask = pos < prompt_lens[:, None]
    pad_mask = pos >= total_lens[:, None]
    target_mask = ~prompt_mask & ~pad_mask
    return prompt_mask, target_mask, pad_mask


def sample_train_pairs(ds: AudioMNISTMel, speakers: list[int], n: int, rng: random.Random):
    """同じ話者の別の発話を2つ選び、(参照, 生成対象)の組にする。"""
    pairs = []
    for _ in range(n):
        ip, it = rng.sample(ds.by_speaker[rng.choice(speakers)], 2)
        pairs.append((ip, it))
    return pairs


# ---------------------------------------------------------------- モデル

class VelocityTransformer(nn.Module):
    """フレームごとに [今の状態 x_t, 参照Mel, 条件] を並べ、Transformerで速度[B, L, 80]を出す。
    時刻tは埋め込んで全フレームに足す。"""

    def __init__(self, dim: int = 256, layers: int = 6, heads: int = 4):
        super().__init__()
        self.in_proj = nn.Linear(N_MELS * 2 + COND_DIM, dim)
        self.pos = nn.Embedding(MAX_LEN, dim)
        self.t_proj = Timesteps(dim, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.t_embed = TimestepEmbedding(dim, dim)
        layer = nn.TransformerEncoderLayer(dim, heads, dim * 4, dropout=0.0,
                                           batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.out = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, N_MELS))

    def forward(self, xt, prompt_x, cond, t, pad_mask):
        h = self.in_proj(torch.cat([xt, prompt_x, cond], dim=-1))
        h = h + self.pos.weight[None] + self.t_embed(self.t_proj(t * 1000))[:, None]
        return self.out(self.encoder(h, src_key_padding_mask=pad_mask))


def flow_matching_loss(model, x1, cond, prompt_lens, total_lens, p_uncond: float):
    """段階1・2と同じ損失。参照区間は状態を0にして別入力で渡し、損失は生成区間だけで取る。"""
    prompt_mask, target_mask, pad_mask = region_masks(prompt_lens, total_lens)
    B = x1.shape[0]
    x0 = torch.randn_like(x1)
    t = torch.rand(B, device=x1.device)
    tb = t[:, None, None]
    xt = ((1 - tb) * x0 + tb * x1) * target_mask[..., None]   # 参照区間とpaddingは0
    prompt_x = x1 * prompt_mask[..., None]
    # CFG用：一部のサンプルで参照Melとフレームごとの条件を全部0にする
    keep = (torch.rand(B, device=x1.device) >= p_uncond).float()[:, None, None]
    v = model(xt, prompt_x * keep, cond * keep, t, pad_mask)
    err = ((v - (x1 - x0)) ** 2).mean(-1)
    return (err * target_mask).sum() / target_mask.sum()


@torch.no_grad()
def generate(model, x1, cond, prompt_lens, total_lens, w: float, n_steps: int, noise=None):
    """参照区間を固定し、生成区間をEuler積分+CFGで作る。戻り値は全体のMel[B, L, 80](生成区間以外は0)。
    x1 は参照区間だけ使う(生成区間の値は見ない)。"""
    prompt_mask, target_mask, pad_mask = region_masks(prompt_lens, total_lens)
    tm = target_mask[..., None]
    prompt_x = x1 * prompt_mask[..., None]
    x = (torch.randn_like(x1) if noise is None else noise) * tm
    dt = 1.0 / n_steps
    for i in range(n_steps):
        t = torch.full((x.shape[0],), i * dt, device=x.device)
        if w == 0:
            v = model(x, prompt_x, cond, t, pad_mask)
        else:
            v_cond, v_uncond = model(torch.cat([x, x]), torch.cat([prompt_x, torch.zeros_like(prompt_x)]),
                                     torch.cat([cond, torch.zeros_like(cond)]), torch.cat([t, t]),
                                     torch.cat([pad_mask, pad_mask])).chunk(2)
            v = (1 + w) * v_cond - w * v_uncond
        x = (x + v * dt) * tm
    return x


def target_part(x: torch.Tensor, prompt_len: int, total_len: int) -> torch.Tensor:
    return x[prompt_len:total_len]


# ---------------------------------------------------------------- 評価用

class DigitClassifier(nn.Module):
    """Mel[1, 80, MAX_TARGET]から数字を当てる小さいCNN。"""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
            nn.Flatten(), nn.Linear(64, 10),
        )

    def forward(self, x):
        return self.net(x)


def pad_target(mels: list[torch.Tensor], device) -> torch.Tensor:
    """長さの違うMel[T, 80]のリストを[N, 1, 80, MAX_TARGET]にそろえる(足りない部分は0=平均値)。"""
    out = torch.zeros(len(mels), 1, N_MELS, MAX_TARGET)
    for i, m in enumerate(mels):
        m = m[:MAX_TARGET]
        out[i, 0, :, :len(m)] = m.T.cpu()
    return out.to(device)


def train_digit_classifier(ds, device) -> tuple[nn.Module, float]:
    """学習用話者の実データで3epoch学習し、評価用話者の実データでの正解率と一緒に返す。"""
    train_idx = [i for s in ds.train_speakers for i in ds.by_speaker[s]]
    test_idx = [i for s in ds.test_speakers for i in ds.by_speaker[s]]
    x_tr = pad_target([ds.mels[i] for i in train_idx], "cpu")  # 約500MBなのでCPUに置き、バッチごとに送る
    y_tr = torch.tensor([ds.digits[i] for i in train_idx])
    clf = DigitClassifier().to(device)
    opt = torch.optim.Adam(clf.parameters(), lr=1e-3)
    for _ in range(3):
        perm = torch.randperm(len(x_tr))
        for i in range(0, len(perm), 128):
            idx = perm[i:i + 128]
            loss = F.cross_entropy(clf(x_tr[idx].to(device)), y_tr[idx].to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()
    clf.eval()
    return clf, classify_acc(clf, [ds.mels[i] for i in test_idx], [ds.digits[i] for i in test_idx], device)


@torch.no_grad()
def classify_acc(clf, mels, digits, device) -> float:
    pred = torch.cat([clf(pad_target(mels[i:i + 256], device)).argmax(1) for i in range(0, len(mels), 256)])
    return (pred.cpu() == torch.tensor(digits)).float().mean().item()


class SpeakerEmbedder:
    """WavLMの話者照合モデル(microsoft/wavlm-base-plus-sv)で、波形から正規化した話者ベクトルを出す。"""

    def __init__(self, device):
        name = "microsoft/wavlm-base-plus-sv"
        self.fe = AutoFeatureExtractor.from_pretrained(name)
        self.model = WavLMForXVector.from_pretrained(name).to(device).eval()
        self.device = device

    @torch.no_grad()
    def __call__(self, waves: list[np.ndarray]) -> torch.Tensor:
        embs = []
        for i in range(0, len(waves), 32):
            inputs = self.fe(waves[i:i + 32], sampling_rate=16000, padding=True, return_tensors="pt")
            emb = self.model(**{k: v.to(self.device) for k, v in inputs.items()}).embeddings
            embs.append(F.normalize(emb, dim=-1).cpu())
        return torch.cat(embs)


def speaker_id_acc(embs: torch.Tensor, true_spk: list[int], centroids: torch.Tensor, spk_list: list[int]):
    """各話者ベクトルを、一番cosine類似度が高い話者の平均ベクトルに割り当てたときの正解率。
    あわせて、正しい話者との類似度の平均と、他の話者との類似度の平均を返す。"""
    sim = embs @ centroids.T  # [N, 話者数]
    true_col = torch.tensor([spk_list.index(s) for s in true_spk])
    acc = (sim.argmax(1) == true_col).float().mean().item()
    own = sim[torch.arange(len(embs)), true_col].mean().item()
    other_mask = torch.ones_like(sim, dtype=torch.bool)
    other_mask[torch.arange(len(embs)), true_col] = False
    return acc, own, sim[other_mask].mean().item()


# ---------------------------------------------------------------- 図

def show_mel(ax, mel: torch.Tensor, title: str = ""):
    ax.imshow(mel.float().cpu().T, origin="lower", aspect="auto", cmap="magma", vmin=-3, vmax=3)
    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])


def plot_example_input(ds, device, out: Path):
    """図1：学習の1サンプル。つないだMel、参照Melの入力、フレームごとの条件、t=0.5の状態 x_t。"""
    rng = random.Random(0)
    x1, cond, pl, tl = build_batch(ds, sample_train_pairs(ds, ds.train_speakers, 1, rng), device)
    prompt_mask, target_mask, _ = region_masks(pl, tl)
    xt = (0.5 * torch.randn_like(x1) + 0.5 * x1) * target_mask[..., None]
    panels = [(x1[0], "x1: prompt mel | target mel (padding = 0)"),
              (x1[0] * prompt_mask[0, :, None], "input 1: prompt_x (only prompt region)"),
              (xt[0], "input 2: x_t at t=0.5 (only target region; loss is computed here)")]
    fig, axes = plt.subplots(4, 1, figsize=(9, 9))
    for ax, (m, title) in zip(axes, panels):
        show_mel(ax, m, title)
        ax.axvline(pl[0].item() - 0.5, c="cyan", lw=1)
        ax.axvline(tl[0].item() - 0.5, c="white", lw=1, ls="--")
    axes[3].imshow(cond[0].cpu().T, origin="lower", aspect="auto", cmap="viridis")
    axes[3].set_yticks([0, 5, 9, 10, 11])
    axes[3].set_yticklabels(["digit 0", "digit 5", "digit 9", "progress", "is_target"], fontsize=8)
    axes[3].set_title("input 3: per-frame condition (digit one-hot, progress in segment, is_target)", fontsize=9)
    axes[3].set_xlabel("frame")
    fig.tight_layout()
    fig.savefig(out / "fig1_example_input.jpg", dpi=110)
    plt.close(fig)


def plot_loss(losses, val_steps, val_losses, out: Path):
    """図2：学習損失(移動平均)と、評価用話者での損失。"""
    fig, ax = plt.subplots(figsize=(7, 4))
    k = min(200, len(losses))
    ax.plot(np.arange(k - 1, len(losses)), np.convolve(losses, np.ones(k) / k, mode="valid"),
            label=f"train (moving avg {k})")
    ax.plot(val_steps, val_losses, "o-", label="unseen speakers (fixed noise & t)")
    ax.set_xlabel("step")
    ax.set_ylabel("MSE on target frames")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "fig2_loss.jpg", dpi=110)
    plt.close(fig)


# ---------------------------------------------------------------- main

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=30000)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--p_uncond", type=float, default=0.1)
    parser.add_argument("--n_eval_per_digit", type=int, default=2, help="評価で話者×数字ごとに生成する数")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = random.Random(args.seed)
    # 既定の20スレッドだと、短いwavごとのresampleが1ファイル16.8ms(4スレッドなら0.8ms)と遅くなる
    torch.set_num_threads(4)
    device = args.device
    out = ROOT / "outputs" / "stage3_audiomnist"
    (out / "audio").mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    ds = AudioMNISTMel(ROOT / "data" / "AudioMNIST" / "data")
    lengths = np.array([len(m) for m in ds.mels])
    print(f"data loaded: {len(ds.mels)} utterances, train speakers {len(ds.train_speakers)}, "
          f"test speakers {ds.test_speakers}, frames max {lengths.max()}  ({time.time() - t0:.1f}s)", flush=True)
    plot_example_input(ds, device, out)

    # 評価用話者での損失を測るための固定バッチ(ノイズと時刻も固定)
    val_batch = build_batch(ds, sample_train_pairs(ds, ds.test_speakers, 256, random.Random(1)), device)
    g = torch.Generator(device=device).manual_seed(1)
    val_noise = torch.randn(val_batch[0].shape, device=device, generator=g)
    val_t = torch.rand(256, device=device, generator=g)

    @torch.no_grad()
    def val_loss():
        x1, cond, pl, tl = val_batch
        prompt_mask, target_mask, pad_mask = region_masks(pl, tl)
        tb = val_t[:, None, None]
        xt = ((1 - tb) * val_noise + tb * x1) * target_mask[..., None]
        v = model(xt, x1 * prompt_mask[..., None], cond, val_t, pad_mask)
        err = ((v - (x1 - val_noise)) ** 2).mean(-1)
        return ((err * target_mask).sum() / target_mask.sum()).item()

    model = VelocityTransformer().to(device)
    print(f"model params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    losses, val_steps, val_losses = [], [], []
    t0 = time.time()
    for step in range(1, args.steps + 1):
        batch = build_batch(ds, sample_train_pairs(ds, ds.train_speakers, args.batch, rng), device)
        loss = flow_matching_loss(model, *batch, args.p_uncond)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
        if step % 1000 == 0 or step == args.steps:
            model.eval()
            val_steps.append(step)
            val_losses.append(val_loss())
            model.train()
            print(f"step {step:6d}  train {np.mean(losses[-1000:]):.4f}  unseen-speaker val {val_losses[-1]:.4f}  "
                  f"{time.time() - t0:.1f}s", flush=True)
    model.eval()
    plot_loss(losses, val_steps, val_losses, out)

    # ---------------- 評価
    t0 = time.time()
    clf, clf_acc = train_digit_classifier(ds, device)
    print(f"digit classifier acc on unseen speakers (real) {clf_acc:.3f}  ({time.time() - t0:.1f}s)", flush=True)
    t0 = time.time()
    embedder = SpeakerEmbedder(device)
    eval_rng = random.Random(2)

    def eval_set(speakers):
        """話者×数字×n_eval_per_digit 個の(参照, 生成対象)の組と、話者の平均ベクトル用の発話を選ぶ。
        生成対象の発話は、数字と長さを決めるため(と比較用の実データ)に使う。"""
        pairs, centroid_idx = [], {}
        for s in speakers:
            idx = ds.by_speaker[s][:]
            eval_rng.shuffle(idx)
            centroid_idx[s] = idx[:20]   # 話者の平均ベクトル用(参照や生成対象には使わない)
            rest = idx[20:]
            for d in range(10):
                cands = [i for i in rest if ds.digits[i] == d]
                for k in range(args.n_eval_per_digit):
                    it = cands[k]
                    ip = eval_rng.choice([i for i in rest if i != it])
                    pairs.append((ip, it))
        return pairs, centroid_idx

    def centroids_for(speakers, centroid_idx):
        """話者ごとに、実データ20発話をGriffin-Limで戻した波形の話者ベクトルを平均する
        (生成側と同じくGriffin-Limを通すことで、波形に戻す処理の劣化の影響をそろえる)。"""
        cs = []
        for s in speakers:
            e = embedder([ds.to_wave(ds.mels[i]) for i in centroid_idx[s]]).mean(0)
            cs.append(F.normalize(e, dim=0))
        return torch.stack(cs)

    metrics = {"digit_classifier_acc_real_unseen": clf_acc, "unseen": {}, "seen": {}}
    results, group_info = {}, {}
    for group, speakers in [("unseen", ds.test_speakers), ("seen", ds.train_speakers[:10])]:
        pairs, centroid_idx = eval_set(speakers)
        cents = centroids_for(speakers, centroid_idx)
        group_info[group] = (centroid_idx, cents)
        true_spk = [ds.speakers[it] for _, it in pairs]
        # 基準：生成対象の実データ(Griffin-Limで戻したもの)
        real_embs = embedder([ds.to_wave(ds.mels[it][:MAX_TARGET]) for _, it in pairs])
        acc, own, other = speaker_id_acc(real_embs, true_spk, cents, speakers)
        metrics[group]["real_griffinlim"] = {"speaker_id_acc": acc, "sim_own": own, "sim_other": other}
        print(f"[{group}] real (Griffin-Lim): speaker-ID acc {acc:.3f}  sim own {own:.3f} / other {other:.3f}",
              flush=True)
        x1, cond, pl, tl = build_batch(ds, pairs, device)
        noise = torch.randn(x1.shape, device=device, generator=torch.Generator(device=device).manual_seed(3))
        for w in (CFG_WS if group == "unseen" else [0.7]):
            gen = torch.cat([generate(model, x1[i:i + 100], cond[i:i + 100], pl[i:i + 100], tl[i:i + 100],
                                      w, N_STEPS, noise[i:i + 100]) for i in range(0, len(pairs), 100)])
            gen_mels = [target_part(gen[b], pl[b], tl[b]) for b in range(len(pairs))]
            digit_acc = classify_acc(clf, gen_mels, [ds.digits[it] for _, it in pairs], device)
            embs = embedder([ds.to_wave(m) for m in gen_mels])
            acc, own, other = speaker_id_acc(embs, true_spk, cents, speakers)
            metrics[group][f"w={w}"] = {"digit_acc": digit_acc, "speaker_id_acc": acc,
                                        "sim_own": own, "sim_other": other}
            results[(group, w)] = (pairs, gen_mels)
            print(f"[{group}] w={w}: digit acc {digit_acc:.3f}  speaker-ID acc {acc:.3f}  "
                  f"sim own {own:.3f} / other {other:.3f}  ({time.time() - t0:.1f}s)", flush=True)

    def examples(group):
        """話者ごとに1つ、話者によって違う数字の生成例を選ぶ(評価の組の並びは 話者→数字→番号)。"""
        n_spk = len(group_info[group][0])
        return [p * 10 * args.n_eval_per_digit + (p % 10) * args.n_eval_per_digit for p in range(n_spk)]

    # ---------------- 図3：評価用話者の生成例(参照 | 生成 | 実データ)
    pairs, gen_mels = results[("unseen", 0.7)]
    show = examples("unseen")[:5]
    fig, axes = plt.subplots(len(show), 3, figsize=(10, 2.0 * len(show)))
    for r, i in enumerate(show):
        ip, it = pairs[i]
        show_mel(axes[r, 0], ds.mels[ip][:MAX_PROMPT], f"prompt (speaker {ds.speakers[ip]}, digit {ds.digits[ip]})")
        show_mel(axes[r, 1], gen_mels[i], f"generated (digit {ds.digits[it]}, w=0.7)")
        show_mel(axes[r, 2], ds.mels[it][:MAX_TARGET], f"real (same speaker, digit {ds.digits[it]})")
    fig.tight_layout()
    fig.savefig(out / "fig3_unseen_examples.jpg", dpi=110)
    plt.close(fig)

    # ---------------- 図4：参照の話者だけを入れ替える(生成する数字・長さ・初期ノイズは同じ)
    # 1発話(0.6秒前後)の話者ベクトルはばらつきが大きいので、参照の話者ごとに数字0〜9の10発話を生成し、
    # その平均ベクトルで話者の平均ベクトルと比べる
    speakers = ds.test_speakers
    centroid_idx, cents = group_info["unseen"]
    digit = 3  # 図に載せるMelの数字
    target_idx = [[i for i in ds.by_speaker[speakers[0]] if ds.digits[i] == d][0] for d in range(10)]
    prompt_idx = [eval_rng.choice([i for i in ds.by_speaker[s] if i not in centroid_idx[s]]) for s in speakers]
    x1, cond, pl, tl = build_batch(ds, [(ip, it) for ip in prompt_idx for it in target_idx], device)
    noise = torch.randn(10, MAX_LEN, N_MELS, device=device,
                        generator=torch.Generator(device=device).manual_seed(4)).repeat(len(speakers), 1, 1)
    gen = generate(model, x1, cond, pl, tl, 0.7, N_STEPS, noise)
    gen_mels = [target_part(gen[b], pl[b], tl[b]) for b in range(len(gen))]  # 並びは(話者, 数字)
    embs = embedder([ds.to_wave(m) for m in gen_mels]).view(len(speakers), 10, -1).mean(1)
    sim = F.normalize(embs, dim=-1) @ cents.T
    swap_mels = gen_mels[digit::10]
    metrics["speaker_swap"] = {"speakers": speakers, "sim": sim.tolist(),
                               "diag_acc": (sim.argmax(1) == torch.arange(len(speakers))).float().mean().item()}
    fig = plt.figure(figsize=(14, 8))
    for k in range(5):
        show_mel(fig.add_subplot(2, 5, k + 1), swap_mels[k], f"prompt: speaker {speakers[k]}")
    ax = fig.add_subplot(2, 1, 2)
    im = ax.imshow(sim, cmap="viridis")
    ax.set_xticks(range(len(speakers)))
    ax.set_xticklabels(speakers, fontsize=8)
    ax.set_yticks(range(len(speakers)))
    ax.set_yticklabels(speakers, fontsize=8)
    ax.set_xlabel("speaker (centroid of real utterances)")
    ax.set_ylabel("prompt speaker of generated\n(mean of digits 0-9)")
    fig.colorbar(im, ax=ax)
    fig.suptitle(f"swap only the prompt speaker (top: digit {digit}; same length & noise, w=0.7)")
    fig.savefig(out / "fig4_speaker_swap.jpg", dpi=110)
    plt.close(fig)

    # ---------------- 試聴用の音声。ファイル名と audio/samples.csv に、参照の話者・数字などを書く
    rows = []

    def write(name, wave, group, ip, gen_digit, w, kind):
        sf.write(out / "audio" / name, wave, SR)
        rows.append([name, group, ds.speakers[ip], ds.gender[ds.speakers[ip]], ds.digits[ip], gen_digit, w, kind])

    for group, ws in [("unseen", CFG_WS), ("seen", [0.7])]:
        pairs = results[(group, ws[0])][0]
        for i in examples(group)[:10 if group == "unseen" else 5]:
            ip, it = pairs[i]
            s, d = ds.speakers[ip], ds.digits[it]
            tag = f"{group}_ref-spk{s:02d}_digit{d}"
            write(f"{tag}_0prompt-digit{ds.digits[ip]}.wav", load_wave(ds.paths[ip]), group, ip, "", "",
                  "prompt (real recording given as reference)")
            for w in ws:
                write(f"{tag}_1generated_w{w}.wav", ds.to_wave(results[(group, w)][1][i]), group, ip, d, w,
                      "generated (Griffin-Lim)")
            write(f"{tag}_2real.wav", load_wave(ds.paths[it]), group, ip, d, "",
                  "real recording of the same speaker & digit")
            write(f"{tag}_3real_griffinlim.wav", ds.to_wave(ds.mels[it][:MAX_TARGET]), group, ip, d, "",
                  "same real recording through Griffin-Lim (vocoder quality reference)")
    for k, s in enumerate(speakers):
        ip = prompt_idx[k]
        write(f"swap_ref-spk{s:02d}_0prompt-digit{ds.digits[ip]}.wav", load_wave(ds.paths[ip]), "swap", ip, "", "",
              "prompt (real recording given as reference)")
        write(f"swap_ref-spk{s:02d}_1generated_digit{digit}_w0.7.wav", ds.to_wave(swap_mels[k]), "swap", ip,
              digit, 0.7, "generated with the same noise/length; only the prompt speaker differs")
    with open(out / "audio" / "samples.csv", "w") as f:
        f.write("file,group,ref_speaker,ref_gender,ref_digit,generated_digit,w,kind\n")
        f.writelines(",".join(str(v) for v in r) + "\n" for r in rows)

    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"outputs saved to {out}  (eval {time.time() - t0:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
