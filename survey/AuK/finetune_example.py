"""
AuK Flow-Matchingファインチューニング (Trainer不使用)
================================================================

音声生成・編集タスクの (指示テキスト, 参照音声[任意], ターゲット音声) 三つ組が
pandas DataFrame として渡される想定の、PyTorch基本機能のみを使った手動訓練
ループです。公式の `src/auk/train/train.py` は使わず、独立に再実装しています。

公式実装参考: gh/AuK/src/auk/train/train.py, README.md "Fine-tuning" セクション
      (公式JSONL形式の `duration`+`messages` 構造を、1行=1サンプルの
       pandas DataFrame に単純化したもの)

============================================================
前提
============================================================
論文 (Sec Unified Pre-Training) の通り、学習対象は Transformer バックボーンと
MLLM層融合パラメータのみで、MLLM本体とVAEは凍結される。本サンプルはQwen2.5-Omni
やBigVGAN-flow VAEそのものは含まないため (サイズ・依存の都合上)、同じ
「層融合」ロジックと「VAE潜在空間」の役割を果たす小さな凍結スタンドインを
用意し、その上で本物のバックボーン+層融合を実際に学習させる。

============================================================
DataFrameの期待するカラム
============================================================
必須:
  - instruction        (str): タスク指示テキスト
  - target_audio_path  (str): ターゲット音声ファイルのパス (学習対象 x1)
オプション:
  - source_audio_path  (str or NaN): 参照/編集元音声のパス (無ければ指示のみタスク)
  - duration           (float): 目安秒数 (本サンプルでは固定長に揃えるため未使用)

============================================================
使い方
============================================================
import pandas as pd
df = pd.DataFrame({
    "instruction": ["Say hello in a cheerful voice."],
    "target_audio_path": ["audio/target1.wav"],
    "source_audio_path": [None],
})
model, fusion, text_enc, vae_enc = load_model_for_finetuning()
train(model, fusion, text_enc, vae_enc, train_df=df, output_dir="./out")
"""

from __future__ import annotations

import argparse
import math
import os
from typing import Any, Dict, List, Optional

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch.utils.data import DataLoader, Dataset

from model import Flux2EditBackbone, SemanticLayerFusion
from loss import build_flow_matching_batch, masked_flow_matching_loss, sample_condition_dropout

SAMPLE_RATE = 16000
MAX_SECONDS = 1.0
MAX_SAMPLES = int(SAMPLE_RATE * MAX_SECONDS)
VAE_DOWNSAMPLE = 320  # 16000/320=50Hz 相当 (公式VAEの50Hzラテントレートを模した比率)


# ============================================================
# 1. 凍結スタンドイン: VAEエンコーダ / MLLMテキストエンコーダ
# ============================================================
#
# 実運用では、ここに事前学習済みで凍結された BigVGAN-flow VAE エンコーダと
# Qwen2.5-Omni (MLLM) をロードする (論文: 両方とも学習中は凍結)。
# 本サンプルはそれらのチェックポイントを前提にできないため、同じ役割
# (生波形->潜在シーケンス / テキスト->複数層の隠れ状態) を果たす小さな
# ランダム初期化ネットワークをその場で作り、重みを凍結して代用する。
# 学習対象 (Flux2EditBackbone + SemanticLayerFusion) は本物の実装である。

class ToyVAEEncoder(nn.Module):
    """生波形 (B, T) -> 潜在シーケンス (B, N, D_lat) の凍結スタンドイン。"""

    def __init__(self, latent_dim: int = 16, hidden: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, hidden, kernel_size=9, stride=4, padding=4),
            nn.GELU(),
            nn.Conv1d(hidden, hidden, kernel_size=9, stride=4, padding=4),
            nn.GELU(),
            nn.Conv1d(hidden, hidden, kernel_size=9, stride=5, padding=4),
            nn.GELU(),
            nn.Conv1d(hidden, hidden, kernel_size=9, stride=4, padding=4),
            nn.GELU(),
            nn.Conv1d(hidden, latent_dim, kernel_size=3, padding=1),
        )  # 総ダウンサンプル率 4*4*5*4=320

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        # waveform: (B, T) -> (B, 1, T) -> conv -> (B, D_lat, N) -> (B, N, D_lat)
        x = self.net(waveform.unsqueeze(1))
        return x.transpose(1, 2)


class ToyTextEncoder(nn.Module):
    """トークンID列 -> 各層の隠れ状態リスト (MLLMの多層出力を模した凍結スタンドイン)。"""

    def __init__(self, vocab_size: int = 2000, dim: int = 32, num_layers: int = 4):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=dim, nhead=4, dim_feedforward=dim * 2, batch_first=True)
            for _ in range(num_layers)
        ])

    def forward(self, token_ids: torch.Tensor) -> List[torch.Tensor]:
        # token_ids: (B, N_txt) -> List[num_layers] of (B, N_txt, dim)
        x = self.embed(token_ids)
        hidden_states = []
        for layer in self.layers:
            x = layer(x)
            hidden_states.append(x)
        return hidden_states


class _WordHashTokenizer:
    """外部依存なしの固定語彙トークナイザ (単語をハッシュで固定サイズ語彙に写す)。"""

    def __init__(self, vocab_size: int = 2000, max_len: int = 16):
        self.vocab_size = vocab_size
        self.max_len = max_len

    def encode(self, text: str) -> List[int]:
        ids = [1 + (hash(w) % (self.vocab_size - 1)) for w in text.strip().split()]
        ids = ids[: self.max_len] or [0]
        return ids + [0] * (self.max_len - len(ids))


# ============================================================
# 2. データセット (実際の音声ファイルI/Oを行う)
# ============================================================

class AuKDataset(Dataset):
    def __init__(self, df: pd.DataFrame, data_root: str = ""):
        self.df = df.reset_index(drop=True)
        self.data_root = data_root

    def __len__(self) -> int:
        return len(self.df)

    def _load_wav(self, path: Optional[str]) -> Optional[torch.Tensor]:
        if path is None or (isinstance(path, float) and math.isnan(path)):
            return None
        wav, sr = torchaudio.load(os.path.join(self.data_root, path))
        wav = wav.mean(dim=0)  # モノラル化: (T,)
        if sr != SAMPLE_RATE:
            wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)
        if wav.shape[0] < MAX_SAMPLES:
            wav = F.pad(wav, (0, MAX_SAMPLES - wav.shape[0]))
        else:
            wav = wav[:MAX_SAMPLES]
        return wav

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        return {
            "instruction": str(row["instruction"]),
            "target_wav": self._load_wav(row["target_audio_path"]),
            "source_wav": self._load_wav(row.get("source_audio_path")),
        }


def collate_batch(features: List[Dict[str, Any]], tokenizer: _WordHashTokenizer) -> Dict[str, Any]:
    token_ids = torch.tensor([tokenizer.encode(f["instruction"]) for f in features], dtype=torch.long)
    target_wavs = torch.stack([f["target_wav"] for f in features])  # (B, T)
    has_source = [f["source_wav"] is not None for f in features]
    source_wavs = torch.stack([f["source_wav"] if s else torch.zeros_like(target_wavs[0])
                                for f, s in zip(features, has_source)])
    return {
        "token_ids": token_ids, "target_wavs": target_wavs,
        "source_wavs": source_wavs, "has_source": torch.tensor(has_source, dtype=torch.bool),
    }


# ============================================================
# 3. モデルロード
# ============================================================

def load_model_for_finetuning(
    dim: int = 64, depth: int = 2, num_single_layers: int = 4,
    heads: int = 4, dim_head: int = 16, latent_dim: int = 16, text_dim: int = 32,
):
    """本物の学習対象 (Flux2EditBackbone + SemanticLayerFusion) と、
    凍結スタンドイン (VAEエンコーダ, テキストエンコーダ) をロードする。"""
    backbone = Flux2EditBackbone(
        dim=dim, depth=depth, num_single_layers=num_single_layers, heads=heads,
        dim_head=dim_head, latent_dim=latent_dim, text_hidden_dim=text_dim,
    )
    fusion = SemanticLayerFusion(num_layers=4)

    text_encoder = ToyTextEncoder(dim=text_dim, num_layers=4)
    vae_encoder = ToyVAEEncoder(latent_dim=latent_dim)
    for p in text_encoder.parameters():
        p.requires_grad = False
    for p in vae_encoder.parameters():
        p.requires_grad = False
    text_encoder.eval()
    vae_encoder.eval()

    return backbone, fusion, text_encoder, vae_encoder


# ============================================================
# 4. メイン訓練ループ
# ============================================================

def train(
    backbone: nn.Module, fusion: nn.Module, text_encoder: nn.Module, vae_encoder: nn.Module,
    train_df: pd.DataFrame, output_dir: str = "./auk-finetuned",
    epochs: int = 3, batch_size: int = 4, lr: float = 1e-4, weight_decay: float = 0.0,
    log_steps: int = 2, max_grad_norm: float = 1.0, num_workers: int = 0,
):
    """手動訓練ループ (Trainer不使用)。

    ========================================
    各バッチのShape
    ========================================
    token_ids:    (B, N_txt)
    target_wavs:  (B, T)
    source_wavs:  (B, T)
    z1 (VAE出力): (B, N_lat, D_lat)
    v_pred:       (B, N_lat, D_lat)
    loss:         scalar
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone, fusion = backbone.to(device), fusion.to(device)
    text_encoder, vae_encoder = text_encoder.to(device), vae_encoder.to(device)

    tokenizer = _WordHashTokenizer()
    dataset = AuKDataset(train_df)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
        collate_fn=lambda f: collate_batch(f, tokenizer),
    )

    trainable_params = list(backbone.parameters()) + list(fusion.parameters())
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay, betas=(0.9, 0.95))

    os.makedirs(output_dir, exist_ok=True)
    print("訓練設定:")
    print(f"  サンプル数: {len(dataset)}  バッチサイズ: {batch_size}")
    print(f"  学習率: {lr}  訓練可能パラメータ (backbone+fusion): "
          f"{sum(p.numel() for p in trainable_params):,}")

    global_step = 0
    for epoch in range(epochs):
        print(f"\n=== Epoch {epoch + 1}/{epochs} ===")
        backbone.train()
        for batch in loader:
            token_ids = batch["token_ids"].to(device)
            target_wavs = batch["target_wavs"].to(device)
            source_wavs = batch["source_wavs"].to(device)
            has_source = batch["has_source"].to(device)

            with torch.no_grad():
                hidden_states = text_encoder(token_ids)          # List[(B,N_txt,text_dim)]
                z1 = vae_encoder(target_wavs)                      # (B, N_lat, D_lat): クリーン潜在
                ref_latent = vae_encoder(source_wavs)               # (B, N_lat, D_lat)

            c_sem = fusion(hidden_states)  # (B, N_txt, text_dim): 本物の学習対象(層融合)

            fm_batch = build_flow_matching_batch(z1)  # z_t, target_velocity, time

            drop_audio_cond, drop_text = sample_condition_dropout()
            ref_for_batch = ref_latent if (has_source.any() and not drop_audio_cond) else ref_latent[:, :0]
            c_for_batch = torch.zeros_like(c_sem) if drop_text else c_sem

            v_pred = backbone(fm_batch.z_t, c_for_batch, fm_batch.time, ref=ref_for_batch)
            loss = masked_flow_matching_loss(v_pred, fm_batch.target_velocity)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_grad_norm)
            optimizer.step()

            global_step += 1
            if global_step % log_steps == 0:
                print(f"  Step {global_step} | loss={loss.item():.4f} | "
                      f"drop_audio={drop_audio_cond} drop_text={drop_text}")

    print("\nTraining complete! (このデモの重みは保存せず、ログのみ表示)")


# ============================================================
# 5. 合成音声での end-to-end 動作確認
# ============================================================

def _make_synthetic_audio_dataset(tmp_dir: str, n_samples: int = 8) -> pd.DataFrame:
    """実際にディスクへ短いサイン波WAVを書き出し、本物のファイルI/Oで検証する。"""
    import numpy as np

    os.makedirs(tmp_dir, exist_ok=True)
    rows = []
    instructions = [
        "Say hello in a cheerful voice.",
        "Read the sentence slowly and calmly.",
        "Remove the background noise from this recording.",
        "Raise the pitch by two semitones.",
    ]
    for i in range(n_samples):
        freq = 220.0 + 40.0 * i
        t = np.linspace(0, MAX_SECONDS, MAX_SAMPLES, endpoint=False)
        target_wav = 0.3 * np.sin(2 * np.pi * freq * t).astype(np.float32)
        target_path = os.path.join(tmp_dir, f"target_{i}.wav")
        torchaudio.save(target_path, torch.from_numpy(target_wav).unsqueeze(0), SAMPLE_RATE)

        has_source = i % 2 == 0
        source_path = None
        if has_source:
            source_wav = 0.3 * np.sin(2 * np.pi * (freq * 0.9) * t).astype(np.float32)
            source_path = os.path.join(tmp_dir, f"source_{i}.wav")
            torchaudio.save(source_path, torch.from_numpy(source_wav).unsqueeze(0), SAMPLE_RATE)

        rows.append({
            "instruction": instructions[i % len(instructions)],
            "target_audio_path": target_path,
            "source_audio_path": source_path,
            "duration": MAX_SECONDS,
        })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser("AuK Flow-Matching Fine-tuning (No Trainer)")
    parser.add_argument("--train_file", type=str, default="",
                        help="CSV/JSON with columns: instruction, target_audio_path, source_audio_path")
    parser.add_argument("--output_dir", type=str, default="./auk-finetuned")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    if args.train_file:
        train_df = pd.read_csv(args.train_file) if args.train_file.endswith(".csv") else pd.read_json(args.train_file)
    else:
        print("--train_file 未指定のため、合成音声データセットで動作確認します。\n")
        train_df = _make_synthetic_audio_dataset("/tmp/auk_finetune_demo")

    backbone, fusion, text_encoder, vae_encoder = load_model_for_finetuning()
    train(backbone, fusion, text_encoder, vae_encoder, train_df,
          output_dir=args.output_dir, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)


if __name__ == "__main__":
    main()
