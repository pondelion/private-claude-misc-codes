"""
CLM 強化学習ファインチューニング (Trainer不使用)
================================================================

ロールアウト済みのエージェント軌跡のメタデータを pandas DataFrame として受け取り、
dual-channel advantage (success-gated efficiency advantage) を使った stepwise GRPO
でポリシーモデルをファインチューニングするサンプルコードです。
transformers の Trainer は使用せず、手動の訓練ループで実装しています
(survey/Qwen3ASR/finetuning_example.py と同じ方針)。

公式実装参考: clm/clm_rl/README.md + patches/ (学習コード自体は非公開;
本ファイルはパッチに書かれたアルゴリズムの独立したPyTorch再実装)

============================================================
前提条件
============================================================
- 実際のRLパイプラインでは、ロールアウトサーバー (SGLang/vLLM) が軌跡を生成し、
  各トークンの旧ポリシー対数確率 (old_logprobs) も一緒に記録する。
  本サンプルでは「旧ポリシー」も「参照(KL)モデル」も、学習開始時点でのポリシーの
  重みを凍結コピーしたものとして扱う (1ステップ分のオンポリシー更新のデモ)。

============================================================
DataFrameの期待するカラム (1行 = 1ロールアウト軌跡)
============================================================
必須:
  - group_id      (str): 同じプロンプトからサンプリングされたGRPOグループのID
  - trajectory_id (str): 軌跡の一意なID
  - success       (bool): タスクに成功したか
  - raw_reward    (float): グループ正規化前の生の報酬 (例: タスクグレーダーの出力)
  - flops         (float): 軌跡の prefix-reuse FLOPs (例: prefix_reuse_flops.py の出力)
  - token_ids     (list[int]): 応答トークンID列 (プロンプト部分を含まない)
  - turn_lengths  (list[int]): token_ids を「ターン」単位に分割した各ターンの長さ
  - turn_is_edit  (list[bool]): 各ターンが「純粋なコンテキスト編集ターン」かどうか
                                 (len(turn_lengths) == len(turn_is_edit))

============================================================
使い方
============================================================
import pandas as pd
df = pd.DataFrame([...])  # 上記カラムを持つ行のリスト
model, vocab = load_model_for_finetuning(vocab_size=2000)
train(model, df, output_dir="./clm-rl-finetuned")
"""

from __future__ import annotations

import argparse
import math
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from rl_loss import (
    TrajectorySample,
    build_role_mask,
    dual_channel_returns,
    grpo_clipped_loss,
    success_gated_efficiency_advantage,
)

IGNORE_INDEX = -100


# ============================================================
# 1. グループ正規化 (GRPO の r_i = (raw_reward - group_mean) / group_std)
# ============================================================

def group_normalize_rewards(df: pd.DataFrame, eps: float = 1e-6) -> pd.Series:
    """``group_id`` 内で raw_reward を正規化した r_i を返す (標準的なGRPO正規化)。"""
    def _norm(g: pd.Series) -> pd.Series:
        mu, sigma = g.mean(), g.std(ddof=0)
        return (g - mu) / (sigma + eps)

    return df.groupby("group_id")["raw_reward"].transform(_norm)


# ============================================================
# 2. Dataset: 1行 = 1軌跡
# ============================================================

class TrajectoryDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        self.df = df.reset_index(drop=True).copy()
        self.df["norm_reward"] = group_normalize_rewards(self.df)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        return {
            "group_id": row["group_id"],
            "trajectory_id": row["trajectory_id"],
            "success": bool(row["success"]),
            "flops": float(row["flops"]),
            "norm_reward": float(row["norm_reward"]),
            "token_ids": list(row["token_ids"]),
            "turn_lengths": list(row["turn_lengths"]),
            "turn_is_edit": list(row["turn_is_edit"]),
        }


# ============================================================
# 3. Collator: グループ単位で dual-channel advantage を計算してバッチ化
# ============================================================

class GRPOCollator:
    """1バッチ = 1つ以上の完全なGRPOグループ。

    ========================================
    処理フロー
    ========================================
    1. バッチ内をグループごとに束ねる
    2. グループごとに success_gated_efficiency_advantage で a_eff を計算 (Eq.5)
    3. 各軌跡の role_mask を turn_lengths/turn_is_edit から構築
    4. dual_channel_returns で advantage (B, T) を構築 (Eq.6, task+efficiency)
    5. token_ids を右パディングしてバッチ化、loss_mask (パディング=0) も作る

    ========================================
    出力 Shape (バッチサイズ B = 全グループの軌跡数合計)
    ========================================
    input_ids:   (B, T_max) int64 - 応答トークン列 (0でパディング)
    loss_mask:   (B, T_max) float - 1=実トークン, 0=パディング
    advantages:  (B, T_max) float - dual-channel advantage
    """

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        groups: Dict[str, List[Dict[str, Any]]] = {}
        for f in features:
            groups.setdefault(f["group_id"], []).append(f)

        all_token_ids: List[torch.Tensor] = []
        all_advantages: List[torch.Tensor] = []
        all_meta: List[Dict[str, Any]] = []

        for gid, members in groups.items():
            success = [m["success"] for m in members]
            flops = [m["flops"] for m in members]
            a_eff = success_gated_efficiency_advantage(success, flops)

            samples = []
            for m in members:
                role_mask = build_role_mask(m["turn_lengths"], m["turn_is_edit"])
                samples.append(
                    TrajectorySample(
                        task_reward=m["norm_reward"], role_mask=role_mask,
                        response_length=len(m["token_ids"]),
                    )
                )
            returns = dual_channel_returns(samples, a_eff, w_eff=0.25)

            for m, ret in zip(members, returns):
                all_token_ids.append(torch.tensor(m["token_ids"], dtype=torch.long))
                all_advantages.append(ret)
                all_meta.append({"group_id": gid, "trajectory_id": m["trajectory_id"],
                                  "success": m["success"], "flops": m["flops"]})

        T_max = max(t.shape[0] for t in all_token_ids)
        B = len(all_token_ids)
        input_ids = torch.zeros(B, T_max, dtype=torch.long)
        loss_mask = torch.zeros(B, T_max, dtype=torch.float32)
        advantages = torch.zeros(B, T_max, dtype=torch.float32)
        for i, (ids, adv) in enumerate(zip(all_token_ids, all_advantages)):
            T_i = ids.shape[0]
            input_ids[i, :T_i] = ids
            loss_mask[i, :T_i] = 1.0
            advantages[i, :T_i] = adv

        return {
            "input_ids": input_ids, "loss_mask": loss_mask, "advantages": advantages,
            "meta": all_meta,
        }


# ============================================================
# 4. ポリシーモデル (デモ用の極小causal LM; 実運用ではQwen3.5-9B等)
# ============================================================

class _ToyPolicyLM(nn.Module):
    """shapeの契約だけ本物のcausal LMに合わせた極小モデル (デモ専用)。

    実運用では ``transformers.AutoModelForCausalLM`` (Qwen3.5-9B等, 論文の
    RL設定) に差し替える。ロールアウトは SGLang/vLLM が生成し、ここでの
    forward はログ確率の再計算 (学習ステップ) のみを担う。
    """

    def __init__(self, vocab_size: int, hidden_size: int = 32, num_layers: int = 2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList(
            nn.TransformerEncoderLayer(
                d_model=hidden_size, nhead=4, dim_feedforward=hidden_size * 2, batch_first=True,
            )
            for _ in range(num_layers)
        )
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, input_ids: torch.LongTensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        ========================================
        Shape
        ========================================
        入力: input_ids (B, T), attention_mask (B, T)
        出力: logits (B, T, V)
        """
        x = self.embed(input_ids)
        T = x.shape[1]
        causal_mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1)
        key_padding_mask = attention_mask == 0
        for layer in self.layers:
            x = layer(x, src_mask=causal_mask, src_key_padding_mask=key_padding_mask)
        return self.lm_head(x)


def token_logprobs(logits: torch.Tensor, input_ids: torch.LongTensor) -> torch.Tensor:
    """各ターゲットトークンの対数確率 (次トークン予測, shift済み)。

    ========================================
    Shape
    ========================================
    入力: logits (B, T, V), input_ids (B, T)
    出力: logprobs (B, T) - 位置 t は「input_ids[:,t] をその直前の文脈から
          予測する対数確率」。先頭位置は常に0で埋める (予測対象が無いため)。
    """
    B, T, V = logits.shape
    logprobs = torch.zeros(B, T, dtype=logits.dtype, device=logits.device)
    if T < 2:
        return logprobs
    log_probs_all = F.log_softmax(logits[:, :-1, :], dim=-1)  # (B, T-1, V)
    targets = input_ids[:, 1:]  # (B, T-1)
    gathered = log_probs_all.gather(-1, targets.unsqueeze(-1)).squeeze(-1)  # (B, T-1)
    logprobs[:, 1:] = gathered
    return logprobs


def load_model_for_finetuning(vocab_size: int = 2000, hidden_size: int = 32):
    """ポリシーモデルをロードする (デモ用の極小LM; 実運用では
    ``transformers.AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.5-9B")`` 等)。"""
    model = _ToyPolicyLM(vocab_size=vocab_size, hidden_size=hidden_size)
    return model


# ============================================================
# 5. 学習率スケジューラ (論文 clm_rl/README "Training configuration")
# ============================================================

def constant_lr(optimizer: torch.optim.Optimizer):
    """論文のRL設定は定数学習率 (lr=1e-6 constant)。スケジューラ不要だが
    インターフェースを統一するためno-opを返す。"""
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0)


# ============================================================
# 6. メイン訓練ループ
# ============================================================

def train(
    model: nn.Module,
    df: pd.DataFrame,
    output_dir: str = "./clm-rl-finetuned",
    epochs: int = 1,
    batch_groups: int = 2,       # 1バッチに束ねるGRPOグループ数
    lr: float = 1e-6,             # 論文: Adam lr 1e-6 constant
    weight_decay: float = 0.1,    # 論文: weight decay 0.1
    clip_low: float = 0.2,        # 論文: clipping eps 0.2/0.28
    clip_high: float = 0.28,
    kl_coef: float = 0.01,        # 論文: KL 0.01 (low-variance KL)
    max_grad_norm: float = 1.0,
    log_steps: int = 1,
):
    """手動訓練ループ (Trainer不使用)。

    ========================================
    各バッチのShape
    ========================================
    input_ids:   (B, T_max)
    loss_mask:   (B, T_max)
    advantages:  (B, T_max)  (dual-channel advantage, Eq.6)
    logits (内部): (B, T_max, V)
    loss (出力):  scalar

    ========================================
    「旧ポリシー」「参照モデル」について
    ========================================
    実際のオンラインRLでは old_logprobs はロールアウトサーバーが記録した値、
    ref_logprobs は固定された参照モデルの値。本デモは1回のオンポリシー更新
    (ロールアウト直後の最初の勾配ステップ) を模しており、学習開始時点の
    ポリシー重みを凍結コピーして both の代わりに使う。
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # 旧ポリシー/参照モデル: 学習開始時点の重みを凍結コピー。
    import copy

    frozen = copy.deepcopy(model).to(device)
    for p in frozen.parameters():
        p.requires_grad = False
    frozen.eval()

    dataset = TrajectoryDataset(df)
    # グループをまとめてバッチ化するため、group_id単位でシャッフルする。
    group_ids = dataset.df["group_id"].unique().tolist()

    collator = GRPOCollator()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay,
                                   betas=(0.9, 0.98))
    scheduler = constant_lr(optimizer)

    os.makedirs(output_dir, exist_ok=True)
    print("訓練設定:")
    print(f"  軌跡数: {len(dataset)}  グループ数: {len(group_ids)}  "
          f"1バッチのグループ数: {batch_groups}")
    print(f"  学習率: {lr} (constant)  weight_decay: {weight_decay}  "
          f"clip: [{1-clip_low:.2f}, {1+clip_high:.2f}]  kl_coef: {kl_coef}")

    global_step = 0
    for epoch in range(epochs):
        print(f"\n=== Epoch {epoch + 1}/{epochs} ===")
        import random

        random.Random(epoch).shuffle(group_ids)
        for b in range(0, len(group_ids), batch_groups):
            batch_gids = set(group_ids[b : b + batch_groups])
            features = [dataset[i] for i in range(len(dataset)) if dataset.df.iloc[i]["group_id"] in batch_gids]
            batch = collator(features)
            batch_t = {k: v.to(device) for k, v in batch.items() if torch.is_tensor(v)}

            attention_mask = batch_t["loss_mask"]  # パディングマスクとして再利用

            logits_new = model(batch_t["input_ids"], attention_mask)
            with torch.no_grad():
                logits_old = frozen(batch_t["input_ids"], attention_mask)
                logits_ref = logits_old  # このデモでは old==ref (最初の更新ステップ)

            logprobs_new = token_logprobs(logits_new, batch_t["input_ids"])
            logprobs_old = token_logprobs(logits_old, batch_t["input_ids"])
            logprobs_ref = token_logprobs(logits_ref, batch_t["input_ids"])

            out = grpo_clipped_loss(
                logprobs_new, logprobs_old, logprobs_ref, batch_t["advantages"],
                batch_t["loss_mask"], clip_low=clip_low, clip_high=clip_high, kl_coef=kl_coef,
            )

            optimizer.zero_grad()
            out["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            scheduler.step()

            global_step += 1
            if global_step % log_steps == 0:
                n_succ = sum(m["success"] for m in batch["meta"])
                print(
                    f"  Step {global_step} | groups={batch_gids} | "
                    f"n_traj={len(batch['meta'])} (succ={n_succ}) | "
                    f"loss={out['loss'].item():.4f} pg={out['pg_loss'].item():.4f} "
                    f"kl={out['kl_loss'].item():.4f}"
                )

    print(f"\nTraining complete! (このデモの重みは保存せず、ログのみ表示)")


# ============================================================
# 7. 合成データでの end-to-end 動作確認
# ============================================================

def _make_synthetic_trajectory_df(n_groups: int = 4, group_size: int = 4, seed: int = 0) -> pd.DataFrame:
    """合成ロールアウトDataFrameを作る (本物のエージェント軌跡の代わり)。

    各グループは同じ(仮想の)プロンプトからのロールアウト。成功軌跡はFLOPsに
    バリエーションを持たせ、dual-channel advantageが実際に差をつけられるようにする。
    """
    rng = np.random.RandomState(seed)
    rows = []
    for g in range(n_groups):
        for i in range(group_size):
            success = bool(rng.rand() < 0.6)
            flops = float(rng.uniform(0.5e14, 2.0e14))
            raw_reward = (1.0 if success else 0.0) + rng.uniform(-0.05, 0.05)
            n_turns = rng.randint(3, 6)
            turn_lengths = [int(rng.randint(3, 8)) for _ in range(n_turns)]
            # 奇数番目のターンをコンテキスト編集ターンとして印付ける (デモ用)
            turn_is_edit = [bool(t % 2 == 1) for t in range(n_turns)]
            total_len = sum(turn_lengths)
            token_ids = rng.randint(4, 500, size=total_len).tolist()
            rows.append(dict(
                group_id=f"g{g}", trajectory_id=f"g{g}_t{i}", success=success,
                raw_reward=raw_reward, flops=flops, token_ids=token_ids,
                turn_lengths=turn_lengths, turn_is_edit=turn_is_edit,
            ))
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser("CLM RL Fine-tuning (No Trainer)")
    parser.add_argument("--train_file", type=str, default="",
                        help="Parquet/JSON file readable by pandas with the required columns")
    parser.add_argument("--output_dir", type=str, default="./clm-rl-finetuned")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_groups", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-6)
    args = parser.parse_args()

    if args.train_file:
        if args.train_file.endswith(".parquet"):
            df = pd.read_parquet(args.train_file)
        else:
            df = pd.read_json(args.train_file, lines=True)
    else:
        print("--train_file 未指定のため、合成ロールアウトDataFrameで動作確認します。\n")
        df = _make_synthetic_trajectory_df()

    model = load_model_for_finetuning(vocab_size=600)
    train(model, df, output_dir=args.output_dir, epochs=args.epochs, batch_groups=args.batch_groups, lr=args.lr)


if __name__ == "__main__":
    main()
