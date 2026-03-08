"""
Kronos Understanding - Training Pipeline (簡略化疑似コード)

2段階学習パイプライン:
    Stage 1: Tokenizer学習 (再構成品質 + BSQ正則化)
    Stage 2: Predictor学習 (自己回帰トークン予測)

対応する公式実装:
  - finetune/train_tokenizer.py: Tokenizer学習
  - finetune/train_predictor.py: Predictor学習
  - finetune_csv/train_sequential.py: 2段階逐次学習

論文参照:
  - Section 3: Methodology (Tokenizer損失, Eq.2; AR損失, Eq.8)
  - Appendix C: Implementation Details (ハイパーパラメータ)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ============================================================
# データセット
# ============================================================

class KlineDataset(Dataset):
    """
    K線データセット

    入力CSV形式:
        timestamps, open, high, low, close, volume, amount

    前処理:
        1. Z-score正規化 (各特徴独立)
        2. クリッピング [-5, 5]
        3. 時間特徴抽出 [minute, hour, weekday, day, month]

    出力 (1サンプル):
        x:     (lookback + predict, 6) - OHLCVA
        stamp: (lookback + predict, 5) - 時間特徴
    """

    def __init__(self, data, timestamps, lookback_window=512, predict_window=48, clip=5):
        """
        Args:
            data: (N, 6) - 全K線データ [open, high, low, close, vol, amt]
            timestamps: (N, 5) - 時間特徴 [min, hour, wday, day, month]
            lookback_window: 履歴ウィンドウ長
            predict_window: 予測ウィンドウ長
            clip: クリッピング閾値
        """
        self.data = data
        self.timestamps = timestamps
        self.lookback = lookback_window
        self.predict = predict_window
        self.total_len = lookback_window + predict_window
        self.clip = clip
        self.n_samples = len(data) - self.total_len + 1

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        """
        出力:
            x:     (lookback + predict, 6) - 正規化済みK線
            stamp: (lookback + predict, 5) - 時間特徴

        正規化: lookback部分の統計量で全体を正規化
        """
        # ウィンドウ切り出し
        segment = self.data[idx:idx + self.total_len]  # (total_len, 6)
        stamp = self.timestamps[idx:idx + self.total_len]  # (total_len, 5)

        # Lookback部分の統計量で正規化
        lookback_data = segment[:self.lookback]  # (lookback, 6)
        mean = np.mean(lookback_data, axis=0)    # (6,)
        std = np.std(lookback_data, axis=0)      # (6,)

        x = (segment - mean) / (std + 1e-5)
        x = np.clip(x, -self.clip, self.clip)

        return (
            torch.from_numpy(x.astype(np.float32)),      # (total_len, 6)
            torch.from_numpy(stamp.astype(np.float32)),   # (total_len, 5)
            torch.from_numpy(mean.astype(np.float32)),    # (6,) - 逆正規化用
            torch.from_numpy(std.astype(np.float32)),     # (6,)
        )


# ============================================================
# Stage 1: Tokenizer 学習
# ============================================================

def train_tokenizer(
    tokenizer,
    train_loader,
    val_loader=None,
    epochs=100,
    lr=1e-3,
    weight_decay=0.01,
    lambda_quant=1.0,   # BSQ損失の重み
):
    """
    Tokenizer学習

    損失関数 (Eq. 2):
        L_tokenizer = L_coarse + L_fine + λ * L_quant

    各損失の意味:
        L_coarse = MSE(x, Decode(BSQ_s1(Encode(x))))
            → s1 (coarseサブトークン) だけで主要構造を捉える
        L_fine = MSE(x, Decode(BSQ_all(Encode(x))))
            → s1 + s2 (全体) で高精度な再構成
        L_quant = commit_loss + entropy_penalty (BSQ由来)
            → エンコーダ出力を量子化結果に近づける + コードブック使用率最大化

    階層的再構成の意義:
        L_coarseの最適化により、s1が「粗い価格構造」をキャプチャするよう強制
        L_fineの最適化により、s2が「s1の残差（細かい変動）」をキャプチャ
        → 自己回帰モデルでのCoarse-to-Fine予測に必要な階層構造が形成

    ハイパーパラメータ (Appendix C):
        Tokenizer Encoder/Decoder: 3層, d=256, 4ヘッド, ff=512
        BSQ: β=0.05, γ0=1.0, γ=1.1, ζ=0.05, group_size=5
        λ (L_quant重み) = 1.0

    入力:
        x: (B, T, 6) - 正規化済みK線 (学習時はlookback+predictの全区間)

    出力:
        (z_pre, z_fine): 再構成
            z_pre:  (B, T, 6) - s1のみからの再構成
            z_fine: (B, T, 6) - s1+s2全体からの再構成
        bsq_loss: BSQ損失
        quantized: (B, T, 20) - 量子化済みベクトル
    """
    optimizer = torch.optim.AdamW(tokenizer.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, epochs=epochs, steps_per_epoch=len(train_loader)
    )

    for epoch in range(epochs):
        tokenizer.train()
        total_loss = 0.0

        for batch_idx, (x, stamp, mean, std) in enumerate(train_loader):
            # x: (B, T, 6), stamp: (B, T, 5)
            x = x.to(tokenizer.embed.weight.device)

            # === Forward ===
            (z_pre, z_fine), bsq_loss, quantized, z_indices = tokenizer(x)
            # z_pre:  (B, T, 6) - s1のみからの粗い再構成
            # z_fine: (B, T, 6) - 全体からの精密再構成

            # === Loss計算 ===
            L_coarse = F.mse_loss(x, z_pre)     # 粗粒度再構成損失
            L_fine = F.mse_loss(x, z_fine)       # 精密再構成損失
            L_quant = bsq_loss                   # BSQ損失 (commit + entropy)

            loss = L_coarse + L_fine + lambda_quant * L_quant

            # === Backward ===
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"[Tokenizer] Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

        # === Validation ===
        if val_loader is not None:
            val_loss = _evaluate_tokenizer(tokenizer, val_loader)
            print(f"  Val Loss: {val_loss:.4f}")


def _evaluate_tokenizer(tokenizer, val_loader):
    """Tokenizer検証"""
    tokenizer.eval()
    total_loss = 0.0
    with torch.no_grad():
        for x, stamp, mean, std in val_loader:
            x = x.to(next(tokenizer.parameters()).device)
            (z_pre, z_fine), bsq_loss, _, _ = tokenizer(x)
            loss = F.mse_loss(x, z_pre) + F.mse_loss(x, z_fine) + bsq_loss
            total_loss += loss.item()
    return total_loss / len(val_loader)


# ============================================================
# Stage 2: Predictor (Kronos) 学習
# ============================================================

def train_predictor(
    model,
    tokenizer,
    train_loader,
    val_loader=None,
    epochs=30,
    lr=1e-3,
    weight_decay=0.01,
    use_teacher_forcing=False,  # デフォルトはサンプリング (exposure bias軽減)
):
    """
    Predictor学習

    損失関数 (Eq. 8):
        L_ar = -E_{b~D} Σ_t [log p(s1_t | b_{<t}) + log p(s2_t | b_{<t}, s1_t)]
             = (CE_s1 + CE_s2) / 2

    処理フロー:
        1. Tokenizer (凍結) でK線データをトークン化
        2. 入力: tokens[0:T-1], ターゲット: tokens[1:T] (自己回帰)
        3. s1予測 → s1サンプリング → s2予測 (Coarse-to-Fine)
        4. Cross-Entropy損失で最適化

    s1サンプリングの工夫:
        - Teacher Forcing (use_teacher_forcing=True):
            正解s1をDependencyAwareLayerに渡す
            → 学習初期の安定化に有効だが exposure bias のリスク
        - サンプリング (use_teacher_forcing=False, デフォルト):
            予測分布からサンプリングしたs1を使用
            → 推論時との分布差を軽減 (exposure bias対策)

    ハイパーパラメータ (Table 5):
        Kronos_small:  lr=1e-3, dropout=0.25, weight_decay=0.01
        Kronos_base:   lr=5e-4, dropout=0.20, weight_decay=0.05
        Kronos_large:  lr=2e-4, dropout=0.00, weight_decay=0.10

    入力:
        s1_ids, s2_ids: (B, T) - トークンID
        stamp: (B, T, 5) - 時間特徴
    """
    # Tokenizer凍結
    tokenizer.eval()
    for param in tokenizer.parameters():
        param.requires_grad = False

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Cosine LR with Linear Warmup (15000 steps)
    warmup_steps = 15000
    total_steps = epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, total_iters=warmup_steps
    )

    device = next(model.parameters()).device

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_ce_s1, total_ce_s2 = 0.0, 0.0

        for batch_idx, (x, stamp, mean, std) in enumerate(train_loader):
            x = x.to(device)         # (B, T, 6)
            stamp = stamp.to(device)  # (B, T, 5)

            # === Step 1: Tokenizer でトークン化 (凍結) ===
            with torch.no_grad():
                s1_ids, s2_ids = tokenizer.encode(x, half=True)
                # s1_ids: (B, T) ∈ [0, 1023]
                # s2_ids: (B, T) ∈ [0, 1023]

            # === Step 2: 自己回帰のための入力/ターゲット分割 ===
            # 入力: tokens[0:T-1] → 予測ターゲット: tokens[1:T]
            input_s1 = s1_ids[:, :-1]     # (B, T-1)
            input_s2 = s2_ids[:, :-1]     # (B, T-1)
            target_s1 = s1_ids[:, 1:]     # (B, T-1)
            target_s2 = s2_ids[:, 1:]     # (B, T-1)
            input_stamp = stamp[:, :-1]   # (B, T-1, 5)

            # === Step 3: Forward ===
            s1_logits, s2_logits = model(
                input_s1, input_s2, input_stamp,
                use_teacher_forcing=use_teacher_forcing,
                s1_targets=target_s1 if use_teacher_forcing else None
            )
            # s1_logits: (B, T-1, 1024)
            # s2_logits: (B, T-1, 1024)

            # === Step 4: Cross-Entropy損失 ===
            ce_s1 = F.cross_entropy(
                s1_logits.reshape(-1, 1024),  # (B*(T-1), 1024)
                target_s1.reshape(-1)          # (B*(T-1),)
            )
            ce_s2 = F.cross_entropy(
                s2_logits.reshape(-1, 1024),
                target_s2.reshape(-1)
            )
            loss = (ce_s1 + ce_s2) / 2

            # === Step 5: Backward ===
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Warmup + Cosine schedule
            global_step = epoch * len(train_loader) + batch_idx
            if global_step < warmup_steps:
                warmup_scheduler.step()
            else:
                scheduler.step()

            total_loss += loss.item()
            total_ce_s1 += ce_s1.item()
            total_ce_s2 += ce_s2.item()

        n = len(train_loader)
        print(f"[Predictor] Epoch {epoch+1}/{epochs}, "
              f"Loss: {total_loss/n:.4f} "
              f"(CE_s1={total_ce_s1/n:.4f}, CE_s2={total_ce_s2/n:.4f})")

        # === Validation ===
        if val_loader is not None:
            val_loss = _evaluate_predictor(model, tokenizer, val_loader)
            print(f"  Val Loss: {val_loss:.4f}")


def _evaluate_predictor(model, tokenizer, val_loader):
    """Predictor検証"""
    model.eval()
    device = next(model.parameters()).device
    total_loss = 0.0

    with torch.no_grad():
        for x, stamp, mean, std in val_loader:
            x = x.to(device)
            stamp = stamp.to(device)

            s1_ids, s2_ids = tokenizer.encode(x, half=True)

            s1_logits, s2_logits = model(
                s1_ids[:, :-1], s2_ids[:, :-1], stamp[:, :-1]
            )

            ce_s1 = F.cross_entropy(s1_logits.reshape(-1, 1024), s1_ids[:, 1:].reshape(-1))
            ce_s2 = F.cross_entropy(s2_logits.reshape(-1, 1024), s2_ids[:, 1:].reshape(-1))
            loss = (ce_s1 + ce_s2) / 2
            total_loss += loss.item()

    return total_loss / len(val_loader)


# ============================================================
# 2段階逐次学習
# ============================================================

def train_sequential(tokenizer, model, train_data, val_data=None, config=None):
    """
    2段階逐次学習パイプライン

    Stage 1: Tokenizer学習
        - 再構成品質を最大化
        - BSQエントロピー正則化でコードブック使用率を高める

    Stage 2: Predictor学習
        - Tokenizer凍結
        - 自己回帰トークン予測精度を最大化

    Args:
        tokenizer: KronosTokenizer (未学習 or 事前学習済み)
        model: Kronos (未学習 or 事前学習済み)
        train_data: (data, timestamps) tuple
        config: 学習設定 dict
    """
    if config is None:
        config = {
            # データ設定
            'lookback_window': 512,
            'predict_window': 48,
            'batch_size': 50,
            # Tokenizer学習
            'tokenizer_epochs': 100,
            'tokenizer_lr': 1e-3,
            # Predictor学習
            'predictor_epochs': 30,
            'predictor_lr': 1e-3,
        }

    # データセット作成
    train_dataset = KlineDataset(
        train_data[0], train_data[1],
        config['lookback_window'], config['predict_window']
    )
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)

    val_loader = None
    if val_data is not None:
        val_dataset = KlineDataset(
            val_data[0], val_data[1],
            config['lookback_window'], config['predict_window']
        )
        val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)

    # === Stage 1: Tokenizer学習 ===
    print("=" * 60)
    print("Stage 1: Tokenizer Training")
    print("=" * 60)
    train_tokenizer(
        tokenizer, train_loader, val_loader,
        epochs=config['tokenizer_epochs'],
        lr=config['tokenizer_lr'],
    )

    # === Stage 2: Predictor学習 ===
    print("=" * 60)
    print("Stage 2: Predictor Training")
    print("=" * 60)
    train_predictor(
        model, tokenizer, train_loader, val_loader,
        epochs=config['predictor_epochs'],
        lr=config['predictor_lr'],
    )

    print("Training complete!")
    return tokenizer, model


# ============================================================
# データ前処理パイプライン
# ============================================================

def preprocess_kline_data(df):
    """
    K線データの前処理

    入力: DataFrame with [timestamps, open, high, low, close, volume, amount]
    出力: (data, timestamps) tuple
        data: (N, 6) float32
        timestamps: (N, 5) float32 [minute, hour, weekday, day, month]

    低品質データフィルタリング (Algorithm 1):
        1. 構造的ブレイク検出 (|open_t / close_{t-1} - 1| > θ)
        2. 非流動性期間除去 (連続ゼロ出来高)
        3. 価格停滞期間除去 (連続同一終値)
        4. 最小長要件チェック

    Volume/Amountの欠損値処理:
        - 価格 (OHLC): 欠損は境界として分割
        - Volume/Amount: 欠損はゼロ埋め
        - 学習時: 5%の確率でVol/Amtをゼロ化 (正則化)
    """
    # 時間特徴抽出
    ts = pd.to_datetime(df['timestamps'])
    stamps = np.stack([
        ts.dt.minute.values,
        ts.dt.hour.values,
        ts.dt.weekday.values,
        ts.dt.day.values,
        ts.dt.month.values,
    ], axis=1).astype(np.float32)

    # OHLCVA抽出
    data = df[['open', 'high', 'low', 'close', 'volume', 'amount']].values.astype(np.float32)

    # 欠損値処理: Volume/Amountのゼロ埋め
    data[:, 4] = np.nan_to_num(data[:, 4], nan=0.0)  # volume
    data[:, 5] = np.nan_to_num(data[:, 5], nan=0.0)  # amount

    return data, stamps


def apply_volume_dropout(x, dropout_rate=0.05):
    """
    Volume/Amount ドロップアウト (学習時正則化)

    5%の確率でVolume, Amountをゼロに設定。
    価格情報のみからの予測能力を学習させる。

    入力: x (B, T, 6)
    出力: x (B, T, 6) - Vol/Amtが一部ゼロ化
    """
    mask = torch.rand(x.size(0)) < dropout_rate  # (B,) 5%の確率でTrue
    x[mask, :, 4] = 0.0  # volume
    x[mask, :, 5] = 0.0  # amount
    return x


# ============================================================
# 事前学習データの統計
# ============================================================

"""
事前学習データ概要 (Table 13 / Section 3 "Model Pre-training"):

| 項目          | 値                                      |
|---------------|----------------------------------------|
| 総レコード数   | 12億本超                                |
| 取引所数      | 45以上 (30カ国超)                        |
| 時間粒度      | 1min, 5min, 10min, 15min, 20min, 30min,|
|               | 40min, 60min, 2H, 4H, Daily, Weekly   |
| 資産クラス    | 株式, 暗号通貨, FX, 先物                |

データリバランス:
    - 株式が多数派 → 暗号通貨/先物/FXのサンプリング重みを増加
    - 資産クラス間の不均衡を補正

低品質フィルタリング (Table 4):
    | 頻度    | 最小長(本)| 価格Jump閾値 | 非流動性(本) | 停滞(本) |
    |---------|----------|-------------|-------------|---------|
    | 1min    | 2048     | 0.10        | 15          | 45      |
    | 5min    | 1024     | 0.15        | 3           | 10      |
    | 15min   | 512      | 0.15        | 2           | 5       |
    | Daily   | 128      | 0.30        | 1           | 3       |
    | Weekly  | 16       | 0.50        | 0           | 2       |
"""


# ============================================================
# 使用例
# ============================================================

if __name__ == "__main__":
    import pandas as pd

    print("=" * 60)
    print("Kronos Training Pipeline (Pseudo-code)")
    print("=" * 60)

    print("""
    # 実際の学習コマンド (公式リポジトリ):

    # === Fine-tune on CSV data ===
    # 逐次学習 (推奨)
    python finetune_csv/train_sequential.py \\
        --config finetune_csv/configs/config_ali09988_candle-5min.yaml

    # === Fine-tune on Qlib data (A-Share market) ===
    # Step 1: Tokenizer
    python finetune/train_tokenizer.py

    # Step 2: Predictor
    python finetune/train_predictor.py

    # Step 3: Backtest
    python finetune/qlib_test.py

    # === Distributed Training ===
    torchrun --nproc_per_node=8 finetune/train_predictor.py
    """)

    print("Training configurations (Table 5):")
    print("  Kronos_small:  lr=1e-3, dropout=0.25, weight_decay=0.01")
    print("  Kronos_base:   lr=5e-4, dropout=0.20, weight_decay=0.05")
    print("  Kronos_large:  lr=2e-4, dropout=0.00, weight_decay=0.10")
    print()
    print("All models: AdamW optimizer, Cosine LR, 15k warmup steps")
