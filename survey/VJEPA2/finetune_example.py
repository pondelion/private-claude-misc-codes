"""
V-JEPA 2.1 ファインチューニングサンプル
=======================================

V-JEPA 2.1 の凍結エンコーダを使った下流タスクのファインチューニング例。
V-JEPA 2.1 の評価では「frozen encoder + 軽量な下流ヘッド」が基本戦略。

対応タスク:
  1. 動画分類 (Video Classification) ← 主な例
  2. 密予測タスク (Semantic Segmentation) の概要

データセットの前提:
  - 動画ファイルが適当なディレクトリに格納されている
  - ラベルCSV: video_path, label (0-indexed)
  - 例: /data/video_dataset/{train,val}/*.mp4

対応する公式実装:
  - evals/video_classification_frozen/eval.py
  - evals/video_classification_frozen/models.py
"""

import os
import math
import random
import csv
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# ============================================================
# データセット
# ============================================================

class VideoDataset(Dataset):
    """
    動画分類用データセット。

    CSVファイルフォーマット:
        video_path,label
        /data/videos/train/cat_001.mp4,0
        /data/videos/train/dog_002.mp4,1
        ...

    動画を T フレームのクリップとして読み込む。
    簡略化のため、実際の動画読み込みは torch.randn で代替。
    実環境では torchvision.io.read_video や decord を使用すること。

    出力 (各サンプル):
        frames: (3, T, H, W)  float32, 正規化済み
        label:  int
    """

    def __init__(
        self,
        csv_path: str,
        num_frames: int = 16,
        img_size: int = 256,
        fps: int = 4,
        split: str = "train",
        random_crop: bool = True,
        horizontal_flip: bool = True,
    ):
        self.num_frames = num_frames
        self.img_size = img_size
        self.fps = fps
        self.split = split
        self.is_train = (split == "train")

        # CSVからパスとラベルを読み込む
        self.samples = []
        if os.path.exists(csv_path):
            with open(csv_path, "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    self.samples.append((row["video_path"], int(row["label"])))
        else:
            # ダミーデータ (CSV未存在の場合)
            print(f"[Warning] CSV not found: {csv_path}. Using dummy data.")
            self.samples = [("/dummy/video_{:04d}.mp4".format(i), i % 10)
                            for i in range(100)]

        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1, 1)
        self.std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1, 1)

    def _load_video(self, video_path: str) -> torch.Tensor:
        """
        動画を読み込んでフレームをサンプリングする。

        出力: (3, T, H, W) float32 [0, 1]

        実環境では:
            import decord
            vr = decord.VideoReader(video_path)
            frame_ids = np.linspace(0, len(vr)-1, self.num_frames, dtype=int)
            frames = vr.get_batch(frame_ids).asnumpy()  # (T, H, W, 3)
            frames = torch.from_numpy(frames).permute(3, 0, 1, 2).float() / 255.0
        """
        # ダミー実装: ランダムフレームを生成
        frames = torch.rand(3, self.num_frames, self.img_size, self.img_size)
        return frames

    def _augment(self, frames: torch.Tensor) -> torch.Tensor:
        """
        入力:  frames (3, T, H, W)
        出力:  frames (3, T, H_out, W_out)  augment済み
        """
        _, T, H, W = frames.shape

        if self.is_train:
            # ランダムクロップ
            if H > self.img_size:
                i = random.randint(0, H - self.img_size)
                j = random.randint(0, W - self.img_size)
                frames = frames[:, :, i:i + self.img_size, j:j + self.img_size]
            # ランダム水平反転
            if random.random() < 0.5:
                frames = torch.flip(frames, dims=[3])
        else:
            # センタークロップ
            if H > self.img_size:
                i = (H - self.img_size) // 2
                j = (W - self.img_size) // 2
                frames = frames[:, :, i:i + self.img_size, j:j + self.img_size]

        return frames

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        video_path, label = self.samples[idx]

        # 動画読み込み: (3, T, H, W)
        frames = self._load_video(video_path)
        # オーグメンテーション
        frames = self._augment(frames)
        # 正規化
        frames = (frames - self.mean) / (self.std + 1e-8)

        return frames, label


class ImageNetStyleDataset(Dataset):
    """
    画像分類用データセット (V-JEPA 2.1 の画像評価用)。

    出力:
        image: (3, H, W)
        label: int
    """

    def __init__(self, csv_path: str, img_size: int = 224, split: str = "train"):
        self.img_size = img_size
        self.is_train = (split == "train")
        self.samples = []
        if os.path.exists(csv_path):
            with open(csv_path, "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    self.samples.append((row["image_path"], int(row["label"])))
        else:
            self.samples = [("/dummy/img_{:04d}.jpg".format(i), i % 1000)
                            for i in range(200)]
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # ダミー実装
        image = torch.rand(3, self.img_size, self.img_size)
        image = (image - self.mean) / (self.std + 1e-8)
        label = self.samples[idx][1]
        return image, label


# ============================================================
# Attentive Pooler (V-JEPA 2.1 の推奨ヘッド)
# ============================================================

class AttentivePooler(nn.Module):
    """
    Attentive Pooler: 可変長トークン列をfixedサイズの表現に集約。

    V-JEPA 2.1 の論文では下流タスクに Attentive Pooler を使用。
    Cross-Attentionによってトークン列から重要な情報を選択的に集約する。

    入力:
        x: (B, N, D)  エンコーダのトークン出力

    出力:
        pooled: (B, num_queries, D)  集約された特徴

    通常は num_queries=1 で使用し (B, D) に reshape。
    """

    def __init__(
        self,
        embed_dim: int = 1024,
        num_heads: int = 16,
        num_queries: int = 1,
        mlp_ratio: float = 4.0,
        depth: int = 1,
    ):
        super().__init__()
        self.num_queries = num_queries

        # クエリトークン (学習可能)
        self.query_tokens = nn.Parameter(torch.zeros(1, num_queries, embed_dim))
        nn.init.trunc_normal_(self.query_tokens, std=0.02)

        # Cross-Attention層
        self.cross_attn = nn.MultiheadAttention(
            embed_dim, num_heads, batch_first=True
        )
        self.norm_q = nn.LayerNorm(embed_dim)
        self.norm_kv = nn.LayerNorm(embed_dim)

        # FFN
        hidden_dim = int(embed_dim * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        入力:
            x: (B, N, D)  エンコーダトークン

        出力:
            pooled: (B, num_queries, D)
        """
        B = x.shape[0]

        # クエリをバッチに合わせて複製
        q = self.query_tokens.expand(B, -1, -1)  # (B, num_queries, D)

        # Cross-Attention: クエリがトークン列からattend
        q_norm  = self.norm_q(q)
        kv_norm = self.norm_kv(x)
        attn_out, _ = self.cross_attn(q_norm, kv_norm, kv_norm)  # (B, num_queries, D)

        # 残差接続 + FFN
        q = q + attn_out
        q = q + self.ffn(q)

        return q  # (B, num_queries, D)


# ============================================================
# 動画分類モデル (Frozen Encoder + Attentive Head)
# ============================================================

class VideoClassifier(nn.Module):
    """
    V-JEPA 2.1 frozen encoder + Attentive Pooler + Classifier ヘッド。

    学習時: encoder は凍結 (no_grad), pooler と classifier のみ学習。

    入力:
        x: (B, 3, T, H, W)  動画クリップ

    出力:
        logits: (B, num_classes)
    """

    def __init__(
        self,
        encoder: nn.Module,
        embed_dim: int = 1024,
        num_classes: int = 400,
        pooler_depth: int = 4,
        pooler_num_heads: int = 16,
        dropout: float = 0.5,
        freeze_encoder: bool = True,
    ):
        super().__init__()
        self.encoder = encoder
        self.embed_dim = embed_dim
        self.freeze_encoder = freeze_encoder

        # エンコーダを凍結
        if freeze_encoder:
            for p in encoder.parameters():
                p.requires_grad = False

        # Attentive Pooler: (B, N, D) → (B, 1, D) → (B, D)
        self.pooler = AttentivePooler(
            embed_dim=embed_dim,
            num_heads=pooler_num_heads,
            num_queries=1,
            depth=pooler_depth,
        )

        # 分類ヘッド
        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes),
        )

        self._init_head_weights()

    def _init_head_weights(self):
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        入力:
            x: (B, 3, T, H, W)  動画クリップ

        出力:
            logits: (B, num_classes)

        処理:
          1. Frozen encoder で特徴抽出: (B, N, D)
          2. Attentive Pooler で集約: (B, 1, D) → (B, D)
          3. Classifier で分類: (B, num_classes)
        """
        # ========================================
        # Step 1: エンコーダで特徴抽出 (凍結)
        # ========================================
        if self.freeze_encoder:
            with torch.no_grad():
                feats = self.encoder(x)  # (B, N, D)
        else:
            feats = self.encoder(x)

        # out_layersが指定されている場合は最終層のみ使用
        if isinstance(feats, list):
            feats = feats[-1]  # 最終層: (B, N, D)
        # feats: (B, N, D)

        # ========================================
        # Step 2: Attentive Pooler で集約
        # ========================================
        pooled = self.pooler(feats)         # (B, 1, D)
        pooled = pooled.squeeze(1)          # (B, D)

        # ========================================
        # Step 3: 分類
        # ========================================
        logits = self.classifier(pooled)    # (B, num_classes)
        return logits


# ============================================================
# マルチクリップ評価 (推論時)
# ============================================================

class MultiClipVideoClassifier(nn.Module):
    """
    複数クリップのアンサンブルによる動画分類モデル (推論用)。

    動画全体から複数のクリップを抽出し、各クリップのlogitsを平均する。
    V-JEPA 2.1 の評価では multi-clip inference が標準。

    入力:
        clips: (B, n_clips, 3, T, H, W)

    出力:
        logits: (B, num_classes)  n_clips分の平均
    """

    def __init__(self, base_classifier: VideoClassifier):
        super().__init__()
        self.classifier = base_classifier

    def forward(self, clips: torch.Tensor) -> torch.Tensor:
        """
        入力:
            clips: (B, n_clips, 3, T, H, W)
        出力:
            logits: (B, num_classes)
        """
        B, n_clips, C, T, H, W = clips.shape
        # クリップを結合: (B*n_clips, 3, T, H, W)
        clips_flat = clips.view(B * n_clips, C, T, H, W)
        logits_flat = self.classifier(clips_flat)  # (B*n_clips, num_classes)
        # 元の形状に戻して平均
        logits = logits_flat.view(B, n_clips, -1).mean(dim=1)  # (B, num_classes)
        return logits


# ============================================================
# 学習ユーティリティ
# ============================================================

class AverageMeter:
    """実行平均を計算するユーティリティ"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def topk_accuracy(output: torch.Tensor, target: torch.Tensor, topk=(1, 5)) -> list:
    """
    Top-K精度を計算する。

    入力:
        output: (B, num_classes)  logits
        target: (B,)              正解ラベル
        topk:   計算するKのリスト

    出力:
        list of float  各Kの精度 [%]
    """
    maxk = max(topk)
    B = target.size(0)
    _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)
    pred = pred.t()  # (maxk, B)
    correct = pred.eq(target.unsqueeze(0).expand_as(pred))

    results = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum()
        results.append(correct_k.mul_(100.0 / B).item())
    return results


def train_one_epoch(
    model: VideoClassifier,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    epoch: int,
    scheduler=None,
) -> dict:
    """
    1エポックの学習。

    入力:
        model:     VideoClassifier
        loader:    DataLoader (frames: (B, 3, T, H, W), labels: (B,))
        criterion: 損失関数 (CrossEntropyLoss)
        optimizer: AdamW等
        scaler:    混合精度用
        device:    デバイス
        epoch:     現在エポック
        scheduler: LRスケジューラ (batch単位の場合)

    出力:
        dict: {"loss": float, "top1": float, "top5": float}
    """
    model.train()
    # エンコーダは凍結なのでeval modeにする
    if model.freeze_encoder:
        model.encoder.eval()

    loss_meter = AverageMeter()
    top1_meter = AverageMeter()
    top5_meter = AverageMeter()

    criterion = criterion.to(device)

    for i, (frames, labels) in enumerate(loader):
        # frames: (B, 3, T, H, W)
        # labels: (B,)
        frames = frames.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        B = frames.shape[0]

        with torch.cuda.amp.autocast(enabled=True):
            logits = model(frames)         # (B, num_classes)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        if scheduler is not None and hasattr(scheduler, 'step_batch'):
            scheduler.step()

        # 精度計算
        acc1, acc5 = topk_accuracy(logits.detach(), labels, topk=(1, 5))
        loss_meter.update(loss.item(), B)
        top1_meter.update(acc1, B)
        top5_meter.update(acc5, B)

        if i % 50 == 0:
            print(f"  [{epoch}][{i}/{len(loader)}] "
                  f"loss={loss_meter.avg:.4f} "
                  f"top1={top1_meter.avg:.2f}% "
                  f"top5={top5_meter.avg:.2f}%")

    return {"loss": loss_meter.avg, "top1": top1_meter.avg, "top5": top5_meter.avg}


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict:
    """
    検証/テストの評価。

    入力:
        model:  VideoClassifier または MultiClipVideoClassifier
        loader: DataLoader
        device: デバイス

    出力:
        dict: {"top1": float, "top5": float}
    """
    model.eval()
    top1_meter = AverageMeter()
    top5_meter = AverageMeter()

    for frames, labels in loader:
        frames = frames.to(device)
        labels = labels.to(device)
        B = frames.shape[0]

        logits = model(frames)
        acc1, acc5 = topk_accuracy(logits, labels, topk=(1, 5))
        top1_meter.update(acc1, B)
        top5_meter.update(acc5, B)

    return {"top1": top1_meter.avg, "top5": top5_meter.avg}


# ============================================================
# メイン: ファインチューニング実行
# ============================================================

def finetune_video_classification(
    # データ設定
    train_csv: str = "train.csv",
    val_csv:   str = "val.csv",
    num_classes: int = 400,         # Kinetics-400
    num_frames: int = 16,
    img_size: int = 256,
    batch_size: int = 32,
    num_workers: int = 4,
    # モデル設定
    checkpoint_path: str = None,    # 事前学習済みチェックポイント
    model_name: str = "vit_large",  # "vit_large" / "vit_gigantic"
    embed_dim: int = 1024,          # ViT-Lの場合1024
    pooler_depth: int = 4,
    freeze_encoder: bool = True,    # エンコーダを凍結するか
    # 学習設定
    num_epochs: int = 30,
    lr: float = 1e-4,
    weight_decay: float = 0.01,
    warmup_epochs: int = 3,
    dropout: float = 0.5,
    # その他
    device_str: str = "cuda",
    save_path: str = "./checkpoint_finetune.pth",
):
    """
    V-JEPA 2.1 frozen encoder を用いた動画分類のファインチューニング。

    学習戦略:
      - エンコーダ: 凍結 (no_grad)
      - Attentive Pooler + Classifier: 学習
      - 損失: CrossEntropyLoss
      - 最適化: AdamW + CosineAnnealing (warmup込み)

    入力:
        train_csv:        訓練データのCSVパス (video_path, label)
        val_csv:          検証データのCSVパス
        num_classes:      クラス数
        num_frames:       フレーム数
        img_size:         入力解像度
        batch_size:       バッチサイズ
        num_workers:      DataLoaderのワーカー数
        checkpoint_path:  V-JEPA 2.1 事前学習済みチェックポイント
        model_name:       エンコーダのモデル名
        embed_dim:        エンコーダの次元
        pooler_depth:     Attentive Poolerの深さ
        freeze_encoder:   エンコーダを凍結するか
        num_epochs:       エポック数
        lr:               最大学習率
        weight_decay:     重み減衰
        warmup_epochs:    warmupエポック数
        dropout:          ドロップアウト率
        device_str:       デバイス文字列
        save_path:        チェックポイント保存先
    """
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    print(f"使用デバイス: {device}")

    # ========================================
    # データセット・ローダー構築
    # ========================================
    print("\n[1] データセット構築")
    train_dataset = VideoDataset(train_csv, num_frames=num_frames, img_size=img_size, split="train")
    val_dataset   = VideoDataset(val_csv,   num_frames=num_frames, img_size=img_size, split="val")

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    print(f"  訓練サンプル: {len(train_dataset)}")
    print(f"  検証サンプル: {len(val_dataset)}")

    # ========================================
    # エンコーダ構築と事前学習済み重みのロード
    # ========================================
    print("\n[2] エンコーダ構築")
    from encoder import VisionTransformer
    encoder = VisionTransformer(
        img_size=img_size,
        patch_size=16,
        num_frames=num_frames,
        tubelet_size=2,
        embed_dim=embed_dim,
        depth={"vit_large": 24, "vit_gigantic": 48}.get(model_name, 24),
        num_heads={"vit_large": 16, "vit_gigantic": 26}.get(model_name, 16),
        out_layers=None,  # 最終層のみ使用
    )

    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"  チェックポイント読み込み: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        # エンコーダの重みを取り出す (DDP対応)
        state_dict = ckpt.get("encoder", ckpt)
        if any(k.startswith("module.") for k in state_dict.keys()):
            state_dict = {k[len("module."):]: v for k, v in state_dict.items()}
        encoder.load_state_dict(state_dict, strict=False)
        print("  エンコーダの重みをロード完了")
    else:
        print("  チェックポイントなし: ランダム初期化")

    # ========================================
    # 分類モデル構築
    # ========================================
    print("\n[3] VideoClassifier 構築")
    model = VideoClassifier(
        encoder=encoder,
        embed_dim=embed_dim,
        num_classes=num_classes,
        pooler_depth=pooler_depth,
        pooler_num_heads=16 if embed_dim == 1024 else 26,
        dropout=dropout,
        freeze_encoder=freeze_encoder,
    ).to(device)

    # 学習可能パラメータ数
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params     = sum(p.numel() for p in model.parameters())
    print(f"  全パラメータ: {total_params / 1e6:.1f}M")
    print(f"  学習可能パラメータ: {trainable_params / 1e6:.1f}M")

    # ========================================
    # 最適化・スケジューラ設定
    # ========================================
    print("\n[4] 最適化設定")
    # エンコーダは凍結 → poolerとclassifierのみ最適化
    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, weight_decay=weight_decay, betas=(0.9, 0.999),
    )

    # Warmup + Cosine Annealing
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return float(epoch + 1) / max(1, warmup_epochs)
        progress = float(epoch - warmup_epochs) / max(1, num_epochs - warmup_epochs)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler    = torch.cuda.amp.GradScaler()

    print(f"  学習率: {lr}, warmup: {warmup_epochs}epochs")

    # ========================================
    # 学習ループ
    # ========================================
    print("\n[5] 学習開始")
    best_top1 = 0.0

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}  (lr={scheduler.get_last_lr()[0]:.2e})")

        # 訓練
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device, epoch + 1
        )
        scheduler.step()

        # 検証
        if (epoch + 1) % 5 == 0 or epoch == num_epochs - 1:
            val_metrics = evaluate(model, val_loader, device)
            print(f"  [Val] top1={val_metrics['top1']:.2f}%  top5={val_metrics['top5']:.2f}%")

            # ベストモデルを保存
            if val_metrics["top1"] > best_top1:
                best_top1 = val_metrics["top1"]
                if save_path:
                    torch.save({
                        "epoch": epoch + 1,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "top1": best_top1,
                    }, save_path)
                    print(f"  ベストモデル保存: {save_path} (top1={best_top1:.2f}%)")

    print(f"\n学習完了! ベストTop-1精度: {best_top1:.2f}%")
    return model


# ============================================================
# 密予測タスクの概要 (セグメンテーション等)
# ============================================================

class DensePredictionHead(nn.Module):
    """
    密予測タスク用の軽量ヘッド (セグメンテーション等)。

    V-JEPA 2.1の Dense Features を活用するため、
    複数中間層の特徴マップを使用する。

    入力:
        feats: list of (B, N, D)  エンコーダの各中間層出力
               N = (H/patch) * (W/patch)

    出力:
        pred: (B, num_classes, H, W)  密予測マップ
    """

    def __init__(
        self,
        embed_dim: int = 1024,
        num_classes: int = 150,    # ADE20K
        img_size: int = 384,
        patch_size: int = 16,
        num_levels: int = 4,       # 使用する中間層数
    ):
        super().__init__()
        self.patch_size = patch_size
        self.img_size = img_size
        self.grid_size = img_size // patch_size   # 例: 384//16 = 24

        # 複数レベルを1次元に投影して融合
        self.proj = nn.Linear(embed_dim * num_levels, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

        # アップサンプリング用の畳み込み
        self.decode_head = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, embed_dim // 2, kernel_size=2, stride=2),
            nn.GroupNorm(16, embed_dim // 2),
            nn.GELU(),
            nn.ConvTranspose2d(embed_dim // 2, embed_dim // 4, kernel_size=2, stride=2),
            nn.GroupNorm(8, embed_dim // 4),
            nn.GELU(),
            nn.Conv2d(embed_dim // 4, num_classes, kernel_size=1),
        )

    def forward(self, feats: list) -> torch.Tensor:
        """
        入力:
            feats: list of (B, N, D)  各中間層の出力
                   N = grid_size * grid_size

        出力:
            pred: (B, num_classes, H_out, W_out)
                  H_out = W_out = grid_size * 4 (ConvTranspose2d × 2 で4倍)
        """
        B = feats[0].shape[0]
        N = self.grid_size * self.grid_size

        # ========================================
        # 複数レベルの特徴を連結・投影
        # ========================================
        feat_concat = torch.cat(feats, dim=-1)  # (B, N, D*K)
        feat_fused = self.proj(feat_concat)      # (B, N, D)
        feat_fused = self.norm(feat_fused)

        # ========================================
        # 2D特徴マップに reshape
        # ========================================
        # (B, N, D) → (B, D, grid_size, grid_size)
        feat_2d = feat_fused.permute(0, 2, 1)           # (B, D, N)
        feat_2d = feat_2d.reshape(B, -1, self.grid_size, self.grid_size)
        # feat_2d: (B, D, H_grid, W_grid) = (B, D, 24, 24) for 384px

        # ========================================
        # デコードヘッドでアップサンプリング
        # ConvTranspose2d × 2 で 24 → 48 → 96
        # ========================================
        pred = self.decode_head(feat_2d)  # (B, num_classes, H_grid*4, W_grid*4)
        return pred


# ============================================================
# 動作確認 example
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("V-JEPA 2.1 ファインチューニング 動作確認")
    print("=" * 60)

    device = torch.device("cpu")

    # ========================================
    # [1] VideoDataset の動作確認
    # ========================================
    print("\n[1] VideoDataset")
    dataset = VideoDataset("non_existent.csv", num_frames=16, img_size=224, split="train")
    frames, label = dataset[0]
    print(f"  frames: {frames.shape}  (3, T, H, W)")
    print(f"  label:  {label}")
    assert frames.shape == (3, 16, 224, 224)

    loader = DataLoader(dataset, batch_size=4, shuffle=True)
    batch_frames, batch_labels = next(iter(loader))
    print(f"  batch frames: {batch_frames.shape}  labels: {batch_labels.shape}")
    assert batch_frames.shape == (4, 3, 16, 224, 224)

    # ========================================
    # [2] VideoClassifier の動作確認
    # ========================================
    print("\n[2] VideoClassifier (Frozen Encoder)")
    from encoder import VisionTransformer
    encoder = VisionTransformer(
        img_size=224, patch_size=16, num_frames=16, tubelet_size=2,
        embed_dim=1024, depth=4, num_heads=16, out_layers=None,
    )
    classifier = VideoClassifier(
        encoder=encoder,
        embed_dim=1024,
        num_classes=400,
        pooler_depth=2,
        freeze_encoder=True,
    )
    classifier.eval()

    x = torch.randn(2, 3, 16, 224, 224)
    with torch.no_grad():
        logits = classifier(x)
    print(f"  入力:    {x.shape}")
    print(f"  logits:  {logits.shape}")
    assert logits.shape == (2, 400)

    # ========================================
    # [3] MultiClipVideoClassifier
    # ========================================
    print("\n[3] MultiClipVideoClassifier (推論時アンサンブル)")
    multi_clf = MultiClipVideoClassifier(classifier)
    n_clips = 3
    clips = torch.randn(2, n_clips, 3, 16, 224, 224)
    with torch.no_grad():
        logits_multi = multi_clf(clips)
    print(f"  入力:   {clips.shape}")
    print(f"  logits: {logits_multi.shape}")
    assert logits_multi.shape == (2, 400)

    # ========================================
    # [4] Top-K 精度計算
    # ========================================
    print("\n[4] Top-K 精度")
    pred = torch.randn(8, 400)
    target = torch.randint(0, 400, (8,))
    acc1, acc5 = topk_accuracy(pred, target, topk=(1, 5))
    print(f"  Top-1: {acc1:.1f}%  Top-5: {acc5:.1f}%")

    # ========================================
    # [5] Dense Prediction Head (セグメンテーション)
    # ========================================
    print("\n[5] DensePredictionHead (セグメンテーション)")
    encoder_seg = VisionTransformer(
        img_size=384, patch_size=16, num_frames=1, tubelet_size=1,
        embed_dim=1024, depth=4, num_heads=16,
        out_layers=[1, 2, 3],  # 3中間層 + 最終層 = 4レベル
    )
    dense_head = DensePredictionHead(
        embed_dim=1024, num_classes=150, img_size=384, patch_size=16, num_levels=4
    )
    encoder_seg.eval()
    dense_head.eval()

    x_img = torch.randn(2, 3, 384, 384)
    with torch.no_grad():
        feats = encoder_seg(x_img)  # list of (B, N, D)
        print(f"  エンコーダ出力: {len(feats)} levels, 各 {feats[0].shape}")
        pred_seg = dense_head(feats)
    print(f"  セグメンテーション出力: {pred_seg.shape}  (B, classes, H, W)")
    # grid=24, ConvTranspose2d×2 → 24*4=96
    assert pred_seg.shape[1] == 150

    # ========================================
    # [6] ファインチューニングの概要実行
    # ========================================
    print("\n[6] ファインチューニング (ダミーデータで2エポック)")
    model = finetune_video_classification(
        train_csv="non_existent_train.csv",
        val_csv="non_existent_val.csv",
        num_classes=10,
        num_frames=16,
        img_size=64,    # 小さいサイズでテスト
        batch_size=4,
        num_workers=0,
        checkpoint_path=None,
        model_name="vit_large",
        embed_dim=1024,
        pooler_depth=2,
        freeze_encoder=True,
        num_epochs=2,
        lr=1e-3,
        warmup_epochs=1,
        device_str="cpu",
        save_path=None,
    )
    print("  ファインチューニング完了 ✓")

    print("\n全テスト通過!")
