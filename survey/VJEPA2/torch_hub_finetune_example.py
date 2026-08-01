"""
V-JEPA 2 / V-JEPA 2.1 torch.hub 本物のEncoderを使った分類ファインチューニング
================================================================================

finetune_example.py は encoder.py のローカル疑似実装 VisionTransformer +
手動チェックポイント読み込みを使っていたが、本スクリプトはそれを
torch_hub_predict_demo.py と同じ torch.hub.load による「本物の学習済み重み」
に差し替えたもの。基本構成(AttentivePooler + Classifier で frozen encoder
の上に軽量ヘッドを学習する)は finetune_example.py と同じ。

finetune_example.py からの変更点:
  1. `from encoder import VisionTransformer` + 手動 VisionTransformer(...) 構築
     + 手動 load_state_dict(...) を丸ごと廃止し、
     `encoder, _ = torch.hub.load("facebookresearch/vjepa2", model_name, pretrained=...)`
     に置き換え (アーキテクチャ構築と重みロードを一括で行ってくれるため)。
     戻り値の2つ目(predictor)はSSL事前学習専用なので下流タスクでは破棄する
     ([[torch_hub_predict_demo.py]] の --mode downstream と同じ扱い)。
  2. model_name はローカル命名("vit_large"等)ではなく torch.hub のエントリ
     ポイント名("vjepa2_1_vit_base_384"等)を使う。
  3. embed_dim / img_size / num_frames を手動指定するのをやめ、
     MODEL_CONFIGS でmodel_nameから自動的に正しい値を引くようにした。
     (公式の事前学習済み重みは固定の解像度・フレーム数で学習されているため、
      ここをズラすと精度が出ない・形状エラーになる)

対応する公式実装:
  - evals/video_classification_frozen/eval.py, models.py (AttentiveClassifier)
  - src/hub/backbones.py (torch.hub エントリポイント)
"""

import os
import math
import random
import csv

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

# torch.hub のモデル名 → (埋め込み次元, Attentive Poolerのヘッド数, 学習時の解像度, フレーム数)
# 値は src/hub/backbones.py の ARCH_NAME_MAP と src/models/vision_transformer.py の
# 各 vit_* 関数のデフォルト引数(embed_dim, num_heads)に対応。
MODEL_CONFIGS = {
    "vjepa2_vit_large": dict(embed_dim=1024, num_heads=16, img_size=256, num_frames=64),
    "vjepa2_vit_huge": dict(embed_dim=1280, num_heads=16, img_size=256, num_frames=64),
    "vjepa2_vit_giant": dict(embed_dim=1408, num_heads=22, img_size=256, num_frames=64),
    "vjepa2_vit_giant_384": dict(embed_dim=1408, num_heads=22, img_size=384, num_frames=64),
    "vjepa2_1_vit_base_384": dict(embed_dim=768, num_heads=12, img_size=384, num_frames=64),
    "vjepa2_1_vit_large_384": dict(embed_dim=1024, num_heads=16, img_size=384, num_frames=64),
    "vjepa2_1_vit_giant_384": dict(embed_dim=1408, num_heads=22, img_size=384, num_frames=64),
    "vjepa2_1_vit_gigantic_384": dict(embed_dim=1664, num_heads=26, img_size=384, num_frames=64),
}


# ============================================================
# データセット (finetune_example.py の VideoDataset と同一)
# ============================================================

class VideoDataset(Dataset):
    """
    動画分類用データセット。

    CSVファイルフォーマット:
        video_path,label
        /data/videos/train/cat_001.mp4,0
        ...

    簡略化のため、実際の動画読み込みは torch.randn で代替。
    実環境では torchvision.io.read_video や decord を使用すること。
    img_size / num_frames は MODEL_CONFIGS[model_name] の値と一致させる
    必要がある (torch.hub の事前学習済み重みは固定解像度・フレーム数のため)。

    出力 (各サンプル):
        frames: (3, T, H, W)  float32, ImageNet正規化済み
        label:  int
    """

    def __init__(
        self,
        csv_path: str,
        num_frames: int,
        img_size: int,
        split: str = "train",
    ):
        self.num_frames = num_frames
        self.img_size = img_size
        self.is_train = (split == "train")

        self.samples = []
        if os.path.exists(csv_path):
            with open(csv_path, "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    self.samples.append((row["video_path"], int(row["label"])))
        else:
            print(f"[Warning] CSV not found: {csv_path}. Using dummy data.")
            self.samples = [("/dummy/video_{:04d}.mp4".format(i), i % 10) for i in range(20)]

        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1, 1)

    def _load_video(self, video_path: str) -> torch.Tensor:
        """
        ダミー実装: ランダムフレームを生成 (3, T, H, W) float32 [0, 1]

        実環境では:
            import decord
            vr = decord.VideoReader(video_path)
            frame_ids = np.linspace(0, len(vr)-1, self.num_frames, dtype=int)
            frames = vr.get_batch(frame_ids).asnumpy()  # (T, H, W, 3)
            frames = torch.from_numpy(frames).permute(3, 0, 1, 2).float() / 255.0
        """
        return torch.rand(3, self.num_frames, self.img_size, self.img_size)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        video_path, label = self.samples[idx]
        frames = self._load_video(video_path)
        if self.is_train and random.random() < 0.5:
            frames = torch.flip(frames, dims=[3])  # ランダム水平反転
        frames = (frames - self.mean) / (self.std + 1e-8)
        return frames, label


# ============================================================
# Attentive Pooler (V-JEPA 2/2.1 の推奨ヘッド)
# ============================================================

class AttentivePooler(nn.Module):
    """
    Attentive Pooler: 可変長トークン列をfixedサイズの表現に集約。
    Cross-Attentionによってトークン列から重要な情報を選択的に集約する。

    入力:  x: (B, N, D)  エンコーダのトークン出力
    出力:  pooled: (B, num_queries, D)
    """

    def __init__(self, embed_dim: int, num_heads: int, num_queries: int = 1, mlp_ratio: float = 4.0):
        super().__init__()
        self.query_tokens = nn.Parameter(torch.zeros(1, num_queries, embed_dim))
        nn.init.trunc_normal_(self.query_tokens, std=0.02)

        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm_q = nn.LayerNorm(embed_dim)
        self.norm_kv = nn.LayerNorm(embed_dim)

        hidden_dim = int(embed_dim * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        q = self.query_tokens.expand(B, -1, -1)  # (B, num_queries, D)

        q_norm = self.norm_q(q)
        kv_norm = self.norm_kv(x)
        attn_out, _ = self.cross_attn(q_norm, kv_norm, kv_norm)

        q = q + attn_out
        q = q + self.ffn(q)
        return q  # (B, num_queries, D)


# ============================================================
# 動画分類モデル (Frozen Encoder + Attentive Head)
# ============================================================

class VideoClassifier(nn.Module):
    """
    torch.hub の frozen encoder + Attentive Pooler + Classifier ヘッド。

    学習時: encoder は凍結(no_grad), pooler と classifier のみ学習。
    Encoderへの forward には masks を渡さない (下流タスクでは動画全体を
    そのまま見せる。masks_enc/masks_pred はSSL事前学習専用の概念であり
    ここでは不要 — [[torch_hub_predict_demo.py]] --mode downstream 参照)。

    入力:  x: (B, 3, T, H, W)
    出力:  logits: (B, num_classes)
    """

    def __init__(
        self,
        encoder: nn.Module,
        embed_dim: int,
        num_classes: int,
        pooler_num_heads: int,
        pooler_depth: int = 1,
        dropout: float = 0.5,
        freeze_encoder: bool = True,
    ):
        super().__init__()
        self.encoder = encoder
        self.freeze_encoder = freeze_encoder

        if freeze_encoder:
            for p in encoder.parameters():
                p.requires_grad = False

        # pooler_depthは互換性のため受け取るが、AttentivePooler自体は1層のcross-attn
        self.pooler = AttentivePooler(embed_dim=embed_dim, num_heads=pooler_num_heads, num_queries=1)

        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes),
        )
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Step 1: Frozen encoder で特徴抽出 (masks=None → 全パッチを処理)
        if self.freeze_encoder:
            with torch.no_grad():
                feats = self.encoder(x)  # (B, N, D)
        else:
            feats = self.encoder(x)

        if isinstance(feats, list):  # out_layers指定時などリストが返るケース対策
            feats = feats[-1]

        # Step 2: Attentive Pooler で集約
        pooled = self.pooler(feats).squeeze(1)  # (B, D)

        # Step 3: 分類
        return self.classifier(pooled)  # (B, num_classes)


class MultiClipVideoClassifier(nn.Module):
    """
    複数クリップのアンサンブルによる動画分類モデル (推論用)。
    入力: clips (B, n_clips, 3, T, H, W) → 出力: logits (B, num_classes)
    """

    def __init__(self, base_classifier: VideoClassifier):
        super().__init__()
        self.classifier = base_classifier

    def forward(self, clips: torch.Tensor) -> torch.Tensor:
        B, n_clips, C, T, H, W = clips.shape
        clips_flat = clips.view(B * n_clips, C, T, H, W)
        logits_flat = self.classifier(clips_flat)
        return logits_flat.view(B, n_clips, -1).mean(dim=1)


# ============================================================
# 学習ユーティリティ
# ============================================================

class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def topk_accuracy(output: torch.Tensor, target: torch.Tensor, topk=(1, 5)) -> list:
    maxk = max(topk)
    B = target.size(0)
    _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)
    pred = pred.t()
    correct = pred.eq(target.unsqueeze(0).expand_as(pred))

    results = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum()
        results.append(correct_k.mul_(100.0 / B).item())
    return results


def train_one_epoch(model, loader, criterion, optimizer, device, epoch) -> dict:
    model.train()
    if model.freeze_encoder:
        model.encoder.eval()  # 凍結エンコーダはeval mode固定

    loss_meter, top1_meter, top5_meter = AverageMeter(), AverageMeter(), AverageMeter()

    for i, (frames, labels) in enumerate(loader):
        frames = frames.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        B = frames.shape[0]

        logits = model(frames)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        acc1, acc5 = topk_accuracy(logits.detach(), labels, topk=(1, min(5, logits.shape[1])))
        loss_meter.update(loss.item(), B)
        top1_meter.update(acc1, B)
        top5_meter.update(acc5, B)

        if i % 10 == 0:
            print(f"  [{epoch}][{i}/{len(loader)}] loss={loss_meter.avg:.4f} top1={top1_meter.avg:.2f}%")

    return {"loss": loss_meter.avg, "top1": top1_meter.avg, "top5": top5_meter.avg}


@torch.no_grad()
def evaluate(model, loader, device) -> dict:
    model.eval()
    top1_meter, top5_meter = AverageMeter(), AverageMeter()

    for frames, labels in loader:
        frames = frames.to(device)
        labels = labels.to(device)
        logits = model(frames)
        acc1, acc5 = topk_accuracy(logits, labels, topk=(1, min(5, logits.shape[1])))
        top1_meter.update(acc1, frames.shape[0])
        top5_meter.update(acc5, frames.shape[0])

    return {"top1": top1_meter.avg, "top5": top5_meter.avg}


# ============================================================
# メイン: ファインチューニング実行
# ============================================================

def finetune_video_classification(
    # データ設定
    train_csv: str = "train.csv",
    val_csv: str = "val.csv",
    num_classes: int = 400,
    batch_size: int = 8,
    num_workers: int = 0,
    # モデル設定
    model_name: str = "vjepa2_1_vit_base_384",  # torch.hub のエントリポイント名 (MODEL_CONFIGSのキー)
    pretrained: bool = True,                     # torch.hubから本物の学習済み重みをダウンロードするか
    pooler_depth: int = 1,
    freeze_encoder: bool = True,
    # 学習設定
    num_epochs: int = 10,
    lr: float = 1e-4,
    weight_decay: float = 0.01,
    warmup_epochs: int = 1,
    dropout: float = 0.5,
    # その他
    device_str: str = "cuda",
    save_path: str = "./checkpoint_finetune.pth",
):
    """
    torch.hub の本物の V-JEPA 2/2.1 encoder を frozen backbone として使う
    動画分類ファインチューニング。

    学習戦略:
      - エンコーダ: 凍結 (no_grad, masks=None で全パッチを見る)
      - Attentive Pooler + Classifier: 学習
    """
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"未対応のmodel_name: {model_name}. 選択肢: {list(MODEL_CONFIGS.keys())}")
    cfg = MODEL_CONFIGS[model_name]

    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    print(f"使用デバイス: {device}")

    # ========================================
    # データセット・ローダー構築
    # img_size/num_framesはmodel_nameの事前学習設定に自動で合わせる
    # ========================================
    print("\n[1] データセット構築")
    train_dataset = VideoDataset(train_csv, num_frames=cfg["num_frames"], img_size=cfg["img_size"], split="train")
    val_dataset = VideoDataset(val_csv, num_frames=cfg["num_frames"], img_size=cfg["img_size"], split="val")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    print(f"  訓練サンプル: {len(train_dataset)}, 検証サンプル: {len(val_dataset)}")

    # ========================================
    # エンコーダ構築 (torch.hub 本物の学習済み重み)
    # finetune_example.py の「VisionTransformer構築 + 手動load_state_dict」を
    # torch.hub.load 一発に置き換え
    # ========================================
    print("\n[2] エンコーダ構築 (torch.hub)")
    print(f"  encoder, _ = torch.hub.load('facebookresearch/vjepa2', '{model_name}', pretrained={pretrained})")
    encoder, _ = torch.hub.load("facebookresearch/vjepa2", model_name, pretrained=pretrained)
    # 2つ目の戻り値(predictor)はSSL事前学習専用のため下流タスクでは使わず破棄する
    encoder = encoder.to(device)

    # ========================================
    # 分類モデル構築
    # ========================================
    print("\n[3] VideoClassifier 構築")
    model = VideoClassifier(
        encoder=encoder,
        embed_dim=cfg["embed_dim"],
        num_classes=num_classes,
        pooler_num_heads=cfg["num_heads"],
        pooler_depth=pooler_depth,
        dropout=dropout,
        freeze_encoder=freeze_encoder,
    ).to(device)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  全パラメータ: {total_params / 1e6:.1f}M, 学習可能パラメータ: {trainable_params / 1e6:.1f}M")

    # ========================================
    # 最適化・スケジューラ設定
    # ========================================
    print("\n[4] 最適化設定")
    optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=weight_decay)

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return float(epoch + 1) / max(1, warmup_epochs)
        progress = float(epoch - warmup_epochs) / max(1, num_epochs - warmup_epochs)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # ========================================
    # 学習ループ
    # ========================================
    print("\n[5] 学習開始")
    best_top1 = 0.0

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}  (lr={scheduler.get_last_lr()[0]:.2e})")
        train_one_epoch(model, train_loader, criterion, optimizer, device, epoch + 1)
        scheduler.step()

        if (epoch + 1) % 5 == 0 or epoch == num_epochs - 1:
            val_metrics = evaluate(model, val_loader, device)
            print(f"  [Val] top1={val_metrics['top1']:.2f}%  top5={val_metrics['top5']:.2f}%")

            if val_metrics["top1"] > best_top1:
                best_top1 = val_metrics["top1"]
                if save_path:
                    torch.save({"epoch": epoch + 1, "model_state_dict": model.state_dict(), "top1": best_top1}, save_path)
                    print(f"  ベストモデル保存: {save_path} (top1={best_top1:.2f}%)")

    print(f"\n学習完了! ベストTop-1精度: {best_top1:.2f}%")
    return model


if __name__ == "__main__":
    # 動作確認 (ダミーデータ + pretrained=False でランダム初期化のまま2エポック学習)
    # 実行例:
    #   python -m survey.VJEPA2.torch_hub_finetune_example
    finetune_video_classification(
        train_csv="non_existent_train.csv",
        val_csv="non_existent_val.csv",
        num_classes=10,
        batch_size=2,
        num_workers=0,
        model_name="vjepa2_1_vit_base_384",
        pretrained=False,
        pooler_depth=1,
        freeze_encoder=True,
        num_epochs=2,
        lr=1e-3,
        warmup_epochs=1,
        device_str="cpu",
        save_path=None,
    )
    print("\n全テスト通過!")
