"""
VisionHOPE 画像分類ファインチューニング (Trainer不使用)
================================================================

画像パス+ラベルのペアデータセットが pandas DataFrame として渡される想定の、
PyTorch基本機能のみを使った手動訓練ループです。transformers/timmのTrainerは
使用しません (survey/Qwen3ASR/finetuning_example.py と同じ方針)。

公式実装参考: gh/VisionHOPE/train/classification.py, scripts/train/visionhope_*.sh

============================================================
DataFrameの期待するカラム
============================================================
必須:
  - image_path (str): 画像ファイルの絶対/相対パス
  - label       (int): クラスID (0 ~ num_classes-1)

============================================================
使い方
============================================================
import pandas as pd
df = pd.DataFrame({
    "image_path": ["images/cat1.jpg", "images/dog1.jpg"],
    "label": [0, 1],
})
model = load_model_for_finetuning("visionhope_tiny", num_classes=2)
train(model, train_df=df, output_dir="./out")
"""

from __future__ import annotations

import argparse
import math
import os
from typing import Any, Dict, Optional

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import torchvision.transforms as T

from model import HierarchicalVisionHOPE, MODEL_CONFIGS, create_model
from loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy, mixup_cutmix

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# ============================================================
# 1. データセット定義
# ============================================================

class ImageFolderDataset(Dataset):
    """画像パス+ラベルの pandas DataFrame から読み込むデータセット。"""

    def __init__(self, df: pd.DataFrame, img_size: int = 224, data_root: str = "", train: bool = True):
        self.df = df.reset_index(drop=True)
        self.data_root = data_root
        if train:
            self.transform = T.Compose([
                T.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ])
        else:
            self.transform = T.Compose([
                T.Resize(int(img_size * 1.14)),
                T.CenterCrop(img_size),
                T.ToTensor(),
                T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ])

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        path = os.path.join(self.data_root, row["image_path"])
        image = Image.open(path).convert("RGB")
        return {"image": self.transform(image), "label": int(row["label"])}


def collate_batch(features):
    images = torch.stack([f["image"] for f in features])        # (B, 3, H, W)
    labels = torch.tensor([f["label"] for f in features], dtype=torch.long)  # (B,)
    return {"images": images, "labels": labels}


# ============================================================
# 2. モデルロード
# ============================================================

def load_model_for_finetuning(
    model_name: str = "visionhope_tiny",
    num_classes: int = 1000,
    checkpoint_path: Optional[str] = None,
    freeze_stages: int = 0,
) -> HierarchicalVisionHOPE:
    """VisionHOPEモデルをロードし、必要なら事前学習済み重みを読み込む。

    ``freeze_stages`` で先頭からそのステージ数だけ凍結できる (転移学習向け)。
    """
    model = create_model(model_name, num_classes=num_classes)

    if checkpoint_path:
        state = torch.load(checkpoint_path, map_location="cpu")
        state = state.get("model", state)
        # 分類ヘッドのクラス数が異なる場合は読み込みをスキップする。
        own_state = model.state_dict()
        filtered = {
            k: v for k, v in state.items()
            if k in own_state and own_state[k].shape == v.shape
        }
        missing = set(own_state) - set(filtered)
        model.load_state_dict(filtered, strict=False)
        print(f"チェックポイントから{len(filtered)}/{len(own_state)}テンソルを読み込み "
              f"({len(missing)}個はshape不一致または新規のためランダム初期化を維持)")

    if freeze_stages > 0:
        for i in range(min(freeze_stages, 4)):
            for p in model.downsample_layers[i].parameters():
                p.requires_grad = False
            for p in model.stages[i].parameters():
                p.requires_grad = False
        n_frozen = sum(1 for p in model.parameters() if not p.requires_grad)
        print(f"先頭{freeze_stages}ステージを凍結 ({n_frozen}テンソル)")

    return model


# ============================================================
# 3. 学習率スケジューラ (公式 train/learning_rate.py のcosine decayに準拠)
# ============================================================

def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps: int, num_training_steps: int):
    def lr_lambda(step: int) -> float:
        if step < num_warmup_steps:
            return float(step) / float(max(1, num_warmup_steps))
        progress = float(step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ============================================================
# 4. メイン訓練ループ
# ============================================================

def train(
    model: nn.Module,
    train_df: pd.DataFrame,
    eval_df: Optional[pd.DataFrame] = None,
    output_dir: str = "./visionhope-finetuned",
    epochs: int = 3,
    batch_size: int = 8,
    grad_acc: int = 1,
    lr: float = 5e-4,
    weight_decay: float = 0.05,
    warmup_ratio: float = 0.05,
    smoothing: float = 0.1,
    use_mixup_cutmix: bool = False,
    num_classes: int = 1000,
    img_size: int = 224,
    data_root: str = "",
    max_grad_norm: float = 1.0,
    log_steps: int = 5,
    num_workers: int = 2,
):
    """手動訓練ループ (Trainer不使用)。

    ========================================
    各バッチのShape
    ========================================
    images: (B, 3, img_size, img_size)
    labels: (B,)
    logits (出力): (B, num_classes)
    loss (出力):   scalar
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    compute_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    use_amp = compute_dtype == torch.bfloat16

    train_dataset = ImageFolderDataset(train_df, img_size=img_size, data_root=data_root, train=True)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, collate_fn=collate_batch,
    )
    eval_loader = None
    if eval_df is not None:
        eval_dataset = ImageFolderDataset(eval_df, img_size=img_size, data_root=data_root, train=False)
        eval_loader = DataLoader(
            eval_dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, collate_fn=collate_batch,
        )

    model = model.to(device)
    model.train()
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay, betas=(0.9, 0.999))

    steps_per_epoch = math.ceil(len(train_loader) / grad_acc)
    total_steps = max(1, steps_per_epoch * epochs)
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    ce_loss = LabelSmoothingCrossEntropy(smoothing) if smoothing > 0 else nn.CrossEntropyLoss()
    soft_loss = SoftTargetCrossEntropy()

    os.makedirs(output_dir, exist_ok=True)
    print("訓練設定:")
    print(f"  サンプル数: {len(train_dataset)}  バッチサイズ: {batch_size}  勾配累積: {grad_acc}")
    print(f"  総ステップ数: {total_steps}  Warmupステップ: {warmup_steps}  lr: {lr}")
    print(f"  訓練可能パラメータ: {sum(p.numel() for p in trainable_params):,}")

    global_step = 0
    accumulated_loss = 0.0

    for epoch in range(epochs):
        print(f"\n=== Epoch {epoch + 1}/{epochs} ===")
        model.train()
        for batch_idx, batch in enumerate(train_loader):
            images = batch["images"].to(device)
            labels = batch["labels"].to(device)

            if use_mixup_cutmix:
                images, soft_targets = mixup_cutmix(images, labels, num_classes=num_classes)
                soft_targets = soft_targets.to(device)

            with torch.autocast(device_type=device.type, dtype=compute_dtype, enabled=use_amp):
                logits = model(images)  # (B, num_classes)
                if use_mixup_cutmix:
                    loss = soft_loss(logits, soft_targets)
                else:
                    loss = ce_loss(logits, labels)
                loss = loss / grad_acc

            loss.backward()
            accumulated_loss += loss.item()

            if (batch_idx + 1) % grad_acc == 0 or (batch_idx + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(trainable_params, max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % log_steps == 0:
                    print(f"  Step {global_step}/{total_steps} | loss={accumulated_loss / log_steps:.4f} "
                          f"| lr={scheduler.get_last_lr()[0]:.2e}")
                    accumulated_loss = 0.0

        if eval_loader is not None:
            acc = evaluate(model, eval_loader, device)
            print(f"  Epoch {epoch + 1} eval top-1 accuracy: {acc:.2%}")

    print(f"\nTraining complete! (このデモの重みは保存せず、ログのみ表示)")


def evaluate(model: nn.Module, eval_loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch in eval_loader:
            images = batch["images"].to(device)
            labels = batch["labels"].to(device)
            logits = model(images)
            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.shape[0]
    model.train()
    return correct / max(1, total)


# ============================================================
# 5. 合成データでの end-to-end 動作確認
# ============================================================

def _make_synthetic_dataset(tmp_dir: str, n_per_class: int = 6, num_classes: int = 3, size=(96, 96)) -> pd.DataFrame:
    """単色+ノイズの合成画像を実際にディスクへ書き出し、本物のファイルI/Oで検証する。"""
    import numpy as np

    os.makedirs(tmp_dir, exist_ok=True)
    rows = []
    for c in range(num_classes):
        base_color = [50 + c * 80, 255 - c * 60, 30 + c * 40]
        for i in range(n_per_class):
            arr = (
                np.array(base_color, dtype=np.float32)[None, None, :]
                + np.random.randn(size[1], size[0], 3) * 15
            ).clip(0, 255).astype("uint8")
            path = os.path.join(tmp_dir, f"class{c}_{i}.png")
            Image.fromarray(arr).save(path)
            rows.append({"image_path": path, "label": c})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser("VisionHOPE Fine-tuning (No Trainer)")
    parser.add_argument("--train_file", type=str, default="",
                        help="CSV/JSON with columns: image_path, label")
    parser.add_argument("--model", type=str, default="visionhope_tiny", choices=tuple(MODEL_CONFIGS))
    parser.add_argument("--num_classes", type=int, default=3)
    # 32x32: 本物のSRNL再帰(逐次python loop)がCPUでも数秒で終わる最小サイズ。
    # 実運用は224x224 (スクリプト先頭のdocstring参照); 大きな画像サイズでは
    # 本参照実装の逐次ループが遅いので、速度が必要な場合は公式CUDA拡張を使うこと。
    parser.add_argument("--img_size", type=int, default=32)
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--output_dir", type=str, default="./visionhope-finetuned")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-4)
    args = parser.parse_args()

    if args.train_file:
        train_df = pd.read_csv(args.train_file) if args.train_file.endswith(".csv") else pd.read_json(args.train_file)
    else:
        print("--train_file 未指定のため、合成画像データセットで動作確認します。\n")
        train_df = _make_synthetic_dataset(
            "/tmp/visionhope_finetune_demo", num_classes=args.num_classes,
            size=(args.img_size, args.img_size),
        )

    # CPU/短時間での確認用に層数を削減した極小構成 (depths削減以外は公式構成を継承)。
    cfg = dict(MODEL_CONFIGS[args.model])
    cfg["depths"] = tuple(min(d, 1) for d in cfg["depths"])
    model = HierarchicalVisionHOPE(num_classes=args.num_classes, head_dim=16, **cfg)

    if args.checkpoint:
        state = torch.load(args.checkpoint, map_location="cpu")
        model.load_state_dict(state.get("model", state), strict=False)

    train(
        model, train_df, output_dir=args.output_dir, epochs=args.epochs,
        batch_size=args.batch_size, lr=args.lr, num_classes=args.num_classes,
        img_size=args.img_size, use_mixup_cutmix=False, num_workers=0,
    )


if __name__ == "__main__":
    main()
