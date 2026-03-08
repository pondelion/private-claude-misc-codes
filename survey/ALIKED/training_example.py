"""
ALIKED Training Example - 簡略化学習サンプル
===========================================

HomographicDatasetを使ったALIKEDの学習例
CPU環境でも動作するように調整済み
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
from typing import Dict
import time

from main_flow import ALIKED
from training_data import HomographicDataset, warp_keypoints_homography
from loss_computation import ALIKEDLossWrapper


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: ALIKEDLossWrapper,
    optimizer: optim.Optimizer,
    device: str,
    epoch: int
) -> Dict[str, float]:
    """
    1エポック分の学習

    Args:
        model: ALIKEDモデル
        dataloader: データローダー
        criterion: 損失関数
        optimizer: オプティマイザー
        device: デバイス
        epoch: エポック番号

    Returns:
        平均損失の辞書
    """
    model.train()

    losses_sum = {
        'total_loss': 0.0,
        'loss_rp': 0.0,
        'loss_pk': 0.0,
        'loss_ds': 0.0,
        'loss_re': 0.0
    }

    num_batches = len(dataloader)

    for batch_idx, batch in enumerate(dataloader):
        # データをデバイスに転送
        image_a = batch['image_a'].to(device)
        image_b = batch['image_b'].to(device)
        H_ab = batch['H_ab'].to(device)
        valid_mask = batch['valid_mask'].to(device)

        # フォワードパス
        outputs_a = model(image_a, top_k=300, scores_th=0.2)
        outputs_b = model(image_b, top_k=300, scores_th=0.2)

        # 損失計算
        losses = criterion(
            outputs_a=outputs_a,
            outputs_b=outputs_b,
            homography_ab=H_ab
        )

        total_loss = losses['total_loss']

        # バックプロパゲーション
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # 損失を累積
        for key in losses_sum.keys():
            losses_sum[key] += losses[key].item()

        # 進捗表示
        if (batch_idx + 1) % 10 == 0:
            avg_loss = losses_sum['total_loss'] / (batch_idx + 1)
            print(f"  Batch [{batch_idx + 1}/{num_batches}] "
                  f"Loss: {total_loss.item():.4f} "
                  f"(Avg: {avg_loss:.4f})")

    # 平均損失を計算
    losses_avg = {k: v / num_batches for k, v in losses_sum.items()}

    return losses_avg


def validate_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: ALIKEDLossWrapper,
    device: str
) -> Dict[str, float]:
    """
    1エポック分の検証

    Args:
        model: ALIKEDモデル
        dataloader: データローダー
        criterion: 損失関数
        device: デバイス

    Returns:
        平均損失の辞書
    """
    model.eval()

    losses_sum = {
        'total_loss': 0.0,
        'loss_rp': 0.0,
        'loss_pk': 0.0,
        'loss_ds': 0.0,
        'loss_re': 0.0
    }

    num_batches = len(dataloader)

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            # データをデバイスに転送
            image_a = batch['image_a'].to(device)
            image_b = batch['image_b'].to(device)
            H_ab = batch['H_ab'].to(device)

            # フォワードパス
            outputs_a = model(image_a, top_k=300, scores_th=0.2)
            outputs_b = model(image_b, top_k=300, scores_th=0.2)

            # 損失計算
            losses = criterion(
                outputs_a=outputs_a,
                outputs_b=outputs_b,
                homography_ab=H_ab
            )

            # 損失を累積
            for key in losses_sum.keys():
                losses_sum[key] += losses[key].item()

    # 平均損失を計算
    losses_avg = {k: v / num_batches for k, v in losses_sum.items()}

    return losses_avg


def train_aliked(
    train_image_dir: str,
    val_image_dir: str = None,
    num_epochs: int = 10,
    batch_size: int = 2,
    learning_rate: float = 1e-4,
    device: str = 'cpu',
    save_dir: str = './checkpoints'
):
    """
    ALIKED学習のメイン関数

    Args:
        train_image_dir: 学習用画像ディレクトリ
        val_image_dir: 検証用画像ディレクトリ（Noneの場合は学習データから分割）
        num_epochs: エポック数
        batch_size: バッチサイズ
        learning_rate: 学習率
        device: 'cpu' or 'cuda'
        save_dir: チェックポイント保存先
    """
    print("=" * 60)
    print("ALIKED Training Example")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Epochs: {num_epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print()

    # 保存ディレクトリ作成
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    # データセット作成
    print("Creating datasets...")
    train_dataset = HomographicDataset(image_dir=train_image_dir)

    if val_image_dir is not None:
        val_dataset = HomographicDataset(image_dir=val_image_dir)
    else:
        # 学習データの一部を検証用に分割
        total_size = len(train_dataset)
        val_size = int(total_size * 0.1)
        train_size = total_size - val_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            train_dataset, [train_size, val_size]
        )

    # データローダー作成
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0  # CPUで動作させる場合は0
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print()

    # モデル作成
    print("Creating model...")
    model = ALIKED(
        c1=16, c2=32, c3=64, c4=128,
        dim=128,
        M=16,
        K=3
    )
    model.to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    print()

    # 損失関数
    criterion = ALIKEDLossWrapper(
        w_rp=1.0,
        w_pk=0.5,
        w_ds=5.0,
        w_re=1.0
    )

    # オプティマイザー
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 学習率スケジューラー
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    # 学習ループ
    print("Starting training...")
    print("=" * 60)

    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        epoch_start_time = time.time()

        print(f"\nEpoch [{epoch + 1}/{num_epochs}]")
        print("-" * 60)

        # 学習
        train_losses = train_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epoch=epoch
        )

        # 検証
        val_losses = validate_epoch(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device
        )

        # 学習率更新
        scheduler.step()

        epoch_time = time.time() - epoch_start_time

        # 結果表示
        print(f"\n  Train Loss: {train_losses['total_loss']:.4f} "
              f"(RP: {train_losses['loss_rp']:.4f}, "
              f"PK: {train_losses['loss_pk']:.4f}, "
              f"DS: {train_losses['loss_ds']:.4f}, "
              f"RE: {train_losses['loss_re']:.4f})")

        print(f"  Val Loss:   {val_losses['total_loss']:.4f} "
              f"(RP: {val_losses['loss_rp']:.4f}, "
              f"PK: {val_losses['loss_pk']:.4f}, "
              f"DS: {val_losses['loss_ds']:.4f}, "
              f"RE: {val_losses['loss_re']:.4f})")

        print(f"  Time: {epoch_time:.2f}s")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")

        # ベストモデル保存
        if val_losses['total_loss'] < best_val_loss:
            best_val_loss = val_losses['total_loss']
            checkpoint_path = save_path / 'best_model.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
            }, checkpoint_path)
            print(f"  → Best model saved! (Val loss: {best_val_loss:.4f})")

        # 定期的にチェックポイント保存
        if (epoch + 1) % 5 == 0:
            checkpoint_path = save_path / f'checkpoint_epoch_{epoch + 1}.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_losses['total_loss'],
            }, checkpoint_path)
            print(f"  → Checkpoint saved: {checkpoint_path.name}")

    print("\n" + "=" * 60)
    print("Training completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    # 実際の使用例
    # train_image_dir には R2D2 Homographic dataset のパス (oxford/paris/aachen) を指定
    # 例: train_aliked(train_image_dir='/path/to/oxford', num_epochs=10)

    # テスト用（画像ディレクトリが存在しない場合はエラーになります）
    import sys
    if len(sys.argv) > 1:
        train_image_dir = sys.argv[1]
        train_aliked(
            train_image_dir=train_image_dir,
            num_epochs=10,
            batch_size=2,
            learning_rate=1e-4,
            device='cpu',
            save_dir='./checkpoints'
        )
    else:
        print("Usage: python training_example.py <train_image_dir>")
        print("\nExample:")
        print("  python training_example.py /path/to/oxford")
        print("\nNote:")
        print("  train_image_dir should contain .jpg images")
        print("  (e.g., R2D2 Homographic dataset: oxford/paris/aachen)")
