"""
MOTIP Training Data Format

MOTIPは複数のMOTデータセットで訓練可能:
- DanceTrack: ダンス動画 (100本)
- SportsMOT: スポーツ放送 (240本)
- BFT: 鳥類トラッキング (106本)
- CrowdHuman: 混雑シーン (静止画、Data Augmentation用)

【データセット構造】
各データセットは以下の情報を提供:
1. 画像パス
2. アノテーション (バウンディングボックス + IDラベル)
3. シーケンス情報 (幅、高さ、長さ)

【アノテーションフォーマット (フレームごと)】
{
    'id': tensor([id1, id2, ...]),              # (M,) - 物体ID
    'category': tensor([cat1, cat2, ...]),      # (M,) - クラスラベル (常に1: 人物)
    'bbox': tensor([[x, y, w, h], ...]),        # (M, 4) - バウンディングボックス
    'visibility': tensor([0.5, 1.0, ...]),      # (M,) - 可視性スコア
    'is_legal': bool,                            # フレームの有効性フラグ
}

【訓練用軌跡フォーマット】
訓練時には、連続フレームから軌跡アノテーションを構築:
{
    'trajectory_id_labels': (G, 1, N),          # グループ化された軌跡ID
    'trajectory_id_masks': (G, 1, N),           # パディングマスク
    'trajectory_times': (G, 1, N),              # タイムスタンプ
    'unknown_id_labels': (G, 1, N),             # 現在フレームのID (教師信号)
    'unknown_id_masks': (G, 1, N),              # 現在フレームのマスク
}

G: 軌跡グループ数 (可変長履歴を効率的にバッチ処理)
N: フレームあたりの最大物体数 (パディング済み)

【Data Augmentation】
1. MultiSimulate: 静止画から動画を生成 (shift + rotation)
2. MultiRandomHorizontalFlip: 水平反転
3. MultiRandomResizedCrop: ランダムクロップ
4. MultiRandomColorJitter: 色調整
5. Trajectory Occlusion: 軌跡の一部をランダムにマスク (prob=0.5)
6. Trajectory Switching: 軌跡IDをランダムに入れ替え (prob=0.5)
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import json


class OneDataset:
    """
    MOTデータセットの基底クラス

    【共通インターフェース】
    - get_sequence_infos(): シーケンス情報
    - get_image_paths(): 画像パスリスト
    - get_annotations(): フレームごとのアノテーション
    """

    def __init__(self, root: str, split: str = 'train'):
        self.root = Path(root)
        self.split = split

    def get_sequence_infos(self) -> Dict[str, Dict]:
        """
        シーケンス情報を取得

        Returns:
            {
                'seq_name': {
                    'width': int,
                    'height': int,
                    'length': int,
                    'is_static': bool,  # 静止画か動画か
                },
                ...
            }
        """
        raise NotImplementedError

    def get_image_paths(self) -> Dict[str, List[str]]:
        """
        画像パスを取得

        Returns:
            {
                'seq_name': [path1, path2, ...],
                ...
            }
        """
        raise NotImplementedError

    def get_annotations(self) -> Dict[str, List[Dict]]:
        """
        アノテーションを取得

        Returns:
            {
                'seq_name': [
                    {  # Frame 0
                        'id': tensor([id1, id2, ...]),
                        'category': tensor([1, 1, ...]),
                        'bbox': tensor([[x, y, w, h], ...]),
                        'visibility': tensor([0.5, 1.0, ...]),
                        'is_legal': True,
                    },
                    ...
                ],
                ...
            }
        """
        raise NotImplementedError


class DanceTrackDataset(OneDataset):
    """
    DanceTrack: ダンス動画のMOTデータセット

    【特徴】
    - 頻繁なオクルージョン
    - 不規則な動き
    - 類似した外観 (同じ衣装のダンサー)

    【データ構造】
    dancetrack/
    ├── train/
    │   ├── dancetrack0001/
    │   │   ├── img1/
    │   │   │   ├── 00000001.jpg
    │   │   │   └── ...
    │   │   └── gt/
    │   │       └── gt.txt
    │   └── ...
    └── test/
        └── ...

    【gt.txtフォーマット】
    <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>

    例:
    1,1,100,200,50,100,1,-1,-1,-1
    1,2,300,400,60,120,1,-1,-1,-1
    2,1,105,205,50,100,1,-1,-1,-1
    ...
    """

    def __init__(self, root: str, split: str = 'train'):
        super().__init__(root, split)
        self.seq_dir = self.root / split

    def get_sequence_infos(self) -> Dict[str, Dict]:
        seq_infos = {}

        for seq_path in sorted(self.seq_dir.iterdir()):
            if not seq_path.is_dir():
                continue

            seq_name = seq_path.name

            # seqinfo.iniから情報を読み取り (簡略化)
            img_dir = seq_path / 'img1'
            img_files = sorted(img_dir.glob('*.jpg'))

            if len(img_files) > 0:
                # 最初の画像から解像度を取得 (実際はPILで読み込み)
                seq_infos[seq_name] = {
                    'width': 1920,  # 仮の値
                    'height': 1080,
                    'length': len(img_files),
                    'is_static': False,
                }

        return seq_infos

    def get_image_paths(self) -> Dict[str, List[str]]:
        image_paths = {}

        for seq_path in sorted(self.seq_dir.iterdir()):
            if not seq_path.is_dir():
                continue

            seq_name = seq_path.name
            img_dir = seq_path / 'img1'
            img_files = sorted(img_dir.glob('*.jpg'))

            image_paths[seq_name] = [str(f) for f in img_files]

        return image_paths

    def get_annotations(self) -> Dict[str, List[Dict]]:
        annotations = {}

        for seq_path in sorted(self.seq_dir.iterdir()):
            if not seq_path.is_dir():
                continue

            seq_name = seq_path.name
            gt_file = seq_path / 'gt' / 'gt.txt'

            if not gt_file.exists():
                continue

            # gt.txtを読み込み
            # フレームごとにグループ化
            frame_anns = {}

            with open(gt_file, 'r') as f:
                for line in f:
                    parts = line.strip().split(',')
                    frame_id = int(parts[0])
                    obj_id = int(parts[1])
                    bb_left = float(parts[2])
                    bb_top = float(parts[3])
                    bb_width = float(parts[4])
                    bb_height = float(parts[5])
                    conf = float(parts[6])

                    if frame_id not in frame_anns:
                        frame_anns[frame_id] = {
                            'ids': [],
                            'bboxes': [],
                        }

                    # xywh形式
                    frame_anns[frame_id]['ids'].append(obj_id)
                    frame_anns[frame_id]['bboxes'].append([bb_left, bb_top, bb_width, bb_height])

            # フレーム順に整理
            max_frame = max(frame_anns.keys()) if frame_anns else 0
            seq_annotations = []

            for frame_id in range(1, max_frame + 1):
                if frame_id in frame_anns:
                    ann = frame_anns[frame_id]
                    seq_annotations.append({
                        'id': torch.tensor(ann['ids'], dtype=torch.long),
                        'category': torch.ones(len(ann['ids']), dtype=torch.long),
                        'bbox': torch.tensor(ann['bboxes'], dtype=torch.float32),
                        'visibility': torch.ones(len(ann['ids']), dtype=torch.float32),
                        'is_legal': True,
                    })
                else:
                    # 空フレーム
                    seq_annotations.append({
                        'id': torch.tensor([], dtype=torch.long),
                        'category': torch.tensor([], dtype=torch.long),
                        'bbox': torch.zeros(0, 4, dtype=torch.float32),
                        'visibility': torch.tensor([], dtype=torch.float32),
                        'is_legal': True,
                    })

            annotations[seq_name] = seq_annotations

        return annotations


class SportsMOTDataset(OneDataset):
    """
    SportsMOT: スポーツ放送のMOTデータセット

    【特徴】
    - 頻繁なカメラ移動
    - 高速移動
    - 繰り返されるインタラクション

    【アノテーション対象】
    - アスリートのみ (審判や観客は除外)

    データ構造はDanceTrackと同様
    """
    pass  # DanceTrackと同じ実装


class BFTDataset(OneDataset):
    """
    BFT (Bird Flight Tracking): 鳥類トラッキングデータセット

    【特徴】
    - 高機動性 (3次元空間での動き)
    - 類似した外観 (同種の鳥)
    - 22種の鳥類

    データ構造はDanceTrackと同様
    """
    pass  # DanceTrackと同じ実装


class CrowdHumanDataset:
    """
    CrowdHuman: 混雑シーンの人物検出データセット (静止画)

    【MOTIPでの使用】
    - Data Augmentationのみ (MultiSimulateで動画を生成)
    - 訓練データの多様性向上

    【注意】
    MOTデータセット (SportsMOT) とアノテーション基準が異なる:
    - CrowdHuman: すべての人物をアノテーション
    - SportsMOT: アスリートのみ

    → 混在訓練時は注意が必要 (論文Appendix C参照)

    【データ構造】
    crowdhuman/
    ├── Images/
    │   ├── 273271,1a0d6000b9e1f5b7.jpg
    │   └── ...
    └── annotation_train.odgt  # JSON Lines形式
    """

    def __init__(self, root: str, split: str = 'train'):
        self.root = Path(root)
        self.split = split
        self.annotation_file = self.root / f'annotation_{split}.odgt'

    def load_annotations(self) -> List[Dict]:
        """
        アノテーションをロード

        Returns:
            [
                {
                    'ID': str,
                    'gtboxes': [
                        {
                            'tag': 'person',
                            'fbox': [x, y, w, h],  # Full body box
                            'vbox': [x, y, w, h],  # Visible box
                            ...
                        },
                        ...
                    ],
                },
                ...
            ]
        """
        annotations = []

        with open(self.annotation_file, 'r') as f:
            for line in f:
                ann = json.loads(line)
                annotations.append(ann)

        return annotations


class JointDataset:
    """
    複数データセットの統合ローダー

    【使用例】
    ```python
    dataset = JointDataset(
        datasets=['dancetrack', 'sportsmot', 'crowdhuman', 'bft'],
        dataset_weights=[1.0, 1.0, 0.5, 0.5],  # サンプリング重み
    )
    ```

    【サンプリング戦略】
    - 各データセットから重み付けでサンプリング
    - Epoch依存のサンプリング長調整
    """

    def __init__(
        self,
        datasets: List[str],
        dataset_weights: List[float],
        sample_lengths: List[int] = [30],
        sample_intervals: List[int] = [1, 2, 3, 4],
    ):
        self.datasets = datasets
        self.dataset_weights = dataset_weights
        self.sample_lengths = sample_lengths
        self.sample_intervals = sample_intervals

    def sample_sequence(
        self,
        epoch: int,
    ) -> Tuple[List[torch.Tensor], List[Dict]]:
        """
        シーケンスをサンプリング

        Args:
            epoch: 現在のエpoch

        Returns:
            images: List[(3, H, W)] - T+1フレームの画像
            annotations: List[Dict] - 対応するアノテーション
        """
        # データセットをランダムに選択 (重み付け)
        dataset_idx = np.random.choice(
            len(self.datasets),
            p=np.array(self.dataset_weights) / sum(self.dataset_weights)
        )

        # サンプリング長を選択
        sample_length = np.random.choice(self.sample_lengths)

        # サンプリング間隔を選択
        sample_interval = np.random.choice(self.sample_intervals)

        # シーケンスとフレームをランダムに選択
        # (実際の実装では各データセットからロード)

        images = []
        annotations = []

        # プレースホルダー
        return images, annotations


def build_trajectory_annotations(
    annotations: List[Dict],        # T+1フレームのアノテーション
    num_train_frames: int = 4,      # DETR訓練フレーム数
) -> Dict[str, torch.Tensor]:
    """
    軌跡アノテーションを構築

    【処理フロー】
    1. 最初のTフレームを履歴軌跡、最後のフレームを現在フレームとする
    2. 各軌跡をグループ化 (同じIDを持つ物体)
    3. パディング (最大物体数に合わせる)

    Args:
        annotations: T+1フレームのアノテーション
        num_train_frames: DETR訓練フレーム数

    Returns:
        trajectory_annotations: {
            'trajectory_id_labels': (G, T, N),
            'trajectory_id_masks': (G, T, N),
            'trajectory_times': (G, T, N),
            'unknown_id_labels': (G, 1, N),
            'unknown_id_masks': (G, 1, N),
        }
    """
    T = len(annotations) - 1  # 履歴フレーム数
    current_ann = annotations[-1]  # 現在フレーム

    # 軌跡をグループ化
    # 簡略化: すべての物体を1つのグループにまとめる
    # 実際の実装では、より複雑なグループ化戦略を使用

    # プレースホルダー
    trajectory_annotations = {
        'trajectory_id_labels': torch.zeros(1, T, 100, dtype=torch.long),
        'trajectory_id_masks': torch.zeros(1, T, 100, dtype=torch.bool),
        'trajectory_times': torch.zeros(1, T, 100, dtype=torch.long),
        'unknown_id_labels': torch.zeros(1, 1, 100, dtype=torch.long),
        'unknown_id_masks': torch.zeros(1, 1, 100, dtype=torch.bool),
    }

    return trajectory_annotations


# ========================================
# Data Augmentation
# ========================================

class MultiSimulate:
    """
    静止画から動画を生成 (CrowdHuman用)

    【方法】
    - ランダムなシフト (平行移動) + スケーリング
    - 同じ画像から複数フレームをサンプリング

    【問題点】 (論文Appendix C参照)
    - 過度に単純化された動き (平行移動とスケールのみ)
    - 実際の動画とは大きく異なる
    - 長期モデリングには不利

    Args:
        shift_ratio: 0.06 - シフト量 (画像サイズの6%)
        overflow_bbox: False - ボックスが画像外に出るのを許可するか
    """

    def __init__(self, shift_ratio: float = 0.06, overflow_bbox: bool = False):
        self.shift_ratio = shift_ratio
        self.overflow_bbox = overflow_bbox

    def __call__(
        self,
        image: torch.Tensor,      # (3, H, W)
        annotation: Dict,
        num_frames: int = 30,
    ) -> Tuple[List[torch.Tensor], List[Dict]]:
        """
        静止画から動画を生成

        Returns:
            images: List[(3, H, W)]
            annotations: List[Dict]
        """
        H, W = image.shape[1:]

        images = []
        annotations = []

        for _ in range(num_frames):
            # ランダムなシフト
            shift_x = np.random.uniform(-self.shift_ratio, self.shift_ratio) * W
            shift_y = np.random.uniform(-self.shift_ratio, self.shift_ratio) * H

            # 画像をクロップ
            # (実際の実装では、シフトに応じてクロップ領域を調整)

            images.append(image)  # 簡略化
            annotations.append(annotation)

        return images, annotations


class TrajectoryOcclusion:
    """
    Trajectory Occlusion: 軌跡の一部をランダムにマスク

    【目的】
    - オクルージョンへの頑健性向上
    - 訓練時に部分的な軌跡情報でID予測を学習

    Args:
        prob: 0.5 - 各軌跡トークンをマスクする確率
    """

    def __init__(self, prob: float = 0.5):
        self.prob = prob

    def __call__(
        self,
        trajectory_tokens: torch.Tensor,  # (G, T, N, C)
    ) -> torch.Tensor:
        """
        軌跡トークンをランダムにマスク

        Returns:
            masked_tokens: (G, T, N, C)
        """
        G, T, N, C = trajectory_tokens.shape

        # ランダムマスク
        mask = torch.rand(G, T, N, 1) > self.prob
        masked_tokens = trajectory_tokens * mask

        return masked_tokens


class TrajectorySwitch:
    """
    Trajectory Switching: 軌跡IDをランダムに入れ替え

    【目的】
    - ID割り当てエラーへの頑健性向上
    - 推論時のID割り当てミスをシミュレート

    Args:
        prob: 0.5 - 同一フレーム内で2つの軌跡IDを入れ替える確率
    """

    def __init__(self, prob: float = 0.5):
        self.prob = prob

    def __call__(
        self,
        trajectory_id_labels: torch.Tensor,  # (G, T, N)
    ) -> torch.Tensor:
        """
        軌跡IDをランダムに入れ替え

        Returns:
            switched_labels: (G, T, N)
        """
        G, T, N = trajectory_id_labels.shape

        # 各フレームで確率probでIDを入れ替え
        switched_labels = trajectory_id_labels.clone()

        for g in range(G):
            for t in range(T):
                if torch.rand(1).item() < self.prob:
                    # 2つのIDをランダムに選択して入れ替え
                    valid_indices = (trajectory_id_labels[g, t] > 0).nonzero(as_tuple=True)[0]
                    if len(valid_indices) >= 2:
                        idx1, idx2 = torch.randperm(len(valid_indices))[:2]
                        switched_labels[g, t, valid_indices[idx1]], \
                        switched_labels[g, t, valid_indices[idx2]] = \
                            switched_labels[g, t, valid_indices[idx2]], \
                            switched_labels[g, t, valid_indices[idx1]]

        return switched_labels


# ========================================
# 形状ガイド
# ========================================
"""
【フレームアノテーション】
{
    'id': (M,) - 物体ID (1~max_id)
    'category': (M,) - クラスラベル (常に1: 人物)
    'bbox': (M, 4) - バウンディングボックス [x, y, w, h]
    'visibility': (M,) - 可視性スコア [0, 1]
    'is_legal': bool - フレームの有効性
}

【軌跡アノテーション】
{
    'trajectory_id_labels': (G, T, N) - 軌跡IDラベル
    'trajectory_id_masks': (G, T, N) - パディングマスク (True=有効)
    'trajectory_times': (G, T, N) - タイムスタンプ (フレームインデックス)
    'unknown_id_labels': (G, 1, N) - 現在フレームのID (教師信号)
    'unknown_id_masks': (G, 1, N) - 現在フレームのマスク
}

G: 軌跡グループ数 (可変)
T: 履歴フレーム数 (30)
N: フレームあたりの最大物体数 (パディング後)

【訓練サンプリング】
- シーケンス長: T+1 (30+1=31フレーム)
- サンプリング間隔: 1~4フレーム (ランダム)
- 最初のTフレーム: 履歴軌跡
- 最後のフレーム: 現在フレーム (ID予測対象)

【DETR訓練フレーム】
- 全フレームでDETR forward
- 最初の4フレームのみ勾配計算
- 残りのT-3フレームは torch.no_grad()
- メモリ効率化のため

【バッチ構成】
images: (B, T+1, 3, H, W)
annotations: List[Dict] - B*(T+1)個の要素
"""
