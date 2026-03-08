"""
ALIKED Training Data Format - 簡略化疑似コード
==============================================

学習データのフォーマットと前処理

ALIKEDは2種類のデータセットで訓練:
1. Homographicデータセット: 合成Homography変換
2. Perspectiveデータセット: 実カメラパラメータ + Depth

各データセットの詳細とラベル取得方法を説明
"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset
from typing import Dict, Tuple
from pathlib import Path

# ============================================
# 1. Homographicデータセット
# ============================================

class HomographicDataset(Dataset):
    """
    Homographicデータセット

    特徴:
    - 1枚の画像から合成ペアを生成
    - Homography行列をランダム生成
    - ラベルデータ不要 (自己教師あり)

    使用データセット:
    - R2D2 Homographic dataset
      * Oxford & Paris retrieval datasets
      * Aachen dataset
      * Style-transferred images

    ディレクトリ構造:
    homographic_dataset/
    ├── oxford/
    │   ├── image_000000.jpg
    │   ├── image_000001.jpg
    │   └── ...
    ├── paris/
    │   └── ...
    └── aachen/
        └── ...

    ⚠️ 重要: カメラパラメータ不要!
    """

    def __init__(self, image_dir: str, image_size: Tuple[int, int] = (512, 512)):
        self.image_paths = list(Path(image_dir).glob('*.jpg'))
        self.image_size = image_size  # (H, W)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Dict:
        """
        1サンプル取得

        処理フロー:
        1. 画像1枚を読み込み
        2. ランダムHomography行列生成
        3. Homography変換で2枚目の画像を生成
        4. ペアとHomography行列を返す

        出力:
        {
            'image_a': (3, H, W) - 元画像
            'image_b': (3, H, W) - Homography変換後の画像
            'H_ab': (3, 3) - Homography行列 (A→Bへの変換)
            'valid_mask': (H, W) - 有効領域マスク
        }
        """

        # ========================================
        # Step 1: 画像読み込み
        # ========================================
        image = self._load_image(self.image_paths[idx])
        # image: (3, H, W) - RGB画像

        # ========================================
        # Step 2: ランダムHomography生成
        # ========================================
        H_ab = self._generate_random_homography(
            image_size=(image.shape[1], image.shape[2]),
            max_rotation=45,        # 最大回転角度 (度)
            max_scale_change=0.5,   # 最大スケール変化
            max_shear=0.3,          # 最大せん断
            max_perspective=0.0005  # 最大透視変換
        )
        # H_ab: (3, 3) - Homography行列

        # ========================================
        # Step 3: Homography変換
        # ========================================
        image_b, valid_mask = self._warp_image(image, H_ab)
        # image_b: (3, H, W) - 変換後の画像
        # valid_mask: (H, W) - 有効ピクセル (bool)

        return {
            'image_a': image,
            'image_b': image_b,
            'H_ab': H_ab,
            'valid_mask': valid_mask
        }

    def _generate_random_homography(
        self,
        image_size: Tuple[int, int],
        max_rotation: float = 45,
        max_scale_change: float = 0.5,
        max_shear: float = 0.3,
        max_perspective: float = 0.0005
    ) -> torch.Tensor:
        """
        ランダムHomography行列生成

        Homography = Translation × Rotation × Scale × Shear × Perspective

        数学的表現:
        H = [a11  a12  tx ]
            [a21  a22  ty ]
            [p1   p2   1  ]

        where:
          Rotation: θ ∈ [-max_rotation, max_rotation]
          Scale: s ∈ [1-max_scale_change, 1+max_scale_change]
          Shear: shear ∈ [-max_shear, max_shear]
          Perspective: p1, p2 ∈ [-max_perspective, max_perspective]
          Translation: tx, ty ∈ [-H/4, H/4]

        出力:
            H: (3, 3) - Homography行列
        """

        H, W = image_size

        # ========================================
        # 1. Rotation
        # ========================================
        theta = np.random.uniform(-max_rotation, max_rotation)
        theta_rad = np.deg2rad(theta)

        cos_theta = np.cos(theta_rad)
        sin_theta = np.sin(theta_rad)

        R = np.array([
            [cos_theta, -sin_theta, 0],
            [sin_theta,  cos_theta, 0],
            [0,          0,         1]
        ], dtype=np.float32)

        # ========================================
        # 2. Scale
        # ========================================
        scale = np.random.uniform(1 - max_scale_change, 1 + max_scale_change)

        S = np.array([
            [scale, 0,     0],
            [0,     scale, 0],
            [0,     0,     1]
        ], dtype=np.float32)

        # ========================================
        # 3. Shear
        # ========================================
        shear_x = np.random.uniform(-max_shear, max_shear)
        shear_y = np.random.uniform(-max_shear, max_shear)

        Sh = np.array([
            [1,       shear_x, 0],
            [shear_y, 1,       0],
            [0,       0,       1]
        ], dtype=np.float32)

        # ========================================
        # 4. Translation (画像中心基準)
        # ========================================
        tx = np.random.uniform(-W / 4, W / 4)
        ty = np.random.uniform(-H / 4, H / 4)

        T = np.array([
            [1, 0, tx],
            [0, 1, ty],
            [0, 0, 1]
        ], dtype=np.float32)

        # ========================================
        # 5. Perspective
        # ========================================
        p1 = np.random.uniform(-max_perspective, max_perspective)
        p2 = np.random.uniform(-max_perspective, max_perspective)

        P = np.array([
            [1,  0, 0],
            [0,  1, 0],
            [p1, p2, 1]
        ], dtype=np.float32)

        # ========================================
        # 6. 合成 (画像中心を原点とする)
        # ========================================
        # 中心への移動
        T_center = np.array([
            [1, 0, -W / 2],
            [0, 1, -H / 2],
            [0, 0, 1]
        ], dtype=np.float32)

        # 中心から戻す
        T_back = np.array([
            [1, 0, W / 2],
            [0, 1, H / 2],
            [0, 0, 1]
        ], dtype=np.float32)

        # 合成: T_back × T × P × Sh × S × R × T_center
        H = T_back @ T @ P @ Sh @ S @ R @ T_center

        return torch.from_numpy(H)

    def _warp_image(
        self,
        image: torch.Tensor,
        H: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Homography変換で画像をワープ

        入力:
            image: (3, H, W)
            H: (3, 3)

        出力:
            warped: (3, H, W) - 変換後の画像
            valid_mask: (H, W) - 有効領域マスク
        """
        import torch.nn.functional as F

        C, H_img, W_img = image.shape

        # グリッド生成
        y, x = torch.meshgrid(
            torch.arange(H_img, dtype=torch.float32),
            torch.arange(W_img, dtype=torch.float32),
            indexing='ij'
        )

        # Homogeneous coordinates
        coords = torch.stack([x, y, torch.ones_like(x)], dim=-1)
        # coords: (H, W, 3)

        # Homography変換
        coords_warped = coords @ H.T
        # coords_warped: (H, W, 3)

        # 正規化
        coords_warped = coords_warped[..., :2] / coords_warped[..., 2:3]
        # coords_warped: (H, W, 2) - [x, y]

        # Grid sample用に正規化 [-1, 1]
        coords_normalized = coords_warped.clone()
        coords_normalized[..., 0] = 2.0 * coords_normalized[..., 0] / (W_img - 1) - 1.0
        coords_normalized[..., 1] = 2.0 * coords_normalized[..., 1] / (H_img - 1) - 1.0

        # Grid sample
        warped = F.grid_sample(
            image.unsqueeze(0),
            coords_normalized.unsqueeze(0),
            mode='bilinear',
            align_corners=False
        ).squeeze(0)

        # 有効領域マスク
        valid_mask = (
            (coords_warped[..., 0] >= 0) &
            (coords_warped[..., 0] < W_img) &
            (coords_warped[..., 1] >= 0) &
            (coords_warped[..., 1] < H_img)
        )

        return warped, valid_mask

    def _load_image(self, path: Path) -> torch.Tensor:
        """画像読み込み（指定サイズにリサイズ）"""
        from PIL import Image
        import torchvision.transforms as T

        image = Image.open(path).convert('RGB')
        transform = T.Compose([
            T.Resize(self.image_size),  # (H, W)
            T.ToTensor()
        ])
        return transform(image)

# ============================================
# 2. Perspectiveデータセット (MegaDepth)
# ============================================

class MegaDepthDataset(Dataset):
    """
    MegaDepthデータセット

    特徴:
    - 実世界の画像ペア
    - COLMAP で 3D再構成済み
    - カメラパラメータ + Depthマップ付き

    ディレクトリ構造:
    megadepth/
    ├── scene_0000/
    │   ├── images/
    │   │   ├── image_0.jpg
    │   │   ├── image_1.jpg
    │   │   └── ...
    │   ├── depths/
    │   │   ├── depth_0.npy
    │   │   ├── depth_1.npy
    │   │   └── ...
    │   └── scene_info.npz  ← カメラパラメータ
    ├── scene_0001/
    │   └── ...
    └── ...

    scene_info.npz の内容:
    {
        'image_paths': List[str]
        'intrinsics': (N, 3, 3) - カメラ内部パラメータ K
        'poses': (N, 4, 4) - カメラポーズ [R|t]
        'depth_paths': List[str]
        'pairs': List[(i, j)] - ペアインデックス
    }

    ⚠️ 重要: COLMAPによる事前計算が必要!
    """

    def __init__(self, scene_dir: str):
        self.scene_dir = Path(scene_dir)
        self.scene_info = np.load(self.scene_dir / 'scene_info.npz', allow_pickle=True)

        self.image_paths = self.scene_info['image_paths']
        self.intrinsics = self.scene_info['intrinsics']     # (N, 3, 3)
        self.poses = self.scene_info['poses']               # (N, 4, 4)
        self.depth_paths = self.scene_info['depth_paths']
        self.pairs = self.scene_info['pairs']               # [(i, j), ...]

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Dict:
        """
        1サンプル取得

        処理フロー:
        1. ペアインデックス取得
        2. 画像とDepthマップ読み込み
        3. カメラパラメータ取得
        4. 相対ポーズ計算 (R_ab, t_ab)

        出力:
        {
            'image_a': (3, H, W)
            'image_b': (3, H, W)
            'depth_a': (H, W) - Depthマップ
            'K_a': (3, 3) - カメラ内部パラメータ
            'K_b': (3, 3)
            'R_ab': (3, 3) - 相対回転行列 (A→B)
            't_ab': (3,) - 相対並進ベクトル (A→B)
        }
        """

        # ========================================
        # Step 1: ペア選択
        # ========================================
        idx_a, idx_b = self.pairs[idx]

        # ========================================
        # Step 2: 画像 & Depth読み込み
        # ========================================
        image_a = self._load_image(self.scene_dir / 'images' / self.image_paths[idx_a])
        image_b = self._load_image(self.scene_dir / 'images' / self.image_paths[idx_b])

        depth_a = self._load_depth(self.scene_dir / 'depths' / self.depth_paths[idx_a])
        # depth_a: (H, W) - メートル単位

        # ========================================
        # Step 3: カメラパラメータ取得
        # ========================================
        K_a = torch.from_numpy(self.intrinsics[idx_a]).float()  # (3, 3)
        K_b = torch.from_numpy(self.intrinsics[idx_b]).float()  # (3, 3)

        # カメラ内部パラメータ K:
        # K = [fx  0   cx]
        #     [0   fy  cy]
        #     [0   0   1 ]
        #
        # fx, fy: 焦点距離 (pixel)
        # cx, cy: 主点 (pixel)

        pose_a = torch.from_numpy(self.poses[idx_a]).float()  # (4, 4)
        pose_b = torch.from_numpy(self.poses[idx_b]).float()  # (4, 4)

        # カメラポーズ (World → Camera):
        # pose = [R  t]  (4, 4)
        #        [0  1]
        #
        # R: (3, 3) - 回転行列
        # t: (3, 1) - 並進ベクトル

        # ========================================
        # Step 4: 相対ポーズ計算
        # ========================================
        R_a = pose_a[:3, :3]  # (3, 3)
        t_a = pose_a[:3, 3]   # (3,)

        R_b = pose_b[:3, :3]
        t_b = pose_b[:3, 3]

        # 相対変換 (Camera A → Camera B):
        # P_b = R_ab @ P_a + t_ab
        #
        # where:
        #   R_ab = R_b @ R_a^T
        #   t_ab = t_b - R_ab @ t_a

        R_ab = R_b @ R_a.T
        t_ab = t_b - R_ab @ t_a

        return {
            'image_a': image_a,
            'image_b': image_b,
            'depth_a': depth_a,
            'K_a': K_a,
            'K_b': K_b,
            'R_ab': R_ab,
            't_ab': t_ab
        }

    def _load_image(self, path: Path) -> torch.Tensor:
        """画像読み込み"""
        from PIL import Image
        import torchvision.transforms as T

        image = Image.open(path).convert('RGB')
        transform = T.ToTensor()
        return transform(image)

    def _load_depth(self, path: Path) -> torch.Tensor:
        """Depthマップ読み込み"""
        depth = np.load(path)
        return torch.from_numpy(depth).float()

# ============================================
# 3. キーポイントのワープ (損失計算で使用)
# ============================================

def warp_keypoints_homography(
    keypoints: torch.Tensor,
    H: torch.Tensor
) -> torch.Tensor:
    """
    Homographyでキーポイントをワープ

    入力:
        keypoints: (N, 2) - [x, y]
        H: (3, 3) - Homography行列

    出力:
        warped: (N, 2) - [x', y']

    数式:
        [x']   [h11  h12  h13] [x]
        [y'] = [h21  h22  h23] [y]
        [w ]   [h31  h32  h33] [1]

        x' = (h11*x + h12*y + h13) / (h31*x + h32*y + h33)
        y' = (h21*x + h22*y + h23) / (h31*x + h32*y + h33)
    """

    N = keypoints.shape[0]

    # Homogeneous coordinates
    kpts_homo = torch.cat([
        keypoints,
        torch.ones(N, 1, device=keypoints.device)
    ], dim=1)  # (N, 3)

    # Homography変換
    kpts_warped_homo = kpts_homo @ H.T  # (N, 3)

    # 正規化
    kpts_warped = kpts_warped_homo[:, :2] / kpts_warped_homo[:, 2:3]

    return kpts_warped

def warp_keypoints_perspective(
    keypoints: torch.Tensor,
    depth: torch.Tensor,
    K_a: torch.Tensor,
    K_b: torch.Tensor,
    R_ab: torch.Tensor,
    t_ab: torch.Tensor
) -> torch.Tensor:
    """
    3D Perspective変換でキーポイントをワープ

    入力:
        keypoints: (N, 2) - Image Aのキーポイント [x, y]
        depth: (H, W) - Image Aのdepthマップ
        K_a: (3, 3) - Image Aの内部パラメータ
        K_b: (3, 3) - Image Bの内部パラメータ
        R_ab: (3, 3) - 相対回転
        t_ab: (3,) - 相対並進

    出力:
        warped: (N, 2) - Image Bでのキーポイント [x', y']

    処理フロー:
        1. Image A pixel → 3D point (Camera A座標系)
        2. 3D point → Camera B座標系に変換
        3. Camera B → Image B pixel に投影
    """

    N = keypoints.shape[0]

    # ========================================
    # Step 1: Pixel → 3D Point (Camera A)
    # ========================================

    # キーポイント位置のdepth取得
    x_pix = keypoints[:, 0].long()
    y_pix = keypoints[:, 1].long()
    d = depth[y_pix, x_pix]  # (N,)

    # Pixel → Normalized camera coordinates
    # [x_norm]   [x_pix - cx]
    # [y_norm] = [y_pix - cy]
    # [1     ]   [1         ]

    fx_a = K_a[0, 0]
    fy_a = K_a[1, 1]
    cx_a = K_a[0, 2]
    cy_a = K_a[1, 2]

    x_norm = (keypoints[:, 0] - cx_a) / fx_a
    y_norm = (keypoints[:, 1] - cy_a) / fy_a

    # 3D Point (Camera A)
    P_a = torch.stack([
        x_norm * d,
        y_norm * d,
        d
    ], dim=1)  # (N, 3)

    # ========================================
    # Step 2: Camera A → Camera B
    # ========================================

    P_b = (R_ab @ P_a.T).T + t_ab  # (N, 3)

    # ========================================
    # Step 3: 3D Point → Pixel (Image B)
    # ========================================

    fx_b = K_b[0, 0]
    fy_b = K_b[1, 1]
    cx_b = K_b[0, 2]
    cy_b = K_b[1, 2]

    x_b = fx_b * (P_b[:, 0] / P_b[:, 2]) + cx_b
    y_b = fy_b * (P_b[:, 1] / P_b[:, 2]) + cy_b

    warped = torch.stack([x_b, y_b], dim=1)

    return warped

# ============================================
# 4. 使用例
# ============================================

def example_training_data():
    """学習データの使用例"""

    print("=" * 60)
    print("1. Homographicデータセット")
    print("=" * 60)

    # Homographicデータセット
    homographic_dataset = HomographicDataset('path/to/oxford')

    sample = homographic_dataset[0]

    print(f"Image A: {sample['image_a'].shape}")         # (3, 800, 800)
    print(f"Image B: {sample['image_b'].shape}")         # (3, 800, 800)
    print(f"Homography: {sample['H_ab'].shape}")         # (3, 3)
    print(f"Valid mask: {sample['valid_mask'].shape}")   # (800, 800)

    print(f"\nHomography matrix:")
    print(sample['H_ab'])

    # キーポイントワープ例
    keypoints_a = torch.rand(100, 2) * 800
    keypoints_b_warped = warp_keypoints_homography(keypoints_a, sample['H_ab'])
    print(f"\nKeypoints A: {keypoints_a[:3]}")
    print(f"Warped to B: {keypoints_b_warped[:3]}")

    print("\n" + "=" * 60)
    print("2. Perspectiveデータセット (MegaDepth)")
    print("=" * 60)

    # MegaDepthデータセット
    megadepth_dataset = MegaDepthDataset('path/to/megadepth/scene_0000')

    sample = megadepth_dataset[0]

    print(f"Image A: {sample['image_a'].shape}")         # (3, 800, 800)
    print(f"Image B: {sample['image_b'].shape}")         # (3, 800, 800)
    print(f"Depth A: {sample['depth_a'].shape}")         # (800, 800)
    print(f"K_a: {sample['K_a'].shape}")                 # (3, 3)
    print(f"K_b: {sample['K_b'].shape}")                 # (3, 3)
    print(f"R_ab: {sample['R_ab'].shape}")               # (3, 3)
    print(f"t_ab: {sample['t_ab'].shape}")               # (3,)

    print(f"\nCamera intrinsics K_a:")
    print(sample['K_a'])
    print(f"\nRelative rotation R_ab:")
    print(sample['R_ab'])
    print(f"\nRelative translation t_ab:")
    print(sample['t_ab'])

    # キーポイントワープ例
    keypoints_a = torch.rand(100, 2) * 800
    keypoints_b_warped = warp_keypoints_perspective(
        keypoints_a,
        sample['depth_a'],
        sample['K_a'],
        sample['K_b'],
        sample['R_ab'],
        sample['t_ab']
    )
    print(f"\nKeypoints A: {keypoints_a[:3]}")
    print(f"Warped to B: {keypoints_b_warped[:3]}")

if __name__ == "__main__":
    example_training_data()
