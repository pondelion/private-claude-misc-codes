"""
InternVL3.5 Dynamic High Resolution + Pixel Shuffle
=====================================================

このファイルは InternVL3.5 の「Dynamic High Resolution」戦略と
Pixel Shuffle によるトークン圧縮を実装しています。

Dynamic High Resolution:
  高解像度画像を固定サイズ (448×448) のタイルに分割して処理することで、
  ViT の固定解像度という制約を克服しながら細部情報を保持します。

Pixel Shuffle (Token Compression):
  ViT 出力の空間方向トークン (1024個) を再配置・圧縮して
  LLM へのコストを削減します (1024 → 256)。

公式実装参考:
  internvl_chat/internvl/train/dataset.py (dynamic_preprocess)
  internvl_chat/internvl/model/internvl_chat/modeling_internvl_chat.py (pixel_shuffle)

============================================================
テンソル形状記法
============================================================
  H_orig, W_orig : 元画像の高さ・幅 (ピクセル)
  tile_size      : タイルサイズ = 448 px
  n_tiles        : タイル数 (アスペクト比に基づいて決定)
  P              : 全パッチ数 = n_tiles + 1 (サムネイル込み)
  S              : ViT 系列長 = (tile_size/patch_size)^2 = 1024
  D_v            : ViT hidden size
"""

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import numpy as np


# ============================================================
# 1. タイル数の計算 (アスペクト比ベース)
# ============================================================

def find_best_aspect_ratio(
    aspect_ratio: float,
    target_ratios: List[Tuple[int, int]],
    image_size: Tuple[int, int],
    tile_size: int,
) -> Tuple[int, int]:
    """
    画像のアスペクト比に最も近いタイルグリッドを選択する。

    引数:
      aspect_ratio   : 元画像の W/H
      target_ratios  : 候補グリッド (n_w, n_h) のリスト
                       例: [(1,1),(1,2),(2,1),(1,3),(3,1),(2,2),(1,4)...]
      image_size     : 元画像の (H, W)
      tile_size      : タイルサイズ (448)

    返値:
      (n_w, n_h) : 最適なグリッドの幅・高さ方向のタイル数
    """
    best_ratio = None
    best_ratio_diff = float('inf')

    for ratio in target_ratios:
        n_w, n_h = ratio
        # このグリッド配置のアスペクト比
        target_aspect = n_w / n_h
        ratio_diff = abs(aspect_ratio - target_aspect)

        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            # 同等の場合はタイル数が多い方を選択 (より高解像度)
            if n_w * n_h > best_ratio[0] * best_ratio[1]:
                best_ratio = ratio

    return best_ratio


def get_target_ratios(
    min_tiles: int = 1,
    max_tiles: int = 6,
) -> List[Tuple[int, int]]:
    """
    使用可能なタイルグリッドのリストを生成。

    min_tiles から max_tiles の範囲で、
    合計タイル数が max_tiles 以下になるグリッドを全列挙。

    例 (max_tiles=6):
      (1,1), (1,2), (2,1), (1,3), (3,1), (2,2), (2,3), (3,2), (1,4), (4,1),
      (2,4), (4,2), (3,3), (1,5), (5,1), (2,5), (5,2), (3,4), (4,3), (1,6),
      (6,1), (2,6), (6,2), (3,5), (5,3)
      ※ ただし n_w*n_h <= max_tiles
    """
    target_ratios = set()
    for n in range(min_tiles, max_tiles + 1):
        for n_w in range(1, n + 1):
            n_h = n  # n_w * n_h = n となるのは n_h = n の場合のみではない
            # 全ての (n_w, n_h) where n_w * n_h <= max_tiles
    # 正しい実装:
    target_ratios = set()
    for n_w in range(1, max_tiles + 1):
        for n_h in range(1, max_tiles + 1):
            if min_tiles <= n_w * n_h <= max_tiles:
                target_ratios.add((n_w, n_h))
    return sorted(target_ratios, key=lambda x: x[0] * x[1])


# ============================================================
# 2. Dynamic High Resolution タイリング
# ============================================================

def dynamic_preprocess(
    image: Image.Image,
    tile_size: int = 448,
    min_tiles: int = 1,
    max_tiles: int = 6,
    use_thumbnail: bool = True,
) -> List[Image.Image]:
    """
    高解像度画像を 448×448 タイルに分割する Dynamic High Resolution の中核処理。

    処理フロー:
      1. 元画像のアスペクト比を計算
      2. 最適なグリッド (n_w, n_h) を選択
      3. 元画像をグリッドサイズ (tile_size*n_w, tile_size*n_h) にリサイズ
      4. グリッドから個々のタイルを切り出す
      5. サムネイル (tile_size×tile_size の縮小版) を先頭に追加

    引数:
      image         : 元画像 (PIL.Image)
      tile_size     : タイルサイズ = 448
      min_tiles     : 最小タイル数 = 1
      max_tiles     : 最大タイル数 = 6 (デフォルト)
      use_thumbnail : True = サムネイルを先頭に追加

    返値:
      tiles : List[PIL.Image]
              長さ = n_tiles + 1 (サムネイル込み)
              各画像のサイズ = (tile_size, tile_size) = (448, 448)

    例:
      入力: 1344×672 画像 (アスペクト比 2.0)
      → グリッド (2, 1) を選択 (n_w=2, n_h=1, 合計2タイル)
      → 896×448 にリサイズ後、448×448 を2枚切り出し
      → サムネイル1枚追加 = 合計3枚

    トークン数:
      1タイル = 1024 ViTトークン → 256 圧縮後トークン (LLM用)
      合計 = (n_tiles + 1) × 256 テキスト位置 (<IMG_CONTEXT> tokens)
    """
    W_orig, H_orig = image.size   # PIL は (W, H) 順
    aspect_ratio = W_orig / H_orig

    # ステップ1: 候補グリッドリストを生成
    target_ratios = get_target_ratios(min_tiles, max_tiles)

    # ステップ2: 最適グリッドを選択
    n_w, n_h = find_best_aspect_ratio(
        aspect_ratio=aspect_ratio,
        target_ratios=target_ratios,
        image_size=(H_orig, W_orig),
        tile_size=tile_size,
    )
    n_tiles = n_w * n_h
    print(f"  動的解像度: 元サイズ=({W_orig}x{H_orig}), "
          f"グリッド=({n_w}x{n_h}), タイル数={n_tiles}")

    # ステップ3: グリッドサイズにリサイズ
    target_W = tile_size * n_w
    target_H = tile_size * n_h
    # LANCZOS は高品質なダウンサンプリングフィルター
    resized = image.resize((target_W, target_H), Image.LANCZOS)

    # ステップ4: タイルに分割
    tiles = []
    for row in range(n_h):
        for col in range(n_w):
            left = col * tile_size
            upper = row * tile_size
            right = left + tile_size
            lower = upper + tile_size
            tile = resized.crop((left, upper, right, lower))
            tiles.append(tile)
    # tiles: List[PIL.Image], 各 = (tile_size, tile_size)

    # ステップ5: サムネイルを先頭に追加
    if use_thumbnail and n_tiles > 1:
        thumbnail = image.resize((tile_size, tile_size), Image.LANCZOS)
        # サムネイルを先頭に: [thumbnail, tile1, tile2, ...]
        tiles = [thumbnail] + tiles

    return tiles  # List of (n_tiles + 1) PIL images, each (448, 448)


# ============================================================
# 3. 画像前処理 (正規化 + テンソル変換)
# ============================================================

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def preprocess_image(pil_image: Image.Image) -> torch.Tensor:
    """
    PIL 画像を正規化済みテンソルに変換。

    入力: PIL.Image (W, H) mode='RGB'
    出力: (3, H, W) float32 テンソル, ImageNet正規化済み
    """
    # numpy配列に変換
    img_np = np.array(pil_image).astype(np.float32) / 255.0   # (H, W, 3), [0, 1]
    # ImageNet 正規化
    mean = np.array(IMAGENET_MEAN, dtype=np.float32)
    std = np.array(IMAGENET_STD, dtype=np.float32)
    img_np = (img_np - mean) / std
    # (H, W, 3) → (3, H, W)
    img_tensor = torch.from_numpy(img_np.transpose(2, 0, 1))
    return img_tensor  # (3, H, W)


def prepare_image_patches(
    image: Image.Image,
    tile_size: int = 448,
    max_tiles: int = 6,
    use_thumbnail: bool = True,
) -> Tuple[torch.Tensor, int]:
    """
    1枚の画像を ViT 入力テンソルに変換する完全な前処理パイプライン。

    入力:
      image       : PIL.Image (任意サイズ)
      tile_size   : 448
      max_tiles   : 最大タイル数 (デフォルト 6)
      use_thumbnail: True

    出力:
      pixel_values : (P, 3, 448, 448)  ※ P = タイル数 + 1 (サムネイル込み)
      num_patches  : P (パッチ数、LLM への IMG_CONTEXT 数の計算に使用)
    """
    # RGB変換 (グレースケール・RGBA等に対応)
    image = image.convert('RGB')

    # Dynamic High Resolution タイリング
    tiles = dynamic_preprocess(
        image,
        tile_size=tile_size,
        max_tiles=max_tiles,
        use_thumbnail=use_thumbnail,
    )
    # tiles: List[PIL.Image], 各 (448, 448)

    # 各タイルをテンソルに変換
    tile_tensors = [preprocess_image(tile) for tile in tiles]

    # (P, 3, 448, 448) に結合
    pixel_values = torch.stack(tile_tensors, dim=0)

    return pixel_values, len(tiles)


# ============================================================
# 4. マルチ画像・バッチ処理
# ============================================================

def batch_prepare_images(
    images: List[Image.Image],
    tile_size: int = 448,
    max_tiles: int = 6,
    use_thumbnail: bool = True,
) -> Tuple[torch.Tensor, List[int]]:
    """
    複数枚の画像を一括でバッチテンソルに変換。

    入力:
      images        : List[PIL.Image]  長さ B
      tile_size     : 448
      max_tiles     : 最大タイル数
      use_thumbnail : True

    出力:
      pixel_values    : (sum(P_i), 3, 448, 448)
                        ※ 全サンプルの全パッチを結合
      num_patches_list: List[int] 長さ B
                        各サンプルのパッチ数
    """
    all_patches = []
    num_patches_list = []

    for image in images:
        patches, n_patches = prepare_image_patches(
            image,
            tile_size=tile_size,
            max_tiles=max_tiles,
            use_thumbnail=use_thumbnail,
        )
        all_patches.append(patches)          # (P_i, 3, 448, 448)
        num_patches_list.append(n_patches)

    # 全パッチを結合: (sum(P_i), 3, 448, 448)
    pixel_values = torch.cat(all_patches, dim=0)

    return pixel_values, num_patches_list


# ============================================================
# 5. <IMG_CONTEXT> トークン列の構築
# ============================================================

def build_image_token_string(
    num_patches: int,
    num_image_token: int = 256,
    img_start: str = '<img>',
    img_end: str = '</img>',
    img_context: str = '<IMG_CONTEXT>',
) -> str:
    """
    1枚の画像 (複数パッチ) に対する IMG_CONTEXT トークン列を構築。

    引数:
      num_patches     : パッチ数 P (サムネイル込み)
      num_image_token : 1パッチあたりの IMG_CONTEXT 数 = 256 (デフォルト)
      img_start       : 画像開始トークン
      img_end         : 画像終了トークン
      img_context     : 画像コンテキストプレースホルダー

    返値:
      "<img><IMG_CONTEXT><IMG_CONTEXT>...(256×P個)...</img>"

    例: num_patches=3, num_image_token=256
      → "<img>" + "<IMG_CONTEXT>" * 768 + "</img>"
      → LLM から見ると 768 個の視覚トークン

    テキスト例:
      "この画像を説明してください。<img><IMG_CONTEXT>×768</img>"
      → tokenize 後、IMG_CONTEXT の埋め込みを ViT 出力で上書き
    """
    total_context_tokens = num_image_token * num_patches
    image_tokens = img_start + img_context * total_context_tokens + img_end
    return image_tokens


# ============================================================
# 6. Pixel Shuffle の詳細解説
# ============================================================

class PixelShuffle(nn.Module):
    """
    ViT 出力トークンの空間方向圧縮モジュール。

    概念: PixelShuffle (Super-Resolution で使われる) の逆操作。
          画像の空間解像度を下げて (H/s, W/s) にする代わりに
          チャンネル数を (s^2) 倍に増やす。

    具体例 (scale_factor=0.5, InternViT-6B):
      入力:  (B*P, 32, 32, 3200)   ← (空間32×32グリッド, 3200チャンネル)
      出力:  (B*P, 16, 16, 12800)  ← (空間16×16グリッド, 12800チャンネル)
      flatten後: (B*P, 256, 12800)  ← 256トークン/パッチ

    数学的には以下と等価:
      各 2×2 ブロックの4ピクセルを結合して1ピクセルにする
      (空間1/4 → チャンネル4倍)
    """
    def __init__(self, scale_factor: float = 0.5, version: str = 'v2'):
        super().__init__()
        self.scale_factor = scale_factor
        self.version = version

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        入力: x  (B*P, H_t, W_t, D_v)
              ※ H_t = W_t = image_size/patch_size = 32 (448px/14px)
        出力:    (B*P, H_t*s, W_t*s, D_v/(s^2))
              ※ s = scale_factor = 0.5
              = (B*P, 16, 16, 12800)
        """
        s = self.scale_factor
        n, w, h, c = x.size()
        # B*P, W, H, C → B*P, W, H*s, C/s
        x = x.view(n, w, int(h * s), int(c / s))
        # B*P, W, H*s, C/s → B*P, H*s, W, C/s
        x = x.permute(0, 2, 1, 3).contiguous()
        # B*P, H*s, W, C/s → B*P, H*s, W*s, C/(s^2)
        x = x.view(n, int(h * s), int(w * s), int(c / (s * s)))
        if self.version == 'v2':
            # v2: 正しい向きに戻す (H と W を入れ替える)
            x = x.permute(0, 2, 1, 3).contiguous()
        return x


class HighCompressionPixelShuffle(nn.Module):
    """
    InternVL3.5-Flash 用の高圧縮 Pixel Shuffle。

    ViR により選択された高圧縮パッチに適用:
      1024 → 64 トークン (16倍圧縮, scale_factor=0.25)
    """
    def __init__(self, scale_factor: float = 0.25, version: str = 'v2'):
        super().__init__()
        self.ps = PixelShuffle(scale_factor=scale_factor, version=version)
        self.scale_factor = scale_factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        入力: (B*P, H_t=32, W_t=32, D_v=3200)
        出力: (B*P, H_t*0.25=8, W_t*0.25=8, D_v*16=51200)
        flatten後: (B*P, 64, 51200)
        """
        return self.ps(x)


# ============================================================
# 使用例
# ============================================================

if __name__ == '__main__':
    print("=" * 60)
    print("Dynamic High Resolution + Pixel Shuffle 動作確認")
    print("=" * 60)

    # --- 1. タイルグリッド候補の確認 ---
    print("\n[1] タイルグリッド候補 (max_tiles=6)")
    ratios = get_target_ratios(min_tiles=1, max_tiles=6)
    print(f"  候補数: {len(ratios)}")
    for r in ratios:
        print(f"    ({r[0]}×{r[1]}) = {r[0]*r[1]}タイル")

    # --- 2. Dynamic Preprocessing テスト ---
    print("\n[2] Dynamic High Resolution タイリングテスト")

    # テスト用ダミー画像 (ランダムRGB)
    test_cases = [
        (448, 448, "正方形"),
        (1344, 448, "横長 3:1"),
        (896, 672, "横長 4:3"),
        (672, 1344, "縦長 1:3"),
    ]

    for W, H, name in test_cases:
        dummy_img = Image.fromarray(
            np.random.randint(0, 255, (H, W, 3), dtype=np.uint8)
        )
        tiles = dynamic_preprocess(dummy_img, tile_size=448, max_tiles=6, use_thumbnail=True)
        print(f"  {name} ({W}×{H}): {len(tiles)}枚のタイル (各 448×448)")
        assert all(t.size == (448, 448) for t in tiles), "タイルサイズが不正"

    # --- 3. 前処理パイプライン テスト ---
    print("\n[3] prepare_image_patches テスト")
    test_img = Image.fromarray(np.random.randint(0, 255, (672, 1344, 3), dtype=np.uint8))
    pixel_values, num_patches = prepare_image_patches(test_img, tile_size=448, max_tiles=6)
    print(f"  入力画像サイズ: (1344×672)")
    print(f"  出力 pixel_values: {pixel_values.shape}")
    print(f"  パッチ数: {num_patches}")
    assert pixel_values.shape[1:] == (3, 448, 448)
    print("  OK: (P, 3, 448, 448)")

    # --- 4. バッチ処理テスト ---
    print("\n[4] batch_prepare_images テスト")
    images = [
        Image.fromarray(np.random.randint(0, 255, (448, 448, 3), dtype=np.uint8)),  # 正方形
        Image.fromarray(np.random.randint(0, 255, (448, 1344, 3), dtype=np.uint8)),  # 横長
    ]
    batch_pixels, n_patches_list = batch_prepare_images(images, tile_size=448, max_tiles=6)
    total_patches = sum(n_patches_list)
    print(f"  バッチサイズ: {len(images)}枚")
    print(f"  各画像のパッチ数: {n_patches_list}")
    print(f"  結合後 pixel_values: {batch_pixels.shape}")
    assert batch_pixels.shape == (total_patches, 3, 448, 448)
    print(f"  OK: ({total_patches}, 3, 448, 448)")

    # --- 5. IMG_CONTEXT トークン列テスト ---
    print("\n[5] IMG_CONTEXT トークン列構築テスト")
    for n_pat in [1, 3, 7]:
        token_str = build_image_token_string(num_patches=n_pat, num_image_token=256)
        expected_context_count = 256 * n_pat
        actual_count = token_str.count('<IMG_CONTEXT>')
        print(f"  パッチ数={n_pat}: IMG_CONTEXTトークン数={actual_count} (期待値={expected_context_count})")
        assert actual_count == expected_context_count

    # --- 6. Pixel Shuffle テスト ---
    print("\n[6] Pixel Shuffle テスト")
    ps_v2 = PixelShuffle(scale_factor=0.5, version='v2')
    ps_hi = HighCompressionPixelShuffle(scale_factor=0.25, version='v2')

    # ダミー ViT 特徴 (B*P=3, H_t=32, W_t=32, D_v=3200)
    feat = torch.randn(3, 32, 32, 3200)

    # 標準圧縮 (256トークン/パッチ)
    out_std = ps_v2(feat)
    out_std_flat = out_std.reshape(3, -1, out_std.shape[-1])
    print(f"  入力:               {feat.shape}")
    print(f"  標準圧縮 (scale=0.5): {out_std.shape}")
    print(f"  → flatten:         {out_std_flat.shape}")
    assert out_std_flat.shape == (3, 256, 12800), f"期待: (3, 256, 12800), 実際: {out_std_flat.shape}"
    print("  OK: 1024 → 256 トークン (4倍圧縮)")

    # 高圧縮 (64トークン/パッチ, InternVL3.5-Flash 用)
    out_hi = ps_hi(feat)
    out_hi_flat = out_hi.reshape(3, -1, out_hi.shape[-1])
    print(f"  高圧縮  (scale=0.25): {out_hi.shape}")
    print(f"  → flatten:         {out_hi_flat.shape}")
    assert out_hi_flat.shape == (3, 64, 51200), f"期待: (3, 64, 51200), 実際: {out_hi_flat.shape}"
    print("  OK: 1024 → 64 トークン (16倍圧縮)")

    print("\n全テスト完了!")
