# gpu_batched_image_aug

albumentations とほぼ同じ API・同じパラメータの意味で、画像拡張を **GPU 上でバッチ (B, C, H, W) 単位に一括実行** する
ライブラリ。バッチ内の各サンプルは独立に「適用するか (p)」と「乱数パラメータ」を引くので、albumentations を
サンプルごとに呼んだのと同じ分布の拡張がかかる。マスク・bbox・keypoints にも対応 (幾何変換・dropout 系)。

> **速度について**: GPU に移せば常に速くなるわけではない。DataLoader の worker で並列化した albumentations の方が
> 速い変換も多い。GPU が明確に速いのは ShotNoise / ElasticTransform / GaussNoise など一部で、
> 計測結果と使い分けの目安は「[速度: いつ GPU に回すべきか](#速度-いつ-gpu-に回すべきか)」を参照。

## インストール

```bash
pip install -e .   # このディレクトリで実行 (依存は torch のみ。テストには albumentations / opencv-python が必要)
```

インストールせずに使う場合は、このディレクトリを `sys.path` に追加して import する。

## 使い方

```python
import gpu_batched_image_aug as G

gpu_aug = G.Compose(
    [
        G.HorizontalFlip(p=0.5),
        G.Affine(scale=(0.8, 1.2), rotate=(-20, 20), translate_percent=(-0.1, 0.1), p=0.7),
        G.OneOf([G.GaussianBlur(p=1), G.MotionBlur(p=1), G.MedianBlur(p=1)], p=0.3),
        G.ColorJitter(p=0.5),
        G.CoarseDropout(num_holes_range=(1, 4), hole_height_range=(0.05, 0.2), hole_width_range=(0.05, 0.2), fill_mask=0, p=0.5),
        G.ImageCompression(quality_range=(40, 90), p=0.3),
    ],
    bbox_params=G.BboxParams(format="pascal_voc", min_visibility=0.3),
    keypoint_params=G.KeypointParams(format="xy"),
    seed=0,  # 省略時は torch のグローバル乱数から初期化
)

out = gpu_aug(image=images, mask=masks, bboxes=boxes, bbox_valid=valid, keypoints=kps, keypoint_valid=kp_valid)  # 全て CUDA テンソル
images, masks, boxes, valid = out["image"], out["mask"], out["bboxes"], out["bbox_valid"]
kps, kp_valid = out["keypoints"], out["keypoint_valid"]
```

### 想定する構成: CPU (albumentations) と GPU (本ライブラリ) の併用

CPU で重く、かつ GPU 版が速い変換 (下の「速度」節を参照) だけをバッチ作成後に GPU で一括適用し、それ以外は
Dataset 内の albumentations (DataLoader の worker で並列化される) に残す。

```python
import albumentations as A

class MyDataset(torch.utils.data.Dataset):
    cpu_aug = A.Compose([A.HorizontalFlip(), A.Affine(p=0.5), A.ColorJitter(p=0.5)])  # CPU で十分速いもの
    def __getitem__(self, i):
        img = self.cpu_aug(image=load(i))["image"]                      # HWC uint8
        return torch.from_numpy(img).permute(2, 0, 1)                    # CHW uint8 (転送量が小さい)

for images in loader:                                                    # (B, C, H, W) uint8
    images = gpu_aug(image=images.cuda(non_blocking=True))["image"]
```

## 入出力の仕様

- **image**: `(B, C, H, W)` または `(B, H, W)`。uint8 (0..255) か浮動小数 ([0, 1])。出力は入力と同じ dtype・形状
  (uint8 は丸めて戻す)。内部は float32 [0, 1] で計算し、各変換の結果は [0, 1] に clip する。
- **mask**: `(B, H, W)` / `(B, Cm, H, W)`、dtype 任意。幾何変換は既定 nearest で再標本化し dtype を保つ。
  `fill_mask` / `mask_drop_value` を指定した dropout 系もマスクに作用する。
- **bboxes**: `(B, N, 4)` のパディングされたテンソル + `bbox_valid` `(B, N)` bool (省略時は全て有効)。
  albumentations のように除外された box を削除する代わりに `bbox_valid=False` にする (座標は clip 済みの値が残る)。
  ラベル等は呼び出し側で `(B, N, ...)` のまま並びを保てばよい。format は `pascal_voc` / `albumentations` /
  `coco` / `yolo`。各変換の後と最後に albumentations と同じ基準 (`min_area` / `min_visibility` / `min_width` /
  `min_height` / `max_accept_ratio`) でフィルタする。
- **keypoints**: `(B, K, F)` のパディングされたテンソル + `keypoint_valid` `(B, K)` bool (省略時は全て有効)。
  `KeypointParams(format, remove_invisible=True, angle_in_degrees=True, check_each_transform=True)` を Compose に渡す。
  format は `xy` / `yx` / `xya` / `xys` / `xyas` / `xysa` (a = 角度、s = スケール)。座標は albumentations と同じく
  画素中心が整数 (左上画素の中心が (0, 0))。format の列より後ろの列 (ラベル等) はそのまま素通しされる。
  画像外 (x ∉ [0, W) または y ∉ [0, H)) に出た点は削除の代わりに `keypoint_valid=False` になる (座標は残る)。
  角度は [0, 360) (または [0, 2π)) に正規化され、スケールは拡大率に応じて変わる (albumentations と同じ)。
- **fill 値** (`fill`, `drop_value` など) は albumentations と同じく入力画像の値のスケールで指定する
  (uint8 入力なら 0..255、float 入力なら 0..1)。`fill_mask` はマスクの値そのもの。
- **乱数**: `Compose(seed=...)` で再現可能。サンプルごとに独立にパラメータを引く。
- **サイズを変える変換** (`RandomCrop` / `CenterCrop` / `Crop` / `Resize` / `RandomResizedCrop`) はバッチ全体に
  適用する必要がある (`p=1`、`OneOf` 等の中には置けない)。ランダムなクロップ位置はサンプルごとに別々。
- interpolation / border_mode は cv2 の定数 (`cv2.INTER_LINEAR`, `cv2.BORDER_REFLECT_101` 等) か文字列で指定できる。

## 対応している変換

引数名・意味・既定値は albumentations 2.0.8 と同じ (下表の「主な引数」以外も同名で受け付ける)。
対象列: ○ = 作用する、− = 素通し、△ = 条件付き (備考参照)。dropout 系の keypoints △ は
`remove_invisible=True` のとき穴の中の点を無効にする (座標は変えない)。

### 幾何変換 (Flip / 回転 / Affine / 歪み)

| 変換 | image | mask | bboxes | keypoints | 主な引数 | 備考 |
|---|:-:|:-:|:-:|:-:|---|---|
| HorizontalFlip | ○ | ○ | ○ | ○ | | |
| VerticalFlip | ○ | ○ | ○ | ○ | | |
| Transpose | ○ | ○ | ○ | ○ | | 正方形画像のみ |
| RandomRotate90 | ○ | ○ | ○ | ○ | | 0/90/180/270 度。正方形画像のみ |
| D4 | ○ | ○ | ○ | ○ | | 正方形の 8 対称から一様選択。正方形画像のみ |
| Rotate | ○ | ○ | ○ | ○ | limit, interpolation, border_mode, fill, fill_mask, rotate_method | `crop_border=True` 未対応 |
| Affine | ○ | ○ | ○ | ○ | scale, translate_percent / translate_px, rotate, shear, keep_ratio, balanced_scale, rotate_method, border_mode, fill, fill_mask | `fit_output=True` 未対応 |
| ShiftScaleRotate | ○ | ○ | ○ | ○ | shift_limit(_x/_y), scale_limit, rotate_limit, border_mode | Affine の特殊ケース |
| Perspective | ○ | ○ | ○ | ○ | scale, interpolation, border_mode, fill, fill_mask | `keep_size=True, fit_output=False` のみ |
| ElasticTransform | ○ | ○ | ○ | ○ | alpha, sigma, approximate, same_dxdy, noise_distribution, border_mode | bbox はマスク化して remap、keypoints も同様 (albumentations の既定 `keypoint_remapping_method="mask"` と同じ方式) |
| GridDistortion | ○ | ○ | ○ | ○ | num_steps, distort_limit, normalized, border_mode | 同上 |
| OpticalDistortion | ○ | ○ | ○ | ○ | distort_limit, mode ("camera" / "fisheye"), border_mode | 同上 |

border_mode は constant / replicate / reflect / reflect_101 / wrap、interpolation は nearest / linear / cubic。

### クロップ / リサイズ (出力サイズが変わる: バッチ全体に p=1 で適用)

| 変換 | image | mask | bboxes | keypoints | 主な引数 | 備考 |
|---|:-:|:-:|:-:|:-:|---|---|
| RandomCrop | ○ | ○ | ○ | ○ | height, width | クロップ位置はサンプルごとにランダム。`pad_if_needed` 未対応 |
| CenterCrop | ○ | ○ | ○ | ○ | height, width | `pad_if_needed` 未対応 |
| Crop | ○ | ○ | ○ | ○ | x_min, y_min, x_max, y_max | 固定領域 |
| RandomResizedCrop | ○ | ○ | ○ | ○ | size, scale, ratio, interpolation, mask_interpolation | 領域はサンプルごとにランダム |
| Resize | ○ | ○ | ○ | ○ | height, width, interpolation, mask_interpolation | area 補間は整数倍の縮小で cv2 と一致 |

### Dropout

| 変換 | image | mask | bboxes | keypoints | 主な引数 | 備考 |
|---|:-:|:-:|:-:|:-:|---|---|
| CoarseDropout | ○ | △ | ○ | △ | num_holes_range, hole_height_range, hole_width_range, fill, fill_mask | fill: 値 / per-channel / "random" / "random_uniform"。mask は fill_mask 指定時。bbox は隠れた割合でフィルタし可視範囲に縮小 |
| GridDropout | ○ | △ | ○ | △ | ratio, unit_size_range, holes_number_xy, random_offset, shift_xy, fill, fill_mask | 同上 |
| XYMasking | ○ | △ | ○ | △ | num_masks_x/y, mask_x/y_length, fill, fill_mask | 同上 |
| PixelDropout | ○ | △ | △ | − | dropout_prob, per_channel, drop_value, mask_drop_value | mask は mask_drop_value 指定時。bbox は per_channel=False のとき可視率でフィルタ |
| ChannelDropout | ○ | − | − | − | channel_drop_range, fill | |

### 色 / 階調

| 変換 | image | mask | bboxes | keypoints | 主な引数 | 備考 |
|---|:-:|:-:|:-:|:-:|---|---|
| RandomBrightnessContrast | ○ | − | − | − | brightness_limit, contrast_limit, brightness_by_max | `ensure_safe_range` 未対応 |
| ColorJitter | ○ | − | − | − | brightness, contrast, saturation, hue | 4 操作の順序もサンプルごとにランダム |
| HueSaturationValue | ○ | − | − | − | hue_shift_limit, sat_shift_limit, val_shift_limit | 単位は albumentations と同じ (hue は OpenCV の 1 単位 = 2 度) |
| RGBShift | ○ | − | − | − | r/g/b_shift_limit | |
| RandomGamma | ○ | − | − | − | gamma_limit | |
| ChannelShuffle | ○ | − | − | − | | |
| ToGray | ○ | − | − | − | method, num_output_channels | method: weighted_average / desaturation / average / max / from_lab |
| ToSepia | ○ | − | − | − | | |
| InvertImg | ○ | − | − | − | | |
| Solarize | ○ | − | − | − | threshold_range | |
| Posterize | ○ | − | − | − | num_bits | チャネルごとの範囲指定も可 |
| Equalize | ○ | − | − | − | by_channels | mode="cv" のみ、mask 指定は未対応 |
| FancyPCA | ○ | − | − | − | alpha | |
| RandomToneCurve | ○ | − | − | − | scale, per_channel | |
| Normalize | ○ | − | − | − | mean, std | 出力は [0, 1] 外になるので最後に置き、float で受け取る |

### シャープ / エンボス

| 変換 | image | mask | bboxes | keypoints | 主な引数 | 備考 |
|---|:-:|:-:|:-:|:-:|---|---|
| Sharpen | ○ | − | − | − | alpha, lightness, method ("kernel" / "gaussian"), kernel_size, sigma | |
| UnsharpMask | ○ | − | − | − | blur_limit, sigma_limit, alpha, threshold | |
| Emboss | ○ | − | − | − | alpha, strength | |

### ノイズ

| 変換 | image | mask | bboxes | keypoints | 主な引数 | 備考 |
|---|:-:|:-:|:-:|:-:|---|---|
| GaussNoise | ○ | − | − | − | std_range, mean_range, per_channel, noise_scale_factor | |
| MultiplicativeNoise | ○ | − | − | − | multiplier, per_channel, elementwise | |
| ISONoise | ○ | − | − | − | color_shift, intensity | RGB のみ |
| SaltAndPepper | ○ | − | − | − | amount, salt_vs_pepper | 画素数は albumentations と同じく厳密に int(H*W*amount) |
| ShotNoise | ○ | − | − | − | scale_range | |

### Blur

| 変換 | image | mask | bboxes | keypoints | 主な引数 | 備考 |
|---|:-:|:-:|:-:|:-:|---|---|
| Blur | ○ | − | − | − | blur_limit | カーネルサイズはサンプルごとに別 (1 回の grouped conv で処理) |
| GaussianBlur | ○ | − | − | − | blur_limit, sigma_limit | |
| MedianBlur | ○ | − | − | − | blur_limit | |
| MotionBlur | ○ | − | − | − | blur_limit, angle_range, direction_range, allow_shifted | |
| Defocus | ○ | − | − | − | radius, alias_blur | |
| ZoomBlur | ○ | − | − | − | max_factor, step_factor | |

### 影 / 天候

| 変換 | image | mask | bboxes | keypoints | 主な引数 | 備考 |
|---|:-:|:-:|:-:|:-:|---|---|
| RandomShadow | ○ | − | − | − | shadow_roi, num_shadows_limit, shadow_dimension, shadow_intensity_range | |
| RandomFog | ○ | − | − | − | alpha_coef, fog_coef_range | |
| RandomRain | ○ | − | − | − | slant_range, drop_length, drop_width, drop_color, blur_value, brightness_coefficient, rain_type | |
| RandomSnow | ○ | − | − | − | brightness_coeff, snow_point_range | method="bleach" のみ |

### 圧縮 / 劣化

| 変換 | image | mask | bboxes | keypoints | 主な引数 | 備考 |
|---|:-:|:-:|:-:|:-:|---|---|
| ImageCompression | ○ | − | − | − | quality_range, treat_as_bgr | JPEG のみ (webp 未対応)。DCT ベースのシミュレーション |
| Downscale | ○ | − | − | − | scale_range, interpolation_pair | |

### 構成 (Composition)

| クラス | 動作 (サンプルごとに判定) |
|---|---|
| Compose(transforms, p=1, bbox_params, keypoint_params, seed) | 入口。確率 p でパイプライン全体を適用 |
| Sequential(transforms, p=0.5) | 入れ子用の Compose |
| OneOf(transforms, p=0.5) | 確率 p で、子の p に比例した確率で 1 つ選んで適用 |
| SomeOf(transforms, n=1, replace=False, p=1) | 確率 p で、子を n 個一様に選び、インデックス順に各自の p で適用 |

### 未対応 (GPU 版が無い) もの

- GPU バッチ化が難しい/不自然なもの: GlassBlur (画素の逐次入れ替え)、CLAHE (タイルごとのヒストグラム処理)、
  Superpixels、inpainting による fill (`fill="inpaint_*"`)
- 出力サイズがサンプルごとに変わる設定: `Affine(fit_output=True)`、`Perspective(keep_size=False / fit_output=True)`、
  `Rotate(crop_border=True)`、`RandomCrop/CenterCrop(pad_if_needed=True)`、LongestMaxSize / SmallestMaxSize / PadIfNeeded
- 今回は実装を見送ったもの: RandomSunFlare, RandomGravel, Spatter, PlanckianJitter, Illumination,
  PlasmaBrightnessContrast, ChromaticAberration, AdvancedBlur, ThinPlateSpline, PiecewiseAffine, GridElasticDeform,
  SafeRotate, AutoContrast
- 一部オプション: `RandomBrightnessContrast(ensure_safe_range=True)`, `ToGray(method="pca")`,
  `Equalize(mode="pil")`・`mask`, `RandomSnow(method="texture")`, `ImageCompression(compression_type="webp")`,
  歪み系の `keypoint_remapping_method="direct"`、keypoints の `xyz` 形式・`label_fields`

これらは上の「併用」構成で Dataset 側の albumentations に残す。

## albumentations との違い (意図的なもの・既知の差)

検証方法は「テスト」節を参照。数値は tests の実測。

- 乱数生成器が異なるので、同じ seed でも出力は一致しない (パラメータ分布は一致することをテストで確認)。
- 確率 p はサンプルごとに判定する (albumentations を 1 枚ずつ呼んだのと同じ意味)。
- bbox:
  - 除外は削除ではなく `bbox_valid=False`。
  - 変換後に完全に画像外へ出た box は常に無効 (albumentations は Affine 等の中で削除する)。
  - reflect 系 border で鏡像に写り込んだ物体の box を追加生成する処理 (albumentations の Affine 等) は行わない。
  - 歪み系で box のマスクが完全に消えた場合は無効にする (albumentations は元の座標を残す)。
  - 入力 box が画像外にはみ出していてもエラーにせず、最後のフィルタで clip する。
- keypoints:
  - 除外は削除ではなく `keypoint_valid=False`。ラベルは `label_fields` の代わりに `(B, K, ...)` の並びを保つか、
    format の後ろの列として渡す (素通しされる)。
  - 入力の時点で画像外にある点はエラーにせず無効にする (`remove_invisible=True` の場合)。
  - reflect 系 border で鏡像の位置に keypoints を追加生成する処理 (albumentations の Affine 等) は行わない。
  - 歪み系 (Elastic / Grid / Optical) で消えた点は `remove_invisible` に関係なく無効 (albumentations も削除する)。
- 浮動小数画像の結果は常に [0, 1] に clip する (albumentations は float 画像で clip しない変換がある)。
- ImageCompression: libjpeg を DCT ベースで再現したシミュレーション (エントロピー符号化は可逆なので省略)。
  cv2 の JPEG 出力との PSNR は quality 10〜95 で 32dB 超、元画像からの劣化量 (PSNR) の差は 1.5dB 未満。
  albumentations は RGB 配列をそのまま cv2 (BGR 前提) に渡しているため、既定 (`treat_as_bgr=True`) でその挙動を
  再現している。正しい RGB の JPEG にしたい場合は `treat_as_bgr=False`。
- RandomRain: 線は cv2 の LINE_4 と同じアルゴリズムで描画 (画像内に収まる線分は cv2 と完全一致)。画像の境界を
  またぐ線分は cv2 が描画前に clip するため経路が少し違う。`drop_width>1` は膨張で近似 (出力変化量の平均で 5% 程度の差)。
- RandomShadow: 多角形の塗りつぶしは even-odd 規則で、cv2.fillPoly との IoU は 0.95 超 (境界画素の差)。
- Downscale の `INTER_AREA` は adaptive average pooling (整数倍の縮小なら cv2 と一致、非整数倍は重みが少し違う)。
- ZoomBlur / Downscale / Resize / RandomResizedCrop は cv2.resize の座標の取り方 (half-pixel、nearest は floor、
  `fx` 指定時は `1/fx`) を再現している。
- FancyPCA: albumentations の式は固有ベクトルの符号に依存し、numpy (LAPACK) と cuSOLVER で符号が異なるため、
  3x3 の固有値分解だけ CPU で解いている (バッチごとに 1 回の小さな同期が入る)。
- 畳み込み (blur 等) は TF32 を無効にした完全な fp32 で計算する (TF32 だと cv2 との差が 1e-4 程度出たため)。

## 自作の変換

```python
class MyBrightness(G.ImageOnlyTransform):
    def __init__(self, limit=0.2, p=0.5):
        super().__init__(p)
        self.limit = limit

    def get_params(self, batch, rng):            # batch: 適用対象のサンプルだけの AugBatch
        return {"beta": rng.uniform(-self.limit, self.limit, batch.batch_size)}

    def apply(self, img, beta, **params):        # img: (n, C, H, W) float [0, 1]
        return (img + beta.view(-1, 1, 1, 1)).clamp(0, 1)
```

幾何変換は `G.DualTransform` を継承し、`apply_to_mask` / `apply_to_bboxes` / `apply_to_keypoints` も実装する
(keypoints は内部形式 `(n, K, 4)` = [x, y, 角度 (rad), スケール])。乱数は必ず `rng`
(デバイス上の `torch.Generator`) から引く。`gpu_batched_image_aug.functional` に warp / remap / per-sample
畳み込み / 色空間変換などの部品がある。

## テスト

```bash
python tests/run_tests.py            # pytest が無くても動く簡易ランナー
python tests/run_tests.py affine     # 名前で絞り込み
pytest tests                         # pytest があればそのままでも可
```

albumentations と OpenCV が必要。CUDA があれば GPU で実行する (`GBIA_TEST_DEVICE=cpu` で CPU)。

- `test_geometric_vs_albumentations.py`: 同じパラメータ (行列・座標マップ・クロップ位置) を与えて、
  albumentations / cv2 の関数と画像・マスク・bbox を比較 (flip / rot90 / D4 / transpose は完全一致、
  affine・perspective・remap は全 border mode × 補間で MAE < 1 (nearest は < 2.5)、変換行列・歪みマップ・bbox 一致)。
- `test_keypoints_vs_albumentations.py`: 同じパラメータで albumentations の keypoints 関数と比較 (flip / rot90 /
  D4 / transpose、Affine / ShiftScaleRotate / Rotate / Perspective の座標・角度・スケール、歪み系の mask 方式で
  残る点とその座標が完全一致、dropout の穴による除外が一致)、format 変換、パイプライン全体 (度数の角度・
  クロップ・リサイズ・フィルタ) を albumentations の Compose と比較。
- `test_pixel_vs_albumentations.py`: 色・ノイズ・blur・dropout・天候・圧縮を同じパラメータで比較
  (Equalize・Posterize・ToneCurve・メディアン等は完全一致、その他は最大 1 階調以内など)。
- `test_distributions_vs_albumentations.py`: 同じ引数の変換を albumentations で 300 回、こちらで 300 枚の
  バッチに 1 回かけ、出力の変化量の平均・標準偏差が一致することを確認 (61 設定)。パラメータの範囲・分布・
  確率の取り違えを検出する。
- `test_pipeline.py`: dtype / 形状、p と OneOf / SomeOf の確率、seed の再現性、サンプルごとの独立性、
  画像とマスクの幾何の一致、bbox の形式変換とフィルタ、fill 値のスケール、全変換のスモークテスト。

## 速度: いつ GPU に回すべきか

`benchmarks/bench_vs_albumentations.py` による実測値をまとめる。

**計測条件**
- 環境: CPU 20 コア、GPU は RTX 5090 1 枚 (他のジョブなし)。
- 入力: 一様乱数の uint8 RGB 画像を事前に生成して使い回す (ディスク読み込み・デコードは含まない)。
- albumentations 側: DataLoader の各 worker の `__getitem__` で 1 枚ずつ処理し、cv2 は 1 スレッド。
- GPU 側: 拡張なしの DataLoader → `.cuda()` → 本ライブラリでバッチ全体に 1 回処理。
- 数値は環境によって変わるので、目安として扱うこと。

### 結論

- **GPU が明確に速い変換**: ShotNoise (30 倍超)、ElasticTransform (3〜5 倍)、GaussNoise (約 3 倍)。
  いずれも albumentations の 16 workers 相当との比較。CPU で重い変換が多く、GPU に回す価値がある。
- **少し速い程度の変換**: ISONoise、MotionBlur、Affine (cubic)。
- **CPU で重いのに GPU でも速くない変換** (現状の実装の効率が悪い): RandomFog、MedianBlur、Defocus、ZoomBlur。
- **それ以外**: flip、crop、軽い色変換、GridDistortion / Perspective など。CPU でも 1 枚 1ms 程度以下で、
  DataLoader の worker で並列化すれば albumentations の方が速い。
- **GPU に移す意味が出るのは、CPU の albumentations がボトルネックになっているときだけ**。軽いパイプラインなら
  albumentations × n_workers で十分速く、GPU 版はかえって遅い (下の 1. を参照)。
- **本ライブラリを CPU テンソルで使う意味はない** (下の 3. を参照)。

### 1. 軽い典型パイプライン: albumentations × workers vs GPU

パイプラインは `HorizontalFlip(p=0.5)`、`Affine(scale=(0.8, 1.2), rotate=(-20, 20), translate_percent=(-0.1, 0.1), p=0.7)`、
`OneOf([GaussianBlur, MotionBlur, MedianBlur], p=0.3)`、`ColorJitter(p=0.5)`、`GaussNoise(std_range=(0.02, 0.08), p=0.5)`、
`CoarseDropout(p=0.5)`、`ImageCompression(quality_range=(40, 90), p=0.3)` の 7 変換。画像は 256×256、batch は 64。

albumentations は 1 コアで 0.94 ms/img。

| DataLoader workers | albumentations (img/s) | 読み込みのみ (img/s) | 読み込み + GPU 拡張 (img/s) |
|---:|---:|---:|---:|
| 1 | 869 | 6,912 | 1,697 |
| 4 | 3,444 | 15,414 | 1,780 |
| 8 | 7,658 | 17,299 | 1,705 |
| 16 | **15,742** | 18,043 | 1,650 |

GPU 側は、拡張自体の処理時間 (batch 64 で約 37ms) で頭打ちになる。worker 数を増やしても速くならないのは、
worker が読み込みしかしていないため。

### 2. GPU 版の batch size 依存 (同じパイプライン、256×256)

GPU 版では batch size が実質的な並列数になる。

| batch | ms/batch | img/s | 確保メモリの増分 |
|---:|---:|---:|---:|
| 1 | 11.8 | 85 | 42 MiB |
| 16 | 29.6 | 541 | 137 MiB |
| 64 | 38.2 | 1,673 | 401 MiB |
| 256 | 47.1 | 5,436 | 1.5 GiB |
| 1024 | 113.4 | 9,032 | 5.7 GiB |
| 2048 | 226 | 9,068 | (reserved 27 GiB) |

- **batch 128 程度まで**: 時間がほぼ一定。GPU の計算ではなく、呼び出しごとの固定オーバーヘッドが支配的。
  プロファイルでは、batch 64 の 1 回 (約 37ms) のうち GPU が計算している時間は約 5ms だけだった。
  残りはカーネル起動 (1 回あたり約 1,200 回) と同期 (約 120 回) で、主な原因はサンプルごとの適用判定と部分集合の取り出し。
- **batch 1024 以降**: 約 9,000 img/s で飽和する。albumentations の 16 workers (15,742 img/s) には届かない。
- **飽和の理由**: 1 枚あたりの増分コストは 0.088 ms/img (16 workers の albumentations は 0.059 ms/img)。
  - float32 画像を 1 回読み書きするだけの処理は 1.04 µs/img (実効 1.5 TB/s) なので、0.088 ms/img は
    画像全体を約 84 回読み書きするのに相当する。
  - 演算が融合されていないこと、float32 で計算していること、部分集合のコピー、MedianBlur のような効率の悪いカーネル
    (プロファイルで GPU 時間の 30%) が積み重なっている。
- **注意**: 大きい batch はメモリを大量に使う。同じプロセスで batch 1024 → 2048 と続けて回したところ、
  キャッシュが残って VRAM を超え (reserved 39.5 GiB)、batch 2048 が 1,723ms まで遅くなった。

### 3. 本ライブラリを CPU テンソルで使った場合 (同じパイプライン、256×256)

| batch | GPU (img/s) | CPU、torch 20 スレッド (img/s) | CPU、1 スレッド (img/s) |
|---:|---:|---:|---:|
| 1 | 85 | 275 | 234 |
| 16 | 541 | 355 | 76 |
| 64 | 1,673 | 310 | 66 |
| 256 | 5,436 | 249 | − |
| 1024 | 9,032 | 199 | − |

- CPU で動かすと、20 スレッドでも albumentations の 1 コア (約 1,065 img/s) より遅い。
- batch を増やすほど 1 枚あたりの時間が増える (1 スレッドだと全ての変換で batch 1 → 16 で 1.4〜4.5 倍)。原因は未確認。
- 速度の利点は GPU の並列性から来ている。バッチ化そのものによる高速化ではない。

### 4. 変換ごとの比較 (単体、p=1)

- albu ms/img: albumentations の 1 コアでの時間。
- 倍率: GPU のスループットを albumentations の 16 workers 相当 (1 コアの 16 倍の理想値) で割ったもの。
  1 より大きければ GPU の方が速い。実際の 16 workers は理想値の約 0.9 倍。
- GPU の batch は 256×256 で 256、512×512 で 64 (どちらも GPU が飽和する量)。

| 変換 | albu ms/img (256²) | 倍率 (256²) | albu ms/img (512²) | 倍率 (512²) |
|---|---:|---:|---:|---:|
| ShotNoise | 8.21 | **30.5** | 35.85 | **33.4** |
| ElasticTransform (alpha=30, sigma=5) | 1.78 | **3.7** | 9.44 | **5.0** |
| ElasticTransform (既定 sigma=50) | 7.64 | **3.1** | 31.60 | **3.0** |
| GaussNoise | 0.53 | **2.7** | 2.79 | **3.3** |
| MotionBlur (3〜21) | 3.36 | **3.4** | 6.48 | 1.5 |
| ISONoise | 1.43 | 1.4 | 6.95 | 1.6 |
| Affine (cubic) | 0.74 | 1.2 | 2.81 | 1.3 |
| ImageCompression | 0.83 | 1.2 | 3.13 | 1.1 |
| FancyPCA | 2.72 | 1.3 | 12.68 | 0.9 |
| Defocus | 5.58 | 0.6 | 13.39 | 1.1 |
| OpticalDistortion (0.3) | 0.32 | 0.9 | 1.37 | 1.0 |
| GridDistortion | 0.27 | 0.9 | 1.07 | 0.7 |
| RandomSnow | 0.59 | 0.7 | 2.37 | 0.8 |
| RandomFog | 15.04 | 0.5 | 58.07 | 0.6 |
| Perspective | 0.30 | 0.6 | 1.07 | 0.5 |
| RandomShadow | 0.50 | 0.5 | 1.91 | 0.5 |
| HueSaturationValue | 0.32 | 0.5 | 1.11 | 0.4 |
| MedianBlur (3〜7) | 1.37 | 0.4 | 5.75 | 0.4 |
| GaussianBlur (3〜15) | 0.12 | 0.3 | 0.50 | 0.4 |
| ZoomBlur | 0.76 | 0.2 | 4.00 | 0.3 |
| RandomRain | 0.18 | 0.1 | 0.63 | 0.1 |

RandomFog・MedianBlur・Defocus・ZoomBlur は、CPU で重いのに GPU 版も遅い。GPU 実装を改善すれば効果が大きい
候補だが、遅い原因はまだ調べていない。

### 再計測

```bash
python benchmarks/bench_vs_albumentations.py --batch 64 --workers 1 4 8 16       # 1. albumentations × workers vs GPU
python benchmarks/bench_vs_albumentations.py --sweep 1 16 64 256 1024             # 2. GPU、batch size 依存
python benchmarks/bench_vs_albumentations.py --sweep 1 16 64 --device cpu         # 3. CPU テンソルで実行
python benchmarks/bench_vs_albumentations.py --sweep 1 16 64 --device cpu --threads 1
python benchmarks/bench_vs_albumentations.py --per-transform --batch 256          # 4. 変換ごと (256×256)
python benchmarks/bench_vs_albumentations.py --per-transform --batch 64 --height 512 --width 512
```

他のジョブが GPU / CPU を使っていない状態で実行すること。並行して動くものがあると数値は比較できない。
