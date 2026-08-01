"""
V-JEPA 2 / V-JEPA 2.1 torch.hub 推論サンプル (ダミーデータ)
=============================================================

公式README「Pretrained backbones (via PyTorch Hub)」に記載の torch.hub.load
によるモデル読み込み方法を使い、ダミー動画データに対して実際に推論(predict)を
行う最小サンプル。encoder.py 等の疑似コードと異なり、本スクリプトは
torch.hub 経由でダウンロードした本物の学習済み重みに対して forward を実行する。

2つのモードを用意している (--mode で切り替え):

  --mode pretrain (デフォルト): 事前学習(SSL)時の "predict" を再現するモード。
    動画パッチを masks_enc(コンテキスト)/masks_pred(ターゲット) に分割し、
    Encoderにはコンテキストのみ見せ、Predictorにマスクされた部分の特徴を
    予測させる。V-JEPAの自己教師あり学習そのものを体験するためのモード。

  --mode downstream: 分類などの下流タスクで実際にモデルを使う時の形を
    再現するモード。masks は一切使わず(= None)、Encoderに動画全体を
    そのまま入力してパッチ特徴 (B, N, D) を得るだけ。predictorは呼ばない
    (下流タスクでは通常使わない。破棄してよい)。実運用ではこの特徴の上に
    分類ヘッド等 ([[finetune_example.py]] の AttentivePooler+Classifier
    のようなもの) を新たに学習して使う。

  つまり masks_enc/masks_pred は事前学習(pretrainモード)専用の概念であり、
  下流タスク(downstreamモード)では指定不要 (None) でよい、という対比を
  このスクリプト1本で確認できるようにしてある。

事前準備:
  pip install torch timm einops
  (torch.hub.load はコードと学習済み重み(数百MB~数GB)をネットワーク経由で
   ダウンロードするため、インターネット接続とディスク容量が必要)

動作確認は /home/ym/infra/jupyter/ai_sandbox の Docker イメージ内で行う想定。

対応する公式実装:
  - README.md の "Pretrained backbones (via PyTorch Hub)" セクション
  - notebooks/vjepa2_demo.py
  - src/hub/backbones.py, evals/hub/preprocessor.py
"""

import argparse

import numpy as np
import torch

# モデルごとの入力設定 (src/hub/backbones.py の _make_vjepa2_model 引数に対応)
MODEL_CONFIGS = {
    "vjepa2_vit_large": dict(img_size=256, num_frames=64),
    "vjepa2_vit_huge": dict(img_size=256, num_frames=64),
    "vjepa2_vit_giant": dict(img_size=256, num_frames=64),
    "vjepa2_vit_giant_384": dict(img_size=384, num_frames=64),
    "vjepa2_1_vit_base_384": dict(img_size=384, num_frames=64),
    "vjepa2_1_vit_large_384": dict(img_size=384, num_frames=64),
    "vjepa2_1_vit_giant_384": dict(img_size=384, num_frames=64),
    "vjepa2_1_vit_gigantic_384": dict(img_size=384, num_frames=64),
}
PATCH_SIZE = 16
TUBELET_SIZE = 2


def build_dummy_clip(img_size: int, num_frames: int) -> list:
    """(H, W, C) uint8 numpy 配列のリストとしてダミー動画フレーム列を生成する。"""
    return [np.random.randint(0, 256, (img_size, img_size, 3), dtype=np.uint8) for _ in range(num_frames)]


def build_dummy_masks(img_size: int, num_frames: int, batch_size: int, device: torch.device):
    """
    masks_enc / masks_pred とは何か:
      動画は (T/tubelet) x (H/patch) x (W/patch) 個のパッチに分割され、
      1〜n_patches のインデックスが振られる (パッチのID)。
      V-JEPAは動画の一部パッチだけをEncoderに見せ (= コンテキスト)、
      残りのパッチをPredictorに予測させる (= ターゲット/マスク) ことで学習する。

        masks_enc  : Encoderに見せる「可視」パッチのインデックス   shape=(B, N_ctx)
        masks_pred : Predictorが予測すべき「マスクされた」パッチのインデックス  shape=(B, N_pred)
        どちらも 0 <= index < n_patches の整数で、1パッチは両方に重複しない。

    ここでの実装は「全パッチをランダムにシャッフルして半分に割る」だけの
    簡略版であり、本物のV-JEPA(2/2.1)が学習時に使う手法ではない。
    本物は空間的に連続した矩形ブロックを時間方向に貫通させてマスクする
    "3D スパシオテンポラルブロックマスキング" ([[mask_generator.py]] の
    MaskGenerator、公式実装は src/masks/multiseq_multiblock3d.py) を使う。

    実際に使う場合にどうするか:
      - このダミー関数をそのまま使うのは動作確認・API理解のためだけに留めること。
      - 本物の(ランダムでなく意味のある)マスクで推論・学習したい場合は、
        同ディレクトリの mask_generator.py の MaskGenerator/MaskCollator、
        または公式リポジトリの src/masks/multiseq_multiblock3d.py の
        MaskCollator をDataLoaderのcollate_fnとして使い、
        (masks_enc, masks_pred) を生成すること。生成されるテンソルの
        形式 (インデックスのlong tensor, shape=(B, N)) はここと同じなので、
        置き換えるだけでこのスクリプトの encoder/predictor 呼び出しに渡せる。
    """
    n_patches = (num_frames // TUBELET_SIZE) * (img_size // PATCH_SIZE) ** 2
    n_ctx = n_patches // 2
    # 本来は「空間的に連続したブロック」を選ぶが、ここでは簡略化のため
    # 1〜n_patchesをシャッフルして前半をコンテキスト、後半をターゲットにするだけ
    perm = torch.stack([torch.randperm(n_patches, device=device) for _ in range(batch_size)])
    masks_enc = perm[:, :n_ctx]
    masks_pred = perm[:, n_ctx:]
    return masks_enc, masks_pred


def main():
    parser = argparse.ArgumentParser(description="V-JEPA 2 torch.hub ダミーデータ推論サンプル")
    parser.add_argument(
        "--model",
        default="vjepa2_1_vit_base_384",
        choices=list(MODEL_CONFIGS.keys()),
        help="torch.hub.load に渡すモデル名 (デフォルト: vjepa2_1_vit_base_384, 80M params)",
    )
    parser.add_argument(
        "--mode",
        default="pretrain",
        choices=["pretrain", "downstream"],
        help=(
            "pretrain: masks_enc/masks_predを使いEncoder+Predictorで事前学習時のpredictを再現 (デフォルト)。"
            "downstream: masks=Noneでencoder(x)のみ呼び出し、分類等の下流タスクでの使い方を再現"
        ),
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--pretrained",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="学習済み重みをダウンロードするか (--no-pretrained でランダム初期化のみ確認)",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    cfg = MODEL_CONFIGS[args.model]

    # 公式READMEのサンプル通り torch.hub.load でモデルを読み込む
    print("processor = torch.hub.load('facebookresearch/vjepa2', 'vjepa2_preprocessor')")
    processor = torch.hub.load("facebookresearch/vjepa2", "vjepa2_preprocessor", crop_size=cfg["img_size"])

    print(f"encoder, predictor = torch.hub.load('facebookresearch/vjepa2', '{args.model}', pretrained={args.pretrained})")
    encoder, predictor = torch.hub.load("facebookresearch/vjepa2", args.model, pretrained=args.pretrained)
    encoder = encoder.to(device).eval()
    predictor = predictor.to(device).eval()

    # ダミー動画データの用意 (実データの代わりに乱数フレームを使用)
    frames = build_dummy_clip(cfg["img_size"], cfg["num_frames"])
    clip = processor(frames)[0]  # (C, T, H, W)
    x = clip.unsqueeze(0).repeat(args.batch_size, 1, 1, 1, 1).to(device)  # (B, C, T, H, W)
    print(f"dummy input shape: {tuple(x.shape)}  (B, C, T, H, W)")

    if args.mode == "downstream":
        # ------------------------------------------------------------
        # 下流タスク(分類など)での使い方の再現
        # masks は使わない (= None)。動画全体をそのままEncoderに入力するだけ。
        # predictorは呼ばない(下流タスクでは通常不要、破棄してよい)。
        # ------------------------------------------------------------
        print("mode=downstream: masks は指定せず encoder(x) のみ呼び出す (predictorは未使用)")
        with torch.inference_mode():
            feats = encoder(x)  # masks=None → 全パッチをそのまま処理

        print(f"encoder output (all patch features) shape: {tuple(feats.shape)}  (B, N, D)")
        print("→ この特徴量に分類ヘッド等 (finetune_example.py の AttentivePooler+Classifier 相当) を")
        print("  新たに乗せて学習するのが、下流タスクでの実際の使い方。")
        return

    # ------------------------------------------------------------
    # mode=pretrain: 事前学習(SSL)時の "predict" の再現
    # masks_enc/masks_pred でパッチをコンテキスト/ターゲットに分割する。
    # ------------------------------------------------------------
    masks_enc, masks_pred = build_dummy_masks(
        img_size=cfg["img_size"], num_frames=cfg["num_frames"], batch_size=args.batch_size, device=device
    )
    print(f"masks_enc (context patch indices) shape: {tuple(masks_enc.shape)}  例: {masks_enc[0, :5].tolist()} ...")
    print(f"masks_pred (target patch indices) shape:  {tuple(masks_pred.shape)}  例: {masks_pred[0, :5].tolist()} ...")

    with torch.inference_mode():
        # 1. Encoder: コンテキストトークンのみを処理して特徴を抽出
        z_ctx = encoder(x, masks=masks_enc)
        # 2. Predictor: コンテキスト特徴からマスクされたターゲットトークンを予測
        # V-JEPA 2 predictor は Tensor を返すが、V-JEPA 2.1 predictor は
        # (予測ターゲット特徴, 予測コンテキスト特徴) のタプルを返す (Dense Prediction Loss用)
        pred_out = predictor(z_ctx, masks_x=masks_enc, masks_y=masks_pred)
        z_pred = pred_out[0] if isinstance(pred_out, tuple) else pred_out

    print(f"encoder output (context features) shape: {tuple(z_ctx.shape)}  (B, N_ctx, D)")
    print(f"predictor output (predicted target features) shape: {tuple(z_pred.shape)}")
    print(f"z_pred stats: mean={z_pred.mean().item():.4f}, std={z_pred.std().item():.4f}")


if __name__ == "__main__":
    # 実行例:
    #   python -m survey.VJEPA2.torch_hub_predict_demo                          # 事前学習(mask+predict)の再現
    #   python -m survey.VJEPA2.torch_hub_predict_demo --mode downstream        # 下流タスクでの使い方の再現 (masks不要)
    #   python -m survey.VJEPA2.torch_hub_predict_demo --model vjepa2_vit_large --no-pretrained
    main()
