"""VoxCPM2の本文→連続音声latent生成器をDataFrameからscratch学習する。

学習する範囲:
    本文と既知音声をLocEnc→TSLM→FSQ→RALMへ通し、次の音声patchを
    LocDiTのFlow Matchingで学習する。LocEnc、言語モデル、LocDiT、stop
    headはすべてrandom初期化する。AudioVAEは別途固定したものを使い、
    この学習では波形decoderやVAEの重みを更新しない。

生データから始める場合:
    必須列はaudio_path/textだけ。prepare_data.pyが本文IDと実VAE latentを
    計算しtrain.parquetへ保存する。speaker_id/感情ラベルは必須でない。
    以下のcache列は前処理の出力であり、手元に揃っていることを前提としない。
    生成されたconfig_scratch.jsonは実tokenizer/VAEのIDと特徴次元数に対応する。

必要条件とデータ準備:
    PyTorch、einops、pandasを用意し、このディレクトリで実行する。
    本文を実tokenizerでID化し、16kHz mono音声を対応するAudioVAE V2で
    encodeする。audio_vae.pyのadapterはそのlatentをpatchへ並べ替える。
    生音声の読み込み/token化/VAE抽出は同梱のprepare_data.pyが担当する。
    各cacheはtorch.save(tensor, path)で保存したTensor単体とする。

DataFrame/Parquetの列（1行=1発話）:
    text_ids_path: int64[L]の本文ID cache。
    latent_path: float32[N,P,C]のtarget音声latent cache。
    reference_path（任意）: float32[Nr,P,C]の別参照音声cache。
    L=本文token数、N/Nr=target/reference patch数、P=1patch内frame数、
    C=VAE特徴次元数。公式V2との接続はP=4、C=64、1frame=40msが基準。
    参照のない行はreference_pathを空欄にする。CLIのpathはメタデータファイルの親基準。
    referenceはloss教師ではなく声の条件。target最後のpatchはstopのGTにもなる。

config.json:
    VoxConfigのfieldを書き、base/localはTransformerConfigの辞書にする。
    小型実験の例（vocab_sizeと境界IDは実tokenizerへ合わせる）:
        {"vocab_size": 73448, "residual_layers": 2, "fsq_dim": 128,
         "base": {"dim": 256, "intermediate": 768, "layers": 4,
                  "heads": 4, "kv_heads": 2, "head_dim": 64},
         "local": {"dim": 256, "intermediate": 768, "layers": 2,
                   "heads": 4, "kv_heads": 2, "head_dim": 64}}
    省略した値はVoxConfigの既定を使う。既定は大きいので、初回の構成選択では
    base/local両方の特徴次元数と層数を確認する。上記は公開重みの構成ではない。

使用例:
    python training_example.py --metadata data/train.parquet \
        --config configs/vox.json --output checkpoints/voxcpm2.pt \
        --epochs 10 --batch-size 2 --device cuda

    Pythonから呼ぶ場合:
        import json
        import pandas as pd
        from model import config_from_dict
        from training_example import load_metadata, train
        with open("configs/vox.json", encoding="utf-8") as handle:
            config = config_from_dict(json.load(handle))
        model = train(load_metadata("data/train.parquet"), config,
                      "checkpoints/voxcpm2.pt", device="cuda")

1更新で何を予測するか:
    Datasetがreference・本文・targetを混在列にし、collateが右paddingする。
    patches:[B,S,P,C]、text IDs/各mask:[B,S]（B=発話数、S=混在列最大長）。
    1位置前のLM状態を次patchの条件にし、GT patchにnoiseを加えてvelocityを
    回帰する。CFMはtarget音声位置だけのMSE。stop headはtarget各位置で
    継続/終了の2クラスCEを取り、両lossの和で更新する。
    reference、本文、境界、paddingは音声生成lossの教師にしない。

出力と利用:
    model/config/optimizer/stepをoutputへ保存し、学習済みVoxCPM2を返す。
    再利用時はconfig_from_dict(checkpoint["config"])で構築してmodelの重みを
    load_state_dictする。合成では同じtokenizer/VAEをmain_flow.pyへ接続する。
    本例は単一processで、resumeや検証分割は実装していない。
"""
import argparse
import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from data import LatentDataset, collate_sequences
from model import VoxConfig, VoxCPM2, config_from_dict


def load_metadata(path: str | Path) -> pd.DataFrame:
    """cache Parquetを読み、*_path列をメタデータファイルの親基準で解決する。

    生音声/text表は先にprepare_data.pyへ渡す。trainへはこの戻り値を渡せる。既にDataFrameを持つ呼び出し側は、
    実行時に読めるpathを直接指定してもよい。

    Args:
        path: 読み書きするファイルのパス。

    Returns:
        cacheの相対パスを絶対パスへ解決したDataFrame。
    """
    path = Path(path).resolve()
    dataframe = pd.read_parquet(path, engine="pyarrow")
    for column in dataframe.columns:
        if column.endswith("_path"):
            dataframe[column] = [
                str((path.parent / value).resolve())
                if isinstance(value, str) and value else value
                for value in dataframe[column]
            ]
    return dataframe


def train(
    dataframe: pd.DataFrame,
    config: VoxConfig,
    output: str | Path,
    epochs: int = 1,
    batch_size: int = 1,
    lr: float = 1e-4,
    device: str = "cuda",
) -> VoxCPM2:
    """DataFrameからbackboneをscratch学習し、checkpointと学習済みmoduleを返す。

    dataframeはtext_ids_path/latent_path/任意reference_pathを持つ。
    各batchのpatches:[B,S,P,C]、各mask/ID:[B,S]。B=batch_size、S=batch内最大LM長、
    P=config.patch_size、C=config.latent_dim。epochs=周回数、lr=AdamW学習率、
    output=checkpoint保存先、device=計算device。

    Args:
        dataframe: 1行を1発話とする表。必要列は呼び出し元のstage schemaに従う。
        config: モデルの層数・特徴次元数・条件付け等を保持した設定。
        output: 生成するcache、checkpoint、または保存先ディレクトリ。
        epochs: データセットを反復する回数。
        batch_size: 1回のoptimizer更新に用いる発話数。
        lr: AdamWの学習率。
        device: 処理先。cpu、cuda:0等のPyTorchデバイス指定。

    Returns:
        checkpoint保存後の学習済みnn.Module。
    """
    # 1. cacheを混在列へpackし、batch内の最大列長まで右paddingする。
    # paddingはvalid=Falseとなるので、attentionのkeyやloss教師に含まれない。
    dataset = LatentDataset(dataframe, config)
    loader = DataLoader(dataset,
                        batch_size=batch_size,
                        shuffle=True,
                        collate_fn=collate_sequences)
    # 2. pretrained重みは読まず、VoxCPM2の全学習パラメータを新規に作る。
    # VAE latentの意味は固定し、このoptimizerにはVAEを登録しない。
    model = VoxCPM2(config).to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    # schedulerはepoch単位ではなくoptimizer更新単位で進める。
    total_steps = max(1, epochs * len(loader))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                           T_max=total_steps)
    step = 0
    for _ in range(epochs):
        for batch in loader:
            # 3. IDs、latent、maskを一緒に移動する。shapeは変えない。
            batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            # target patchだけのCFMと、同位置のstop CEをscalarへまとめる。
            losses = model(**batch, progress=step / total_steps)
            # 4. CFM+stop CEを逆伝播し、勾配normを制限してから1回更新する。
            # clipは勾配の発散対策で、loss値を上限で切る操作ではない。
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            step += 1
    # 5. 復元に必要な構成と重みを保存する。tokenizer/VAE資産は別途保持する。
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "config": asdict(config),
            "optimizer": optimizer.state_dict(),
            "step": step
        }, output)
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", default="checkpoints/voxcpm2.pt")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    with open(args.config, encoding="utf-8") as handle:
        config = config_from_dict(json.load(handle))
    train(load_metadata(args.metadata),
          config,
          args.output,
          args.epochs,
          args.batch_size,
          device=args.device)
