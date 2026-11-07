"""Fish AudioのSlow/Fast Dual-ARを、実codec IDからscratch学習する。

学習対象:
    Slow Transformerは本文と過去の音声から次semantic IDを予測する。
    Fast TransformerはそのframeのSlow状態と既知book IDから残りbookを
    予測する。この2つのTransformerとembedding/headをrandom初期化し、
    PyTorchのDataLoader/AdamWで更新する。音声codecはここでは学習しない。

生データから始める場合:
    利用者に必要な列はaudio_path/textだけ。prepare_data.pyが実codec IDと
    公式書式の本文prefixを計算する。speaker_id/感情ラベルは要求しない。
    以下のcache列はその生成物。生成したtrain.parquet/config_scratch.jsonを
    このスクリプトへ渡せばよく、アノテーションを追加して埋める必要はない。

実行前に用意するもの:
    PyTorch、einops、pandas。このディレクトリを作業ディレクトリとする。
    本文は実tokenizerのchat templateに従ってID化し、音声は対応する
    公式codecでID抽出しておく。各.ptは辞書ではなくTensor単体を保存する。
    生のtext/audioからcacheを作る処理は同梱のprepare_data.pyに実装している。

DataFrame/Parquetの列（1行=1発話）:
    prefix_ids_path: int64[L]。system/user/assistant headerを含むprefix。
    target_codes_path: int64[N,T]。発話のGT codec ID、book-first形式。
    reference_codes_path（任意）: int64[N,Tr]の参照音声ID。
    after_reference_ids_path（参照ありなら必須）: int64[La]。
        参照直後の区切り、user本文、assistant headerまでを含むtoken列。
    N=codebook数（通常10）、T/Tr=target/referenceの約21.53Hz frame数、
    L/La=各text segmentのtoken数。CLIのpathはメタデータファイルの親基準。
    参照ありではprefixを参照挿入位置の直前で切る。参照なしではprefixに
    user本文とassistant headerまで含める。参照なしの行は参照pathを空欄にする。

token IDの2種類を区別する:
    textはglobal tokenizer ID。一方codesは各codebook内の0始まりID。
    Datasetが先頭bookへsemantic_beginを加え、Slowのglobal IDへ変換する。
    codec cacheの全bookへこのoffsetを加えてはいけない。
    vocab_size/semantic_begin/semantic_size/eos_idは実tokenizer、
    num_codebooks/codebook_sizeは実codecの定義と一致させる。

config.json:
    DualARConfigのfieldを書き、slow/fastはTransformerConfigの辞書にする。
    小型の特徴次元数・層設定例（語彙/ID fieldは使用資産に合わせて別途指定）:
        {"slow": {"dim": 256, "intermediate": 768, "layers": 4,
                  "heads": 4, "kv_heads": 2, "head_dim": 64},
         "fast": {"dim": 128, "intermediate": 384, "layers": 2,
                  "heads": 2, "kv_heads": 1, "head_dim": 64}}
    これは構造実験用で、S2-Pro公開重みの構成を意味しない。

使用例:
    python training_example.py --metadata data/train.parquet \
        --config configs/fish.json --output checkpoints/fish.pt \
        --epochs 10 --batch-size 2 --device cuda

    Pythonから呼ぶ場合:
        import json
        import pandas as pd
        from dual_ar import config_from_dict
        from training_example import load_metadata, train
        with open("configs/fish.json", encoding="utf-8") as handle:
            config = config_from_dict(json.load(handle))
        model = train(load_metadata("data/train.parquet"), config,
                      "checkpoints/fish.pt", device="cuda")

入力・GT・lossの関係:
    main_ids:[B,S]、codes:[B,S,N]。B=発話数、S=padding後の混在列長。
    Slowは入力を1位置ずらし、次のtarget semantic IDとEOSをCEで学ぶ。
    FastはGT先頭bookを含むbook方向teacher forcingで各bookのCEを取る。
    target音声とEOSだけを教師とし、reference/本文/paddingは条件に使う。
    loss=Slow CE+Fast CE。Python APIのcodebook_weightsでFast各bookの
    相対重みを指定できる（未指定なら均等）。

出力:
    outputにmodel/config/optimizer/stepを保存し、学習済みDualARを返す。
    再利用は保存configでDualARを構築しmodel.load_state_dictする。
    波形合成はmain_flow.pyで同じcodecへ接続する。単一processの学習例で、
    自動resume、検証分割、codec/GAN事前学習は含まない。
"""
import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

import pandas as pd
import torch
from torch.utils.data import DataLoader

from data import SpeechDataset, collate_sequences
from dual_ar import DualAR, DualARConfig, config_from_dict


def load_metadata(path: str | Path) -> pd.DataFrame:
    """準備済みParquetの相対cache pathを、そのメタデータファイルの親基準で解決して返す。

    raw音声/text表はprepare_data.pyへ渡し、そのtrain.parquetをここで読む。
    trainへ既存DataFrameを直接渡す場合は、読める実pathを呼び出し側で用意する。

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
    config: DualARConfig,
    output: str | Path,
    epochs: int = 1,
    batch_size: int = 1,
    lr: float = 1e-4,
    device: str = "cuda",
    codebook_weights: Sequence[float] | torch.Tensor | None = None,
) -> DualAR:
    """Dual-ARをrandom初期化して学習し、checkpointと学習済みmoduleを返す。

    dataframeのpath列はSpeechDatasetに従う。batchのmain_ids/各mask:[B,S]、codes:[B,S,N]。
    B=batch_size、S=batch内最大混在長、N=config.num_codebooks。
    codebook_weights:[N]は各Fast CEの重み。epochs=周回数、lr=AdamW学習率、output=保存先、
    device=計算device。

    Args:
        dataframe: 1行を1発話とする表。必要列は呼び出し元のstage schemaに従う。
        config: モデルの層数・特徴次元数・条件付け等を保持した設定。
        output: 生成するcache、checkpoint、または保存先ディレクトリ。
        epochs: データセットを反復する回数。
        batch_size: 1回のoptimizer更新に用いる発話数。
        lr: AdamWの学習率。
        device: 処理先。cpu、cuda:0等のPyTorchデバイス指定。
        codebook_weights: Fast CEのcodebook別重み[N]。Nはcodebook数。Noneでは既定の重み。

    Returns:
        checkpoint保存後の学習済みnn.Module。
    """
    # 1. Slow/Fastとembedding/headを新規に構築する。音声codecは含めない。
    model = DualAR(config).to(device).train()  # weightsはランダム。downloadなし。
    loader = DataLoader(SpeechDataset(dataframe, config),
                        batch_size=batch_size,
                        shuffle=True,
                        collate_fn=collate_sequences)
    # 2. collate済みbatchは混在位置Sを共有する2track。両trackを同時に使う。
    # AdamWはDual-ARの全パラメータを更新し、cacheのGT ID自体は固定する。
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = max(1, epochs * len(loader))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, total_steps)
    step = 0
    for _ in range(epochs):
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            # 3. main_ids/codes/各maskを同じdeviceへ移す。B/S/Nの軸は維持する。
            batch = {key: value.to(device) for key, value in batch.items()}
            # Slowは次時刻、Fastはその時刻のbook深さを教師にして、scalar CEを得る。
            losses = model(**batch, codebook_weights=codebook_weights)
            # 4. 同一GTから得たSlow/FastのCE和を逆伝播し、1batchで1回更新する。
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            step += 1
    # 5. tokenizerとcodecは別資産。ここにはDual-ARの構成・重み・更新状態を保存。
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
    parser.add_argument("--output", default="checkpoints/fish_dual_ar.pt")
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
