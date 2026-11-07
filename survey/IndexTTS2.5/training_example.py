"""IndexTTSの3段階を、抽出済み特徴とDataFrameから個別に学習する。

何をするスクリプトか:
    音声合成は「本文→semantic ID→Mel→波形」の順に進む。このファイルは
    codec、本文からIDを予測するT2S、ID由来特徴からMelを作るS2Melのうち、
    --stageで指定した1つをrandom初期化して学習する。公式trainerは使わず、
    DataLoader、AdamW、backwardで更新する。BigVGANの学習は含まない。

生データから始める場合:
    入力表はaudio_path/textだけでよい。prepare_data.pyが特徴、本文ID、
    speaker/emotion embedding、Mel、学習済みcodecのIDを計算する。
    speaker_idと感情ラベルは不要。以下の*_path列は前処理の出力であり、
    利用者が最初から持っているべきアノテーションではない。
    prepare_data.py --stage codec→codec学習→--stage allの順に実行する。
    詳しいCLIはprepare_data.pyとREADMEの「生音声と書き起こしからの準備」。

実行前の準備:
    このディレクトリを作業ディレクトリとし、PyTorch、pandas、einopsを
    用意する。GPT-2本体はmodels/t2s/gpt2.pyに実装している。
    生音声/本文からの特徴抽出はprepare_data.pyが行う。公式と同じ
    tokenizer、W2v-BERTの正規化統計、CAMPPlus、感情抽出器を使う。
    各.ptには辞書ではなく以下のTensorを1個ずつtorch.saveする。

固定資産の準備（前処理を省略しない場合の入口）:
    1. READMEの依存ライブラリと固定モデルの重みを準備する。
    2. prepare_assets.py --model-dir checkpoints/IndexTTS-2.5 --downloadで
       利用者が明示的に資産を取得する。既にある場合は--downloadを付けず検査。
    3. 以下の--raw-metadataで、資産検査→特徴抽出→指定stage学習まで実行する。
       固定資産の由来はprepare_assets.pyに列挙している。
       W2v-BERT/CAMPPlus/BigVGANは外部学習済みモデル、感情Conformerと
       PerceiverはIndexTTSのGPT checkpoint内の学習済み部分である。
       この学習loopのoptimizerには、これらの固定資産は含めない。

    python training_example.py --raw-metadata data/raw.parquet --stage codec \
        --model-dir checkpoints/IndexTTS-2.5 --cache-dir data/codec_cache \
        --language JA --config configs/config_codec_example.json \
        --output checkpoints/codec.pt
    python training_example.py --raw-metadata data/raw.parquet --stage t2s \
        --model-dir checkpoints/IndexTTS-2.5 --cache-dir data/t2s_cache \
        --language JA --codec-checkpoint checkpoints/codec.pt \
        --output checkpoints/t2s.pt
    --stage s2melでも同じraw入口を使える。既存cacheの再利用は--metadataで指定。

再現範囲:
    このファイルはcodec/T2S/公開DiT版S2Melのscratch学習loopである。
    既定は公式config全体を使い、小型化する場合だけvariantを明示する。
    論文のZipformer版、感情frontend自体の学習、BigVGANの学習、GRPOの
    rollout/評価更新loopはここには含まれず、公式精度を再現する完全recipeではない。
    モデル本体はローカル実装を使う。数値一致と学習精度はまだ実測していない。

前処理後のDataFrame/Parquet（1行=1発話、CLIのpathはメタデータファイルの親基準）:
    stage=codec:
        features_path: float32[T50,1024]。正規化済みW2v-BERT特徴。
    stage=t2s:
        text_ids_path: int64[L]。本文ID。BOS/EOSは本モデルが追加する。
        codes_path: int64[U]。先に学習・固定したcodecのsemantic ID。
        speaker_path: float32[192]。話者embedding。
        emotion_path: float32[1024]。感情embedding。
        language_id: 公式lang_to_tokenの整数ID。独自compact IDではない。
    stage=s2mel:
        features_path: float32[T50,1024]。正規化済みsemantic特徴。
        mel_path: float32[F,M]。対応vocoderと同じ設定・正規化のGT Mel。
        speaker_path: float32[192]。話者embedding。
    T50=約50Hz frame数、L=本文token数、U=約25Hz code数、
    F=Mel bin数（既定80）、M=Mel frame数。各発話の時間長は異なってよい。

config.jsonの書き方:
    公式config.yamlを参照する例:
        {"official_config_path": "checkpoints/IndexTTS-2.5/config.yaml"}
    公式T2Sを基準にした演算構造を維持し、容量だけを下げる例:
        {"official_config_path": "checkpoints/IndexTTS-2.5/config.yaml",
         "variant": {"layers": 8, "model_dim": 512, "heads": 8}}
    codec/T2S/S2Melで同じ指定形式を使用し、該当sectionを自動選択する。
    未指定fieldは公式値を保持する。語彙・dropout・mask等を独自値へ置換しない。
    学習checkpointには解決済みの全設定を保存する。

使用例（実データとconfigを準備してから実行）:
    python training_example.py --metadata data/t2s.parquet --stage t2s \
        --config configs/t2s.json --output checkpoints/t2s.pt \
        --epochs 10 --batch-size 4 --device cuda

    Pythonから呼ぶ場合:
        import pandas as pd
        from training_example import load_metadata, train
        model = train(load_metadata("data/t2s.parquet"), "t2s",
                      {"official_config_path":
                       "checkpoints/IndexTTS-2.5/config.yaml"},
                      "checkpoints/t2s.pt", device="cuda")

学習時の入力・GT・loss:
    codec: features自身がGT。50→25Hzで量子化し、50Hz特徴を復元する。
        feature再構成誤差とVQ/codebook損失で更新する。
    t2s: 本文/話者/感情/言語を条件に、GT codeを1位置ずらして入力する。
        次codeとEOSのCEに、本文側の次token CEを加える。
    s2mel: GT Melに1回noiseを加え、DiTに復元方向のvelocityを予測させる。
        先頭0〜30%を観測promptとして固定し、残りのvelocity MSEを取る。
    可変長発話を個別にforward/backwardし、batch内平均の勾配で1回更新する。
    batch-sizeはpadding後のTensorサイズではなく、更新に使う発話数である。

保存物と次の段階:
    outputへstage/config/model/optimizerを持つcheckpointを保存する。
    戻り値は学習したnn.Module。再構築後model.load_state_dictで重みを戻せる。
    codec更新後は古いcodesを流用せず再抽出し、T2Sの教師ID空間を固定する。
    合成には各段の学習済み重みと対応vocoderをmain_flow.pyで接続する。
    この例は単一processの学習で、再開、検証分割、音声生成評価は行わない。
"""
from pathlib import Path
from typing import Any, TypedDict, cast
import argparse
import json

from torch import nn
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import torch

from models.codec.model import SemanticCodec
from models.config import stage_config
from models.s2mel.model import S2Mel
from models.t2s.model import T2SConfig, TextToSemantic


def prepare_training_data(
    raw_metadata: Path,
    model_dir: Path,
    cache_dir: Path,
    stage: str,
    language: str,
    codec_checkpoint: Path | None,
    device: str,
) -> Path:
    """raw音声/text表を検査・前処理し、指定stage用Parquetのpathを返す。

    固定重みはprepare_assets.pyで事前準備する。この関数は取得しない。 t2s/s2melは既に学習したscratch codecを固定し
    てGT ID/復元特徴を抽出する。

    Args:
        raw_metadata: audio_pathとtext列を持つ入力表のパス。
        model_dir: 事前配置した公開重みと設定のディレクトリ。
        cache_dir: 生成した学習用cacheの保存先。
        stage: codec、t2s、s2melのいずれか。
        language: 読み上げ言語コードまたはモデル用言語ID。
        codec_checkpoint: 学習済みsemantic codecのcheckpoint。
        device: 処理先。cpu、cuda:0等のPyTorchデバイス指定。

    Returns:
        指定stageの学習用Parquetファイルのパス。
    """
    from prepare_assets import check_assets
    from prepare_data import load_pretrained_assets, prepare, read_source_table

    # 1. 重み不足はモデル構築前にpath一覧で報告する。
    check_assets(model_dir)
    if stage != "codec" and codec_checkpoint is None:
        raise ValueError("Train codec first and supply --codec-checkpoint")
    # 2. 必須データはraw音声path/textだけ。追加感情ラベルは要求しない。
    dataframe = read_source_table(raw_metadata)
    if not {"audio_path", "text"}.issubset(
            dataframe.columns) or dataframe.empty:
        raise ValueError("Nonempty audio_path/text table required")
    # 3. 固定抽出器で前処理する。公式生成重みはscratch学習の初期値に使わない。
    official = load_pretrained_assets(model_dir.resolve(), device)
    prepare(dataframe,
            raw_metadata.resolve().parent, cache_dir, official, language,
            "codec" if stage == "codec" else "all", codec_checkpoint)
    # 4. 公式pipelineの参照はここで解放し、その後scratchモデルを構築する。
    # PyTorchのdevice allocatorは解放済みmemoryを後続の学習に再利用できる。
    return cache_dir / f"{stage}.parquet"


def load_metadata(path: str | Path) -> pd.DataFrame:
    """準備済みParquetを読み、各*_path列をメタデータファイルの親基準の実pathへ解決する。

    prepare_data.pyが出力する相対cache pathを、起動ディレクトリによらず読む。
    Pythonから既存DataFrameをtrainへ直接渡す場合は、path解決を呼び出し側で行う。

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


class CachedSample(TypedDict, total=False):
    """stageに応じた1発話のcache。T50=50Hz長、M=Mel長、L/U=text/audio長。

    Attributes:
        features: semantic特徴[T50,1024]。T50は約50Hzのフレーム数。
        mel: GT Mel[F,M]。FはMel bin数、Mはフレーム数。
        speaker: 話者埋め込み[192]。
        emotion: 感情埋め込み[1024]。
        text_ids: 本文ID[L]。Lは本文token数。
        codes: semantic ID[U]。Uは約25Hzのtoken数。
        language: モデル用の整数言語ID。
    """

    features: torch.Tensor
    mel: torch.Tensor
    speaker: torch.Tensor
    emotion: torch.Tensor
    text_ids: torch.Tensor
    codes: torch.Tensor
    language: int


class CachedDataset(Dataset[CachedSample]):
    """DataFrameの各rowから、選んだstageに必要な実tensorだけを読み込む。"""

    def __init__(self, dataframe: pd.DataFrame, stage: str) -> None:
        """metadataとcodec/t2s/s2melのstage名を保持する。

        Args:
            dataframe: 1行を1発話とする表。必要列は呼び出し元のstage schemaに従う。
            stage: codec、t2s、s2melのいずれか。

        Returns:
            None。層・設定を登録する。ここでは学習・音声生成を行わない。
        """
        self.rows, self.stage = dataframe.to_dict("records"), stage

    def __len__(self) -> int:
        """metadataに登録された発話数を返す。

        Returns:
            登録された発話数。
        """
        return len(self.rows)

    def __getitem__(self, index: int) -> CachedSample:
        """指定rowのtensorをCPUへ読み、CachedSampleのshape契約で返す。

        Args:
            index: DataFrameから取得する発話の行番号。

        Returns:
            指定行の可変長Tensorを持つCachedSample。各フィールドのshapeは同クラスに定義する。
        """
        row = self.rows[index]
        columns = {
            "codec": ("features_path", ),
            "t2s":
            ("text_ids_path", "codes_path", "speaker_path", "emotion_path"),
            "s2mel": ("features_path", "mel_path", "speaker_path")
        }[self.stage]
        result = {
            column.removesuffix("_path"):
            torch.load(row[column], map_location="cpu", weights_only=True)
            for column in columns
        }
        if self.stage == "t2s":
            result["language"] = int(row["language_id"])
        return cast(CachedSample, result)


def keep_samples(samples: list[CachedSample]) -> list[CachedSample]:
    """可変長発話をpaddingせず保持するDataLoaderのcollate関数。

    Args:
        samples: 可変長の1発話cacheを保持するCachedSampleのリスト。

    Returns:
        入力と同じCachedSampleのリスト。可変長の時間軸をpaddingしない。
    """
    return samples


def train(
    dataframe: pd.DataFrame,
    stage: str,
    config: dict[str, Any],
    output: str | Path,
    epochs: int = 1,
    batch_size: int = 1,
    lr: float = 1e-4,
    device: str = "cuda",
) -> nn.Module:
    """指定stageをrandom初期化して学習し、checkpointと学習済みmoduleを返す。

    dataframeの列/shapeはCachedSampleとREADMEのstage別schemaに従う。
    batch_sizeは1optimizer更新あたりの発話数で、可変長発話のlossを平均する。epochs=周回数、lr=AdamW学習率、
    output=保存先、device=計算device。

    Args:
        dataframe: 1行を1発話とする表。必要列は呼び出し元のstage schemaに従う。
        stage: codec、t2s、s2melのいずれか。
        config: モデルの層数・特徴次元数・条件付け等を保持した設定。
        output: 生成するcache、checkpoint、または保存先ディレクトリ。
        epochs: データセットを反復する回数。
        batch_size: 1回のoptimizer更新に用いる発話数。
        lr: AdamWの学習率。
        device: 処理先。cpu、cuda:0等のPyTorchデバイス指定。

    Returns:
        checkpoint保存後の学習済みnn.Module。
    """
    config = stage_config(stage, config)
    # 1. 学習対象を1つ選ぶ。各stageで必要なGTとoptimizer対象は異なる。
    # codec学習とT2S学習を同時に行うとGT IDの意味が変わるため、別々に実施する。
    if stage == "codec":
        model = SemanticCodec(**config)
    elif stage == "t2s":
        model = TextToSemantic(T2SConfig(**config))
    elif stage == "s2mel":
        model = S2Mel(**config)
    else:
        raise ValueError("stage must be codec, t2s, or s2mel")
    model = model.to(device).train()
    # 2. collateでpaddingせず発話のlistを受け取る。元の有効長を保ったまま
    # 1発話ずつlossを計算するので、padding領域を誤ってGTにすることがない。
    loader = DataLoader(CachedDataset(dataframe, stage),
                        batch_size=batch_size,
                        shuffle=True,
                        collate_fn=keep_samples)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = max(1, epochs * len(loader))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, total_steps)
    for _ in range(epochs):
        for samples in loader:
            optimizer.zero_grad(set_to_none=True)
            # variable-length発話を逐次backwardし、batch全体の平均勾配を作る。
            # padding由来の偽教師を作らない。throughput最適化はlength bucketingで可能。
            # 3. 先に各lossを発話数で割ることで、順次backwardしてもbatch平均になる。
            # 全発話の計算graphを同時保持しないので、可変長でも使用memoryを抑える。
            for sample in samples:
                sample = {
                    key:
                    value.to(device)
                    if isinstance(value, torch.Tensor) else value
                    for key, value in sample.items()
                }
                if stage == "codec":
                    loss = model(sample["features"].float()[None])["loss"]
                elif stage == "s2mel":
                    loss = model(sample["features"].float(),
                                 sample["mel"].float(),
                                 sample["speaker"].float())
                else:
                    # L=text token数、U=semantic token数。発話batch軸を追加する。
                    # 本文IDへ発話軸を追加する。[L]→[1,L]。Lは本文token数。
                    text = sample["text_ids"].long()[None]
                    # semantic IDへ発話軸を追加する。[U]→[1,U]。Uは音声token数。
                    codes = sample["codes"].long()[None]
                    loss = model(
                        text, torch.tensor([text.size(1)],
                                           device=device), codes,
                        torch.tensor([codes.size(1)], device=device),
                        sample["speaker"].float()[None],
                        sample["emotion"].float()[None],
                        torch.tensor([sample["language"]],
                                     device=device))["loss"]
                (loss / len(samples)).backward()
            # 4. 発話batch全体の勾配が蓄積してから、1回だけclip/更新する。
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
    # 5. このstageの重みを保存する。次stageでは対応codec/特徴抽出器を固定する。
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "stage": stage,
            "config": config,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict()
        }, output)
    return model


def main() -> None:
    """固定資産とraw表の準備、または既存cacheの読込からstage学習を開始する。

    Returns:
        None。コマンドラインで指定した処理と保存を行う。
    """
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--metadata", type=Path)
    source.add_argument("--raw-metadata", type=Path)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--codec-checkpoint", type=Path)
    parser.add_argument("--language", choices=("ZH", "EN", "JA", "ES", "AR"))
    parser.add_argument("--repo", type=Path)
    parser.add_argument("--stage",
                        choices=("codec", "t2s", "s2mel"),
                        required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", default="checkpoints/index_stage.pt")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.config is not None and not args.config.is_file():
        raise FileNotFoundError(args.config)
    if (args.raw_metadata is not None and args.stage == "codec"
            and args.config is None):
        parser.error("Codec raw preparation requires --config")
    if args.repo:
        import sys

        sys.path.insert(0, str(args.repo.resolve()))
    metadata_path = args.metadata
    if args.raw_metadata is not None:
        if (args.model_dir is None or args.cache_dir is None
                or args.language is None):
            parser.error(
                "Raw input requires --model-dir, --cache-dir, --language")
        if args.cache_dir.exists():
            raise FileExistsError("Use a new --cache-dir or reuse --metadata")
        if args.stage != "codec" and args.codec_checkpoint is None:
            parser.error(
                "T2S/S2Mel raw preparation requires --codec-checkpoint")
        metadata_path = prepare_training_data(
            args.raw_metadata,
            args.model_dir,
            args.cache_dir,
            args.stage,
            args.language,
            args.codec_checkpoint,
            args.device,
        )
        if args.config is None and args.stage != "codec":
            args.config = args.cache_dir / f"config_{args.stage}.json"
    if args.config is None:
        parser.error("Supply --config for prepared metadata or codec training")
    with args.config.open(encoding="utf-8") as handle:
        config = json.load(handle)
    train(load_metadata(metadata_path),
          args.stage,
          config,
          args.output,
          args.epochs,
          args.batch_size,
          device=args.device)


if __name__ == "__main__":
    main()
