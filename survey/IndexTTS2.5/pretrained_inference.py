"""公式配布のpretrained重みをローカル再現実装で読み、参照付きTTSを実行する。

外部IndexTTS2/UnifiedVoiceの実行APIをimportしない。
main_flow.IndexTTSPipeline.from_pretrainedが、ローカルのモデル定義へ
checkpointを読み込む。scratch学習重みを使う場合はmain_flow.pyのCLIを使う。

使用例（この作業ではモデル実行しない）:
    python pretrained_inference.py --model-dir checkpoints/IndexTTS-2.5 \
        --reference data/speaker.wav --language JA \
        --text 'こんにちは。' --output outputs/pretrained.wav

感情参照は任意。未指定なら話者参照音声から感情embeddingを抽出する。
外部pretrainedモデルはW2v-BERTと固定語彙のtokenizer、その他の固定重み。
CAMPPlus/BigVGAN/感情frontendを含むネットワーク構造もローカル実装を使用する。
"""
from __future__ import annotations

from pathlib import Path
import argparse
import os


def parse_args() -> argparse.Namespace:
    """モデル資産、本文、話者/任意感情参照、出力先を読み取る。

    Returns:
        入力資産、本文、参照音声、出力先等を保持するargparse.Namespace。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--emotion-reference", type=Path)
    parser.add_argument("--language",
                        choices=("ZH", "EN", "JA", "ES", "AR"),
                        required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--duration-factor", type=float, default=1.0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    """ローカル実装へpretrained重みを読み込み、mono PCM16 WAVを保存する。

    Returns:
        None。コマンドラインで指定した処理と保存を行う。
    """
    args = parse_args()
    # 1. 入出力pathの妥当性を、重み読込前に確認する。
    for path in (args.reference, args.emotion_reference):
        if path is not None:
            if not path.is_file():
                raise FileNotFoundError(path)
            if path.resolve() == args.output.resolve():
                raise ValueError("Output must differ from reference audio")
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(args.output)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import torch
    import soundfile as sf
    from main_flow import IndexTTSPipeline

    # 2. 許可されたpretrained重みを読み、推論modeのローカルpipelineを構築する。
    torch.manual_seed(args.seed)
    pipeline = IndexTTSPipeline.from_pretrained(args.model_dir, args.device)
    # 3. 本文と音声から全条件を内部抽出し、T2S→S2Mel→vocoderを実行する。
    with torch.inference_mode():
        result = pipeline(args.text,
                          args.reference,
                          language=args.language,
                          emotion_reference_audio=args.emotion_reference,
                          duration_factor=args.duration_factor)
    # 4. [1,1,S]→[S]。S=生成sample数。参照音声は返却波形に含まれない。
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(args.output),
             result["waveform"][0, 0].cpu().float().numpy(),
             pipeline.sample_rate,
             subtype="PCM_16")


if __name__ == "__main__":
    main()
