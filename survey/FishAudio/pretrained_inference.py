"""公式Fish Audio S2-Pro pretrainedモデルで参照音声付きTTSを行う。

使用例（未実行）:
    python pretrained_inference.py --model-dir checkpoints/s2-pro \
        --reference data/speaker.wav --reference-text '参照音声の実際の本文。' \
        --text '<|speaker:0|>こんにちは。[whisper]これは小さな声です。' \
        --output outputs/fish.wav

公式init_model/generate_longと実codecを使う。APIサーバーは不要。
対象はサーベイと同じS2-Pro。後発Drama 3のcheckpointは受け付けない。

事前に用意するもの:
    公式fish-speech packageと依存、S2-Proのbackbone/tokenizer/config、
    対応するcodec.pth。--model-dirにbackbone等を置き、codecはその直下、
    または--codecで別pathを指定する。package未installなら
    --repo repos/fish-speechでrepo rootを渡す。自動取得はofflineで止める。

入力の意味:
    --referenceは声を参照する実音声、--reference-textはその音声で実際に
    発話されている本文。--textは新たに読ませたい本文であり、別の文字列。
    speaker/styleの記法は使用tokenizerの仕様に従う。上の例ではspeaker:0と
    whisper tagを本文へ入れている。参照音声のmono化/resampleは公式codec側。

処理の流れ:
    参照音声→codec IDs[Nbook,Tref]→本文/参照prompt→Slow/Fast AR生成→
    chunkのIDを時間方向に連結→codec decode→PCM16 WAV保存。
    Nbook=book数、Tref=参照frame数。book軸を時間として連結してはいけない。
    --chunk-lengthは本文分割の長さで、波形sample数やcodec frame数ではない。
    --max-new-tokens、--top-p、--top-k、--temperatureはAR sampling条件。

出力と計算設定:
    --outputにmono WAVを保存する。sample rateはcodecから読み出す。
    --deviceで実行先を選び、--precision autoはCUDAのBF16対応状況に応じて
    dtypeを選ぶ（CPUはFP32）。compileは無効。--seedで乱数を指定する。
    出力の親ディレクトリは作り、既存ファイルは--overwriteがある場合だけ
    置換する。scratch側DualARの独自checkpointを読むためのCLIではない。
"""
import argparse
import json
import os
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    """モデル、参照音声と書き起こし、本文、sampling条件を読み取る。

    Returns:
        入力資産、本文、参照音声、出力先等を保持するargparse.Namespace。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo",
                        type=Path,
                        help="公式fish-speech repoのroot。install済みなら不要")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--codec", type=Path, help="省略時はmodel-dir/codec.pth")
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--reference-text", required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--precision",
                        choices=("auto", "float32", "float16", "bfloat16"),
                        default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--top-p", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--chunk-length", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    """公式Dual-ARとcodecで参照音声付きTTSを実行しWAVを保存する。

    参照・生成codec IDは[Nbook,T]。Nbookはcodebook数、Tはcodec frame数。
    生成chunkはframe軸だけで連結する。decode後のmono波形は[S]で、Sは音声sample数。CLIの本文・ファイルパスから生成し、
    戻り値はない。

    Returns:
        None。コマンドラインで指定した処理と保存を行う。
    """
    args = parse_args()
    model_dir = args.model_dir.resolve()
    codec_path = (args.codec or model_dir / "codec.pth").resolve()
    output = args.output.resolve()
    if not args.text.strip() or not args.reference_text.strip():
        raise ValueError(
            "text and the actual reference transcript are required")
    if (not 0 < args.top_p <= 1 or not 0 < args.temperature < 2
            or min(args.top_k, args.max_new_tokens, args.chunk_length) < 1):
        raise ValueError("invalid S2-Pro sampling parameters")
    if not args.reference.is_file():
        raise FileNotFoundError(args.reference)
    if not codec_path.is_file():
        raise FileNotFoundError(codec_path)
    if output == args.reference.resolve():
        raise ValueError("output must differ from reference audio")
    if output.exists() and not args.overwrite:
        raise FileExistsError(
            f"output already exists: {output}; use --overwrite")
    with (model_dir / "config.json").open(encoding="utf-8") as handle:
        model_type = json.load(handle).get("model_type")
    if model_type not in {"dual_ar", "fish_qwen3_omni"}:
        raise ValueError(
            f"expected S2-Pro Dual-AR checkpoint, got {model_type!r}")
    if args.repo is not None:
        repo = args.repo.resolve()
        if not (repo / "fish_speech").is_dir():
            raise FileNotFoundError("--repo must contain fish_speech/")
        sys.path.insert(0, str(repo))

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import soundfile as sf
    import torch
    from fish_speech.models.text2semantic.inference import (
        decode_to_audio,
        encode_audio,
        generate_long,
        init_model,
        load_codec_model,
    )

    device = torch.device(args.device)
    if args.precision == "auto":
        precision = torch.float32
        if device.type == "cuda":
            with torch.cuda.device(device):
                precision = (torch.bfloat16 if torch.cuda.is_bf16_supported()
                             else torch.float16)
    else:
        precision = getattr(torch, args.precision)
    torch.manual_seed(args.seed)
    # 公式loaderがtokenizerとsemantic ID bandも読み込む。手動でIDを決めない。
    model, decode_one_token = init_model(model_dir,
                                         str(device),
                                         precision,
                                         compile=False)
    codec = load_codec_model(codec_path, str(device), precision=precision)
    with torch.inference_mode():
        # 参照音声→IDs[Nbook,Tref]。Nbook=codebook数、Tref=参照frame数。
        # mono化/resampleも公式処理。
        reference_codes = encode_audio(args.reference, codec,
                                       str(device)).cpu()
        responses = generate_long(
            model=model,
            device=str(device),
            decode_one_token=decode_one_token,
            text=args.text,
            num_samples=1,
            max_new_tokens=args.max_new_tokens,
            top_p=args.top_p,
            top_k=args.top_k,
            temperature=args.temperature,
            compile=False,
            iterative_prompt=True,
            chunk_length=args.chunk_length,
            prompt_text=[args.reference_text],
            prompt_tokens=[reference_codes],
        )
        chunks = []
        try:
            for response in responses:
                if response.action == "sample":
                    if response.codes is None:
                        raise RuntimeError(
                            "official generator returned no code tensor")
                    # [Nbook,Tchunk]。Tchunk=このchunkの生成codec frame数。
                    chunks.append(response.codes)
                elif response.action == "next":
                    break  # 1sampleを構成する全text chunksが完了。
                else:
                    raise RuntimeError(
                        f"unexpected generation action: {response.action}")
        finally:
            responses.close()
        if not chunks:
            raise RuntimeError("EOS was emitted before any audio frame")
        codes = torch.cat(chunks, dim=1)  # [Nbook,sum(Tchunk)]。book軸は連結しない。
        waveform = decode_to_audio(codes.to(device),
                                   codec)  # 実codec -> mono[S]。
    output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output),
             waveform.cpu().float().numpy(),
             codec.sample_rate,
             subtype="PCM_16")
    print(f"Saved pretrained Fish S2-Pro audio: {output} "
          f"({codec.sample_rate} Hz)")


if __name__ == "__main__":
    main()
