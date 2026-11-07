"""公式VoxCPM2 pretrainedモデルでTTS / reference / continuationを行う。

使用例（未実行）:
    python pretrained_inference.py --model-dir checkpoints/VoxCPM2 \
        --text 'こんにちは。' --output outputs/voxcpm.wav

--referenceはtranscript不要の話者参照。continuationは--prompt-audioと
--prompt-textを両方渡す。自然言語の声/style指定は本文先頭の括弧内へ書く。

必要な準備:
    公式VoxCPM packageと依存、VoxCPM2のモデル資産をローカルへ用意する。
    --model-dirはconfig.json、backbone、AudioVAE、tokenizer等を含む
    公式loader用ディレクトリ。architectureがvoxcpm2かを先に確認する。
    package未installの場合は--repo repos/VoxCPMで公式repo rootを渡す。
    自動downloadはoffline指定で無効化し、denoiserもロードしない。

入力の3通り:
    本文だけ: 上の最小CLI例。
    声の参照: 最小例へ--reference data/speaker.wavを追加。
        この音声の書き起こしは不要で、独立した話者条件として使う。
    続きの生成: --prompt-audio data/prefix.wavと、その実際の書き起こしを
        --prompt-textで両方指定する。--textには新たに読む本文を渡す。
    音声の前処理は公式generateへ委譲する。Tensorを手動で作る必要はない。

生成条件と出力:
    --stepsはpatch内のflow更新回数、--cfgは条件の強さ、--max-patchesは
    生成patch数の上限。速度・品質・生成長を変える別の設定である。
    --deviceで計算先を選び、--seedで生成乱数を指定する。
    モデル読込→prefix構築→連続latent生成→AudioVAE decodeの後、monoの
    NumPy波形[S]を得る。S=生成sample数。モデルのsample rateでPCM16 WAVへ
    保存し、既存出力は--overwriteなしには置換しない。
    optimize=Falseによりロード時のcompile/warmupを無効にしている。
    scratch側の重みはmodel.py/main_flow.pyで扱い、このCLIには渡さない。
"""
import argparse
import json
import os
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    """本文、参照・継続音声、モデル資産と生成条件をCLIから読み取る。

    Returns:
        入力資産、本文、参照音声、出力先等を保持するargparse.Namespace。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo",
                        type=Path,
                        help="公式VoxCPM repoのroot。install済みなら不要")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--prompt-audio", type=Path)
    parser.add_argument("--prompt-text")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cfg", type=float, default=2.0)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--max-patches", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    """公式モデルを読み込み、生成したmono波形をWAVへ保存する。

    入力はCLIで指定する本文・音声ファイル。公式generateの戻り値はNumPy配列[S]で、Sは生成波形のsample数。
    batch軸・channel軸はない。sample rateはモデルの設定を使い、配列の時間軸を変更せず保存する。

    Returns:
        None。コマンドラインで指定した処理と保存を行う。
    """
    args = parse_args()
    model_dir, output = args.model_dir.resolve(), args.output.resolve()
    if not args.text.strip() or args.steps < 1 or args.max_patches < 1:
        raise ValueError(
            "nonempty text and positive steps/max-patches are required")
    if (args.prompt_audio is None) != (args.prompt_text is None):
        raise ValueError(
            "--prompt-audio and --prompt-text must be supplied together")
    if args.prompt_text is not None and not args.prompt_text.strip():
        raise ValueError(
            "prompt-text must contain the actual prompt transcript")
    for audio in (args.reference, args.prompt_audio):
        if audio is not None:
            if not audio.is_file():
                raise FileNotFoundError(audio)
            if audio.resolve() == output:
                raise ValueError(
                    "output must differ from reference/prompt audio")
    if output.exists() and not args.overwrite:
        raise FileExistsError(
            f"output already exists: {output}; use --overwrite")
    with (model_dir / "config.json").open(encoding="utf-8") as handle:
        if json.load(handle).get("architecture", "").lower() != "voxcpm2":
            raise ValueError("--model-dir must contain a VoxCPM2 checkpoint")
    if args.repo is not None:
        source = args.repo.resolve() / "src"
        if not (source / "voxcpm" / "core.py").is_file():
            raise FileNotFoundError("--repo must contain src/voxcpm/core.py")
        sys.path.insert(0, str(source))

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import soundfile as sf
    import torch
    from voxcpm import VoxCPM

    # 実checkpointを読む。scratchのmodel.pyとは別。denoiserの自動取得を無効化。
    # optimize=Falseでconstructor内のcompile/warmupも行わない。
    model = VoxCPM.from_pretrained(
        str(model_dir),
        local_files_only=True,
        load_denoiser=False,
        optimize=False,
        device=args.device,
    )
    with torch.inference_mode():
        waveform = model.generate(
            text=args.text,
            reference_wav_path=(str(args.reference.resolve())
                                if args.reference else None),
            prompt_wav_path=(str(args.prompt_audio.resolve())
                             if args.prompt_audio else None),
            prompt_text=args.prompt_text,
            cfg_value=args.cfg,
            inference_timesteps=args.steps,
            max_len=args.max_patches,
            denoise=False,
            seed=args.seed,
        )
    # 公式generate()はmono waveform[S]のNumPy配列を返す。rateはモデルから読む。
    sample_rate = model.tts_model.sample_rate  # VoxCPM2は通常48,000Hz。
    output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output), waveform, sample_rate, subtype="PCM_16")
    print(f"Saved pretrained VoxCPM2 audio: {output} ({sample_rate} Hz)")


if __name__ == "__main__":
    main()
