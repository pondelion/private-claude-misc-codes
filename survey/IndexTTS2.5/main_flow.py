"""学習済みIndexTTSの各段を接続し、本文から波形まで合成する。

処理順:
    本文ID→T2Sのsemantic ID→codecの復元特徴→Mel長への整列→
    参照条件付きCFMでMel生成→vocoderで波形生成。
    参照側のsemantic特徴は量子化せず整列し、生成側と時間方向に連結する。

利用準備:
    TextToSemantic、SemanticCodec、LengthRegulator、CFM、対応BigVGANの
    各学習済みmoduleをIndexTTSPipelineへ渡す。pipeline.eval()にして
    全moduleを同じdeviceへ移す。本文token化と参照特徴抽出は内部で行う。
    未学習moduleを接続するだけでは意味のある音声にならない。

入出力:
    forwardは本文text、話者参照の音声path、言語を受け取る。
    感情参照音声は任意。未指定なら話者参照音声から感情embeddingを計算する。
    話者/感情embeddingや参照Melを利用者が事前準備する必要はない。
    L=本文token数、Tr50=参照50Hz長、F=Mel bin数、R=参照Mel長。
    codes[1,U]、mel[1,F,M]、waveform[1,1,S]をdictで返す。
    U=生成semantic長、M=生成Mel長、S=波形sample数、1channel=mono。
    参照部分は返す波形に含めない。Mel長は指定、または公開推論の経験則で決める。
    波形の保存は呼び出し側が担当する。公式重みでの合成は別ファイルの
    pretrained_inference.pyを使う。このファイルはscratch学習した各段を接続する。

使用例（先にtraining_example.pyで各段を学習する）:
    python main_flow.py --model-dir checkpoints/IndexTTS-2.5 \
        --codec-checkpoint checkpoints/codec.pt \
        --t2s-checkpoint checkpoints/t2s.pt \
        --s2mel-checkpoint checkpoints/s2mel.pt \
        --text 'こんにちは。' --reference data/speaker.wav \
        --language JA --output outputs/scratch.wav
    感情参照を別にしたい場合だけ--emotion-reference data/emotion.wavを追加。
    固定抽出器とvocoderは公式資産、codec/T2S/S2Melはscratch checkpointを使う。
    モデル自動取得や追加installは行わない。出力はmono WAV。
"""
from __future__ import annotations

from pathlib import Path
import argparse
import sys

from torch import nn
import torch
import torchaudio

from data_processing.frontend import LANGUAGES, IndexTTSFrontend, language_id
from models.codec.model import SemanticCodec
from models.conditioning import render_mel
from models.s2mel.model import ConditionalFlowMatching, LengthRegulator
from models.t2s.model import TextToSemantic


class IndexTTSPipeline(nn.Module):
    """構築済みのT2S・codec・S2M・vocoderを接続する単一発話pipeline。"""

    def __init__(self, t2s: TextToSemantic, codec: SemanticCodec,
                 regulator: LengthRegulator, cfm: ConditionalFlowMatching,
                 vocoder: nn.Module, frontend: IndexTTSFrontend) -> None:
        """学習済み各段と共通frontendを登録する。

        自動構築はfrom_checkpointsを使う。

        frontendはprepare_data.pyと同じ固定抽出器・tokenizer・Mel設定を持つ。
        このconstructorへは実モジュールを渡し、参照embeddingは渡さない。

        Args:
            t2s: 本文と参照条件からsemantic IDを生成するモデル。
            codec: semantic IDと連続音声特徴を相互変換するモデル。
            regulator: semantic条件をMelのフレーム長へ整列するモジュール。
            cfm: 条件付きflow matchingによるMel生成モデル。
            vocoder: Mel[B,F,M]を波形[B,1,S]へ変換するモジュール。
            frontend: 音声・本文から条件特徴を計算する共通前処理。

        Returns:
            None。層・設定を登録する。ここでは学習・音声生成を行わない。
        """
        super().__init__()
        self.t2s, self.codec, self.regulator = t2s, codec, regulator
        self.cfm, self.vocoder = cfm, vocoder
        self.frontend = frontend
        self.sample_rate = frontend.mel_sample_rate

    @classmethod
    def from_pretrained(cls,
                        model_dir: str | Path,
                        device: str = "cuda:0") -> IndexTTSPipeline:
        """公式配布重みをローカル再現モデルへ読み、合成pipelineを構築する。

        外部IndexTTS pipelineをimportしない。重み以外の実行処理は本資料のローカル実装を使う。
        欠落した学習パラメータはrandom値で補完しない。

        Args:
            model_dir: 事前配置した公開重みと設定のディレクトリ。
            device: 処理先。cpu、cuda:0等のPyTorchデバイス指定。

        Returns:
            公開重みを読み込んだIndexTTSPipeline。
        """
        from data_processing.resources import load_assets
        from models.t2s.model import T2SConfig
        from models.s2mel.model import S2Mel
        from omegaconf import OmegaConf

        root = Path(model_dir).resolve()
        assets = load_assets(root, device)
        config = assets.cfg
        t2s = TextToSemantic(
            T2SConfig(official=OmegaConf.to_container(config.gpt)))
        codec = SemanticCodec(
            official=OmegaConf.to_container(config.semantic_codec))
        s2mel = S2Mel(official=OmegaConf.to_container(config.s2mel))
        checkpoint = torch.load(root / str(config.gpt_checkpoint),
                                map_location="cpu",
                                weights_only=True)
        state = checkpoint.get("model", checkpoint)
        # 配布時のinference cache用aliasは同じ学習パラメータの重複保存である。
        # ローカルGPTには不要だが、欠落した生成層を無視する処理とは区別する。
        aliases = {
            "inference_model.transformer.": "gpt.",
            "inference_model.embeddings.": "mel_embedding.",
            "inference_model.text_pos_embedding.": "mel_pos_embedding.",
            "inference_model.lm_head.0.": "final_norm.",
            "inference_model.lm_head.1.": "mel_head.",
        }
        cleaned = {}
        for name, value in state.items():
            canonical = ("mel_embedding.weight"
                         if name == "gpt.wte.weight" else name)
            for prefix, target in aliases.items():
                if name.startswith(prefix):
                    canonical = target + name[len(prefix):]
                    break
            if canonical == "gpt.wte.weight":
                canonical = "mel_embedding.weight"
            if canonical != name:
                if canonical not in state or not torch.equal(
                        value, state[canonical]):
                    raise ValueError(
                        f"Conflicting pretrained parameter alias: {name}")
                continue
            cleaned[name] = value
        state = cleaned
        t2s.core.load_state_dict(state, strict=True)
        codec_state = torch.load(root / "codec.pth",
                                 map_location="cpu",
                                 weights_only=True)
        codec.core.load_state_dict(codec_state["model"], strict=True)
        mel_state = torch.load(root / str(config.s2mel_checkpoint),
                               map_location="cpu",
                               weights_only=True)["net"]
        for name, network in s2mel.core.models.items():
            state = {
                key.removeprefix("module."): value
                for key, value in mel_state[name].items()
            }
            network.load_state_dict(state, strict=True)
        return cls(t2s, codec, s2mel.regulator, s2mel.flow, assets.bigvgan,
                   IndexTTSFrontend(assets)).to(device).eval()

    @classmethod
    def from_checkpoints(
        cls,
        model_dir: str | Path,
        codec_checkpoint: str | Path,
        t2s_checkpoint: str | Path,
        s2mel_checkpoint: str | Path,
        device: str = "cuda:0",
    ) -> IndexTTSPipeline:
        """学習済みscratch各段と、公式frontend/vocoderを読み込む。

        model_dirはprepare_data.pyと同じ公式資産。各checkpointはtraining_example.pyが保存した
        stage/config/modelを含む辞書。 公式T2Sの重みをscratch T2Sへ代用せず、指定した学習重みを使用する。

        Args:
            model_dir: 事前配置した公開重みと設定のディレクトリ。
            codec_checkpoint: 学習済みsemantic codecのcheckpoint。
            t2s_checkpoint: 学習済みT2Sのcheckpoint。
            s2mel_checkpoint: 学習済みS2Melのcheckpoint。
            device: 処理先。cpu、cuda:0等のPyTorchデバイス指定。

        Returns:
            指定したscratch学習の各stage重みを読み込んだIndexTTSPipeline。
        """
        from prepare_data import load_pretrained_assets
        from models.t2s.model import T2SConfig
        from models.s2mel.model import S2Mel

        # 1. データ準備と同じ固定抽出器・正規化統計・tokenizer・vocoderを読む。
        assets = load_pretrained_assets(Path(model_dir).resolve(), device)
        frontend = IndexTTSFrontend(assets)
        # 2. 保存configから各scratchモデルを再構築する。重みは厳密に照合する。
        codec_state = torch.load(codec_checkpoint,
                                 map_location="cpu",
                                 weights_only=True)
        t2s_state = torch.load(t2s_checkpoint,
                               map_location="cpu",
                               weights_only=True)
        s2mel_state = torch.load(s2mel_checkpoint,
                                 map_location="cpu",
                                 weights_only=True)
        if (codec_state["stage"], t2s_state["stage"],
                s2mel_state["stage"]) != ("codec", "t2s", "s2mel"):
            raise ValueError("Expected codec, t2s, s2mel stage checkpoints")
        codec = SemanticCodec(**codec_state["config"])
        t2s = TextToSemantic(T2SConfig(**t2s_state["config"]))
        config = s2mel_state["config"]
        s2mel = S2Mel(**config)
        codec.load_state_dict(codec_state["model"], strict=True)
        t2s.load_state_dict(t2s_state["model"], strict=True)
        s2mel.load_state_dict(s2mel_state["model"], strict=True)
        mel_bins = int(
            assets.cfg.s2mel["preprocess_params"]["spect_params"]["n_mels"])
        if s2mel.flow.estimator.in_channels != mel_bins:
            raise ValueError(
                "S2Mel and fixed vocoder Mel-bin definitions differ")
        # 3. 公式と同じ言語ID・codec語彙の構成であることを確認する。
        if (t2s.config.semantic_vocab != codec.vocabulary
                or t2s.core.text_embedding.num_embeddings
                < assets.tokenizer.encoding.n_vocab):
            raise ValueError("Checkpoint vocabulary configuration mismatch")
        return cls(t2s, codec, s2mel.regulator, s2mel.flow, assets.bigvgan,
                   frontend).to(device).eval()

    @torch.no_grad()
    def forward(
        self,
        text: str,
        reference_audio: str | Path,
        language: str = "JA",
        emotion_reference_audio: str | Path | None = None,
        target_mel_length: int | None = None,
        duration_factor: float = 1.0,
    ) -> dict[str, torch.Tensor]:
        """本文と参照音声から合成する。

        感情ラベル/embeddingの指定は不要。

        reference_audioは話者参照。感情参照を省略すると同じ音声を兼用する。
        languageはprepare_data.pyと共通のZH/EN/JA/ES/AR。戻り値はcodes[1,U]、Mel[1,F,M]、
        waveform[1,1,S]。 U=semantic frame数、F=Mel bin数、M=生成Mel長、S=波形sample数。

        Args:
            text: 読み上げる本文。
            reference_audio: 話者参照音声のファイルパス。
            language: 読み上げる言語コード。
            emotion_reference_audio: 任意の感情参照音声。Noneでは話者参照を兼用する。
            target_mel_length: 生成するMelのフレーム数。Noneではsemantic長から決める。
            duration_factor: 自動計算した生成長へ掛ける正の倍率。

        Returns:
            codes[1,U]、mel[1,F,M]、waveform[1,1,S]の辞書。1は単一発話、Uはsemantic長、
            FはMel bin数、Mは生成Mel長、Sは波形サンプル数。
        """
        # 1. 推論modeを確認し、文字列の言語指定を学習時と同じIDへ変換する。
        if self.training:
            raise ValueError("Call eval() before synthesis")
        language = language.upper()
        if language not in LANGUAGES:
            raise ValueError("Supported languages: ZH, EN, JA, ES, AR")
        device = next(self.t2s.parameters()).device
        # 2. 本文の正規化・G2P・tokenizer処理を内部で実行する。
        # text_ids[1,L]。L=言語prefixを含む本文token数、BOS/EOSはT2Sが追加。
        text_ids = self.frontend.tokenize(text, language)
        language_ids = torch.tensor([language_id(language)], device=device)
        # 3. 話者参照を読む。[C,Sref]、C=channel数、Sref=実sample数。
        waveform, sample_rate = torchaudio.load(str(reference_audio))
        # 4. 感情参照は任意。未指定では別ファイルを要求せず、frontend内で
        # 話者参照の特徴を再利用する。指定時のみ別音声[Ce,Se]を読み込む。
        emotion_waveform, emotion_rate = None, None
        if emotion_reference_audio is not None:
            emotion_waveform, emotion_rate = torchaudio.load(
                str(emotion_reference_audio))
        # 5. mono化・resample・W2v-BERT・CAMPPlus・感情抽出・Mel計算を実行。
        # 単一発話の話者[1,192]・感情[1,1024]・参照特徴[1,Tr50,1024]・Mel[1,F,R]。Tr50は約50Hz長、Fは
        # Mel bin数、Rは参照Mel長。
        # Tr50=参照約50Hz長、R=参照Mel長。感情embeddingを音声から自動計算する。
        conditions = self.frontend(waveform, sample_rate, emotion_waveform,
                                   emotion_rate)
        # 6. 抽出した条件をT2S→S2M→vocoderへ渡す。下の関数に各段を詳記する。
        return self.generate_from_features(
            text_ids,
            language_ids,
            conditions["speaker"],
            conditions["emotion"],
            conditions["reference_features"],
            conditions["reference_mel"],
            target_mel_length=target_mel_length,
            duration_factor=duration_factor,
        )

    @torch.no_grad()
    def generate_from_features(
        self,
        text_ids: torch.Tensor,
        language: torch.Tensor,
        speaker: torch.Tensor,
        emotion: torch.Tensor,
        reference_features: torch.Tensor,
        reference_mel: torch.Tensor,
        target_mel_length: int | None = None,
        duration_factor: float = 1.0,
    ) -> dict[str, torch.Tensor]:
        """単一発話。

        参照Mel[B,F,R]をCFM条件にし、target波形を返す。

        text_ids:[1,L]、language:[1]、speaker:[1,192]、emotion:[1,1024]。
        reference_features:[1,Tr50,1024]は話者参照の正規化W2v-BERT特徴。
        公開推論はreference側をcodec量子化せずregulatorへ渡す。 target_mel_lengthは学習済みduration
        predictorではなく利用者指定。 L=本文token数、Tr50=参照の約50Hz時刻数、F=Mel bin数、R=参照Mel長。
        返り値はcodes:[1,U]、mel:[1,F,M]、waveform:[1,1,S]。U=生成25Hz時刻数、M=生成Mel長、
        S=波形sample数、1channel=mono。

        Args:
            text_ids: 本文ID[B,L]。Bは発話数、Lはpadding込みの本文token数。
            language: モデル用言語ID[B]。Bは発話数。本文tokenと同じ発話順。
            speaker: 話者埋め込み[B,192]。Bは発話数、192はCAMPPlusの出力の特徴次元数。
            emotion: 感情埋め込み[B,1024]。Bは発話数、1024は感情encoderの出力の特徴次元数。
            reference_features: 話者参照特徴[1,Tr50,1024]。1は単一発話、
                Tr50は参照の約50Hzフレーム数。
            reference_mel: 話者参照Mel[1,F,R]。1は単一発話、FはMel bin数、Rは参照フレーム数。
            target_mel_length: 生成するMelのフレーム数。Noneではsemantic長から決める。
            duration_factor: 自動計算した生成長へ掛ける正の倍率。

        Returns:
            codes[1,U]、mel[1,F,M]、waveform[1,1,S]の辞書。1は単一発話、Uはsemantic長、
            FはMel bin数、Mは生成Mel長、Sは波形サンプル数。
        """
        # 1. 単一発話の推論として実行できるか確認する。
        # batch軸は1固定。eval()で学習時のdropout等を無効にしてから呼び出す。
        # 生成Mel長の指定値と、長さを調整する倍率は正の値に制限する。
        if self.training or text_ids.size(0) != 1:
            raise ValueError("pipeline requires eval() and batch=1")
        if duration_factor <= 0 or (target_mel_length is not None
                                    and target_mel_length < 1):
            raise ValueError(
                "duration_factor and requested Mel length must be positive")
        # 2. T2S：本文と話者・感情・言語の条件からsemantic ID列を生成する。
        # 入力は本文[1,L]、話者[1,192]、感情[1,1024]、言語ID[1]。1は単一発話、Lは本文長。
        # L=本文token数。次のIDを自己回帰生成し、終了tokenまで繰り返す。
        # 出力：codes[1,U]。U=約25Hzの生成frame数で、終了tokenは含まない。
        codes = self.t2s.generate(
            text_ids,
            speaker,
            emotion,
            language,
            reference_features,
        )
        if codes.size(1) == 0:
            raise RuntimeError("T2S generated EOS before any semantic frame")
        # 3. S2Mの前処理：離散IDを連続semantic特徴へ復元する。
        # codebook lookupとcodec decoderで[1,U]→[1,2U,1024]へ変換する。
        # 2U=復元後の約50Hz frame数、1024=各frameの特徴次元数。
        # この時点の出力はsemantic特徴であり、Melや波形への変換は後段で行う。
        semantic = self.codec.decode(codes)

        # 4. 生成対象のMel frame数Mを決める。
        # target_mel_lengthの指定があれば、その値をMとして使う。
        # 未指定なら公開推論の規則「復元frame数×1.72×duration_factor」を使う。
        # 1.72は約50Hz特徴長からMel長への経験的な比率。最小長は1frameとする。
        length = (target_mel_length if target_mel_length is not None else max(
            1, int(semantic.size(1) * 1.72 * duration_factor)))
        device = semantic.device

        # 5. 生成側のsemantic特徴を、生成対象のMel時間軸へ整列する。
        # 入力特徴[1,2U,1024]と、その有効長[1]、整列先のMel長[1]を渡す。
        # length regulatorが時間補間と学習済みConvで変換する。
        # 出力：target_condition[1,M,C]。C=regulatorの出力の特徴次元数。
        target_condition = self.regulator(
            semantic, torch.tensor([semantic.size(1)], device=device),
            torch.tensor([length], device=device))
        # 6. 参照側のsemantic特徴を、参照Melの時間軸へ整列する。
        # 参照特徴[1,Tr50,1024]をMel条件[1,R,C]へ整列する。Tr50は参照特徴長、Rは参照Mel長、Cは条件の特徴次元数。
        # Tr50=参照音声の約50Hz frame数、R=reference_mel[1,F,R]のframe数。
        # F=Mel bin数。参照側は正規化W2v-BERT特徴を使い、codecを通さない。
        ref_condition = self.regulator(
            reference_features,
            torch.tensor([reference_features.size(1)], device=device),
            torch.tensor([reference_mel.size(-1)], device=device))
        # 7. 参照区間を先頭、生成区間を後半に配置して条件列を構築する。
        # [1,R,C]と[1,M,C]を時間軸dim=1で連結し、[1,R+M,C]にする。
        # この順序により、CFM内で先頭R frameを観測済み参照として扱える。
        condition = torch.cat((ref_condition, target_condition), dim=1)

        # 8. S2Mの生成処理：参照条件付きFlow MatchingでMelを生成する。
        # 条件[1,R+M,C]、条件列の有効長[1]、参照Mel[1,F,R]、話者[1,192]。
        # sample()内でnoiseからvelocity推定とEuler更新を繰り返す。
        # 参照Melは観測条件として与え、生成状態とは別の入力で扱う。
        # 出力：mel[1,F,M]。参照区間Rを除いた、新規生成区間だけが返る。
        mel = self.cfm.sample(condition,
                              torch.tensor([condition.size(1)], device=device),
                              reference_mel, speaker)
        # 9. vocoder：生成MelをBigVGAN等の対応する学習済みvocoderへ渡す。
        # [1,F,M]→waveform[1,1,S]。S=生成波形のsample数、中央の1=mono。
        # Melの正規化・hop長・sample rateはvocoderの学習設定に揃える。
        waveform = render_mel(self.vocoder, mel)

        # 10. 各段階の出力を返す。参照音声は生成波形に含めない。
        # codes[1,U]、mel[1,F,M]、waveform[1,1,S]を呼び出し側で利用・保存する。
        return {"codes": codes, "mel": mel, "waveform": waveform}


def main() -> None:
    """本文と参照音声からscratch各段による音声を合成し、WAVへ保存する。

    Returns:
        None。コマンドラインで指定した処理と保存を行う。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--codec-checkpoint", type=Path, required=True)
    parser.add_argument("--t2s-checkpoint", type=Path, required=True)
    parser.add_argument("--s2mel-checkpoint", type=Path, required=True)
    parser.add_argument("--repo", type=Path)
    parser.add_argument("--text", required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--emotion-reference", type=Path)
    parser.add_argument("--language", choices=tuple(LANGUAGES), default="JA")
    parser.add_argument("--duration-factor", type=float, default=1.0)
    parser.add_argument("--target-mel-length", type=int)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    # 1. 出力の上書きと参照音声の破壊を防ぐ。モデル読込前に確認する。
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(args.output)
    for path in (args.reference, args.emotion_reference):
        if path is not None:
            if not path.is_file():
                raise FileNotFoundError(path)
            if path.resolve() == args.output.resolve():
                raise ValueError("Output must differ from reference audio")
    if args.repo is not None:
        sys.path.insert(0, str(args.repo.resolve()))
    # 2. scratch各段の学習済み重みと、データ準備に使った固定抽出器を読む。
    pipeline = IndexTTSPipeline.from_checkpoints(
        args.model_dir,
        args.codec_checkpoint,
        args.t2s_checkpoint,
        args.s2mel_checkpoint,
        args.device,
    )
    # 3. forward内で本文と音声を前処理する。感情参照の指定は省略できる。
    result = pipeline(
        args.text,
        args.reference,
        language=args.language,
        emotion_reference_audio=args.emotion_reference,
        target_mel_length=args.target_mel_length,
        duration_factor=args.duration_factor,
    )
    # 4. 発話batch軸とmono軸を除き、波形[1,1,S]→[S]をPCM16 WAVへ保存する。
    import soundfile as sf

    args.output.parent.mkdir(parents=True, exist_ok=True)
    waveform = result["waveform"][0, 0].cpu().float().numpy()
    sf.write(str(args.output),
             waveform,
             pipeline.sample_rate,
             subtype="PCM_16")


if __name__ == "__main__":
    main()
