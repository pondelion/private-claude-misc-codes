"""公式実体との数値比較を、利用者が実行可能になった時点で行うスクリプト。

この作業中には実行しない。--run指定なしではモデルをimport/構築しない。
例:
    python -m verification.verify_official --run \
        --model-dir checkpoints/IndexTTS-2.5 \
        --report outputs/official_comparison.json
    --full-sizeで公式容量を比較する。未指定では層数・特徴次元数だけを小さくした
    同じ容量設定を公式側とローカル実装側に使う。Tensor演算を行うため資源に余裕が
    あるときだけ実行する。既存の学習processと並行実行しない。

比較方法:
    参照側は公式クラスを直接構築し、そのforward/quantize/decode等を呼ぶ。
    自作の参照式、dummy依存、mockによる置換は使わない。
    対応するstate_dictをstrict=Trueで移し、同じ入力・mode・乱数を使う。
    codec、T2S logits、regulator、DiT velocity、CFM出力を比較する。
    完了した場合だけ実測の最大絶対誤差をJSONへ保存する。
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
import argparse
import json
import sys

if TYPE_CHECKING:
    import torch


def compare(actual: torch.Tensor, expected: torch.Tensor) -> float:
    """同shapeの実Tensor出力を比較し、最大絶対誤差を返す。

    shapeは各呼び出し箇所に記す。許容値はFP32でrtol=1e-5/atol=1e-6。 ID列は浮動小数点へ変換せず完全一致で検査する。

    Args:
        actual: 再現実装の出力。shapeは比較箇所ごとに定義し、expectedと同一にする。
        expected: 公式側の出力。shapeとdtypeはactualと同一にする。

    Returns:
        actualとexpectedの最大絶対誤差。ID出力は完全一致を要求する。
    """
    import torch

    if actual.is_floating_point():
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    else:
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    return float((actual.float() - expected.float()).abs().max())


def verify(model_dir: Path, full_size: bool) -> dict[str, float]:
    """公式全設定、同じ重みと入力で比較する。

    モデルの学習や資産取得はしない。

    Args:
        model_dir: 事前配置した公開重みと設定のディレクトリ。
        full_size: 容量variantを使わず公式容量で比較するか。

    Returns:
        比較項目名をkey、最大絶対誤差をvalueとする辞書。
    """
    import torch
    from omegaconf import OmegaConf
    from torch.nn import functional as F

    from indextts.codec.models import EnhancedCodec
    from indextts.gpt.model_v2 import UnifiedVoice
    from indextts.s2mel.modules.commons import (
        MyModel as OfficialSemanticToMelNetwork, )

    from models.s2mel.model import S2Mel
    from models.config import stage_config
    from models.codec.model import SemanticCodec
    from models.t2s.model import T2SConfig, TextToSemantic

    spec = {"official_config_path": str(model_dir / "config.yaml")}
    variants = {
        "codec": {
            "vocos_dim": 64,
            "vocos_intermediate_dim": 128,
            "vocos_num_layers": 2
        },
        "t2s": {
            "model_dim": 128,
            "heads": 2,
            "layers": 2
        },
        "s2mel": {
            "DiT.hidden_dim": 128,
            "DiT.num_heads": 2,
            "DiT.depth": 2
        },
    }
    configs = {
        stage:
        stage_config(stage, {
            **spec,
            "variant": {} if full_size else variant,
        })
        for stage, variant in variants.items()
    }
    results = {}
    with torch.no_grad():
        # 1. Codec：公式側とローカル実装側を構築し、対応する全重みを移す。
        codec = SemanticCodec(**configs["codec"]).eval()
        reference = EnhancedCodec(**configs["codec"]["official"]).eval()
        reference.load_state_dict(codec.core.state_dict(), strict=True)
        for length in (8, 9):
            # 特徴[1,T50,D]。偶数/奇数の時間長で公式の末尾処理も比較する。
            features = torch.randn(1, length, reference.hidden_size)
            output = codec(features)
            expected = reference(features)
            results[f"codec_reconstruction_{length}"] = compare(
                output["reconstruction"], expected[0])
            quantized, ids = codec.encode(features)
            ref_ids, ref_quantized = reference.quantize(features)
            results[f"codec_ids_{length}"] = compare(ids, ref_ids)
            results[f"codec_quantized_{length}"] = compare(
                quantized, ref_quantized.transpose(1, 2))
            results[f"codec_decode_{length}"] = compare(
                codec.decode(ids), reference.decode(ref_ids))
        del codec, reference

        # 2. T2S：参照には実際のGPT2InferenceModelのfull-prefix forwardを使う。
        model = TextToSemantic(T2SConfig(**configs["t2s"])).eval()
        reference = UnifiedVoice(**configs["t2s"]["official"],
                                 spk_cond_mode="campplus").eval()
        reference.load_state_dict(model.core.state_dict(), strict=True)
        reference.post_init_gpt2_config(kv_cache=False, half=False)
        # B=2、L=4。0/1の境界IDと左paddingを含む本文でも比較する。
        text = torch.tensor([[2, 3, 4, 1], [5, 6, 0, 1]])
        speaker, emotion = torch.randn(2, 192), torch.randn(2, 1024)
        language = torch.zeros(2, dtype=torch.long)
        audio = torch.tensor([[reference.start_mel_token, 2, 3],
                              [reference.start_mel_token, 4, 5]])
        actual = model.token_logits(text, speaker, emotion, language, audio)
        # この条件配置は公式inference_speechのCAMPPlus経路からの同一演算。
        style = reference.spk_emb_proj(speaker)[:, None]
        projected = reference.emo_layer(reference.emovec_layer(emotion))
        condition = torch.cat(
            (style + projected[:, None], torch.zeros(2, 2,
                                                     reference.model_dim)),
            dim=1)
        fake, prefix, mask = reference.prepare_gpt_inputs(
            condition, text, language)
        reference.inference_model.store_mel_emb(prefix)
        # fake末尾はBOSなので、既知codeだけを後ろへ足す。[B,P+A]。
        ids = torch.cat((fake, audio[:, 1:]), dim=1)
        expected = reference.inference_model(
            input_ids=ids,
            attention_mask=F.pad(mask, (0, 2), value=1),
            use_cache=False,
            return_dict=True,
        ).logits[:, -audio.size(1):]
        results["t2s_logits"] = compare(actual, expected)
        del model, reference

        # 3. S2Mel：公式S2Melネットワークを参照にし、regulatorとDiTの出力を直接比較。
        model = S2Mel(**configs["s2mel"]).eval()
        reference = OfficialSemanticToMelNetwork(
            OmegaConf.create(configs["s2mel"]["official"])).eval()
        reference.load_state_dict(model.core.state_dict(), strict=True)
        regulator = reference.models["length_regulator"]
        config = configs["s2mel"]["official"]
        features = torch.randn(2, 7, config["length_regulator"]["in_channels"])
        source_lengths = torch.tensor([7, 5])
        lengths = torch.tensor([12, 9])
        actual = model.regulator(features, source_lengths, lengths)
        expected = regulator(features, ylens=lengths, n_quantizers=3,
                             f0=None)[0]
        results["length_regulator"] = compare(actual, expected)
        cfm = reference.models["cfm"]
        mel_bins = cfm.in_channels
        semantic = actual[:1]
        lengths = lengths[:1]
        noisy = torch.randn(1, mel_bins, 12)
        prompt_full = torch.zeros_like(noisy)
        prompt_full[:, :, :3] = torch.randn(1, mel_bins, 3)
        style = torch.randn(1, config["style_encoder"]["dim"])
        time = torch.tensor([0.4])
        for estimator in (model.flow.estimator, cfm.estimator):
            estimator.setup_caches(max_batch_size=1, max_seq_length=12)
        results["dit_velocity"] = compare(
            model.flow.estimator(noisy, prompt_full, lengths, time, style,
                                 semantic),
            cfm.estimator(noisy, prompt_full, lengths, time, style, semantic),
        )
        # 4. CFM：同じ乱数noiseとCFG設定で公式inferenceの最終Melを比較する。
        prompt = prompt_full[:, :, :3]
        torch.manual_seed(42)
        actual = model.flow.sample(semantic.clone(),
                                   lengths,
                                   prompt,
                                   style,
                                   steps=2,
                                   cfg=0.7)
        torch.manual_seed(42)
        expected = cfm.inference(semantic.clone(),
                                 lengths,
                                 prompt,
                                 style,
                                 None,
                                 2,
                                 inference_cfg_rate=0.7)[:, :, 3:]
        results["cfm_mel"] = compare(actual, expected)
    return results


def main() -> None:
    """明示的な--run指定時だけ比較し、成功した全項目の実測誤差を保存する。

    Returns:
        None。コマンドラインで指定した処理と保存を行う。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--repo", type=Path)
    parser.add_argument("--full-size", action="store_true")
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if not args.run:
        parser.error("Model execution is disabled without explicit --run")
    if args.repo:
        sys.path.insert(0, str(args.repo.resolve()))
    results = verify(args.model_dir, args.full_size)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
