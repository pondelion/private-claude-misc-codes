"""Offline shape, gradient, mathematical-inverse, and official-core checks.

Run from this directory with a Python environment containing torch/einops.
No package installation, benchmark dataset, or network access is needed.
"""
from __future__ import annotations

import argparse
import ast
import importlib
import importlib.util
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parent


def load_source_definitions(path: Path, names: set[str], namespace: dict):
    """Execute named, unchanged definitions from an official Python file.

    Some official model files import unavailable wavelet packages at module
    import time. Extracting their original class/function AST nodes lets us
    compare the actual released core code without a dependency or a mock.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    selected = [
        node for node in tree.body
        if isinstance(node, (ast.ClassDef,
                             ast.FunctionDef)) and node.name in names
    ]
    assert {node.name for node in selected} == names
    module = ast.fix_missing_locations(
        ast.Module(body=selected, type_ignores=[]))
    exec(compile(module, str(path), "exec"), namespace)
    return namespace


def load_block(name: str):
    path = ROOT / name / "ts_block.py"
    spec = importlib.util.spec_from_file_location(f"survey_{name}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check_shapes_and_backward() -> None:
    cases = [
        ("AdaWaveNet", "AdaWaveNetBlock", lambda t: dict(seq_len=t,
                                                         levels=2,
                                                         d_model=8,
                                                         heads=2,
                                                         attention_layers=1,
                                                         trend_kernel=5)),
        ("WFTNet", "WFTNetBlock", lambda t: dict(
            seq_len=t, d_model=8, d_ff=16, top_k=2, wavelet_scale=2,
            num_kernels=2)),
        ("WPMixer", "WPMixerBlock",
         lambda t: dict(seq_len=t, levels=2, patch_len=4, patch_stride=2)),
        ("CycleNet", "CycleNetBlock",
         lambda t: dict(seq_len=t, cycle_len=4, d_model=8)),
        ("TimeMixer", "TimeMixerBlock",
         lambda t: dict(seq_len=t, d_model=8, d_ff=16, e_layers=1)),
        ("FITS", "FITSBlock", lambda t: dict(cut_freq=4)),
    ]
    for name, class_name, options in cases:
        cls = getattr(load_block(name), class_name)
        for length in (16, 17):
            model = cls(3, 5, **options(length))
            x = torch.randn(2, length, 3, requires_grad=True)
            if name == "FITS" and length % 2:
                try:
                    model(x)
                except ValueError as error:
                    assert "even T only" in str(error)
                else:
                    raise AssertionError("odd FITS length must be rejected")
                continue
            y = model(x)
            assert y.shape == (2, length, 5), (name, y.shape)
            assert torch.isfinite(y).all(), name
            y.square().mean().backward()
            assert x.grad is not None and torch.isfinite(x.grad).all(), name
            assert any(p.grad is not None for p in model.parameters()), name
        suffix = ("even shape/backward, odd rejected" if name == "FITS"
                  else "even/odd shape and backward")
        print(f"{name}: {suffix} OK")


def check_transforms() -> None:
    lifting = load_block("AdaWaveNet").AdaptiveLifting(3, 17)
    x = torch.randn(2, 3, 17, requires_grad=True)
    coarse, detail = lifting(x)
    reconstruction = lifting.inverse(coarse, detail)
    torch.testing.assert_close(reconstruction, x, atol=1e-5, rtol=1e-5)
    reconstruction.square().mean().backward()
    assert torch.isfinite(x.grad).all()

    wpmixer = load_block("WPMixer")
    for length in (16, 17):
        x = torch.randn(2, 3, length, requires_grad=True)
        low, details, lengths = wpmixer.haar_decompose(x, levels=2)
        reconstruction = wpmixer.haar_reconstruct(low, details, lengths)
        torch.testing.assert_close(reconstruction, x, atol=1e-6, rtol=1e-6)
        reconstruction.sum().backward()
        torch.testing.assert_close(x.grad, torch.ones_like(x))
    print("adaptive lifting and Haar exact reconstruction OK")


def check_fits_lowpass() -> None:
    block = load_block("FITS").FITSBlock(1, 1, cut_freq=3)
    with torch.no_grad():
        block.freq_upsampler.weight.copy_(torch.eye(3, dtype=torch.cfloat))
        block.freq_upsampler.bias.zero_()
        block.output.weight.fill_(1)
        block.output.bias.zero_()
    time = torch.arange(16, dtype=torch.float32)
    # Bin 6 is above the first three retained frequency bins.
    wave = torch.sin(2 * torch.pi * 6 * time / 16)[None, :, None]
    output = block(wave)
    assert output.abs().max() < 1e-5
    print("FITS high-frequency cut-off OK")


def check_ada_official_full(source: Path) -> None:
    """Compare the same-length AdaWaveNet path with released class bodies."""
    root = source / "AdaWaveNet" / "gh" / "AdaWaveNet"
    if not (root / "models" / "AdaWaveNet.py").exists():
        return

    class UnusedKMeans:
        """Only construction is needed; clusters are explicitly supplied."""

        def __init__(self, n_clusters: int) -> None:
            self.n_clusters = n_clusters

    namespace = {
        "torch": torch,
        "nn": torch.nn,
        "F": torch.nn.functional,
        "math": math,
        "KMeans": UnusedKMeans,
        "sqrt": math.sqrt,
    }
    load_source_definitions(root / "layers" / "LiftingScheme.py",
                            {"Splitting", "LiftingScheme",
                             "InverseLiftingScheme"}, namespace)
    load_source_definitions(root / "layers" / "Invertible.py",
                            {"RevIN"}, namespace)
    load_source_definitions(root / "layers" / "Embed.py",
                            {"DataEmbedding_inverted"}, namespace)
    load_source_definitions(root / "layers" / "SelfAttention_Family.py",
                            {"FullAttention", "AttentionLayer"}, namespace)
    load_source_definitions(root / "layers" / "Transformer_EncDec.py",
                            {"EncoderLayer", "Encoder"}, namespace)
    load_source_definitions(root / "models" / "AdaWaveNet.py",
                            {"normalization", "moving_avg", "series_decomp",
                             "AdpWaveletBlock", "InverseAdpWaveletBlock",
                             "ClusteredLinear", "Model"}, namespace)
    cfg = SimpleNamespace(
        task_name="long_term_forecast", seq_len=16, pred_len=16,
        output_attention=False, n_clusters=1, enc_in=3,
        lifting_levels=2, lifting_kernel_size=3,
        regu_details=0.0, regu_approx=0.0, d_model=8, n_heads=2,
        d_ff=16, e_layers=1, dropout=0.0, activation="gelu",
        embed="fixed", freq="h", factor=5)
    official = namespace["Model"](cfg)
    official.clusters = torch.zeros(3, dtype=torch.long)
    ada = load_block("AdaWaveNet")
    ours = ada.AdaWaveNetBlock(
        3, 3, seq_len=16, levels=2, d_model=8, heads=2,
        attention_layers=1, trend_kernel=24, lifting_kernel=3, dropout=0.0,
        d_ff=16, activation="gelu", n_clusters=1)

    for index in range(2):
        source_lifting = list(official.encoder_levels)[index].wavelet
        target_lifting = ours.lifting[index]
        for target, source_filter in (
                (target_lifting.update, source_lifting.U),
                (target_lifting.predict, source_lifting.P)):
            target.conv.load_state_dict(source_filter[1].state_dict())
            target.norm.load_state_dict(source_filter[3].state_dict())
        source_inverse = list(official.decoder_levels)[
            1 - index].inverse_wavelet
        ours.inverse_lifting[index].update.load_state_dict(
            source_inverse.wavelet.U.state_dict())
        ours.inverse_lifting[index].predict.load_state_dict(
            source_inverse.wavelet.P.state_dict())
        ours.coarse_skip[index].load_state_dict(
            list(official.linear_levels)[index][0].state_dict())
        ours.detail_mixer[index].load_state_dict(
            list(official.coef_linear_levels)[index][0].state_dict())
        ours.detail_decoder[index].load_state_dict(
            list(official.coef_dec_levels)[index][0].state_dict())

    ours.coarse_embedding.load_state_dict(
        official.enc_embedding.value_embedding.state_dict())
    for target, reference in zip(ours.channel_encoder,
                                 official.encoder.attn_layers):
        for own_proj, ref_proj in (
                (target.attention.query,
                 reference.attention.query_projection),
                (target.attention.key,
                 reference.attention.key_projection),
                (target.attention.value,
                 reference.attention.value_projection),
                (target.attention.output,
                 reference.attention.out_projection)):
            own_proj.load_state_dict(ref_proj.state_dict())
        for attr in ("conv1", "conv2", "norm1", "norm2"):
            getattr(target, attr).load_state_dict(
                getattr(reference, attr).state_dict())
    ours.channel_norm.load_state_dict(official.encoder.norm.state_dict())
    ours.coarse_readout.load_state_dict(
        official.lowrank_projection.state_dict())
    ours.season_projection.load_state_dict(official.projection.state_dict())
    ours.trend_mixer[0].load_state_dict(
        official.trend_linear.linear_layers["0"].state_dict())
    with torch.no_grad():
        ours.trend_affine_weight.copy_(official.rev_trend.affine_weight)
        ours.trend_affine_bias.copy_(official.rev_trend.affine_bias)
        ours.output.weight.copy_(torch.eye(3))
        ours.output.bias.zero_()

    x = torch.randn(2, 16, 3, requires_grad=True)
    actual = ours(x)
    expected = official.forecast(x, None, None, None,
                                 official.clusters)
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
    # Released forecast normalizes via an in-place division, so backward
    # through that exact source raises an autograd version-counter error.
    # The requested parity criterion here is the forward output.
    print("AdaWaveNet official full same-length forward parity OK")


def check_official_cores(source: Path) -> None:
    ada_file = (source / "AdaWaveNet" / "gh" / "AdaWaveNet" / "layers" /
                "LiftingScheme.py")
    if ada_file.exists():
        spec = importlib.util.spec_from_file_location("ada_lifting_official",
                                                      ada_file)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        official = module.LiftingScheme(3,
                                        input_size=16,
                                        modified=True,
                                        k_size=3,
                                        simple_lifting=True)
        ada = load_block("AdaWaveNet")
        lifting = ada.AdaptiveLifting(3, input_length=16, kernel_size=3)
        lifting.update.conv.load_state_dict(official.U[1].state_dict())
        lifting.update.norm.load_state_dict(official.U[3].state_dict())
        lifting.predict.conv.load_state_dict(official.P[1].state_dict())
        lifting.predict.norm.load_state_dict(official.P[3].state_dict())
        x = torch.randn(2, 3, 16, requires_grad=True)
        actual = lifting(x)
        expected = official(x)
        for a, e in zip(actual, expected):
            torch.testing.assert_close(a, e, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(
            torch.autograd.grad(sum(y.square().sum() for y in actual),
                                x,
                                retain_graph=True)[0],
            torch.autograd.grad(sum(y.square().sum() for y in expected), x)[0],
            atol=1e-5,
            rtol=1e-5)
        torch.testing.assert_close(ada._instance_normalize(actual[0]),
                                   torch.nn.InstanceNorm1d(3, affine=False)(
                                       actual[0]),
                                   atol=1e-6,
                                   rtol=1e-6)
        print("AdaWaveNet official modified lifting/normalization parity OK")
        check_ada_official_full(source)

    wft_root = source / "WFTNet" / "gh" / "WFTNet"
    wft_conv_file = wft_root / "layers" / "Conv_Blocks.py"
    wft_model_file = wft_root / "models" / "Wavelet.py"
    if wft_conv_file.exists() and wft_model_file.exists():
        spec = importlib.util.spec_from_file_location("wft_conv_official",
                                                      wft_conv_file)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        official = module.Inception_Block_V1(8, 16, num_kernels=2)
        ours = load_block("WFTNet").InceptionBlock(8, 16, num_kernels=2)
        ours.load_state_dict(official.state_dict())
        grid = torch.randn(2, 8, 4, 7, requires_grad=True)
        own_grid, ref_grid = ours(grid), official(grid)
        torch.testing.assert_close(own_grid, ref_grid)
        torch.testing.assert_close(
            torch.autograd.grad(own_grid.square().sum(),
                                grid,
                                retain_graph=True)[0],
            torch.autograd.grad(ref_grid.square().sum(), grid)[0])
        import numpy as np
        import ptwt

        class PublishedPTWT:
            @staticmethod
            def cwt(data, scales, wavelet):
                # WFTNet pins ptwt 0.1.6 (internal precision=10). In 1.x the
                # explicit keyword reproduces the old coefficients exactly.
                return ptwt.cwt(data, scales, wavelet, precision=10)

        namespace = load_source_definitions(wft_model_file,
                                            {"FFT_for_Period",
                                             "Wavelet_for_Period"},
                                            {
                                                "torch": torch,
                                                "np": np,
                                                "ptwt": PublishedPTWT,
                                            })
        x = torch.randn(2, 16, 8)
        official_periods, official_weights = namespace["FFT_for_Period"](x, 2)
        spectrum = torch.fft.rfft(x, dim=1)
        energy = spectrum.abs().mean(dim=(0, 2))
        energy[0] = -torch.inf
        indices = torch.topk(energy, 2).indices
        torch.testing.assert_close(torch.as_tensor(official_periods),
                                   16 // indices)
        torch.testing.assert_close(official_weights,
                                   spectrum.abs().mean(-1)[:, indices])
        wft_block = load_block("WFTNet").WFTBlock(
            8, d_ff=16, top_k=2, period_coeff=0.8,
            wavelet_scale=2, num_kernels=2)

        namespace.update({
            "nn": torch.nn,
            "F": torch.nn.functional,
            "Inception_Block_V1": module.Inception_Block_V1,
        })
        load_source_definitions(wft_model_file, {"Wavelet"}, namespace)
        config = SimpleNamespace(seq_len=16, pred_len=0, top_k=2,
                                 period_coeff=0.8, wavelet_scale=2,
                                 d_model=8, d_ff=16, num_kernels=2)
        official_wavelet = namespace["Wavelet"](config)
        for own, ref in zip(wft_block.period_conv, official_wavelet.period_conv):
            if isinstance(own, torch.nn.Module) and hasattr(own, "kernels"):
                own.load_state_dict(ref.state_dict())
        wft_block.scale_conv.load_state_dict(
            official_wavelet.scale_conv.state_dict())
        x = torch.randn(2, 16, 8, requires_grad=True)
        actual = wft_block(x)
        expected = official_wavelet(x)
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(
            torch.autograd.grad(actual.square().sum(), x,
                                retain_graph=True)[0],
            torch.autograd.grad(expected.square().sum(), x)[0],
            atol=1e-4, rtol=1e-4)
        print("WFTNet official Wavelet forward with ptwt precision=10 parity OK")

        embed_ns = load_source_definitions(
            wft_root / "layers" / "Embed.py",
            {"PositionalEmbedding", "TokenEmbedding", "FixedEmbedding",
             "TemporalEmbedding", "TimeFeatureEmbedding", "DataEmbedding"},
            {"torch": torch, "nn": torch.nn, "F": torch.nn.functional,
             "math": math})
        namespace["DataEmbedding"] = embed_ns["DataEmbedding"]
        load_source_definitions(wft_model_file, {"Model"}, namespace)
        for top_k in (0, 2):
            candidate = load_block("WFTNet").WFTNetBlock(
                3, 3, d_model=8, d_ff=16, layers=1, top_k=top_k,
                period_coeff=0.8, wavelet_scale=2, num_kernels=2,
                seq_len=16)
            config = SimpleNamespace(
                seq_len=16, pred_len=16, label_len=0, top_k=top_k,
                period_coeff=0.8, wavelet_scale=2, d_model=8, d_ff=16,
                num_kernels=2, enc_in=3, c_out=3, e_layers=1,
                embed="fixed", freq="h", dropout=0.0)
            reference = namespace["Model"](config)
            candidate.input.load_state_dict(
                reference.enc_embedding.value_embedding.tokenConv.state_dict())
            candidate.time_extension.load_state_dict(
                reference.predict_linear.state_dict())
            candidate.projection.load_state_dict(
                reference.projection.state_dict())
            candidate.norm.load_state_dict(reference.layer_norm.state_dict())
            own_wavelet = candidate.layers[0]
            ref_wavelet = reference.wavelet_model[0]
            own_wavelet.period_conv.load_state_dict(
                ref_wavelet.period_conv.state_dict())
            own_wavelet.scale_conv.load_state_dict(
                ref_wavelet.scale_conv.state_dict())
            own_wavelet.projection.load_state_dict(
                ref_wavelet.projection.state_dict())
            with torch.no_grad():
                candidate.positional.copy_(
                    reference.enc_embedding.position_embedding.pe[:, :16])
                candidate.output.weight.copy_(torch.eye(3))
                candidate.output.bias.zero_()
            x = torch.randn(2, 16, 3)
            torch.testing.assert_close(candidate(x),
                                       reference(x, None, None, None),
                                       atol=1e-5, rtol=1e-5)
        print("WFTNet official full forward with ptwt precision=10 parity OK")

    wpm_root = source / "WPMixer" / "gh" / "WPMixer"
    wpm_core_file = wpm_root / "models" / "wavelet_patch_mixer.py"
    wpm_revin_file = wpm_root / "utils" / "RevIN.py"
    if wpm_core_file.exists() and wpm_revin_file.exists():
        spec = importlib.util.spec_from_file_location("wpm_revin_official",
                                                      wpm_revin_file)
        assert spec is not None and spec.loader is not None
        revin_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(revin_module)
        namespace = load_source_definitions(
            wpm_core_file, {"TokenMixer", "Mixer", "ResolutionBranch"}, {
                "torch": torch,
                "nn": torch.nn,
                "RevIN": revin_module.RevIN
            })
        official = namespace["ResolutionBranch"](input_seq=16,
                                                 pred_seq=16,
                                                 batch_size=2,
                                                 channel=3,
                                                 d_model=8,
                                                 dropout=0.0,
                                                 embedding_dropout=0.0,
                                                 tfactor=2,
                                                 dfactor=2,
                                                 patch_len=4,
                                                 patch_stride=2)
        with torch.no_grad():
            official.revin.affine_weight.copy_(torch.tensor([1.2, 0.8, 1.1]))
            official.revin.affine_bias.copy_(torch.tensor([0.1, -0.2, 0.3]))
        ours = load_block("WPMixer").ResolutionBranch(channels=3,
                                                      length=16,
                                                      patch_len=4,
                                                      patch_stride=2,
                                                      d_model=8,
                                                      token_factor=2,
                                                      embed_factor=2,
                                                      dropout=0.0)
        ours.patch_norm.load_state_dict(official.patch_norm.state_dict())
        ours.patch_embed.load_state_dict(
            official.patch_embedding_layer.state_dict())
        for own_mixer, ref_mixer in ((ours.mixer1, official.mixer1),
                                     (ours.mixer2, official.mixer2)):
            own_mixer.norm1.load_state_dict(ref_mixer.norm1.state_dict())
            own_mixer.norm2.load_state_dict(ref_mixer.norm2.state_dict())
            own_mixer.token.load_state_dict(
                ref_mixer.tMixer.layers.state_dict())
            own_mixer.embedding.load_state_dict(
                ref_mixer.embeddingMixer.state_dict())
        ours.norm.load_state_dict(official.norm.state_dict())
        ours.readout.load_state_dict(official.head[1].state_dict())
        with torch.no_grad():
            ours.affine_weight.copy_(official.revin.affine_weight)
            ours.affine_bias.copy_(official.revin.affine_bias)
        x = torch.randn(2, 3, 16, requires_grad=True)
        for training in (True, False):
            official.train(training)
            ours.train(training)
            actual, expected = ours(x), official(x)
            torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
            torch.testing.assert_close(
                torch.autograd.grad(actual.square().sum(),
                                    x,
                                    retain_graph=True)[0],
                torch.autograd.grad(expected.square().sum(), x)[0],
                atol=1e-4,
                rtol=1e-4)
        print("WPMixer official ResolutionBranch train/eval parity OK")

        # Use the real wavelet package. The official vendored build accepts a
        # use_amp flag; the published PyPI 1.3.0 build omits that flag, so
        # only this constructor argument is adapted (use_amp=False here).
        wpm_module = load_block("WPMixer")
        from pytorch_wavelets import DWT1DForward, DWT1DInverse

        class CompatibleDWT1DForward(DWT1DForward):
            def __init__(self, wave, J, use_amp=False):
                if use_amp:
                    raise ValueError("this comparison requires use_amp=False")
                super().__init__(wave=wave, J=J)

        class CompatibleDWT1DInverse(DWT1DInverse):
            def __init__(self, wave, use_amp=False):
                if use_amp:
                    raise ValueError("this comparison requires use_amp=False")
                super().__init__(wave=wave)

        for length in (31, 32):
            signal = torch.randn(2, 3, length)
            own_low, own_details, _ = wpm_module.haar_decompose(signal, 2)
            ref_low, ref_details = DWT1DForward(wave="haar", J=2)(signal)
            torch.testing.assert_close(own_low, ref_low,
                                       atol=1e-6, rtol=1e-6)
            for own_detail, ref_detail in zip(own_details, ref_details):
                torch.testing.assert_close(own_detail, ref_detail,
                                           atol=1e-6, rtol=1e-6)
            torch.testing.assert_close(
                wpm_module.haar_reconstruct_official(ref_low, ref_details),
                DWT1DInverse(wave="haar")((ref_low, ref_details)),
                atol=1e-6, rtol=1e-6)
        print("WPMixer actual pytorch_wavelets Haar coefficients/inverse OK")

        decomposition_file = wpm_root / "models" / "decomposition.py"
        decomposition_ns = load_source_definitions(
            decomposition_file, {"Decomposition"},
            {"torch": torch, "nn": torch.nn,
             "DWT1DForward": CompatibleDWT1DForward,
             "DWT1DInverse": CompatibleDWT1DInverse,
             "RevIN": revin_module.RevIN})
        namespace["Decomposition"] = decomposition_ns["Decomposition"]
        load_source_definitions(wpm_core_file, {"WPMixerCore"}, namespace)
        official_core = namespace["WPMixerCore"](
            input_length=32,
            pred_length=32,
            wavelet_name="haar",
            level=2,
            batch_size=2,
            channel=3,
            d_model=8,
            dropout=0.0,
            embedding_dropout=0.0,
            tfactor=2,
            dfactor=2,
            device=torch.device("cpu"),
            patch_len=4,
            patch_stride=2,
            no_decomposition=False,
            use_amp=False)
        ours_core = wpm_module.WPMixerBlock(
            3, 3, seq_len=32, levels=2, patch_len=4, patch_stride=2,
            d_model=8, token_factor=2, embed_factor=2, dropout=0.0)
        with torch.no_grad():
            ours_core.input_affine_weight.copy_(
                official_core.revin.affine_weight)
            ours_core.input_affine_bias.copy_(official_core.revin.affine_bias)
            ours_core.output.weight.copy_(torch.eye(3))
            ours_core.output.bias.zero_()
        for own_branch, official_branch in zip(
                ours_core.branches, official_core.resolutionBranch):
            own_branch.patch_norm.load_state_dict(
                official_branch.patch_norm.state_dict())
            own_branch.patch_embed.load_state_dict(
                official_branch.patch_embedding_layer.state_dict())
            for own_mixer, ref_mixer in (
                    (own_branch.mixer1, official_branch.mixer1),
                    (own_branch.mixer2, official_branch.mixer2)):
                own_mixer.norm1.load_state_dict(ref_mixer.norm1.state_dict())
                own_mixer.norm2.load_state_dict(ref_mixer.norm2.state_dict())
                own_mixer.token.load_state_dict(
                    ref_mixer.tMixer.layers.state_dict())
                own_mixer.embedding.load_state_dict(
                    ref_mixer.embeddingMixer.state_dict())
            own_branch.norm.load_state_dict(official_branch.norm.state_dict())
            own_branch.readout.load_state_dict(
                official_branch.head[1].state_dict())
            with torch.no_grad():
                own_branch.affine_weight.copy_(
                    official_branch.revin.affine_weight)
                own_branch.affine_bias.copy_(
                    official_branch.revin.affine_bias)
        x = torch.randn(2, 32, 3)
        for training in (True, False):
            official_core.train(training)
            ours_core.train(training)
            torch.testing.assert_close(
                ours_core(x), official_core(x), atol=1e-5, rtol=1e-5)
        print("WPMixer official full core with real Haar backend parity OK")

    time_root = source / "TimeMixer" / "gh" / "TimeMixer"
    time_file = time_root / "models" / "TimeMixer.py"
    if time_file.exists():
        sys.path.insert(0, str(time_root))
        try:
            spec = importlib.util.spec_from_file_location(
                "time_official", time_file)
            assert spec is not None and spec.loader is not None
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        finally:
            sys.path.pop(0)
        for ci in (False, True):
            config = SimpleNamespace(seq_len=16,
                                     pred_len=16,
                                     down_sampling_window=2,
                                     down_sampling_layers=2,
                                     d_model=8,
                                     d_ff=16,
                                     dropout=0.0,
                                     channel_independence=int(ci),
                                     decomp_method="moving_avg",
                                     moving_avg=5)
            official = module.PastDecomposableMixing(config)
            ours = load_block("TimeMixer").PastDecomposableMixing([16, 8, 4],
                                                                  8, 16, 5,
                                                                  0.0, ci)
            if not ci:
                ours.cross_layer.load_state_dict(
                    official.cross_layer.state_dict())
            ours.out_cross_layer.load_state_dict(
                official.out_cross_layer.state_dict())
            for own_layer, ref_layer in zip(
                    ours.season_down,
                    official.mixing_multi_scale_season.down_sampling_layers):
                own_layer.load_state_dict(ref_layer.state_dict())
            for own_layer, ref_layer in zip(
                    ours.trend_up,
                    official.mixing_multi_scale_trend.up_sampling_layers):
                own_layer.load_state_dict(ref_layer.state_dict())
            scales = [
                torch.randn(2, length, 8, requires_grad=True)
                for length in (16, 8, 4)
            ]
            actual, expected = ours(scales), official(scales)
            for a, e in zip(actual, expected):
                torch.testing.assert_close(a, e, atol=1e-5, rtol=1e-5)
            torch.testing.assert_close(
                torch.autograd.grad(sum(a.square().sum() for a in actual),
                                    scales,
                                    retain_graph=True),
                torch.autograd.grad(sum(e.square().sum() for e in expected),
                                    scales),
                atol=1e-4,
                rtol=1e-4)
        print("TimeMixer official PDM (CI=0/1) forward parity OK")

        for ci in (False, True):
            full_config = SimpleNamespace(
                task_name="long_term_forecast", seq_len=16,
                label_len=0, pred_len=16, enc_in=3, c_out=3,
                down_sampling_window=2, down_sampling_layers=2,
                down_sampling_method="avg", d_model=8, d_ff=16,
                e_layers=1, dropout=0.0, channel_independence=int(ci),
                decomp_method="moving_avg", moving_avg=5,
                use_future_temporal_feature=False, use_norm=1,
                embed="fixed", freq="h")
            reference = module.Model(full_config)
            candidate = load_block("TimeMixer").TimeMixerBlock(
                3, 3, seq_len=16, d_model=8, d_ff=16, e_layers=1,
                down_sampling_layers=2, down_sampling_window=2,
                moving_avg=5, dropout=0.0, channel_independence=ci,
                down_sampling_method="avg", use_norm=True)
            candidate.embedding.load_state_dict(
                reference.enc_embedding.value_embedding.tokenConv.state_dict())
            for index, official_norm in enumerate(reference.normalize_layers):
                with torch.no_grad():
                    candidate.norm_weight[index].copy_(
                        official_norm.affine_weight)
                    candidate.norm_bias[index].copy_(
                        official_norm.affine_bias)
            own_pdm = candidate.pdm_blocks[0]
            ref_pdm = reference.pdm_blocks[0]
            if not ci:
                own_pdm.cross_layer.load_state_dict(
                    ref_pdm.cross_layer.state_dict())
            own_pdm.out_cross_layer.load_state_dict(
                ref_pdm.out_cross_layer.state_dict())
            for own_layer, ref_layer in zip(
                    own_pdm.season_down,
                    ref_pdm.mixing_multi_scale_season.down_sampling_layers):
                own_layer.load_state_dict(ref_layer.state_dict())
            for own_layer, ref_layer in zip(
                    own_pdm.trend_up,
                    ref_pdm.mixing_multi_scale_trend.up_sampling_layers):
                own_layer.load_state_dict(ref_layer.state_dict())

            captured = []
            hook = reference.pdm_blocks[-1].register_forward_hook(
                lambda _module, _inputs, output: captured.append(output))
            x = torch.randn(2, 16, 3)
            reference.forecast(x, None, None, None)
            hook.remove()
            expected = captured[0][0]
            if ci:
                expected = expected.unflatten(0, (2, 3))
                expected = expected.permute(0, 2, 1, 3).flatten(2)
            torch.testing.assert_close(candidate.forward_features(x),
                                       expected, atol=1e-5, rtol=1e-5)
        print("TimeMixer official input-to-PDM-encoder forward parity OK")

    cycle_root = source / "CycleNet" / "gh" / "CycleNet"
    cycle_file = cycle_root / "models" / "CycleNet.py"
    if cycle_file.exists():
        spec = importlib.util.spec_from_file_location("cycle_official",
                                                      cycle_file)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        for model_type in ("linear", "mlp"):
            for length in (16, 17):
                config = SimpleNamespace(seq_len=length,
                                         pred_len=length,
                                         enc_in=3,
                                         cycle=4,
                                         model_type=model_type,
                                         d_model=8,
                                         use_revin=True)
                official = module.Model(config)
                block = load_block("CycleNet").CycleNetBlock(
                    3,
                    3,
                    seq_len=length,
                    cycle_len=4,
                    model_type=model_type,
                    d_model=8,
                    variance_unbiased=True)
                with torch.no_grad():
                    block.cycle.data.copy_(official.cycleQueue.data)
                    block.temporal.load_state_dict(official.model.state_dict())
                    block.output.weight.copy_(torch.eye(3))
                    block.output.bias.zero_()
                x = torch.randn(2, length, 3)
                phase = torch.tensor([0, 2])
                torch.testing.assert_close(block(x, phase),
                                           official(x, phase),
                                           atol=1e-6,
                                           rtol=1e-6)
        print("CycleNet official linear/MLP full forward parity OK")

    fits_root = source / "FITS" / "gh" / "FITS"
    if (fits_root / "models" / "FITS.py").exists():
        sys.path.insert(0, str(fits_root))
        try:
            official_class = importlib.import_module("models.FITS").Model
        finally:
            sys.path.pop(0)
        for individual in (False, True):
            config = SimpleNamespace(seq_len=16,
                                     pred_len=0,
                                     individual=individual,
                                     enc_in=3,
                                     cut_freq=4)
            official = official_class(config)
            block = load_block("FITS").FITSBlock(3,
                                                 3,
                                                 cut_freq=4,
                                                 individual=individual,
                                                 variance_unbiased=True)
            with torch.no_grad():
                block.freq_upsampler.load_state_dict(
                    official.freq_upsampler.state_dict())
                block.output.weight.copy_(torch.eye(3))
                block.output.bias.zero_()
            x = torch.randn(2, 16, 3)
            torch.testing.assert_close(block(x),
                                       official(x)[0],
                                       atol=1e-5,
                                       rtol=1e-5)
        print("FITS official eta=1 shared/individual core parity OK")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--official-root",
        type=Path,
        help="directory containing optional official checkouts")
    args = parser.parse_args()
    torch.manual_seed(7)
    torch.set_num_threads(2)
    check_shapes_and_backward()
    check_transforms()
    check_fits_lowpass()
    if args.official_root is not None:
        check_official_cores(args.official_root)


if __name__ == "__main__":
    main()
