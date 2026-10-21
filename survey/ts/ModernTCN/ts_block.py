"""ModernTCN large-kernel, two-stage grouped-convolution block.

The output is aligned to the input length; this interpolation/readout is an
adaptation of the paper's task-specific heads. Convolutions are noncausal.
"""
from __future__ import annotations

import torch
from einops import rearrange
from torch import nn
from torch.nn import functional as F


def _fuse(conv: nn.Conv1d, bn: nn.BatchNorm1d):
    scale = bn.weight / torch.sqrt(bn.running_var + bn.eps)
    weight = conv.weight * scale[:, None, None]
    bias = bn.bias + (torch.zeros_like(bn.running_mean) if conv.bias is None
                      else conv.bias - bn.running_mean) * scale
    if conv.bias is None:
        bias = bn.bias - bn.running_mean * scale
    return weight, bias


class ReparamDepthwise(nn.Module):
    """Training: large+small Conv/BN; evaluation: one fused depthwise Conv."""

    def __init__(self,
                 channels: int,
                 large_kernel: int = 15,
                 small_kernel: int = 3):
        super().__init__()
        if (large_kernel % 2 != 1 or small_kernel % 2 != 1
                or small_kernel > large_kernel):
            raise ValueError(
                "kernels must be odd and small_kernel <= large_kernel")
        self.large_kernel = large_kernel
        self.large = nn.Sequential(
            nn.Conv1d(channels,
                      channels,
                      large_kernel,
                      padding=large_kernel // 2,
                      groups=channels,
                      bias=False), nn.BatchNorm1d(channels))
        self.small = nn.Sequential(
            nn.Conv1d(channels,
                      channels,
                      small_kernel,
                      padding=small_kernel // 2,
                      groups=channels,
                      bias=False), nn.BatchNorm1d(channels))
        self.fused: nn.Conv1d | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Depthwise temporal filtering: [B, channels, patches] -> same."""
        if self.fused is not None:
            return self.fused(x)
        return self.large(x) + self.small(x)

    @torch.no_grad()
    def merge_kernel(self) -> None:
        """Call after training and model.eval(); do not resume training afterward."""
        if self.fused is not None:
            return
        if self.training:
            raise RuntimeError("call model.eval() before merging BN branches")
        w1, b1 = _fuse(*self.large)
        w2, b2 = _fuse(*self.small)
        pad = (self.large_kernel - w2.size(-1)) // 2
        merged = nn.Conv1d(w1.size(0),
                           w1.size(0),
                           self.large_kernel,
                           padding=self.large_kernel // 2,
                           groups=w1.size(0),
                           bias=True,
                           device=w1.device,
                           dtype=w1.dtype)
        merged.weight.copy_(w1 + F.pad(w2, (pad, pad)))
        merged.bias.copy_(b1 + b2)
        self.fused = merged
        del self.large, self.small


class ModernTCNUnit(nn.Module):
    """One stage unit: [batch, series, latent features, patch positions].

    The input/output shape is [B, Cin, D_stage, N_patch]. ``N_patch`` is
    the number of sliding windows, not the number of points in one window.
    Depthwise convolution runs over patch positions. The first grouped FFN
    mixes latent features within each series; the second mixes series.
    """

    def __init__(self, nvars: int, dim: int, expansion: int, large: int,
                 small: int, dropout: float):
        super().__init__()
        hidden = dim * expansion
        self.nvars, self.dim = nvars, dim
        self.dw = ReparamDepthwise(nvars * dim, large, small)
        self.bn = nn.BatchNorm1d(dim)
        # FFN1: groups=入力系列数。各系列内のD特徴だけを混ぜる。
        # ここまでは別系列の値を参照せず、系列間融合は起きない。
        # Released Block: PWConv -> Dropout -> GELU -> PWConv -> Dropout.
        self.ffn_var = nn.Sequential(
            nn.Conv1d(nvars * dim, nvars * hidden, 1, groups=nvars),
            nn.Dropout(dropout), nn.GELU(),
            nn.Conv1d(nvars * hidden, nvars * dim, 1, groups=nvars),
            nn.Dropout(dropout))
        # FFN2: [B,D,N,P]に並べ替え、groups=Dとする。
        # 同じ潜在特徴位置のN系列を各grouped Conv1dが同時に読む。
        # ここがModernTCN encoder内で最初の系列間融合箇所。
        # 一覧READMEの「out projection前に融合○／FFN2で融合」に対応。
        self.ffn_cross = nn.Sequential(
            nn.Conv1d(dim * nvars, dim * nvars * expansion, 1, groups=dim),
            nn.Dropout(dropout), nn.GELU(),
            nn.Conv1d(dim * nvars * expansion, dim * nvars, 1, groups=dim),
            nn.Dropout(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Mix patch positions, latent features, then variables.

        B=batch数、N=入力系列数、D=各系列の潜在特徴数、P=patch個数。
        この関数の入出力は [B,N,D,P]。Pは1patchの長さではない。
        """
        batch_size, num_series, latent_dim, _ = x.shape
        # [B,N,D,P] -> [B,N*D,P]; depthwise filters each latent stream.
        temporal_streams = rearrange(x, "b n d p -> b (n d) p")
        temporal_streams = self.dw(temporal_streams)  # [B,N*D,P]
        # [B,N*D,P] -> [B*N,D,P]; BN shares D statistics over B*N streams.
        per_series = rearrange(
            temporal_streams, "b (n d) p -> (b n) d p",
            n=num_series, d=latent_dim)
        per_series = self.bn(per_series)  # [B*N,D,P]
        # Restore variable groups: [B*N,D,P] -> [B,N*D,P].
        per_series = rearrange(
            per_series, "(b n) d p -> b (n d) p",
            b=batch_size, n=num_series)
        # FFN1は各系列のD特徴だけを更新。[B,N*D,P]のN系列は独立。
        per_series = self.ffn_var(per_series)
        per_series = rearrange(
            per_series, "b (n d) p -> b n d p",
            n=num_series, d=latent_dim)  # [B,N,D,P]
        # FFN2に備えて[B,N,D,P] -> [B,D*N,P]に並べ替える。
        # 各group内にN系列を隣接させ、同じ潜在特徴の系列間融合を可能にする。
        cross_series = rearrange(per_series, "b n d p -> b (d n) p")
        # このConvで初めて他系列の値が各系列の表現に混ざる。
        # [B,D*N,P] -> [B,D*N,P]、P=patch個数は変えない。
        cross_series = self.ffn_cross(cross_series)  # [B,D*N,P]
        cross_series = rearrange(
            cross_series, "b (d n) p -> b n d p",
            d=latent_dim, n=num_series)  # [B,N,D,P]
        return x + cross_series  # [B,N,D,P]


class ModernTCNBlock(nn.Module):
    """[B,T,Cin] -> [B,T,Cout], length-preserving practical adapter.

    ``patch_size`` is the number of original time points in one patch;
    ``patch_stride`` is the shift in original time points between patches.
    Neither is the number of resulting patches. Stage i has D_i latent
    features per input series and N_i patch positions.
    """

    def __init__(self,
                 in_dim: int,
                 out_dim: int,
                 d_model: int = 32,
                 patch_size: int = 8,
                 patch_stride: int = 4,
                 depth: int = 2,
                 large_kernel: int = 15,
                 small_kernel: int = 3,
                 expansion: int = 2,
                 dropout: float = 0.0,
                 stage_dims: tuple[int, ...] | None = None,
                 stage_depths: tuple[int, ...] | None = None,
                 downsample_ratio: int = 2):
        super().__init__()
        if min(in_dim, out_dim, d_model, patch_size, patch_stride, depth,
               expansion, downsample_ratio) < 1:
            raise ValueError("dimensions must be positive")
        if patch_stride > patch_size:
            raise ValueError(
                "patch_stride must not exceed patch_size; "
                "otherwise samples are uncovered"
            )
        self.in_dim = in_dim
        self.patch_size, self.patch_stride = patch_size, patch_stride
        dims = (d_model, ) if stage_dims is None else tuple(stage_dims)
        depths = (depth, ) * len(dims) if stage_depths is None else tuple(
            stage_depths)
        if not dims or len(dims) != len(depths) or min(*dims, *depths) < 1:
            raise ValueError(
                "stage_dims and stage_depths must have equal nonzero length")
        self.stage_dims, self.downsample_ratio = dims, downsample_ratio
        self.stem = nn.Conv1d(1, dims[0], patch_size, stride=patch_stride)
        self.stem_bn = nn.BatchNorm1d(dims[0])
        # Official ModernTCN: each stage contains its own stack of identical
        # Block architecture, with BN -> strided Conv between stages.
        self.stages = nn.ModuleList(
            nn.ModuleList(
                ModernTCNUnit(in_dim, dim, expansion, large_kernel,
                              small_kernel, dropout) for _ in range(count))
            for dim, count in zip(dims, depths))
        self.downsamples = nn.ModuleList(
            nn.Sequential(
                nn.BatchNorm1d(dims[i]),
                nn.Conv1d(dims[i],
                          dims[i + 1],
                          downsample_ratio,
                          stride=downsample_ratio))
            for i in range(len(dims) - 1))
        # 汎用block専用の学習可能な同長decoder。公式encoder処理は変えない。
        # 各段間Conv1dが減らしたpatch個数を、逆順の転置畳み込みで戻す。
        # [B*Cin,D_i,P_i] -> [B*Cin,D_(i-1),P_i*downsample_ratio]。
        self.patch_upsamples = nn.ModuleList(
            nn.ConvTranspose1d(
                dims[stage_idx], dims[stage_idx - 1],
                downsample_ratio, stride=downsample_ratio)
            for stage_idx in range(len(dims) - 1, 0, -1))
        # stemが圧縮したpatch軸を元の時刻軸へ学習可能に展開する。
        # [B*Cin,D0,P0] -> [B*Cin,1,(P0-1)*stride+patch_size]。
        self.stem_upsample = nn.ConvTranspose1d(
            dims[0], 1, patch_size, stride=patch_stride)
        self.output = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """[B,T,Cin] -> [B,T,Cout]; patch encoder is noncausal.

        B=batch数、T=入力時刻数、Cin=入力系列数、Cout=出力特徴数。
        patch_size=1patchに含む元時刻数、patch_stride=開始位置の移動幅。
        各stageのD_iは1系列当たりの潜在特徴数、N_iはpatch個数。
        """
        if x.ndim != 3 or x.size(-1) != self.in_dim or x.size(1) < 1:
            raise ValueError(f"expected [B,T,{self.in_dim}] with T>=1")
        batch_size, seq_len, num_series = x.shape
        # Separate input series for shared stem: [B,T,Cin] -> [B*Cin,1,T].
        per_series = rearrange(x, "b t n -> (b n) 1 t")
        # 公式stemはpatch_size-patch_stride個だけ末尾の観測値を複製する。
        # 入力Tがpatch_sizeより短い場合に限り、最低1patch作れるまで追加。
        right_padding = (self.patch_size - self.patch_stride
                         if self.patch_size != self.patch_stride else 0)
        right_padding = max(right_padding, self.patch_size - seq_len)
        # pad=right_paddingは追加した時刻数。[B*Cin,1,T+pad]になる。
        per_series = F.pad(
            per_series, (0, right_padding), mode="replicate")
        # stemのConv1dは各系列に長さpatch_sizeの窓をstrideずつずらす。
        # D0=stage_dims[0]：各patchから抽出する潜在特徴の個数。
        # P0=stem後のpatch個数（元時刻数ではない）。
        # P0=floor((T+pad-patch_size)/patch_stride)+1。
        # 例: T=33, patch_size=8, stride=4, pad=4ならP0=8個。
        # [B*Cin,1,T+pad] -> [B*Cin,D0,P0]。
        patch_features = self.stem(per_series)
        patch_features = self.stem_bn(patch_features)  # [B*Cin,D0,P0]
        # Expose the series axis: [B*Cin,D0,P0] -> [B,Cin,D0,P0].
        patch_features = rearrange(
            patch_features, "(b n) d p -> b n d p",
            b=batch_size, n=num_series)
        stage_patch_counts: list[int] = []
        for stage_idx, units in enumerate(self.stages):
            # 最初のstage（index=0）はstemのpatch表現をそのまま処理する。
            # 2段目以降だけ、前stageのpatch個数を縮小して特徴幅を変える。
            if stage_idx > 0:
                # D_prev=直前stageの1系列当たり潜在特徴数。
                # P_prev=直前stageのpatch個数。ここでは元時刻を数えない。
                # [B,Cin,D_prev,P_prev] -> [B*Cin,D_prev,P_prev]。
                per_series = rearrange(
                    patch_features, "b n d p -> (b n) d p")
                stage_padding = (-per_series.size(-1)) % self.downsample_ratio
                if stage_padding:
                    # P_prevが縮小倍率で割り切れるよう、末尾patch列を複製。
                    # [B*Cin,D_prev,P_prev] -> [B*Cin,D_prev,P_prev+pad]。
                    per_series = torch.cat(
                        (per_series, per_series[..., -stage_padding:]),
                        dim=-1)
                # downsampling Convのkernel/strideはdownsample_ratio。
                # D_i=現在stageの潜在特徴数、P_i=縮小後のpatch個数。
                # P_i=(P_prev+stage_padding)/downsample_ratio。
                # [B*Cin,D_prev,P_prev+pad] -> [B*Cin,D_i,P_i]。
                per_series = self.downsamples[stage_idx - 1](per_series)
                patch_features = rearrange(
                    per_series, "(b n) d p -> b n d p",
                    b=batch_size, n=num_series)  # [B,Cin,D_i,P_i]
            for unit in units:
                # 各unit内ではdepthwise時間畳み込み→系列別FFN1の後、
                # FFN2がCin系列を混ぜる。stemや段間downsampleは融合しない。
                # この呼び出し時点でout projectionより前の系列間融合が起きる。
                patch_features = unit(patch_features)  # [B,Cin,D_i,P_i]
            stage_patch_counts.append(patch_features.size(-1))
        # 公式encoderの出力[B,Cin,D_last,P_last]を保持したままdecoderへ。
        # 旧実装のD_last->1と線形補間は、潜在特徴を早期に潰すため使わない。
        decoded_series = rearrange(
            patch_features, "b n d p -> (b n) d p")
        for upsample, previous_patch_count in zip(
                self.patch_upsamples, reversed(stage_patch_counts[:-1])):
            # [B*Cin,D_i,P_i] -> [B*Cin,D_(i-1),P_i*ratio]。
            decoded_series = upsample(decoded_series)
            # encoder側で追加したpad patchを削り、直前stageの個数に揃える。
            decoded_series = decoded_series[..., :previous_patch_count]
        # 転置stemは全D0特徴を使い、patch窓から元時刻への写像を学習。
        # [B*Cin,D0,P0] -> [B*Cin,1,T_dec] -> [B*Cin,1,T]。
        decoded_series = self.stem_upsample(decoded_series)[..., :seq_len]
        # このdecoderも失われた情報を必ず復元できるわけではない。
        # ただし補間と異なり、patch特徴からの展開は学習可能。
        aligned_series = rearrange(
            decoded_series, "(b n) 1 t -> b n t",
            b=batch_size, n=num_series)  # [B,Cin,T]
        # FFN2で既に系列間融合済み。次の出力Linearは追加のlate fusionで、
        # 最初の融合箇所ではない。一覧READMEの分類と混同しない。
        # [B,Cin,T] -> [B,T,Cin] -> [B,T,Cout]、Cin=Coutでも適用。
        aligned_time = rearrange(aligned_series, "b n t -> b t n")
        return self.output(aligned_time)

    def merge_kernels(self) -> None:
        """Reduce each large+small+BN pair to one convolution for inference."""
        if self.training:
            raise RuntimeError("call eval() before merge_kernels()")
        for stage in self.stages:
            for unit in stage:
                unit.dw.merge_kernel()


if __name__ == "__main__":
    model = ModernTCNBlock(3, 5, d_model=8, depth=2)
    x = torch.randn(2, 33, 3, requires_grad=True)
    y = model(x)
    y.square().mean().backward()
    assert y.shape == (2, 33, 5) and x.grad is not None
    assert model.stem_upsample.weight.grad is not None
    # 多段の逆方向decoderでも短い系列・割り切れない系列長を保持する。
    multistage_model = ModernTCNBlock(
        3, 5, stage_dims=(8, 16, 24), stage_depths=(1, 1, 1))
    for sequence_length in (1, 2, 7, 33, 64, 96):
        test_input = torch.randn(2, sequence_length, 3)
        test_output = multistage_model(test_input)
        assert test_output.shape == (2, sequence_length, 5)
        test_output.square().mean().backward()
    assert all(layer.weight.grad is not None
               for layer in multistage_model.patch_upsamples)
    assert multistage_model.stem_upsample.weight.grad is not None
    model.eval()
    with torch.no_grad():
        before = model(x.detach())
        model.merge_kernels()
        after = model(x.detach())
        torch.testing.assert_close(before, after, rtol=1e-4, atol=1e-5)
