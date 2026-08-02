from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F


class TimeSeriesEncoder(nn.Module, ABC):
    """
    Interface every time-series encoder backend must satisfy so it can be swapped (e.g. this
    file's own PatchTimeSeriesEncoder today, a wrapped Chronos/Kronos model later) without
    touching TimeSeriesBranch, the projector, or the MLLM wiring.
    """

    @property
    @abstractmethod
    def output_dim(self) -> int:
        """embedding dim of each token this encoder emits"""

    @abstractmethod
    def n_tokens(self, seq_len: int) -> int:
        """
        how many tokens forward(ohlc) will emit for an input of length seq_len -- must be
        computable without running forward(), the same way an image's token count is known from
        image_grid_thw before the vision tower runs. Needed so the caller can build the prompt
        (how many placeholder tokens to insert) before tokenizing.
        """

    @abstractmethod
    def forward(self, ohlc: torch.Tensor) -> torch.Tensor:
        """
        ohlc: (B, T, n_features), e.g. n_features=4 for OHLC
        returns: (B, n_tokens(T), output_dim)
        """


class PatchTimeSeriesEncoder(TimeSeriesEncoder):
    """
    Reference implementation: non-overlapping patchify over the time axis + a small transformer
    encoder. Not meant to be state-of-the-art -- just a working default so the rest of the
    pipeline (projector, placeholder-token wiring, MLLM) can be built and tested without
    depending on an external time-series foundation model. Swap this out for e.g. a Chronos
    wrapper later; it only needs to implement `output_dim` and `forward`.
    """

    def __init__(self, n_features=4, patch_size=4, hidden_dim=256, n_layers=4, n_heads=8,
                 max_patches=256, dropout=0.1):
        super().__init__()
        self.n_features = n_features
        self.patch_size = patch_size
        self._output_dim = hidden_dim

        self.patch_proj = nn.Linear(n_features * patch_size, hidden_dim)
        self.pos_emb = nn.Parameter(torch.randn(1, max_patches, hidden_dim) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=n_heads, dim_feedforward=hidden_dim * 4,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers, enable_nested_tensor=False)
        self.out_norm = nn.LayerNorm(hidden_dim)

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def n_tokens(self, seq_len: int) -> int:
        return seq_len // self.patch_size

    def forward(self, ohlc: torch.Tensor) -> torch.Tensor:
        B, T, n_feat = ohlc.shape
        assert n_feat == self.n_features, f"expected {self.n_features} features, got {n_feat}"
        n_patches = T // self.patch_size
        assert n_patches > 0, f"sequence length {T} shorter than patch_size {self.patch_size}"
        assert n_patches <= self.pos_emb.shape[1], f"{n_patches} patches > max_patches {self.pos_emb.shape[1]}"

        usable_len = n_patches * self.patch_size
        x = ohlc[:, -usable_len:, :]  # drop any leading remainder that doesn't fill a full patch

        # per-sample z-normalization so absolute price level doesn't leak into patch content;
        # callers who want e.g. log-return scaling instead should pre-transform ohlc themselves
        mean = x.mean(dim=1, keepdim=True)
        std = x.std(dim=1, keepdim=True).clamp_min(1e-6)
        x = (x - mean) / std

        x = x.reshape(B, n_patches, self.patch_size * self.n_features)
        x = self.patch_proj(x) + self.pos_emb[:, :n_patches, :]
        x = self.transformer(x)
        return self.out_norm(x)


class CrossAttnTokenCompressor(nn.Module):
    """
    Compresses an arbitrary-length token sequence into a fixed number of output tokens via
    learnable-query cross attention, then projects to the target (LLM) hidden size. This
    decouples "how many raw tokens the encoder emits" (depends on window length / patch size /
    encoder backend) from "how many placeholder tokens go in the LLM prompt" (must be fixed and
    known before tokenization, same constraint images have via image_grid_thw).
    """

    def __init__(self, in_dim, n_query_tokens, out_dim, hidden_dim=None, num_heads=8, dropout=0.05):
        super().__init__()
        hidden_dim = hidden_dim or in_dim
        assert hidden_dim % num_heads == 0, f"hidden_dim {hidden_dim} not divisible by num_heads {num_heads}"
        self.num_heads = num_heads
        self.n_query_tokens = n_query_tokens
        self.query = nn.Parameter(torch.randn(1, n_query_tokens, hidden_dim) * 0.02)
        self.in_proj = nn.Linear(in_dim, hidden_dim)
        self.norm_in = nn.LayerNorm(hidden_dim)
        self.norm_out = nn.LayerNorm(hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, out_dim)
        self.act = nn.GELU()
        self.dropout = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, N, in_dim) -> (B, n_query_tokens, out_dim)"""
        B = x.shape[0]
        h = self.act(self.norm_in(self.in_proj(x)))  # (B, N, hidden_dim)
        head_dim = h.shape[-1] // self.num_heads

        h = h.view(B, -1, self.num_heads, head_dim).transpose(1, 2)  # (B, heads, N, head_dim)
        q = self.query.expand(B, -1, -1).reshape(B, self.n_query_tokens, self.num_heads, head_dim).transpose(1, 2)
        out = F.scaled_dot_product_attention(q, h, h, dropout_p=self.dropout if self.training else 0.0)
        out = out.transpose(1, 2).reshape(B, self.n_query_tokens, self.num_heads * head_dim)

        out = self.act(self.norm_out(out))
        return self.out_proj(out)


class MLPProjector(nn.Module):
    """
    Plain per-token projector -- LayerNorm -> Linear -> GELU -> Linear, no attention, no change
    in token count. Same shape of block as Qwen3-VL's own vision patch merger
    (Qwen3VLVisionPatchMerger): current mainstream VLM projectors (LLaVA, Qwen-VL) mostly work
    this way, keeping however many tokens the encoder emits rather than compressing to a fixed
    count -- unlike the older Flamingo/BLIP-2 Perceiver-Resampler/Q-Former style, which used
    learnable-query cross-attention to force a fixed token count.
    """

    def __init__(self, in_dim, out_dim, hidden_dim=None):
        super().__init__()
        hidden_dim = hidden_dim or in_dim
        self.norm = nn.LayerNorm(in_dim)
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, N, in_dim) -> (B, N, out_dim), N unchanged"""
        return self.fc2(self.act(self.fc1(self.norm(x))))


class TimeSeriesBranch(nn.Module):
    """
    encoder + projector bundle that plugs into the MLLM. Swap `encoder` for a different
    TimeSeriesEncoder implementation (e.g. a Chronos wrapper) without touching this class or the
    MLLM code, as long as it implements `output_dim`, `n_tokens`, and `forward`.

    Token-count handling is opt-in:
      - n_output_tokens=None (default): variable token count, one per encoder output token
        (mirrors how Qwen3-VL's own image branch works -- token count scales with input size).
      - n_output_tokens=<int>: compress to that fixed count via CrossAttnTokenCompressor
        (Perceiver-Resampler/Q-Former style). Useful if a particular encoder backend emits a
        token count that scales badly with window length (e.g. one token per bar, no
        patchification) and you want to bound prompt length regardless of input length.
    """

    def __init__(self, encoder: TimeSeriesEncoder, llm_hidden_dim: int, n_output_tokens: int | None = None,
                 projector_hidden_dim=None, compressor_hidden_dim=512, num_heads=8):
        super().__init__()
        self.encoder = encoder
        self.n_output_tokens = n_output_tokens
        if n_output_tokens is None:
            self.compressor = None
            self.projector = MLPProjector(encoder.output_dim, llm_hidden_dim, hidden_dim=projector_hidden_dim)
        else:
            self.projector = None
            self.compressor = CrossAttnTokenCompressor(
                in_dim=encoder.output_dim, n_query_tokens=n_output_tokens, out_dim=llm_hidden_dim,
                hidden_dim=compressor_hidden_dim, num_heads=num_heads,
            )

    def n_tokens_for_length(self, seq_len: int) -> int:
        """placeholder-token count for a window of this length -- must be known before running
        forward(), same role image_grid_thw plays for images."""
        return self.n_output_tokens if self.n_output_tokens is not None else self.encoder.n_tokens(seq_len)

    def forward(self, ohlc: torch.Tensor) -> torch.Tensor:
        """ohlc: (B, T, n_features) -> (B, n_tokens_for_length(T), llm_hidden_dim)"""
        feats = self.encoder(ohlc)
        return self.compressor(feats) if self.compressor is not None else self.projector(feats)


class MultiAssetTimeSeriesBranch(nn.Module):
    """
    Adds explicit multivariate target/context handling on top of an existing (univariate)
    TimeSeriesBranch, keeping that class itself untouched:

      - the target asset is encoded exactly as `target_branch` already does (reused as-is, same
        weights) -- full resolution, no forced compression, since it's the one thing predictions
        are about.
      - an arbitrary number N of context assets (other tickers, same calendar period) share the
        *same underlying encoder* (weight-shared, channel-independent -- each asset is encoded on
        its own, the model isn't given a fixed multivariate input vector) and then get pooled
        together and compressed into ONE fixed-size block via learnable-query cross-attention.
        Total context tokens stay bounded regardless of N -- N independent variable-length
        blocks would make prompt length explode as more context assets are added; this caps it.
      - a learned role embedding (target=0 / context=1) is added to every token post-projection,
        so the target/context distinction is explicit in the continuous embedding space itself,
        not just conveyed indirectly through the surrounding prompt text.
    """

    def __init__(self, target_branch: TimeSeriesBranch, llm_hidden_dim: int, n_context_tokens: int = 24,
                 context_projector_hidden_dim=None, compressor_hidden_dim=512, num_heads=8):
        super().__init__()
        self.target_branch = target_branch
        self.encoder = target_branch.encoder  # shared weights with target -- channel-independent
        self.n_context_tokens = n_context_tokens

        self.role_embedding = nn.Embedding(2, llm_hidden_dim)  # 0 = target, 1 = context
        self.context_projector = MLPProjector(
            self.encoder.output_dim, llm_hidden_dim, hidden_dim=context_projector_hidden_dim,
        )
        self.context_compressor = CrossAttnTokenCompressor(
            in_dim=llm_hidden_dim, n_query_tokens=n_context_tokens, out_dim=llm_hidden_dim,
            hidden_dim=compressor_hidden_dim, num_heads=num_heads,
        )

    def n_target_tokens_for_length(self, seq_len: int) -> int:
        return self.target_branch.n_tokens_for_length(seq_len)

    def forward(self, target_ohlc: torch.Tensor, context_ohlc_list: list[torch.Tensor]) -> dict[str, torch.Tensor | None]:
        """
        target_ohlc: (1, T, n_features) -- one sample's target-asset window
        context_ohlc_list: list of (1, T_i, n_features), one per context asset -- both T_i and
            len(context_ohlc_list) (N) may vary freely from call to call
        returns {'target': (n_target_tokens, hidden), 'context': (n_context_tokens, hidden) or
            None if context_ohlc_list is empty}
        """
        target_tokens = self.target_branch(target_ohlc)[0]  # (n_target_tokens, hidden)
        target_tokens = target_tokens + self.role_embedding.weight[0]

        context_tokens = None
        if context_ohlc_list:
            per_asset = [self.context_projector(self.encoder(ohlc))[0] for ohlc in context_ohlc_list]  # each (n_patches_i, hidden)
            pooled = torch.cat(per_asset, dim=0).unsqueeze(0)  # (1, sum_i n_patches_i, hidden)
            context_tokens = self.context_compressor(pooled)[0]  # (n_context_tokens, hidden)
            context_tokens = context_tokens + self.role_embedding.weight[1]

        return {"target": target_tokens, "context": context_tokens}


if __name__ == "__main__":
    torch.manual_seed(0)
    encoder = PatchTimeSeriesEncoder(n_features=4, patch_size=4, hidden_dim=256)

    print("-- variable token count (default, no compression) --")
    branch = TimeSeriesBranch(encoder, llm_hidden_dim=2048)
    for T in (40, 60, 100):
        ohlc = torch.randn(3, T, 4)
        out = branch(ohlc)
        expected_n = branch.n_tokens_for_length(T)
        print(f"T={T} -> {tuple(out.shape)} (expected n_tokens={expected_n})")
        assert out.shape == (3, expected_n, 2048)

    print("-- fixed token count (opt-in compression) --")
    branch_fixed = TimeSeriesBranch(encoder, llm_hidden_dim=2048, n_output_tokens=16)
    for T in (40, 60, 100):
        ohlc = torch.randn(3, T, 4)
        out = branch_fixed(ohlc)
        print(f"T={T} -> {tuple(out.shape)} (expected n_tokens=16)")
        assert out.shape == (3, 16, 2048)

    print("ok")

    print("-- multi-asset target/context branch --")
    target_branch = TimeSeriesBranch(encoder, llm_hidden_dim=2048)  # variable target token count
    multi_branch = MultiAssetTimeSeriesBranch(target_branch, llm_hidden_dim=2048, n_context_tokens=24)

    for n_context, T_target in [(0, 60), (1, 60), (5, 60)]:
        target_ohlc = torch.randn(1, T_target, 4)
        context_ohlc_list = [torch.randn(1, 40 + 4 * i, 4) for i in range(n_context)]  # varying T per context asset
        out = multi_branch(target_ohlc, context_ohlc_list)
        expected_target_n = multi_branch.n_target_tokens_for_length(T_target)
        print(f"n_context={n_context} -> target {tuple(out['target'].shape)} "
              f"(expected {expected_target_n}), context {None if out['context'] is None else tuple(out['context'].shape)}")
        assert out["target"].shape == (expected_target_n, 2048)
        if n_context == 0:
            assert out["context"] is None
        else:
            assert out["context"].shape == (24, 2048)

    print("ok")
