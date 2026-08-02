import torch

from model import TS_TOKEN, FinMLLM
from timeseries_encoder import MultiAssetTimeSeriesBranch, TimeSeriesEncoder

TS_CONTEXT_TOKEN = "<|ts_context_pad|>"


class FinMultiAssetMLLM(FinMLLM):
    """
    Extends FinMLLM with an explicit target/context multivariate time-series branch, keeping
    FinMLLM itself (the univariate model) completely untouched. Reuses everything from FinMLLM
    (tokenizer/processor, base Qwen3-VL, image handling, the "ts" placeholder slot for the target
    asset) via inheritance; this subclass only wraps `self.ts_branch` in a
    MultiAssetTimeSeriesBranch and adds a second placeholder-token slot ("ts_context") for the
    pooled, fixed-size context-asset block.
    """

    def __init__(self, model_name="Qwen/Qwen3-VL-2B-Instruct", ts_encoder: TimeSeriesEncoder = None,
                 n_target_tokens: int | None = None, n_context_tokens: int = 24,
                 dtype=torch.bfloat16, device=None, load_in_8bit=False, attn_implementation=None):
        assert ts_encoder is not None, "FinMultiAssetMLLM requires a ts_encoder"
        super().__init__(model_name=model_name, ts_encoder=ts_encoder, n_ts_tokens=n_target_tokens,
                          dtype=dtype, device=device, load_in_8bit=load_in_8bit,
                          attn_implementation=attn_implementation)
        # super().__init__ already built self.ts_branch as a plain (target-only) TimeSeriesBranch
        # (kept fp32, see FinMLLM.__init__) and registered the "ts" slot/TS_TOKEN for it -- wrap
        # that branch to add context handling on top, reusing the exact same target-side weights
        # and slot. Stays fp32 for the same reason (relies on the training loop's outer autocast).
        self.ts_branch = MultiAssetTimeSeriesBranch(
            target_branch=self.ts_branch, llm_hidden_dim=self.llm.config.text_config.hidden_size,
            n_context_tokens=n_context_tokens,
        ).to(device=self.device)
        self.ts_context_token_id = self._add_placeholder_token(TS_CONTEXT_TOKEN, "ts_context")

    # ---- prompt construction -------------------------------------------------

    def _build_message_content(self, sample: dict, include_image: bool, include_ts: bool,
                                include_period_text: bool, instruction: str) -> list[dict]:
        content = []
        if include_image:
            content.append({"type": "image", "image": sample["image"]})

        text_lines = [f"asset: {sample['ticker'] or 'unknown'}"]
        if include_period_text:
            n_bars = sample.get("n_bars", sample["ohlc"].shape[0])
            text_lines.append(f"period: {n_bars} bars ({sample.get('start_date', '?')} ~ {sample.get('end_date', '?')})")

        context_samples = sample.get("context_samples") or []

        # vision-side context: one extra image tiling every context asset's own small chart
        # (each on its own background color, a visual "slot index" cue independent of which
        # ticker landed there -- see data.tile_context_images), instead of one full image per
        # context asset, so vision cost stays bounded regardless of how many there are.
        if include_image and context_samples and sample.get("context_tile_image") is not None:
            content.append({"type": "image", "image": sample["context_tile_image"]})
            tickers_str = ", ".join(c["ticker"] or "unknown" for c in context_samples)
            text_lines.append(f"context assets tile image ({len(context_samples)}, left-to-right top-to-bottom): {tickers_str}")

        # ts-side context: target_branch is on the fixed "ts" slot, real content unchanged.
        if include_ts:
            assert self.has_ts_branch, "include_ts=True but this FinMultiAssetMLLM has no ts_encoder configured"
            n_target_tokens = self.ts_branch.n_target_tokens_for_length(sample["ohlc"].shape[0])
            text_lines.append(TS_TOKEN * n_target_tokens)

            if context_samples:
                tickers_str = ", ".join(c["ticker"] or "unknown" for c in context_samples)
                text_lines.append(f"context assets ({len(context_samples)}): {tickers_str}")
                text_lines.append(TS_CONTEXT_TOKEN * self.ts_branch.n_context_tokens)

        text_lines.append(instruction)
        content.append({"type": "text", "text": "\n".join(text_lines)})
        return content

    # ---- batch preparation (adds context_ohlc_lists on top of FinMLLM's batch) -----

    def prepare_batch(self, samples: list[dict], instruction: str = "Describe this price chart.",
                       include_image: bool = True, include_ts: bool = False,
                       include_period_text: bool = True) -> dict:
        batch = super().prepare_batch(samples, instruction=instruction, include_image=include_image,
                                       include_ts=include_ts, include_period_text=include_period_text)
        if include_ts:
            batch["context_ohlc_lists"] = [
                [c["ohlc"].to(self.device) for c in (s.get("context_samples") or [])] for s in samples
            ]
        return batch

    # ---- ts-embedding injection (overrides the univariate hook) --------------

    def _inject_ts_embeds(self, ohlc_list: list[torch.Tensor], context_ohlc_lists: list[list[torch.Tensor]] | None = None):
        context_ohlc_lists = context_ohlc_lists or [[] for _ in ohlc_list]
        target_list, context_list = [], []
        for ohlc, context_ohlc in zip(ohlc_list, context_ohlc_lists):
            out = self.ts_branch(
                ohlc.unsqueeze(0),
                [c.unsqueeze(0) for c in context_ohlc],
            )
            target_list.append(out["target"])
            if out["context"] is not None:
                context_list.append(out["context"])

        self._wrapped_embed_tokens.set_pending("ts", torch.cat(target_list, dim=0))
        if context_list:
            self._wrapped_embed_tokens.set_pending("ts_context", torch.cat(context_list, dim=0))


if __name__ == "__main__":
    from data import load_ohlc_frame, FinMultiAssetCandlestickDataset
    from timeseries_encoder import PatchTimeSeriesEncoder

    df = load_ohlc_frame()
    ds = FinMultiAssetCandlestickDataset(df, visible_len=(40, 100), lookback=24, n_samples=4, seed=0,
                                          n_context_range=(0, 4))
    samples = [ds[i] for i in range(len(ds))]
    print("n_context per sample:", [len(s["context_samples"]) for s in samples])

    ts_encoder = PatchTimeSeriesEncoder(n_features=4, patch_size=4, hidden_dim=256)
    model = FinMultiAssetMLLM(ts_encoder=ts_encoder, n_context_tokens=24)
    print("model loaded on", model.device, "vocab size", len(model.tokenizer))

    for include_image, include_ts in [(True, False), (False, True), (True, True)]:
        batch = model.prepare_batch(samples, include_image=include_image, include_ts=include_ts)
        out = model(batch)
        print(f"include_image={include_image} include_ts={include_ts} "
              f"input_ids={tuple(batch['input_ids'].shape)} loss={out.loss.item():.4f}")

    texts = model.generate(samples, include_image=True, include_ts=True, max_new_tokens=30, do_sample=False)
    for t in texts:
        print("---")
        print(repr(t))
