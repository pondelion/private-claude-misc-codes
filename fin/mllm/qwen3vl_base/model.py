import torch
import torch.nn as nn
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration

from timeseries_encoder import TimeSeriesBranch, TimeSeriesEncoder

TS_TOKEN = "<|ts_pad|>"


class _EmbeddingWithPlaceholderInjection(nn.Module):
    """
    Wraps Qwen3-VL's original token-embedding lookup so one or more named placeholder-token ids
    get their embeddings replaced with externally-computed continuous values, while every other
    token (text, image placeholders) goes through the untouched original embedding table. Slots
    are named (e.g. "ts", or "ts_target"/"ts_context" for the multivariate subclass) so several
    independent placeholder kinds can coexist without stepping on each other.

    This is the *only* modification made to Qwen3-VL. Everything else -- get_image_features,
    the image masked_scatter + deepstack injection, and the mrope position-id computation via
    get_rope_index -- runs completely unmodified off the normal `input_ids` path, because we
    still pass `input_ids` (not `inputs_embeds`) into the base model's forward. Exposes `.weight`
    so weight-tying with lm_head (Qwen3-VL ties them) keeps working transparently.
    """

    def __init__(self, base_embed_tokens: nn.Embedding, token_ids: dict[str, int] | None = None):
        super().__init__()
        self.base_embed_tokens = base_embed_tokens
        self.token_ids: dict[str, int] = dict(token_ids or {})
        self._pending: dict[str, torch.Tensor | None] = {name: None for name in self.token_ids}

    @property
    def weight(self):
        return self.base_embed_tokens.weight

    def add_slot(self, name: str, token_id: int):
        self.token_ids[name] = token_id
        self._pending[name] = None

    def set_pending(self, name: str, embeds: torch.Tensor | None):
        """embeds: (n_tokens_total_in_batch, hidden_dim), flattened batch-then-seq order -- i.e.
        matching `(input_ids == token_ids[name]).flatten()` order (row-major over (batch, seq))."""
        assert name in self.token_ids, f"unknown placeholder slot {name!r}, expected one of {list(self.token_ids)}"
        self._pending[name] = embeds

    def forward(self, input_ids: torch.LongTensor) -> torch.Tensor:
        embeds = self.base_embed_tokens(input_ids)
        for name, token_id in self.token_ids.items():
            pending = self._pending.get(name)
            if pending is None:
                continue
            mask = (input_ids == token_id).unsqueeze(-1).expand_as(embeds)
            n_expected = mask.sum().item() // embeds.shape[-1]
            assert n_expected == pending.shape[0], (
                f"{name!r}: placeholder count in input_ids ({n_expected}) != provided embeddings ({pending.shape[0]})"
            )
            embeds = embeds.masked_scatter(mask, pending.to(embeds.dtype))
            self._pending[name] = None
        return embeds


class FinMLLM(nn.Module):
    """
    Qwen3-VL (vision tower + LLM reused as-is, unmodified) plus an optional time-series branch.
    Swap `ts_encoder` for a different TimeSeriesEncoder implementation (e.g. a Chronos wrapper)
    without touching this class.
    """

    def __init__(self, model_name="Qwen/Qwen3-VL-2B-Instruct", ts_encoder: TimeSeriesEncoder | None = None,
                 n_ts_tokens: int | None = None, dtype=torch.bfloat16, device=None, load_in_8bit=False,
                 attn_implementation=None):
        """
        n_ts_tokens: None (default) -> variable placeholder-token count, one per encoder output
            token (mirrors Qwen3-VL's own image branch, whose token count scales with image
            size). Pass an int to compress to that fixed count instead (Perceiver-Resampler /
            Q-Former style) -- useful if `ts_encoder` emits a token count that scales badly with
            window length and prompt length needs to stay bounded regardless of input length.
        load_in_8bit: quantize the frozen base weights to int8 via bitsandbytes (QLoRA-style).
            Default False: for a model this small (~2B params), bitsandbytes' LLM.int8() has to
            dequantize back to fp16 for every matmul during forward/backward (see the
            "MatMul8bitLt: inputs will be cast..." warning), and that dequant overhead can exceed
            the raw weight-storage savings -- measured *higher* peak training memory with this on
            than with plain bf16, not lower. Quantization pays off on much larger models (7B+)
            where the storage savings dwarf the per-call overhead; it isn't a good default here.
        attn_implementation: None (default) lets transformers auto-resolve (currently "sdpa" for
            Qwen3-VL). Pass "flash_attention_2" (needs the flash-attn package) to force it --
            Qwen3-VL's custom mrope/deepstack attention patterns may not hit SDPA's fused kernel
            path, falling back to a less memory-efficient path silently; flash_attention_2 avoids
            that ambiguity. Worth an A/B measurement, since this is orthogonal to (and untested
            against) the dtype/AMP choices elsewhere in this file.
        """
        super().__init__()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype

        self.processor = AutoProcessor.from_pretrained(model_name)
        self.tokenizer = self.processor.tokenizer
        extra_kwargs = {"attn_implementation": attn_implementation} if attn_implementation else {}
        if load_in_8bit:
            quant_config = BitsAndBytesConfig(load_in_8bit=True)
            self.llm = Qwen3VLForConditionalGeneration.from_pretrained(
                model_name, quantization_config=quant_config, device_map={"": self.device}, **extra_kwargs,
            )
        else:
            self.llm = Qwen3VLForConditionalGeneration.from_pretrained(
                model_name, dtype=dtype, **extra_kwargs,
            ).to(self.device)

        self._wrapped_embed_tokens = _EmbeddingWithPlaceholderInjection(self.llm.get_input_embeddings())
        self.llm.set_input_embeddings(self._wrapped_embed_tokens)

        self.ts_branch = None
        if ts_encoder is not None:
            self.ts_token_id = self._add_placeholder_token(TS_TOKEN, "ts")
            # fp32: this is the one part of the model actually being trained from scratch, and
            # gets its precision from the outer torch.autocast wrapped around the whole forward
            # pass in train_eval_demo.py's train() (compute in bf16, gradients land on these fp32
            # leaves) rather than a dedicated cast here.
            self.ts_branch = TimeSeriesBranch(
                encoder=ts_encoder, n_output_tokens=n_ts_tokens,
                llm_hidden_dim=self.llm.config.text_config.hidden_size,
            ).to(device=self.device)

    def _add_placeholder_token(self, token_str: str, slot_name: str) -> int:
        """Registers a new special token + a matching named slot on the wrapped embedding layer.
        Subclasses (e.g. the multivariate FinMultiAssetMLLM) call this to add more placeholder
        kinds beyond the base "ts" slot."""
        n_added = self.tokenizer.add_tokens([token_str], special_tokens=True)
        if n_added:
            # resize_token_embeddings requires a plain nn.Embedding (it type-checks), so
            # temporarily swap our wrapper back out for the resize, then re-wrap the (new,
            # bigger) resized embedding it hands back.
            self.llm.set_input_embeddings(self._wrapped_embed_tokens.base_embed_tokens)
            self.llm.resize_token_embeddings(len(self.tokenizer))
            self._wrapped_embed_tokens.base_embed_tokens = self.llm.get_input_embeddings()
            self.llm.set_input_embeddings(self._wrapped_embed_tokens)
        token_id = self.tokenizer.convert_tokens_to_ids(token_str)
        self._wrapped_embed_tokens.add_slot(slot_name, token_id)
        return token_id

    @property
    def has_ts_branch(self) -> bool:
        return self.ts_branch is not None

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
        if include_ts:
            assert self.has_ts_branch, "include_ts=True but this FinMLLM has no ts_encoder configured"
            n_tokens = self.ts_branch.n_tokens_for_length(sample["ohlc"].shape[0])
            text_lines.append(TS_TOKEN * n_tokens)
        text_lines.append(instruction)

        content.append({"type": "text", "text": "\n".join(text_lines)})
        return content

    def build_raw_prompt(self, sample: dict, instruction: str = "Describe this price chart.",
                          include_image: bool = True, include_ts: bool = False,
                          include_period_text: bool = True) -> str:
        """
        Returns the fully expanded prompt text for one sample -- placeholder tokens spelled out
        exactly as many times as they'll actually appear in input_ids (image tokens expanded per
        image_grid_thw, ts tokens repeated per n_tokens_for_length), i.e. exactly what the model
        receives. Useful for debugging/inspection; not used in the forward/generate path itself.
        """
        sample_instruction = sample.get("instruction", instruction)
        content = self._build_message_content(sample, include_image, include_ts, include_period_text, sample_instruction)
        inputs = self.processor.apply_chat_template(
            [{"role": "user", "content": content}], tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt",
        )
        return self.tokenizer.decode(inputs["input_ids"][0])

    # ---- batch preparation ----------------------------------------------------

    def prepare_batch(self, samples: list[dict], instruction: str = "Describe this price chart.",
                       include_image: bool = True, include_ts: bool = False,
                       include_period_text: bool = True) -> dict:
        """
        samples: list of dataset items (see data.FinCandlestickDataset), each with keys
            'ticker', 'ohlc' (T_i, 4) [T_i may differ per sample], 'image' (PIL.Image),
            'description', 'n_bars', 'start_date', 'end_date'.
        Builds a right-padded training batch: input_ids/attention_mask/labels (-100 outside the
        assistant's response), plus pixel_values/image_grid_thw if include_image.

        Per-sample task mixing: if a sample dict carries its own 'instruction' and/or
        'target_text' keys, those win over this method's `instruction` param and over
        sample['description'] respectively -- so different samples in the same batch can be
        doing different tasks (e.g. captioning vs return-forecasting) with the same modality
        setup. Samples that don't set these fall back to the old behavior exactly.
        """
        assert not include_ts or self.has_ts_branch

        per_sample = []
        for sample in samples:
            sample_instruction = sample.get("instruction", instruction)
            target_text = sample.get("target_text", sample["description"])
            content = self._build_message_content(sample, include_image, include_ts, include_period_text, sample_instruction)
            prompt_messages = [{"role": "user", "content": content}]
            full_messages = prompt_messages + [
                {"role": "assistant", "content": [{"type": "text", "text": target_text}]}
            ]

            prompt_inputs = self.processor.apply_chat_template(
                prompt_messages, tokenize=True, add_generation_prompt=True,
                return_dict=True, return_tensors="pt",
            )
            full_inputs = self.processor.apply_chat_template(
                full_messages, tokenize=True, add_generation_prompt=False,
                return_dict=True, return_tensors="pt",
            )

            prompt_len = prompt_inputs["input_ids"].shape[1]
            input_ids = full_inputs["input_ids"][0]
            labels = input_ids.clone()
            labels[:prompt_len] = -100  # loss only on the assistant's response tokens

            per_sample.append({
                "input_ids": input_ids,
                "labels": labels,
                "pixel_values": full_inputs.get("pixel_values"),
                "image_grid_thw": full_inputs.get("image_grid_thw"),
                "ohlc": sample["ohlc"],
            })

        max_len = max(s["input_ids"].shape[0] for s in per_sample)
        pad_id = self.tokenizer.pad_token_id

        input_ids = torch.full((len(per_sample), max_len), pad_id, dtype=torch.long)
        attention_mask = torch.zeros((len(per_sample), max_len), dtype=torch.long)
        labels = torch.full((len(per_sample), max_len), -100, dtype=torch.long)
        for i, s in enumerate(per_sample):
            n = s["input_ids"].shape[0]
            input_ids[i, :n] = s["input_ids"]
            attention_mask[i, :n] = 1
            labels[i, :n] = s["labels"]

        batch = {
            "input_ids": input_ids.to(self.device),
            "attention_mask": attention_mask.to(self.device),
            "labels": labels.to(self.device),
        }
        if include_image:
            batch["pixel_values"] = torch.cat([s["pixel_values"] for s in per_sample], dim=0).to(self.device, self.dtype)
            batch["image_grid_thw"] = torch.cat([s["image_grid_thw"] for s in per_sample], dim=0).to(self.device)
        if include_ts:
            batch["ohlc_list"] = [s["ohlc"].to(self.device) for s in per_sample]  # T_i varies per sample
        return batch

    # ---- forward ----------------------------------------------------------

    def _inject_ts_embeds(self, ohlc_list: list[torch.Tensor], context_ohlc_lists: list[list[torch.Tensor]] | None = None):
        """
        Univariate (base class) behavior: encode each sample's own asset only, into the single
        "ts" slot. `context_ohlc_lists` is accepted (and ignored) here purely so forward()/
        generate() can pass it through unconditionally; FinMultiAssetMLLM overrides this method
        to also fill the "ts_context" slot from it.
        """
        # T (and, when the ts branch isn't compressing, the resulting token count) varies per
        # sample, so the encoder runs one sample at a time; concatenating along the token axis
        # (not stacking into a batch dim) gives exactly the flattened, batch-then-seq-order
        # tensor set_pending/masked_scatter expects, regardless of whether counts match.
        # fed at natural fp32 (ts_branch's own storage dtype); under the training loop's
        # model-wide autocast this computes in bf16, at inference (no autocast) it just runs
        # fp32-in/fp32-out directly -- either way no manual cast is needed here.
        ts_embeds_flat = torch.cat(
            [self.ts_branch(ohlc.unsqueeze(0))[0] for ohlc in ohlc_list], dim=0,
        )  # (sum_i n_tokens_i, hidden)
        self._wrapped_embed_tokens.set_pending("ts", ts_embeds_flat)

    def forward(self, batch: dict):
        if "ohlc_list" in batch:
            self._inject_ts_embeds(batch["ohlc_list"], batch.get("context_ohlc_lists"))

        return self.llm(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
            pixel_values=batch.get("pixel_values"),
            image_grid_thw=batch.get("image_grid_thw"),
        )

    # ---- inference ----------------------------------------------------------

    @torch.inference_mode()
    def generate(self, samples: list[dict], instruction: str = "Describe this price chart.",
                 include_image: bool = True, include_ts: bool = False,
                 include_period_text: bool = True, max_new_tokens: int = 128, **gen_kwargs) -> list[str]:
        """Prompt-only batch (no assistant/labels) -> decoded continuations, one per sample."""
        assert not include_ts or self.has_ts_branch

        per_sample = []
        for sample in samples:
            sample_instruction = sample.get("instruction", instruction)
            content = self._build_message_content(sample, include_image, include_ts, include_period_text, sample_instruction)
            inputs = self.processor.apply_chat_template(
                [{"role": "user", "content": content}], tokenize=True, add_generation_prompt=True,
                return_dict=True, return_tensors="pt",
            )
            per_sample.append({"inputs": inputs, "ohlc": sample["ohlc"], "context_samples": sample.get("context_samples") or []})

        max_len = max(s["inputs"]["input_ids"].shape[1] for s in per_sample)
        pad_id = self.tokenizer.pad_token_id
        B = len(per_sample)
        input_ids = torch.full((B, max_len), pad_id, dtype=torch.long)
        attention_mask = torch.zeros((B, max_len), dtype=torch.long)
        for i, s in enumerate(per_sample):
            n = s["inputs"]["input_ids"].shape[1]
            # left-pad so generation can start right after the real prompt tokens for every sample
            input_ids[i, max_len - n:] = s["inputs"]["input_ids"][0]
            attention_mask[i, max_len - n:] = 1

        batch = {"input_ids": input_ids.to(self.device), "attention_mask": attention_mask.to(self.device)}
        if include_image:
            batch["pixel_values"] = torch.cat([s["inputs"]["pixel_values"] for s in per_sample], dim=0).to(self.device, self.dtype)
            batch["image_grid_thw"] = torch.cat([s["inputs"]["image_grid_thw"] for s in per_sample], dim=0).to(self.device)
        if include_ts:
            ohlc_list = [s["ohlc"].to(self.device) for s in per_sample]
            context_ohlc_lists = [[c["ohlc"].to(self.device) for c in s["context_samples"]] for s in per_sample]
            self._inject_ts_embeds(ohlc_list, context_ohlc_lists)

        gen_ids = self.llm.generate(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
            pixel_values=batch.get("pixel_values"), image_grid_thw=batch.get("image_grid_thw"),
            max_new_tokens=max_new_tokens, **gen_kwargs,
        )
        new_tokens = gen_ids[:, input_ids.shape[1]:]
        return self.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)


if __name__ == "__main__":
    from data import load_ohlc_frame, FinCandlestickDataset
    from timeseries_encoder import PatchTimeSeriesEncoder

    df = load_ohlc_frame()
    ds = FinCandlestickDataset(df, visible_len=(40, 100), lookback=24, n_samples=4, seed=0)
    samples = [ds[i] for i in range(len(ds))]

    ts_encoder = PatchTimeSeriesEncoder(n_features=4, patch_size=4, hidden_dim=256)
    model = FinMLLM(ts_encoder=ts_encoder)  # n_ts_tokens=None -> variable token count (default)
    print("model loaded on", model.device, "vocab size", len(model.tokenizer))

    for include_image, include_ts in [(True, False), (False, True), (True, True)]:
        batch = model.prepare_batch(samples, include_image=include_image, include_ts=include_ts)
        out = model(batch)
        print(f"include_image={include_image} include_ts={include_ts} "
              f"input_ids={tuple(batch['input_ids'].shape)} loss={out.loss.item():.4f}")
