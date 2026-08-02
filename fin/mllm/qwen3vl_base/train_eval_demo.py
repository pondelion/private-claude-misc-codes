import argparse
import contextlib
import json
import os
import random
import re

# Must be set before any CUDA context is created, so this needs to land before torch touches the
# GPU -- setdefault at import time is early enough, and won't clobber an explicit value the
# caller may have set beforehand. This workload allocates widely varying tensor shapes (image
# size / ts token count / modality combo all vary per batch), which badly fragments PyTorch's
# default CUDA caching allocator: measured ~10GB of reserved-but-unusable memory on top of ~16.5GB
# actually in use after just 30 varied-shape training steps without this; expandable_segments
# eliminates nearly all of it (reserved dropped to ~17.6GB, matching actual usage) for ~0%
# measured slowdown (0.343s/step -> 0.341s/step). This is a separate, much larger effect from a
# different mechanism (allocator fragmentation) than the dtype/AMP/quantization choices elsewhere
# in this file, which address precision/compute cost, not fragmentation.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from tqdm import tqdm

from data import DynamicFinCandlestickSampler, FinCandlestickDataset, load_ohlc_frame, split_by_date
from model import TS_TOKEN, FinMLLM
from timeseries_encoder import PatchTimeSeriesEncoder

DESCRIPTION_INSTRUCTION = "Describe this price chart."
RETURN_INSTRUCTION_TMPL = "Forecast how the price will move over the next {horizon} bars."

# batch-level modality dropout: one of these three is chosen per training/eval step, so the
# model sees image-only, ts-only, and both-modalities examples across the run and learns to
# produce good output from any single modality alone (not just from the full combination).
MODALITY_COMBOS = {
    "image_only": dict(include_image=True, include_ts=False),
    "ts_only": dict(include_image=False, include_ts=True),
    "both": dict(include_image=True, include_ts=True),
}


def assign_task(sample: dict, horizon: int, rng: random.Random, description_prob: float = 0.5) -> dict:
    """
    Mutates `sample` to carry its own 'instruction'/'target_text' (honored per-sample by
    FinMLLM.prepare_batch/generate) -- this is what lets captioning and return-forecasting
    examples coexist in the same training batch, each sample independently choosing its task.
    """
    if rng.random() < description_prob:
        sample["instruction"] = DESCRIPTION_INSTRUCTION
        sample["target_text"] = sample["description"]
    else:
        sample["instruction"] = RETURN_INSTRUCTION_TMPL.format(horizon=horizon)
        sample["target_text"] = sample["future_return_text"]
    return sample


def collapse_repeated_tokens(text: str, tokens: list[str]) -> str:
    """
    Collapses runs of a repeated placeholder token (e.g. 15 copies of <|ts_pad|>, or hundreds of
    <|image_pad|>) into `<|ts_pad|>×15` so a dumped raw prompt stays human-readable instead of
    showing the token spelled out dozens/hundreds of times.
    """
    for token in tokens:
        pattern = re.compile("(?:" + re.escape(token) + ")+")

        def _replace(m, token=token):
            count = len(m.group(0)) // len(token)
            return f"{token}×{count}" if count > 1 else token

        text = pattern.sub(_replace, text)
    return text


def train(model: FinMLLM, sampler: DynamicFinCandlestickSampler, n_steps=40, batch_size=4, lr=5e-5,
          horizon=10, seed=0, log_every=5, grad_accum_steps=1, max_grad_norm=1.0, use_amp=True):
    """
    `sampler` draws a fresh window on every call (see DynamicFinCandlestickSampler) -- unlike
    indexing into a fixed pre-sampled pool, this means the total number of distinct (ticker,
    period) combinations the model sees is bounded only by n_steps x batch_size, not capped by
    an arbitrary pool size set up front.

    use_amp=True wraps the whole forward pass in torch.autocast(bfloat16). LoRA is left at its
    native dtype (not upcast to fp32) so there's no fp32-delta-cascading-through-the-residual-
    -stream risk this time; ts_branch is fp32-stored and picks up bf16 compute from this same
    outer autocast (no dedicated autocast of its own -- see FinMLLM.__init__). This is simpler
    than wrapping ts_branch narrowly, but means the whole forward (not just ts_branch) pays
    whatever autocast overhead/behavior exists -- worth comparing against use_amp=False directly
    if memory or speed looks off, since bnb-quantized bases in particular did NOT play well with
    this (see load_in_8bit's docstring); plain bf16 (the default here) is the case being tried.
    """
    rng = random.Random(seed)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=lr)
    autocast_ctx = (lambda: torch.autocast(device_type="cuda", dtype=torch.bfloat16)) if use_amp else contextlib.nullcontext

    loss_history = []
    pbar = tqdm(range(n_steps), desc="train")
    for step in pbar:
        combo_name = rng.choice(list(MODALITY_COMBOS))
        optimizer.zero_grad()
        step_loss = 0.0

        for _ in range(grad_accum_steps):
            samples = [assign_task(sampler.sample(), horizon, rng) for _ in range(batch_size)]
            batch = model.prepare_batch(samples, **MODALITY_COMBOS[combo_name])
            with autocast_ctx():
                out = model(batch)
            (out.loss / grad_accum_steps).backward()
            step_loss += out.loss.item() / grad_accum_steps

        torch.nn.utils.clip_grad_norm_(trainable_params, max_grad_norm)
        optimizer.step()

        loss_history.append(step_loss)
        pbar.set_postfix(modality=combo_name, loss=f"{step_loss:.4f}")
        if step % log_every == 0 or step == n_steps - 1:
            tqdm.write(f"step {step:3d}  modality={combo_name:10s}  loss={step_loss:.4f}")
    return loss_history


DIRECTION_CLASSES = ("上昇", "下落", "横ばい")


def parse_direction(text: str) -> str | None:
    """Pulls the model's 3-class direction call out of free-form generated text by substring
    presence -- the model is trained to echo describe_window_mock/describe_future_return's own
    "上昇"/"下落"/"横ばい" wording (ground_truth_direction below labels real outcomes across
    these same three classes, using the exact same +-1% threshold those templates use). Returns
    None (truly unparseable, not one of the three classes) if none or more than one appear."""
    found = {word for word in DIRECTION_CLASSES if word in text}
    return found.pop() if len(found) == 1 else None


def ground_truth_direction(pct_change: float) -> str:
    """Real 3-class outcome label. Same +-1% 'flat' threshold describe_window_mock/
    describe_future_return use to write "横ばい" into the target text itself, so every sample
    gets one of the three classes -- none are excluded as 'ambiguous' anymore."""
    if pct_change > 1:
        return "上昇"
    if pct_change < -1:
        return "下落"
    return "横ばい"


def _task_defs(horizon: int):
    """(task_name, instruction, ground-truth-text key, fn(sample) -> ground-truth pct_change)"""
    return [
        ("description", DESCRIPTION_INSTRUCTION, "description",
         lambda s: (s["ohlc"][-1, 3] / s["ohlc"][0, 3] - 1.0).item() * 100),
        ("return_forecast", RETURN_INSTRUCTION_TMPL.format(horizon=horizon), "future_return_text",
         lambda s: s["future_return_pct"]),
    ]


@torch.inference_mode()
def run_eval(model: FinMLLM, dataset, results_dir: str | None = None, loss_history: list[float] | None = None,
             n_qualitative: int = 2, full_accuracy: bool = False, horizon: int = 10,
             max_new_tokens: int = 150, gen_batch_size: int = 8, extra_collapse_tokens=()) -> dict | None:
    """
    One generation pass, reused for every output below -- no sample is ever sent through
    model.generate()/build_raw_prompt() more than once, no matter how many of these are requested.

    full_accuracy=False (quick check, e.g. before training): generates for just the first
        n_qualitative samples of `dataset`, prints them to console, and (if results_dir is given)
        dumps sample_i/chart.png + predictions.txt for them.
    full_accuracy=True (e.g. after training): generates for the *entire* dataset, computes 3-class
        (上昇/下落/横ばい) direction accuracy + confusion matrix + per-class/macro precision-
        recall-f1 per (task, modality) -- see parse_direction / ground_truth_direction -- and
        writes direction_accuracy.json under results_dir. The console printout and sample_i/
        files above are then just the first n_qualitative entries of this same run, not a second
        pass.

    Either way, if loss_history is given and results_dir is set, also writes
    loss_history.json/loss_curve.png. Returns the direction-accuracy dict when full_accuracy=True,
    else None.
    """
    if results_dir is not None:
        os.makedirs(results_dir, exist_ok=True)
        if loss_history is not None:
            with open(os.path.join(results_dir, "loss_history.json"), "w") as f:
                json.dump(loss_history, f, indent=2)
            plt.figure(figsize=(6, 3.5))
            plt.plot(loss_history)
            plt.xlabel("step")
            plt.ylabel("loss")
            plt.title("training loss")
            plt.tight_layout()
            plt.savefig(os.path.join(results_dir, "loss_curve.png"), dpi=120)
            plt.close()

    n_avail = len(dataset)
    n_use = n_avail if full_accuracy else min(n_qualitative, n_avail)
    samples = [dataset[i] for i in range(n_use)]
    tasks = _task_defs(horizon)

    n_calls = len(tasks) * len(MODALITY_COMBOS) * -(-len(samples) // gen_batch_size)
    pbar = tqdm(total=n_calls, desc="run_eval (generate)")

    generated = {}  # {task_name: {combo_name: [text, ...]}}, aligned index-for-index with `samples`
    for task_name, instruction, gt_key, gt_pct_fn in tasks:
        task_samples = [{**s, "instruction": instruction} for s in samples]
        generated[task_name] = {}
        for combo_name, combo in MODALITY_COMBOS.items():
            texts = []
            for i in range(0, len(task_samples), gen_batch_size):
                pbar.set_postfix(task=task_name, modality=combo_name, batch=i // gen_batch_size)
                texts.extend(model.generate(task_samples[i:i + gen_batch_size], max_new_tokens=max_new_tokens,
                                             do_sample=False, **combo))
                pbar.update(1)
            generated[task_name][combo_name] = texts
    pbar.close()

    # qualitative: console printout + (if results_dir given) per-sample files, always just the
    # first n_qualitative of whatever was generated above.
    collapse_tokens = [model.processor.image_token, TS_TOKEN, *extra_collapse_tokens]
    for i in range(min(n_qualitative, len(samples))):
        sample = samples[i]
        lines = [f"ticker: {sample['ticker'] or 'unknown'}",
                 f"period: {sample['n_bars']} bars ({sample['start_date']} ~ {sample['end_date']})", ""]

        for task_name, instruction, gt_key, _ in tasks:
            print(f"\n=== task: {task_name} | sample {i} ({sample['ticker'] or 'unknown'}) ===")
            lines += [f"=== task: {task_name} ===", f"ground truth: {sample[gt_key]}"]
            for combo_name, combo in MODALITY_COMBOS.items():
                text = generated[task_name][combo_name][i]
                print(f"-- modality: {combo_name} --\n  gen: {text!r}\n  gt : {sample[gt_key]!r}")
                lines.append(f"-- modality: {combo_name} --")
                if results_dir is not None:
                    task_sample = {**sample, "instruction": instruction}
                    raw_prompt = model.build_raw_prompt(task_sample, **combo)
                    lines.append(f"[raw prompt] {collapse_repeated_tokens(raw_prompt, collapse_tokens)}")
                lines.append(f"[generated]  {text}")
            lines.append("")

        if results_dir is not None:
            sample_dir = os.path.join(results_dir, f"sample_{i}")
            os.makedirs(sample_dir, exist_ok=True)
            sample["image"].save(os.path.join(sample_dir, "chart.png"))
            with open(os.path.join(sample_dir, "predictions.txt"), "w") as f:
                f.write("\n".join(lines))

    if not full_accuracy:
        return None

    # quantitative: 3-class (上昇/下落/横ばい) accuracy + confusion matrix + per-class/overall
    # precision-recall-f1 (via sklearn) over the full dataset, from the exact same `generated`
    # texts computed above -- no extra generate() calls. Every sample has a definite gt class (see
    # ground_truth_direction) so none are excluded anymore; an unparseable pred maps to its own
    # "unparseable" label (excluded from sklearn's `labels`, so it can't be a false positive for a
    # real class) but never equals any gt, so it still counts as wrong everywhere else.
    labels = list(DIRECTION_CLASSES)
    results = {}
    for task_name, instruction, gt_key, gt_pct_fn in tasks:
        y_true = [ground_truth_direction(gt_pct_fn(s)) for s in samples]
        results[task_name] = {}
        for combo_name in MODALITY_COMBOS:
            y_pred = [parse_direction(t) or "unparseable" for t in generated[task_name][combo_name]]

            cm = confusion_matrix(y_true, y_pred, labels=[*labels, "unparseable"])
            per_p, per_r, per_f1, per_support = precision_recall_fscore_support(
                y_true, y_pred, labels=labels, average=None, zero_division=0)
            macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
                y_true, y_pred, labels=labels, average="macro", zero_division=0)
            micro_p, micro_r, micro_f1, _ = precision_recall_fscore_support(
                y_true, y_pred, labels=labels, average="micro", zero_division=0)

            results[task_name][combo_name] = {
                "accuracy": float(accuracy_score(y_true, y_pred)),
                "n_total": len(y_true),
                "n_unparseable_pred": y_pred.count("unparseable"),
                "confusion_matrix": {
                    gt: dict(zip([*labels, "unparseable"], row.tolist()))
                    for gt, row in zip(labels, cm)
                },
                "overall": {
                    "macro": {"precision": float(macro_p), "recall": float(macro_r), "f1": float(macro_f1)},
                    "micro": {"precision": float(micro_p), "recall": float(micro_r), "f1": float(micro_f1)},
                },
                "per_class": {
                    c: {"precision": float(p), "recall": float(r), "f1": float(f), "support": int(s)}
                    for c, p, r, f, s in zip(labels, per_p, per_r, per_f1, per_support)
                },
            }

    if results_dir is not None:
        out_path = os.path.join(results_dir, "direction_accuracy.json")
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"saved direction accuracy results to {out_path}")
    return results


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n-steps", type=int, default=40, help="training steps")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--horizon", type=int, default=10, help="return-forecast horizon, in bars")
    p.add_argument("--n-eval-samples", type=int, default=20,
                    help="size of the FIXED eval window pool (fixed on purpose, for reproducible before/after comparison)")
    p.add_argument("--cutoff-date", default="2024-06-01", help="train/eval date split (eval is strictly after this)")
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--n-result-samples", type=int, default=2, help="how many eval samples to dump under results/")
    p.add_argument("--results-dir", default=None, help="defaults to <script_dir>/results/univariate")
    p.add_argument("--log-every", type=int, default=5)
    # Default False: measured *higher* peak training memory with this on than with plain bf16 for
    # this ~2B model -- bitsandbytes' LLM.int8() dequantizes back to fp16 for every matmul during
    # forward/backward, and that per-call overhead outweighs the raw weight-storage savings at
    # this size. Quantization pays off on much larger models (7B+); pass --load-in-8bit to try it
    # anyway, but it isn't a good default here.
    p.add_argument("--load-in-8bit", action=argparse.BooleanOptionalAction, default=False,
                    help="quantize the frozen base to int8 via bitsandbytes -- see the comment above this arg before enabling")
    p.add_argument("--grad-accum-steps", type=int, default=1, help="microbatches accumulated per optimizer step")
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True,
                    help="wrap the whole forward pass in torch.autocast(bfloat16); --no-amp to disable")
    p.add_argument("--attn-implementation", default=None,
                    help="e.g. 'flash_attention_2' (needs the flash-attn package). Measured ~300MB peak-memory "
                         "reduction vs the sdpa default, ~0 speed cost. None (default) lets transformers auto-resolve.")
    p.add_argument("--train-vision-projector", action=argparse.BooleanOptionalAction, default=True,
                    help="fully fine-tune Qwen3-VL's vision-to-LLM patch merger alongside LoRA (cheap, small module); "
                         "--no-train-vision-projector to leave it frozen")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    df = load_ohlc_frame()
    train_df, eval_df = split_by_date(df, cutoff_date=args.cutoff_date)  # eval is strictly later in time than train
    print(f"train rows: {len(train_df)} ({train_df.index.get_level_values(1).min()} ~ {train_df.index.get_level_values(1).max()})")
    print(f"eval rows:  {len(eval_df)} ({eval_df.index.get_level_values(1).min()} ~ {eval_df.index.get_level_values(1).max()})")

    # training: dynamic sampler, fresh window every call -- diversity bounded only by the data,
    # not by an arbitrary pool size (see DynamicFinCandlestickSampler's docstring)
    train_sampler = DynamicFinCandlestickSampler(train_df, visible_len=(40, 90), lookback=24, seed=0,
                                                  prediction_horizon=args.horizon)
    # eval: FIXED pool on purpose, so before/after-training comparisons use the exact same windows
    eval_ds = FinCandlestickDataset(eval_df, visible_len=(40, 90), lookback=24, n_samples=args.n_eval_samples,
                                     seed=999, prediction_horizon=args.horizon)

    ts_encoder = PatchTimeSeriesEncoder(n_features=4, patch_size=4, hidden_dim=256)
    model = FinMLLM(ts_encoder=ts_encoder, load_in_8bit=args.load_in_8bit, attn_implementation=args.attn_implementation)

    model.llm.enable_input_require_grads()
    if args.load_in_8bit:
        model.llm = prepare_model_for_kbit_training(model.llm)
    lora_config = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_r, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        # full fine-tune (not LoRA-decomposed -- it's a small 2-layer MLP, too small to bother
        # decomposing) the vision-tower-to-LLM patch merger alongside the LoRA adapters. Qwen3-VL
        # was pretrained on natural images/documents, not synthetic candlestick renders; this
        # merger is the one place vision features get translated into the LLM's embedding space,
        # so it's the cheapest lever for adapting to a new visual domain (same rationale as
        # LLaVA-style training unfreezing the projector while keeping the ViT backbone frozen).
        modules_to_save=["merger"] if args.train_vision_projector else None,
    )
    model.llm = get_peft_model(model.llm, lora_config)
    model.llm.print_trainable_parameters()

    print("\n########## BEFORE TRAINING ##########")
    run_eval(model, eval_ds, n_qualitative=2, horizon=args.horizon)

    print("\n########## TRAINING ##########")
    loss_history = train(model, train_sampler, n_steps=args.n_steps, batch_size=args.batch_size, lr=args.lr,
                          horizon=args.horizon, log_every=args.log_every,
                          grad_accum_steps=args.grad_accum_steps, max_grad_norm=args.max_grad_norm,
                          use_amp=args.amp)

    results_dir = args.results_dir or os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "univariate")
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    print("\n########## AFTER TRAINING (full accuracy pass) ##########")
    run_eval(model, eval_ds, results_dir=results_dir, loss_history=loss_history, n_qualitative=args.n_result_samples,
             full_accuracy=True, horizon=args.horizon)
