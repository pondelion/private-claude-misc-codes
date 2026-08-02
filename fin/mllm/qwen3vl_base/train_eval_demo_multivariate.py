import argparse
import json
import os

# see train_eval_demo.py's top-of-file comment for why -- same variable-shape workload here.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

from data import (
    DynamicFinMultiAssetCandlestickSampler,
    FinMultiAssetCandlestickDataset,
    load_ohlc_frame,
    split_by_date,
)
from model_multivariate import TS_CONTEXT_TOKEN, FinMultiAssetMLLM
from timeseries_encoder import PatchTimeSeriesEncoder
from train_eval_demo import run_eval, train  # reused as-is; both accept any FinMLLM subclass


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
    p.add_argument("--n-context-min", type=int, default=0, help="min random context-asset count per sample")
    p.add_argument("--n-context-max", type=int, default=6, help="max random context-asset count per sample")
    p.add_argument("--n-context-tokens", type=int, default=24, help="fixed pooled-context token count")
    p.add_argument("--n-result-samples", type=int, default=2, help="how many eval samples to dump under results/")
    p.add_argument("--results-dir", default=None, help="defaults to <script_dir>/results/multivariate")
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
    n_context_range = (args.n_context_min, args.n_context_max)

    df = load_ohlc_frame()
    train_df, eval_df = split_by_date(df, cutoff_date=args.cutoff_date)  # eval is strictly later in time than train
    print(f"train rows: {len(train_df)} ({train_df.index.get_level_values(1).min()} ~ {train_df.index.get_level_values(1).max()})")
    print(f"eval rows:  {len(eval_df)} ({eval_df.index.get_level_values(1).min()} ~ {eval_df.index.get_level_values(1).max()})")

    # training: dynamic sampler, fresh (target window + random context combination/count) every
    # call -- see DynamicFinCandlestickSampler's docstring for why this beats a fixed pool here.
    train_sampler = DynamicFinMultiAssetCandlestickSampler(
        train_df, visible_len=(40, 90), lookback=24, seed=0, prediction_horizon=args.horizon,
        n_context_range=n_context_range,
    )
    # eval: FIXED pool on purpose, so before/after-training comparisons use the exact same windows
    # (and the exact same context-asset combinations, for a fair comparison). context count is
    # pinned to the max (not the same random range training uses) so every eval sample exercises
    # the model at full context capacity -- consistent, reproducible, and shows the model's best
    # case rather than an arbitrary random draw.
    eval_ds = FinMultiAssetCandlestickDataset(
        eval_df, visible_len=(40, 90), lookback=24, n_samples=args.n_eval_samples, seed=999,
        prediction_horizon=args.horizon, n_context_range=(args.n_context_max, args.n_context_max),
    )

    ts_encoder = PatchTimeSeriesEncoder(n_features=4, patch_size=4, hidden_dim=256)
    model = FinMultiAssetMLLM(ts_encoder=ts_encoder, n_context_tokens=args.n_context_tokens,
                              load_in_8bit=args.load_in_8bit, attn_implementation=args.attn_implementation)

    model.llm.enable_input_require_grads()
    if args.load_in_8bit:
        model.llm = prepare_model_for_kbit_training(model.llm)
    lora_config = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_r, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        # see train_eval_demo.py's comment on this same arg for why -- same rationale here.
        modules_to_save=["merger"] if args.train_vision_projector else None,
    )
    model.llm = get_peft_model(model.llm, lora_config)
    model.llm.print_trainable_parameters()

    print("\n########## BEFORE TRAINING ##########")
    run_eval(model, eval_ds, n_qualitative=2, horizon=args.horizon, extra_collapse_tokens=[TS_CONTEXT_TOKEN])

    print("\n########## TRAINING ##########")
    loss_history = train(model, train_sampler, n_steps=args.n_steps, batch_size=args.batch_size, lr=args.lr,
                          horizon=args.horizon, log_every=args.log_every,
                          grad_accum_steps=args.grad_accum_steps, max_grad_norm=args.max_grad_norm,
                          use_amp=args.amp)

    results_dir = args.results_dir or os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "multivariate")
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    print("\n########## AFTER TRAINING (full accuracy pass) ##########")
    run_eval(model, eval_ds, results_dir=results_dir, loss_history=loss_history, n_qualitative=args.n_result_samples,
             full_accuracy=True, horizon=args.horizon, extra_collapse_tokens=[TS_CONTEXT_TOKEN])
