import os
import random
import sys

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "candlestick"))
from fast_candlestick_render import render_candlesticks, rolling_mean_last_n  # noqa: E402

DEFAULT_PARQUET_PATH = "/mnt/c/data/finance/yfinance/us_etf_ind_20000101_20251024_db_extended_to_20260516.parquet"
OHLC_COLS = ["Open", "High", "Low", "Close"]  # order matches candlestick_render's expected last-dim layout


def load_ohlc_frame(path: str = DEFAULT_PARQUET_PATH) -> pd.DataFrame:
    """path -> DataFrame indexed by (Ticker, Date), sorted, columns = OHLC_COLS only"""
    df = pd.read_parquet(path, columns=OHLC_COLS)
    df = df.sort_index()
    return df


def split_by_date(df: pd.DataFrame, cutoff_date: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits df into (train_df, eval_df) at `cutoff_date`: rows with Date < cutoff_date go to
    train, rows with Date >= cutoff_date go to eval. This keeps every eval window strictly
    *after* every train window in calendar time -- picking train/eval windows from the same full
    date range (even with different random seeds, as a naive split would) risks the model being
    evaluated on a period it could have effectively already seen data from, which especially
    matters for a forecasting task.
    """
    dates = df.index.get_level_values(1)
    cutoff = pd.Timestamp(cutoff_date)
    cutoff = cutoff.date() if isinstance(dates[0], type(cutoff.date())) else cutoff  # match index element type
    train_df = df[dates < cutoff]
    eval_df = df[dates >= cutoff]
    return train_df, eval_df


def _eligible_tickers(df: pd.DataFrame, lookback: int, min_visible_len: int, future_horizon: int = 0):
    """
    Precomputable once per Dataset/Sampler instance (it's an O(n) scan over the whole frame) and
    then reused across many window draws, rather than recomputed on every single sample.
    """
    tickers = df.index.get_level_values(0).unique().tolist()
    sizes = df.groupby(level=0).size()
    min_total = lookback + min_visible_len + future_horizon
    eligible = [t for t in tickers if sizes[t] >= min_total]
    assert eligible, (
        f"no ticker has >= {min_total} rows (lookback={lookback} + min visible_len={min_visible_len} + future_horizon={future_horizon})"
    )
    return eligible, sizes


def sample_one_window(eligible: list, sizes: pd.Series, visible_len_range, lookback: int,
                       rng: random.Random, future_horizon: int = 0):
    """
    Draws one fresh (ticker, start_pos, visible_len) triple such that
    [start_pos - lookback, start_pos + visible_len + future_horizon) is a valid, in-range slice of
    that ticker's rows. start_pos is a row-position (not a calendar date), so gaps/holidays don't
    matter. `eligible`/`sizes` come from `_eligible_tickers` (computed once, passed in here so
    repeated calls -- e.g. from a dynamic per-step sampler -- don't re-scan the whole frame).
    """
    lo, hi = (visible_len_range, visible_len_range) if isinstance(visible_len_range, int) else visible_len_range
    visible_len = rng.randint(lo, hi)
    candidates = [t for t in eligible if sizes[t] >= lookback + visible_len + future_horizon]
    ticker = rng.choice(candidates) if candidates else rng.choice(eligible)
    visible_len = min(visible_len, sizes[ticker] - lookback - future_horizon)
    n = sizes[ticker]
    start_pos = rng.randint(lookback, n - visible_len - future_horizon)  # first *visible* row
    return ticker, start_pos, visible_len


def build_windows(df: pd.DataFrame, visible_len_range, lookback: int, n_samples: int, seed: int = 0,
                   future_horizon: int = 0):
    """
    Pre-samples a *fixed* pool of n_samples (ticker, start_pos, visible_len) triples, once, via
    repeated sample_one_window() calls. Used by FinCandlestickDataset (fixed & reproducible --
    appropriate for eval, where you want the exact same windows before/after training). For
    training, prefer DynamicFinCandlestickSampler instead: indexing into a pool fixed at
    construction time caps how many distinct windows a training run can ever see at n_samples,
    no matter how many training steps you run, which is not what you want there.

    visible_len_range: either a fixed int (every window has that exact length) or a (min, max)
        tuple, in which case each window independently samples its own length uniformly from
        [min, max].
    future_horizon: extra rows to reserve *after* the visible window (not sampled into it) so a
        caller doing return-prediction (see FinCandlestickDataset's prediction_horizon) always
        has real future data available to compute the realized outcome from -- 0 (default)
        reserves nothing.
    """
    rng = random.Random(seed)
    lo = visible_len_range if isinstance(visible_len_range, int) else visible_len_range[0]
    eligible, sizes = _eligible_tickers(df, lookback, lo, future_horizon)
    return [sample_one_window(eligible, sizes, visible_len_range, lookback, rng, future_horizon)
            for _ in range(n_samples)]


def fit_render_params(visible_len: int, px_per_bar: int = 4, max_width: int = 480, height: int = 160):
    """
    Pick (width, candle_width, gap) for render_candlesticks so that longer windows don't just
    get squeezed into a fixed-width image: width grows with visible_len (at `px_per_bar` px/bar)
    up to `max_width`, beyond which bars shrink instead of the image growing without bound.
    """
    width = visible_len * px_per_bar
    if width <= max_width:
        return width, max(1, px_per_bar - 1), 1
    step = max(2, max_width // visible_len)
    candle_width = max(1, step - 1)
    gap = 1 if step > candle_width else 0
    return step * visible_len, candle_width, gap


def describe_window_mock(ticker: str, ohlc_visible: np.ndarray, dates) -> str:
    """
    Cheap, deterministic placeholder for the "description" output (spec item 5): computes a few
    real summary stats from the window and drops them into a template. Not a trained caption --
    just enough real signal (direction, magnitude, range, volatility) to stand in for one until an
    actual captioning model (e.g. teacher-VLM-generated or human-written) is wired up.
    """
    close = ohlc_visible[:, 3]
    open_ = ohlc_visible[:, 0]
    high = ohlc_visible[:, 1]
    low = ohlc_visible[:, 2]

    pct_change = (close[-1] / close[0] - 1.0) * 100
    daily_ret = np.diff(close) / close[:-1]
    volatility = daily_ret.std() * 100
    up_days = int((close >= open_).sum())
    down_days = len(close) - up_days
    direction = "上昇" if pct_change > 1 else ("下落" if pct_change < -1 else "横ばい")

    return (
        f"[MOCK DESCRIPTION] {ticker or 'unknown asset'} について、{dates[0]}から{dates[-1]}までの"
        f"{len(close)}本の期間は{direction}基調（期間騰落率 {pct_change:+.1f}%）。"
        f"高値 {high.max():.2f} / 安値 {low.min():.2f}、日次リターンのボラティリティ約{volatility:.2f}%。"
        f"陽線 {up_days}本 / 陰線 {down_days}本。"
    )


def describe_future_return(ticker: str, last_close: float, future_close: float, future_high: float,
                            future_low: float, horizon_bars: int) -> str:
    """
    Teacher text for the return-prediction task: not a mock number, this is the *actually realized*
    forward return computed from real future data -- used as the SFT target so the model learns to
    predict it from the visible window (which does not include this future slice) alone. Rule-based
    templating only applies to the wording, not the numbers.
    """
    pct = (future_close / last_close - 1.0) * 100
    direction = "上昇" if pct > 1 else ("下落" if pct < -1 else "横ばい")
    max_up = (future_high / last_close - 1.0) * 100
    max_down = (future_low / last_close - 1.0) * 100

    return (
        f"[FUTURE OUTCOME] {ticker or 'unknown asset'}: 直近終値 {last_close:.2f} を起点に、"
        f"今後{horizon_bars}本の期間で{direction}し、期間終値は {future_close:.2f}（騰落率 {pct:+.1f}%）。"
        f"期間中の最大値幅は上値 {max_up:+.1f}% / 下値 {max_down:+.1f}%。"
    )


def _build_item(df: pd.DataFrame, ticker: str, start_pos: int, visible_len: int, lookback: int,
                 ma_windows, image_height: int, unknown_ticker_prob: float, prediction_horizon: int | None,
                 rng: random.Random) -> dict:
    """
    Turns one (ticker, start_pos, visible_len) window into the actual sample dict: renders the
    candlestick image, computes the mock description, and (if prediction_horizon is set) the
    realized future-return teacher text. Shared by FinCandlestickDataset.__getitem__ (fixed pool)
    and DynamicFinCandlestickSampler.sample() (fresh every call) so this logic lives in one place.
    """
    ticker_df = df.loc[ticker]
    full_slice = ticker_df.iloc[start_pos - lookback: start_pos + visible_len]
    visible_slice = full_slice.iloc[lookback:]

    ohlc_full = torch.as_tensor(full_slice[OHLC_COLS].to_numpy(copy=True), dtype=torch.float32)  # (lookback+T, 4)
    ohlc_visible = ohlc_full[lookback:]  # (T, 4)

    ma_series = None
    if ma_windows:
        close_full = ohlc_full[:, 3].unsqueeze(0)  # (1, lookback+T)
        ma_series = [rolling_mean_last_n(close_full, w, visible_len)[0] for w in ma_windows]

    width, candle_width, gap = fit_render_params(visible_len)
    img_tensor = render_candlesticks(
        ohlc_visible.unsqueeze(0), width=width, height=image_height,
        candle_width=candle_width, gap=gap, ma_series=ma_series,
    )[0]  # (H, W, 3) in [0,1]
    image = Image.fromarray((img_tensor.clamp(0, 1) * 255).byte().numpy())

    shown_ticker = "" if rng.random() < unknown_ticker_prob else ticker
    description = describe_window_mock(ticker, ohlc_visible.numpy(), visible_slice.index.astype(str).tolist())

    result = {
        "ticker": shown_ticker,
        "ohlc": ohlc_visible,  # (T, 4), T varies per sample -- for the optional time-series branch
        "image": image,
        "description": description,
        "n_bars": visible_len,
        "start_date": str(visible_slice.index[0]),
        "end_date": str(visible_slice.index[-1]),
    }

    if prediction_horizon:
        future_slice = ticker_df.iloc[start_pos + visible_len: start_pos + visible_len + prediction_horizon]
        last_close = float(ohlc_visible[-1, 3])
        result["future_return_pct"] = (float(future_slice["Close"].iloc[-1]) / last_close - 1.0) * 100
        result["future_return_text"] = describe_future_return(
            ticker, last_close, float(future_slice["Close"].iloc[-1]),
            float(future_slice["High"].max()), float(future_slice["Low"].min()),
            len(future_slice),
        )

    return result


class FinCandlestickDataset(Dataset):
    """
    Each item: a rendered candlestick image + the raw OHLC window (for the optional time-series
    branch) + a mock description + the asset ticker. Tokenization/prompt-building is the model's
    job (FinMLLM.prepare_batch), not this dataset's -- keeps this class independent of any
    particular LLM/tokenizer.

    Draws a *fixed* pool of n_samples windows once at construction time -- appropriate for eval,
    where you want the exact same windows before/after training so checkpoints are comparable.
    For training, use DynamicFinCandlestickSampler instead (see its docstring for why).
    """

    def __init__(self, df: pd.DataFrame, visible_len=(40, 120), lookback=24, n_samples=256, seed=0,
                 ma_windows=(5, 20), image_height=160, unknown_ticker_prob=0.0, prediction_horizon=None):
        """
        visible_len: fixed int, or (min, max) to sample a different window length per item
            (augmentation during training; also exercises the same variable-length path used at
            inference time, where the caller obviously won't always have exactly one fixed T).
        prediction_horizon: None (default, matches old behavior exactly) -> no future data is
            touched. Set to an int to additionally reserve that many rows *after* each visible
            window and attach the realized forward return (see describe_future_return) as
            'future_return_pct'/'future_return_text' -- the return-prediction task's SFT target.
        """
        self.df = df
        self.lookback = lookback
        self.ma_windows = ma_windows
        self.image_height = image_height
        self.unknown_ticker_prob = unknown_ticker_prob
        self.prediction_horizon = prediction_horizon
        self.windows = build_windows(df, visible_len, lookback, n_samples, seed=seed,
                                      future_horizon=prediction_horizon or 0)
        self._rng = random.Random(seed + 1)

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        ticker, start_pos, visible_len = self.windows[idx]
        return _build_item(self.df, ticker, start_pos, visible_len, self.lookback, self.ma_windows,
                            self.image_height, self.unknown_ticker_prob, self.prediction_horizon, self._rng)


class DynamicFinCandlestickSampler:
    """
    Draws a genuinely fresh (ticker, start_pos, visible_len) window on every single .sample()
    call, instead of pre-building a fixed pool once (as FinCandlestickDataset does). A training
    loop built on top of a fixed pool of n_samples windows can never see more than n_samples
    distinct windows no matter how many training steps you run; this removes that artificial cap
    -- diversity is bounded only by the underlying data (ticker x position x window-length
    combinatorics), which is what you want for training. Use FinCandlestickDataset instead for
    eval, where a fixed/reproducible set of windows is actually the goal.

    Not a torch.utils.data.Dataset (no meaningful fixed __len__) -- call .sample() directly, as
    train_eval_demo.py's train() does.
    """

    def __init__(self, df: pd.DataFrame, visible_len=(40, 120), lookback=24, seed=0,
                 ma_windows=(5, 20), image_height=160, unknown_ticker_prob=0.0, prediction_horizon=None):
        self.df = df
        self.visible_len = visible_len
        self.lookback = lookback
        self.ma_windows = ma_windows
        self.image_height = image_height
        self.unknown_ticker_prob = unknown_ticker_prob
        self.prediction_horizon = prediction_horizon

        lo = visible_len if isinstance(visible_len, int) else visible_len[0]
        self._eligible, self._sizes = _eligible_tickers(df, lookback, lo, prediction_horizon or 0)
        self._window_rng = random.Random(seed)
        self._item_rng = random.Random(seed + 1)

    def sample(self) -> dict:
        ticker, start_pos, visible_len = sample_one_window(
            self._eligible, self._sizes, self.visible_len, self.lookback, self._window_rng,
            future_horizon=self.prediction_horizon or 0,
        )
        return _build_item(self.df, ticker, start_pos, visible_len, self.lookback, self.ma_windows,
                            self.image_height, self.unknown_ticker_prob, self.prediction_horizon, self._item_rng)


def sample_context_assets(df: pd.DataFrame, target_ticker: str, target_dates: pd.Index, n_context: int,
                           rng: random.Random, min_coverage: float = 0.9) -> list[dict]:
    """
    Randomly selects up to n_context *other* tickers, reindexed to exactly `target_dates` so each
    context asset's OHLC lines up bar-for-bar with the target window (calendar-aligned, not just
    "same number of rows" -- different tickers have different trading calendars/inception dates).
    Tickers with insufficient coverage for this date range (e.g. not yet listed, delisted) are
    skipped. Returns fewer than n_context if not enough eligible tickers are found for this
    particular window -- callers that also want the context *count* randomized should draw
    n_context randomly per item to begin with (see FinMultiAssetCandlestickDataset).
    """
    if n_context <= 0:
        return []

    candidates = [t for t in df.index.get_level_values(0).unique() if t != target_ticker]
    rng.shuffle(candidates)  # random combination, not just "first N found"

    selected = []
    for ticker in candidates:
        if len(selected) >= n_context:
            break
        aligned = df.loc[ticker].reindex(target_dates)
        coverage = aligned[OHLC_COLS[0]].notna().mean()
        if coverage < min_coverage:
            continue
        aligned = aligned.ffill().bfill()
        ohlc = torch.as_tensor(aligned[OHLC_COLS].to_numpy(copy=True), dtype=torch.float32)
        selected.append({"ticker": ticker, "ohlc": ohlc, "n_bars": len(target_dates)})
    return selected


# distinct per-tile background colors, cycled by tile index -- acts as a visual "positional
# encoding" for tile slot (paired with the layout description in the prompt text), independent of
# which ticker happens to land in that slot. Kept dark and away from the red/blue candle palette
# (see fast_candlestick_render's bull/bear colors) so bodies/wicks stay readable on every tile.
TILE_BG_PALETTE = [
    (0.05, 0.05, 0.05),  # near-black
    (0.03, 0.14, 0.03),  # dark green
    (0.16, 0.10, 0.00),  # dark brown/orange
    (0.12, 0.00, 0.16),  # dark purple
    (0.00, 0.13, 0.13),  # dark teal
    (0.16, 0.16, 0.00),  # dark olive
    (0.13, 0.13, 0.13),  # neutral gray
    (0.05, 0.00, 0.12),  # dark indigo
]


def tile_context_images(context_samples: list[dict], tile_width=120, tile_height=80, max_cols=4) -> Image.Image | None:
    """
    Renders every context asset's candlestick chart at a small, fixed size -- each on its own
    distinct background color (see TILE_BG_PALETTE, cycled by tile index) -- and tiles them into
    ONE composite image (left-to-right, top-to-bottom) instead of sending N separate full-size
    images. Keeps the vision cost bounded regardless of how many context assets there are (mirrors
    why MultiAssetTimeSeriesBranch pools context tokens into a fixed count on the ts side), and the
    per-slot background gives the model a visual cue for tile position independent of which ticker
    landed there (paired with the layout description in the prompt text). Returns None if
    context_samples is empty.
    """
    if not context_samples:
        return None

    n = len(context_samples)
    T = context_samples[0]["ohlc"].shape[0]
    step = max(1, tile_width // T)
    candle_width = max(1, step - 1)
    gap = 1 if step > candle_width else 0

    cols = min(max_cols, n)
    rows = -(-n // cols)  # ceil division
    canvas = Image.new("RGB", (cols * tile_width, rows * tile_height))
    for i, sample in enumerate(context_samples):
        bg_color = TILE_BG_PALETTE[i % len(TILE_BG_PALETTE)]
        img = render_candlesticks(sample["ohlc"].unsqueeze(0), width=tile_width, height=tile_height,
                                   candle_width=candle_width, gap=gap, bg_color=bg_color)[0]
        tile = Image.fromarray((img.clamp(0, 1) * 255).byte().numpy())
        r, c = divmod(i, cols)
        canvas.paste(tile, (c * tile_width, r * tile_height))
    return canvas


class FinMultiAssetCandlestickDataset(FinCandlestickDataset):
    """
    Extends FinCandlestickDataset with a variable number of randomly-selected context assets per
    item. Both *which* tickers (random combination) and *how many* (random count within
    n_context_range) are re-sampled every time an item is fetched, so a training run sees many
    different context compositions for the same target window -- composition-and-count
    augmentation, not a fixed context set. Context assets carry only ticker + calendar-aligned
    OHLC (no image/description of their own); they feed MultiAssetTimeSeriesBranch's context pool.

    Same fixed-pool-of-n_samples caveat as FinCandlestickDataset applies here -- use
    DynamicFinMultiAssetCandlestickSampler for training instead.
    """

    def __init__(self, df: pd.DataFrame, visible_len=(40, 120), lookback=24, n_samples=256, seed=0,
                 ma_windows=(5, 20), image_height=160, unknown_ticker_prob=0.0, prediction_horizon=None,
                 n_context_range=(0, 6), min_context_coverage=0.9):
        super().__init__(df, visible_len=visible_len, lookback=lookback, n_samples=n_samples, seed=seed,
                          ma_windows=ma_windows, image_height=image_height, unknown_ticker_prob=unknown_ticker_prob,
                          prediction_horizon=prediction_horizon)
        self.n_context_range = n_context_range
        self.min_context_coverage = min_context_coverage

    def __getitem__(self, idx):
        sample = super().__getitem__(idx)
        ticker, start_pos, visible_len = self.windows[idx]
        target_dates = self.df.loc[ticker].iloc[start_pos: start_pos + visible_len].index

        n_context = self._rng.randint(*self.n_context_range)
        sample["context_samples"] = sample_context_assets(
            self.df, ticker, target_dates, n_context, self._rng, min_coverage=self.min_context_coverage,
        )
        sample["context_tile_image"] = tile_context_images(sample["context_samples"])
        return sample


class DynamicFinMultiAssetCandlestickSampler(DynamicFinCandlestickSampler):
    """Dynamic (fresh-every-call) counterpart to FinMultiAssetCandlestickDataset -- see
    DynamicFinCandlestickSampler's docstring for why this exists instead of a fixed pool."""

    def __init__(self, df: pd.DataFrame, visible_len=(40, 120), lookback=24, seed=0,
                 ma_windows=(5, 20), image_height=160, unknown_ticker_prob=0.0, prediction_horizon=None,
                 n_context_range=(0, 6), min_context_coverage=0.9):
        super().__init__(df, visible_len=visible_len, lookback=lookback, seed=seed, ma_windows=ma_windows,
                          image_height=image_height, unknown_ticker_prob=unknown_ticker_prob,
                          prediction_horizon=prediction_horizon)
        self.n_context_range = n_context_range
        self.min_context_coverage = min_context_coverage
        self._context_rng = random.Random(seed + 2)

    def sample(self) -> dict:
        ticker, start_pos, visible_len = sample_one_window(
            self._eligible, self._sizes, self.visible_len, self.lookback, self._window_rng,
            future_horizon=self.prediction_horizon or 0,
        )
        item = _build_item(self.df, ticker, start_pos, visible_len, self.lookback, self.ma_windows,
                            self.image_height, self.unknown_ticker_prob, self.prediction_horizon, self._item_rng)

        target_dates = self.df.loc[ticker].iloc[start_pos: start_pos + visible_len].index
        n_context = self._context_rng.randint(*self.n_context_range)
        item["context_samples"] = sample_context_assets(
            self.df, ticker, target_dates, n_context, self._context_rng, min_coverage=self.min_context_coverage,
        )
        item["context_tile_image"] = tile_context_images(item["context_samples"])
        return item


if __name__ == "__main__":
    df = load_ohlc_frame()
    print(f"loaded {len(df)} rows, {df.index.get_level_values(0).nunique()} tickers")

    ds = FinCandlestickDataset(df, visible_len=(40, 120), lookback=24, n_samples=4, seed=0)
    for i in range(len(ds)):
        sample = ds[i]
        print(f"--- sample {i} ---")
        print("ticker:", sample["ticker"], "n_bars:", sample["n_bars"],
              "range:", sample["start_date"], "->", sample["end_date"])
        print("ohlc shape:", sample["ohlc"].shape, "image size:", sample["image"].size)
        print("description:", sample["description"])

    print("\n=== multi-asset ===")
    ds_multi = FinMultiAssetCandlestickDataset(df, visible_len=(40, 120), lookback=24, n_samples=6, seed=0,
                                                n_context_range=(0, 5))
    for i in range(len(ds_multi)):
        sample = ds_multi[i]
        ctx = sample["context_samples"]
        print(f"--- sample {i} --- ticker: {sample['ticker']} n_context: {len(ctx)} "
              f"context_tickers: {[c['ticker'] for c in ctx]}")
        for c in ctx:
            assert c["ohlc"].shape[0] == sample["n_bars"], "context OHLC must be calendar-aligned to target length"

    print("\n=== dynamic sampler (should show fresh, non-repeating windows across calls) ===")
    dyn = DynamicFinCandlestickSampler(df, visible_len=(40, 120), lookback=24, seed=0, prediction_horizon=10)
    for _ in range(4):
        s = dyn.sample()
        print("ticker:", s["ticker"], "n_bars:", s["n_bars"], "range:", s["start_date"], "->", s["end_date"])

    print("\n=== dynamic multi-asset sampler ===")
    dyn_multi = DynamicFinMultiAssetCandlestickSampler(df, visible_len=(40, 120), lookback=24, seed=0,
                                                        n_context_range=(0, 5))
    for _ in range(4):
        s = dyn_multi.sample()
        ctx = s["context_samples"]
        print(f"ticker: {s['ticker']} n_context: {len(ctx)} context_tickers: {[c['ticker'] for c in ctx]}")
