import torch
from PIL import Image


def render_candlesticks(ohlc, width=224, height=224, candle_width=3, wick_width=1, gap=1,
                         bg_color=(0.0, 0.0, 0.0),
                         bull_color=(0.85, 0.15, 0.15),
                         bear_color=(0.15, 0.4, 0.85),
                         wick_tint=0.5,
                         ma_windows=(), ma_series=None, ma_colors=None, ma_line_width=1):
    """
    ohlc: (B, T, 4) tensor, last dim = [open, high, low, close]
    bull_color / bear_color: RGB in [0, 1] for close >= open / close < open (JP convention: red up, blue down)
    wick_tint: how much to lighten the wick color relative to the body color, in [0, 1] (0 = same as body,
        1 = white). Gives the thin wick contrast against the wider body fill so it doesn't get visually lost.

    ma_windows: tuple of simple-moving-average window sizes computed from the visible `close` only, e.g. (5, 25).
        The first `window - 1` candles have no line since there isn't enough visible history to average over.
    ma_series: optional list/tuple of precomputed MA value tensors, each shaped (T,) or (B, T), price-scale.
        Use this when you have data from before the displayed window and want the line to cover every candle
        (e.g. compute the SMA over the full history, then pass only the slice aligned to this ohlc window).
        Takes precedence over ma_windows if both are given. Each entry's length must equal ohlc's T.
    ma_colors: RGB tuple per MA line (ma_series if given, else ma_windows); defaults to a fixed palette if None
    returns: (B, height, width, 3) RGB image tensor in [0, 1]
    """
    B, T, _ = ohlc.shape
    device = ohlc.device
    step = candle_width + gap
    assert step * T <= width, "increase width or shrink candle_width/gap for this T"

    o, h, l, c = ohlc.unbind(-1)
    lo = l.min(dim=1, keepdim=True).values
    hi = h.max(dim=1, keepdim=True).values
    scale = (hi - lo).clamp_min(1e-8)

    def to_row(x):
        return ((hi - x) / scale * (height - 1)).round().long().clamp(0, height - 1)

    row_o, row_h, row_l, row_c = to_row(o), to_row(h), to_row(l), to_row(c)
    bullish = c >= o
    body_top = torch.minimum(row_o, row_c)
    body_bot = torch.maximum(row_o, row_c)

    row_idx = torch.arange(height, device=device).view(1, height, 1)
    wick_mask = (row_idx >= row_h.unsqueeze(1)) & (row_idx <= row_l.unsqueeze(1))  # (B,H,T)
    body_mask = (row_idx >= body_top.unsqueeze(1)) & (row_idx <= body_bot.unsqueeze(1))  # (B,H,T)

    bull_c = torch.tensor(bull_color, device=device, dtype=ohlc.dtype)
    bear_c = torch.tensor(bear_color, device=device, dtype=ohlc.dtype)
    body_color = torch.where(bullish.unsqueeze(-1), bull_c, bear_c)  # (B,T,3)
    body_color = body_color.unsqueeze(1).expand(-1, height, -1, -1)  # (B,H,T,3)

    bull_wick_c = bull_c + (1.0 - bull_c) * wick_tint
    bear_wick_c = bear_c + (1.0 - bear_c) * wick_tint
    wick_color = torch.where(bullish.unsqueeze(-1), bull_wick_c, bear_wick_c)  # (B,T,3)
    wick_color = wick_color.unsqueeze(1).expand(-1, height, -1, -1)  # (B,H,T,3)

    bg_c = torch.tensor(bg_color, device=device, dtype=ohlc.dtype)
    img = bg_c.view(1, 1, 1, 3).expand(B, height, width, 3).clone()

    col_start = torch.arange(T, device=device) * step
    wick_offset = (candle_width - wick_width) // 2

    for dx in range(wick_width):
        cols = col_start + wick_offset + dx
        img[:, :, cols] = torch.where(wick_mask.unsqueeze(-1), wick_color, img[:, :, cols])
    for dx in range(candle_width):
        cols = col_start + dx
        img[:, :, cols] = torch.where(body_mask.unsqueeze(-1), body_color, img[:, :, cols])

    ma_values = []
    if ma_series is not None:
        for i, s in enumerate(ma_series):
            s = torch.as_tensor(s, device=device, dtype=ohlc.dtype)
            if s.dim() == 1:
                s = s.unsqueeze(0).expand(B, -1)
            assert s.shape[0] == B, f"ma_series[{i}] batch size {s.shape[0]} != ohlc's batch size {B}"
            assert s.shape[-1] == T, f"ma_series[{i}] length {s.shape[-1]} != ohlc's T ({T})"
            ma_values.append(s)
    elif ma_windows:
        for window in ma_windows:
            ma_values.append(_sma(c, window))

    if ma_values:
        if ma_colors is None:
            default_palette = [(1.0, 1.0, 0.2), (0.2, 1.0, 1.0), (1.0, 0.5, 1.0), (0.6, 1.0, 0.3)]
            ma_colors = [default_palette[i % len(default_palette)] for i in range(len(ma_values))]
        center_offset = candle_width // 2
        for values, color in zip(ma_values, ma_colors):
            _draw_ma_line(img, values, hi, scale, col_start, center_offset, step,
                          torch.tensor(color, device=device, dtype=ohlc.dtype), line_width=ma_line_width)

    return img


def _sma(close, window):
    """close: (B, T) -> (B, T) simple moving average, nan-padded for the first `window - 1` steps"""
    B, T = close.shape
    device = close.device
    if window > T:
        return torch.full_like(close, float('nan'))
    cumsum = torch.cat([torch.zeros(B, 1, device=device, dtype=close.dtype), torch.cumsum(close, dim=1)], dim=1)
    sma = (cumsum[:, window:] - cumsum[:, :T - window + 1]) / window  # (B, T-window+1)
    pad = torch.full((B, window - 1), float('nan'), device=device, dtype=close.dtype)
    return torch.cat([pad, sma], dim=1)  # (B, T), aligned to the same time axis as close


def _draw_ma_line(img, values, hi, scale, col_start, center_offset, step, color, line_width=1):
    """
    values: (B, T) price-scale line values, may contain nan for missing points; hi, scale: (B, 1); col_start: (T,)
    draws `values` as a connected polyline, in-place on img (B,H,W,3)
    """
    B, T = values.shape
    device = values.device
    H, W = img.shape[1], img.shape[2]
    if T < 2:
        return

    valid = ~torch.isnan(values)
    row = (hi - values) / scale * (H - 1)  # (B, T) float row, nan where invalid

    n_cols = (T - 1) * step + 1
    rel_cols = torch.arange(n_cols, device=device)
    seg_idx = (rel_cols // step).clamp(max=T - 2)
    frac = ((rel_cols % step).float() / step).view(1, -1)

    r0, r1 = row[:, seg_idx], row[:, seg_idx + 1]
    interp_row = r0 + (r1 - r0) * frac
    seg_valid = valid[:, seg_idx] & valid[:, seg_idx + 1]  # (B, n_cols)

    col_abs = (col_start[0] + center_offset + rel_cols).clamp(0, W - 1)
    row_idx = interp_row.round().long().clamp(0, H - 1)

    batch_idx = torch.arange(B, device=device).view(B, 1).expand(B, n_cols)
    col_idx = col_abs.view(1, -1).expand(B, n_cols)

    half = line_width // 2
    for dy in range(-half, line_width - half):
        r = (row_idx + dy).clamp(0, H - 1)
        img[batch_idx[seg_valid], r[seg_valid], col_idx[seg_valid]] = color


def make_synthetic_ohlc(batch=4, length=60, seed=0):
    g = torch.Generator().manual_seed(seed)
    base = torch.cumsum(torch.randn(batch, length, generator=g) * 0.5, dim=1) + 100
    o = base
    c = base + torch.randn(batch, length, generator=g) * 0.3
    h = torch.maximum(o, c) + torch.rand(batch, length, generator=g) * 0.3
    l = torch.minimum(o, c) - torch.rand(batch, length, generator=g) * 0.3
    return torch.stack([o, h, l, c], dim=-1)


def rolling_mean_last_n(x, window, n):
    """x: (B, L) -> (B, n) SMA for the last n steps of x, using `window`-length lookback (needs L >= n + window - 1)"""
    B, L = x.shape
    assert L >= n + window - 1, f"need at least {n + window - 1} steps of history for window={window}, got {L}"
    return x.unfold(dimension=1, size=window, step=1).mean(dim=-1)[:, -n:]


if __name__ == "__main__":
    import os

    out_dir = os.path.dirname(os.path.abspath(__file__))
    visible_len = 60
    lookback = 24  # extra history before the visible window, only used to seed the MAs
    ohlc_full = make_synthetic_ohlc(batch=4, length=lookback + visible_len)
    ohlc = ohlc_full[:, lookback:]  # the actually displayed window

    ma_windows = (5, 20)
    close_full = ohlc_full[..., 3]  # (B, lookback + visible_len)
    ma_series = [rolling_mean_last_n(close_full, w, visible_len) for w in ma_windows]

    imgs = render_candlesticks(ohlc, width=240, height=160, candle_width=3, wick_width=1, gap=1,
                                ma_series=ma_series)

    for i in range(imgs.shape[0]):
        arr = (imgs[i].clamp(0, 1) * 255).byte().numpy()
        Image.fromarray(arr).save(os.path.join(out_dir, f"candlestick_sample_{i}.png"))

    print(f"saved {imgs.shape[0]} images to {out_dir}")
