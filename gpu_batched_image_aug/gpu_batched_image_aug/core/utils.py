"""Argument normalization helpers mirroring albumentations' pydantic validators."""

from __future__ import annotations

from collections.abc import Sequence
from numbers import Number


def to_range(value: float | Sequence[float], low: float | None = None, name: str = "value") -> tuple[float, float]:
    """albumentations ``to_tuple``: a scalar ``v`` becomes ``(-v, v)`` if ``low`` is None, else ``(low, v)`` sorted;
    a pair is returned as-is (validated)."""
    if isinstance(value, Number):
        v = float(value)
        if low is None:
            return (-v, v)
        return (min(low, v), max(low, v))
    if isinstance(value, Sequence) and len(value) == 2:
        return (float(value[0]), float(value[1]))
    raise ValueError(f"{name} must be a number or a pair, got {value!r}")


def fixed_or_range(value: float | Sequence[float], name: str = "value") -> tuple[float, float]:
    """scalar ``v`` -> ``(v, v)``; pair -> pair (albumentations ``to_tuple(v, v)``)."""
    return to_range(value, low=value if isinstance(value, Number) else None, name=name)


def symmetric_range(value: float | Sequence[float], name: str = "value") -> tuple[float, float]:
    """SymmetricRangeType: scalar ``v`` -> ``(-v, v)``."""
    return to_range(value, None, name)


def check_nondecreasing(rng: tuple[float, float], name: str) -> tuple[float, float]:
    if rng[0] > rng[1]:
        raise ValueError(f"{name}: first value must be <= second value, got {rng}")
    return rng


def check_bounds(rng: tuple[float, float], lo: float, hi: float, name: str) -> tuple[float, float]:
    if not (lo <= rng[0] <= hi and lo <= rng[1] <= hi):
        raise ValueError(f"{name} must be within [{lo}, {hi}], got {rng}")
    return rng


def int_range(value: int | Sequence[int], low: int | None, name: str) -> tuple[int, int]:
    r = to_range(value, low, name)
    return (int(r[0]), int(r[1]))


def odd_blur_range(value: int | Sequence[int], min_value: int, name: str) -> tuple[int, int]:
    """albumentations ``process_blur_limit``: scalar -> (min_value, v); values below min raised to min; even values
    made odd (+1); min > max -> (max, max)."""
    if isinstance(value, Sequence):
        if len(value) != 2:
            raise ValueError(f"{name} must contain exactly 2 elements")
        lo, hi = int(value[0]), int(value[1])
    else:
        lo, hi = min_value, int(value)
    lo, hi = max(lo, min_value), max(hi, min_value)
    lo = lo + 1 if lo != 0 and lo % 2 == 0 else lo
    hi = hi + 1 if hi != 0 and hi % 2 == 0 else hi
    if lo > hi:
        lo = hi
    return lo, hi
