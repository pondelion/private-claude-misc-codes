"""Random number generation on the target device.

Every random draw of the library goes through :class:`Rng`, which wraps a ``torch.Generator`` living on the same
device as the data. Seeding a :class:`~gpu_batched_image_aug.Compose` (``seed=...``) therefore makes the whole
pipeline reproducible; without a seed the generator is seeded from torch's global RNG, so ``torch.manual_seed``
also controls it.

All "range" arguments (``low`` / ``high``) may be python numbers or tensors broadcastable to ``size``, which is how
per-sample parameters are drawn in a single call.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

Size = int | Sequence[int]


def _size(size: Size) -> tuple[int, ...]:
    return (size,) if isinstance(size, int) else tuple(size)


class Rng:
    def __init__(self, device: torch.device | str, seed: int | None = None):
        self.device = torch.device(device)
        self.generator = torch.Generator(device=self.device)
        if seed is None:
            seed = int(torch.randint(0, 2**62, (1,)).item())
        self.generator.manual_seed(seed)

    def rand(self, size: Size, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        return torch.rand(_size(size), generator=self.generator, device=self.device, dtype=dtype)

    def randn(self, size: Size, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        return torch.randn(_size(size), generator=self.generator, device=self.device, dtype=dtype)

    def uniform(self, low: float | torch.Tensor, high: float | torch.Tensor, size: Size) -> torch.Tensor:
        """Uniform in [low, high). ``low``/``high`` may be tensors broadcastable to ``size``."""
        return low + (high - low) * self.rand(size)

    def normal(self, mean: float | torch.Tensor, std: float | torch.Tensor, size: Size) -> torch.Tensor:
        return mean + std * self.randn(size)

    def randint(self, low: int | torch.Tensor, high: int | torch.Tensor, size: Size) -> torch.Tensor:
        """Integers uniform in [low, high] (**both ends inclusive**, same as ``random.randint``).
        ``low``/``high`` may be tensors broadcastable to ``size``."""
        if not torch.is_tensor(low) and not torch.is_tensor(high):
            return torch.randint(int(low), int(high) + 1, _size(size), generator=self.generator, device=self.device)
        low_t = torch.as_tensor(low, device=self.device, dtype=torch.float64)
        high_t = torch.as_tensor(high, device=self.device, dtype=torch.float64)
        u = self.rand(size, dtype=torch.float64)
        return torch.minimum(low_t + torch.floor(u * (high_t - low_t + 1)), high_t).long()

    def bernoulli(self, p: float | torch.Tensor, size: Size) -> torch.Tensor:
        return self.rand(size) < p

    def choice(self, num_options: int, size: Size, probs: Sequence[float] | torch.Tensor | None = None) -> torch.Tensor:
        """Indices in [0, num_options) with optional (unnormalized) probabilities."""
        size = _size(size)
        if probs is None:
            return torch.randint(0, num_options, size, generator=self.generator, device=self.device)
        p = torch.as_tensor(probs, dtype=torch.float32, device=self.device)
        p = p / p.sum()
        n = 1
        for s in size:
            n *= s
        return torch.multinomial(p.expand(n, num_options), 1, replacement=True, generator=self.generator).view(size)

    def permutation(self, n_rows: int, n: int) -> torch.Tensor:
        """``n_rows`` independent random permutations of ``range(n)`` -> (n_rows, n) long."""
        return torch.argsort(self.rand((n_rows, n)), dim=1)

    def poisson(self, rate: torch.Tensor) -> torch.Tensor:
        return torch.poisson(rate, generator=self.generator)
