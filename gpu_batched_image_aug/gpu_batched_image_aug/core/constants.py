"""Interpolation / border mode constants.

Values are identical to the corresponding OpenCV constants, so ``cv2.INTER_LINEAR`` / ``cv2.BORDER_REFLECT_101``
etc. can be passed as-is (same as albumentations). String aliases are also accepted.
"""

INTER_NEAREST = 0
INTER_LINEAR = 1
INTER_CUBIC = 2
INTER_AREA = 3

BORDER_CONSTANT = 0
BORDER_REPLICATE = 1
BORDER_REFLECT = 2
BORDER_WRAP = 3
BORDER_REFLECT_101 = 4

_INTERPOLATION_ALIASES = {
    "nearest": INTER_NEAREST,
    "linear": INTER_LINEAR,
    "bilinear": INTER_LINEAR,
    "cubic": INTER_CUBIC,
    "bicubic": INTER_CUBIC,
    "area": INTER_AREA,
}
_BORDER_ALIASES = {
    "constant": BORDER_CONSTANT,
    "replicate": BORDER_REPLICATE,
    "edge": BORDER_REPLICATE,
    "reflect": BORDER_REFLECT,
    "symmetric": BORDER_REFLECT,
    "wrap": BORDER_WRAP,
    "reflect_101": BORDER_REFLECT_101,
    "reflect101": BORDER_REFLECT_101,
}


def resolve_interpolation(value: int | str, allow_area: bool = False) -> int:
    if isinstance(value, str):
        key = value.lower()
        if key not in _INTERPOLATION_ALIASES:
            raise ValueError(f"Unknown interpolation {value!r}. Use one of {sorted(_INTERPOLATION_ALIASES)}")
        value = _INTERPOLATION_ALIASES[key]
    supported = {INTER_NEAREST, INTER_LINEAR, INTER_CUBIC} | ({INTER_AREA} if allow_area else set())
    if value not in supported:
        raise ValueError(
            f"Interpolation {value} is not supported here (supported: nearest=0, linear=1, cubic=2"
            f"{', area=3' if allow_area else ''}). cv2.INTER_LANCZOS4 etc. have no GPU batched counterpart."
        )
    return int(value)


def resolve_border_mode(value: int | str) -> int:
    if isinstance(value, str):
        key = value.lower()
        if key not in _BORDER_ALIASES:
            raise ValueError(f"Unknown border mode {value!r}. Use one of {sorted(_BORDER_ALIASES)}")
        value = _BORDER_ALIASES[key]
    if value not in {BORDER_CONSTANT, BORDER_REPLICATE, BORDER_REFLECT, BORDER_WRAP, BORDER_REFLECT_101}:
        raise ValueError(f"Unsupported border mode {value}")
    return int(value)
