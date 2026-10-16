"""GroundAnythingの0--999座標tokenと構造化responseを理解するための実装。

画像サイズに依存するpixel座標を1000値の離散語彙へ写すことで、box/pointを通常の
language-model tokenと同じcross entropyで学習できる。B: batchではなくbox数をNと書く。
"""
from __future__ import annotations
from dataclasses import dataclass

GRID_SIZE = 1000


def quantize_coordinate(value: float, extent: int) -> int:
    """pixel indexを``[0,999]``へ丸める。

    ``extent``はxなら画像幅W、yなら画像高さH。pixel indexの有効範囲は
    ``[0, extent-1]``なので分母も``extent-1``になる。
    """
    if extent <= 1 or not 0 <= value <= extent - 1:
        raise ValueError(f"coordinate {value} is outside [0, {extent - 1}]")
    return min(999, max(0, round(999 * value / (extent - 1))))


def dequantize_coordinate(token: int, extent: int) -> float:
    """0--999座標tokenを元画像のpixel indexへ戻す。

    公式visualizerと同じ``token / 999 * (extent - 1)``を使う。返値は浮動小数で、
    raster描画時に必要なら呼び出し側でroundする。量子化のため完全な逆変換ではない。
    """
    if not 0 <= token < GRID_SIZE:
        raise ValueError("coordinate token must be in [0, 999]")
    if extent <= 1:
        raise ValueError("image extent must be greater than one pixel")
    return token / 999 * (extent - 1)


@dataclass(frozen=True)
class GroundingEntry:
    """1 semantic identifierに属する複数box/point。geometryはxyxy(4値)またはxy(2値)。"""
    identifier: str
    geometry: tuple[tuple[int, ...], ...] | None


def serialize_entry(entry: GroundingEntry) -> str:
    """論文の共有grammarへserializeする。Noneは不在categoryを表しpaddingではない。"""
    if entry.geometry is None:
        payload = "None"
    else:
        chunks = []
        for item in entry.geometry:
            if len(item) not in (2, 4) or any(not 0 <= v <= 999 for v in item):
                raise ValueError("geometry must contain xy points or xyxy boxes on the 0--999 grid")
            chunks.append("".join(f"<{v}>" for v in item))
        payload = ",".join(chunks)
    return (f"<|object_ref_start|>{entry.identifier}<|object_ref_end|>"
            f"<|box_start|>{payload}<|box_end|>")


def serialize_response(entries: list[GroundingEntry]) -> str:
    """異なるsemantic entryを``, ``で連結する。boxは入力順を保持する。"""
    return ", ".join(serialize_entry(entry) for entry in entries)


def quantize_box_xyxy(box: tuple[float, float, float, float], width: int, height: int) -> tuple[int, ...]:
    """pixel xyxyを4つのatomic coordinate token値へ変換し、degenerate boxを拒否する。"""
    x1, y1, x2, y2 = box
    if x1 >= x2 or y1 >= y2:
        raise ValueError("box must satisfy x1 < x2 and y1 < y2")
    return (quantize_coordinate(x1, width), quantize_coordinate(y1, height),
            quantize_coordinate(x2, width), quantize_coordinate(y2, height))
