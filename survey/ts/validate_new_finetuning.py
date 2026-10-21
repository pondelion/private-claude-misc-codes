"""Run one real optimizer/validation epoch of each new DataFrame example."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent


def make_dataframe() -> pd.DataFrame:
    rows = []
    for series_id in ("a", "b"):
        for time_index in range(40):
            value = np.sin(2 * np.pi * time_index / 12)
            rows.append({
                "series_id": series_id,
                "timestamp": time_index,
                "x1": float(value),
                "x2": float(time_index / 40),
                "target": float(0.7 * value + 0.2 * time_index / 40),
                "phase": time_index % 24,
            })
    return pd.DataFrame(rows)


def main() -> None:
    torch.manual_seed(11)
    torch.set_num_threads(2)
    df = make_dataframe()
    for name in ("AdaWaveNet", "WFTNet", "WPMixer", "CycleNet", "TimeMixer",
                 "FITS"):
        model_dir = ROOT / name
        spec = importlib.util.spec_from_file_location(
            f"training_{name}", model_dir / "finetuning_example.py")
        assert spec is not None and spec.loader is not None
        sys.path.insert(0, str(model_dir))
        try:
            # The six adapters deliberately share the import name ts_block.
            # Ensure every adapter resolves its own implementation.
            sys.modules.pop("ts_block", None)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            kwargs = {"phase_col": "phase"} if name == "CycleNet" else {}
            result = module.fit(df, ["x1", "x2"], ["target"],
                                seq_len=16,
                                epochs=1,
                                batch_size=16,
                                device="cpu",
                                **kwargs)
            assert np.isfinite(result["best_valid_loss"]), name
            print(f"{name}: DataFrame training/validation OK")
            if name == "FITS":
                classified = df.copy()
                classified["class_label"] = (classified["target"]
                                             > 0).astype(np.int64)
                result = module.fit(classified, ["x1", "x2"], ["class_label"],
                                    seq_len=16,
                                    epochs=1,
                                    batch_size=16,
                                    device="cpu",
                                    task="classification",
                                    num_classes=2)
                assert np.isfinite(result["best_valid_loss"])
                print("shared classification objective OK")
        finally:
            sys.path.pop(0)


if __name__ == "__main__":
    main()
