"""One real DataFrame -> optimization step for all five fine-tuning examples."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent


def main():
    n = 48
    time = np.arange(n)
    df = pd.DataFrame({
        'series_id': np.zeros(n, dtype=np.int64),
        'timestamp': time,
        'f': np.sin(time / 5).astype(np.float32),
        'target': np.cos(time / 5).astype(np.float32),
    })
    for name in ('PsLSTM', 'Mamba', 'DSTMamba', 'WaveTS', 'ModernTCN'):
        path = ROOT / name / 'finetuning_example.py'
        spec = importlib.util.spec_from_file_location('fit_' + name, path)
        module = importlib.util.module_from_spec(spec)
        sys.path.insert(0, str(path.parent))
        try:
            spec.loader.exec_module(module)
            kwargs = dict(lookback=8,
                          horizon=2,
                          epochs=1,
                          batch_size=8,
                          device='cpu')
            if name == 'Mamba':
                kwargs.update(version=3, backend='torch', is_mimo=True)
            _, info = module.fit(df, ['f'], ['target'], **kwargs)
            assert np.isfinite(info['val_mse'])
            print(name, 'DataFrame fit PASS')
        finally:
            sys.path.pop(0)
            # All examples import a same-named local ts_block module.
            sys.modules.pop('ts_block', None)


if __name__ == '__main__':
    main()
