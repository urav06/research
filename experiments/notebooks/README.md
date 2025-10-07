# Local Jupyter Notebooks

This directory contains Jupyter notebooks for local exploratory work and debugging.

Notebooks here directly interface with the `tsckit` package for quick experimentation.

## Usage

```bash
jupyter notebook experiments/notebooks/
```

## Example Template

```python
import sys
sys.path.extend([
    '/Users/urav/code/research',
    '/Users/urav/code/research/quant/code',
    '/Users/urav/code/research/hydra/code',
    '/Users/urav/code/research/aaltd2024/code',
])

from tsckit import Experiment, MonsterDataset
from tsckit.algorithms import QuantAALTD2024, HydraAALTD2024

# Quick test with small dataset
dataset = MonsterDataset("Pedestrian", train_pct=1, test_pct=1)

exp = Experiment(
    name="quick_test",
    datasets=[dataset],
    algorithms=[
        QuantAALTD2024(num_estimators=50),
        HydraAALTD2024(k=4, g=16)
    ]
)

exp.run(verbose=True)
print(exp.summary())
```

## Notes

- Use small datasets or sampling for speed (`train_pct=1`, `test_pct=1`)
- No `output_dir` needed for notebooks (results not saved)
- For production experiments, use `experiments/ensemble_benchmark/` on M3
