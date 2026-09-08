## Sample-based CRPS scoring

Use `score_sample_crps` with a long-form Polars table containing one row per
predictive sample. The observation is repeated for every sample in its forecast
unit.

```python
import numpy as np
import polars as pl

from cfa.stf.forecasttools import score_sample_crps

forecasts = pl.DataFrame(
    {
        "model": ["example", "example"],
        "reference_date": ["2026-01-05", "2026-01-05"],
        "location": ["US", "US"],
        "target": ["cases", "cases"],
        "horizon": [1, 1],
        "sample_id": [0, 1],
        "predicted": [10.0, 14.0],
        "observed": [12.0, 12.0],
    }
)
forecast_unit = ["model", "reference_date", "location", "target", "horizon"]

natural_scores = score_sample_crps(
    forecasts,
    forecast_unit=forecast_unit,
    transform=None,
    scale_name="natural",
)
log1p_scores = score_sample_crps(
    forecasts,
    forecast_unit=forecast_unit,
    transform=np.log1p,
    scale_name="log1p",
)
```

Each result contains the forecast-unit columns followed by `scale` and `crps`,
with one row per forecast unit. Lower CRPS is better and zero is optimal. Scores
from different scales must not be combined or directly compared.

Set `transform=None` to score values as supplied. A transform must accept and
return a same-shaped NumPy array containing finite real values. `scale_name`
labels the transformed scale in the result.

The function does not aggregate across forecast units. Averaging, weighting,
and comparison with a baseline are downstream operations.
