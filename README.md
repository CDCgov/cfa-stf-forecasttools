# cfa-stf-forecasttools

Forecasting helpers for the CFA Short-Term Forecasting Team.

## Sample-based CRPS scoring

Use `score_sample_crps` with a long-form Polars table containing one row per
predictive sample. The observation is repeated for every sample in its forecast
unit.

```python
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
    scale="natural",
)
log1p_scores = score_sample_crps(
    forecasts,
    forecast_unit=forecast_unit,
    scale="log1p",
)
```

Each result contains the forecast-unit columns followed by `scale` and `crps`,
with one row per forecast unit. Lower CRPS is better and zero is optimal. Scores
from different scales must not be combined or directly compared.

The function does not aggregate across forecast units. Averaging, weighting,
and comparison with a baseline are downstream operations.

## License Standard Notice
The repository utilizes code licensed under the terms of the Apache Software
License and therefore is licensed under ASL v2 or later.

This source code in this repository is free: you can redistribute it and/or modify it under
the terms of the Apache Software License version 2, or (at your option) any
later version.

This source code in this repository is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the Apache Software License for more details.

You should have received a copy of the Apache Software License along with this
program. If not, see http://www.apache.org/licenses/LICENSE-2.0.html

The source code forked from other open source projects will inherit its license.

## Privacy Standard Notice
This repository contains only non-sensitive, publicly available data and
information. All material and community participation is covered by the
[Disclaimer](DISCLAIMER.md)
and [Code of Conduct](code-of-conduct.md).
For more information about CDC's privacy policy, please visit [http://www.cdc.gov/other/privacy.html](https://www.cdc.gov/other/privacy.html).

## Contributing Standard Notice
Anyone is encouraged to contribute to the repository by [forking](https://help.github.com/articles/fork-a-repo)
and submitting a pull request. (If you are new to GitHub, you might start with a
[basic tutorial](https://help.github.com/articles/set-up-git).) By contributing
to this project, you grant a world-wide, royalty-free, perpetual, irrevocable,
non-exclusive, transferable license to all users under the terms of the
[Apache Software License v2](http://www.apache.org/licenses/LICENSE-2.0.html) or
later.

All comments, messages, pull requests, and other submissions received through
CDC including this GitHub page may be subject to applicable federal law, including but not limited to the Federal Records Act, and may be archived. Learn more at [http://www.cdc.gov/other/privacy.html](http://www.cdc.gov/other/privacy.html).

## Records Management Standard Notice
This repository is not a source of government records, but is a copy to increase
collaboration and collaborative potential. All government records will be
published through the [CDC web site](http://www.cdc.gov).

## Additional Standard Notices
Please refer to [CDC's Template Repository](https://github.com/CDCgov/template) for more information about [contributing to this repository](https://github.com/CDCgov/template/blob/main/CONTRIBUTING.md), [public domain notices and disclaimers](https://github.com/CDCgov/template/blob/main/DISCLAIMER.md), and [code of conduct](https://github.com/CDCgov/template/blob/main/code-of-conduct.md).
