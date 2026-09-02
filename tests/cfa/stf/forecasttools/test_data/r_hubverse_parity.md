# R Hubverse parity fixtures

These static CSV fixtures exercise the initial eight-column Hubverse profile:

- `r_hubverse_parity_input.csv` contains one-based R draw IDs and the source
  names used by routine forecasting.
- `r_hubverse_sample_expected.csv` contains the corresponding sample output.
- `r_hubverse_quantile_expected.csv` contains the routine converter's 23
  default Hubverse quantiles.

The generator sources and executes `var_to_target()`,
`prelim_to_hub_samples()`, and `prelim_to_hub_quantiles()` from
`cfa-stf-routine-forecasting` at commit
`c52d677a8cb9de256e38ce6efe379521fd9f960e`. It also calls the installed
`forecasttools` and `hubUtils` packages used by those functions. R is needed
only to regenerate the files; Python tests read the committed CSV outputs.

Routine forecasting calls its weekly resolution `epiweekly`; the Python v1
input contract calls it `weekly`. The Python parity tests make that explicit
value normalization before conversion. Expected Hubverse rows are not altered.

Regenerate with:

```sh
Rscript \
  tests/cfa/stf/forecasttools/test_data/generate_r_hubverse_parity.R \
  tests/cfa/stf/forecasttools/test_data \
  ../cfa-stf-routine-forecasting/stfroutineforecasting/R/to_hubverse_tbl.R
```

Generation environment:

- R 4.6.0
- forecasttools 0.1.7
- hubUtils 1.2.0
- dplyr 1.2.1

SHA-256 hashes are recorded after generation:

- input: `ee68fa17d22394d2501a6952aea5538ad8262ef516d4ab66a4bf1da1123d7d38`
- sample expected:
  `7b6cc299aa6e0a0937fbfd19dff4c43096ca779dca05d9752d894d0722b4a5a2`
- quantile expected:
  `8faae04390556d5da3265827180d61096a6ceca7bd9511addc9bf200d4cc6e30`
