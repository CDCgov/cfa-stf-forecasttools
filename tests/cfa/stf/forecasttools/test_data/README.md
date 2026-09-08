# Forecasttools test data

This directory contains small, static forecast fixtures used by
`test_hubverse.py`. Normal Python test runs use only the committed files and do
not require PyRenew or R.

## PyRenew production-shaped samples

- `pyrenew_he_samples_3_draws.parquet` contains 48 synthetic forecast rows
  generated through the PyRenew HEW bootstrap.
- `pyrenew_he_samples_3_draws.md` records the generating repository commit,
  model settings, retained draws, and SHA-256 hash.

The sample and quantile integration tests use this artifact to confirm that
both public converters accept the scaffold's native six-column output.

## R Hubverse parity fixtures

- `r_hubverse_parity_input.csv` is a compact, one-based routine-forecasting
  input with the R source-column names.
- `r_hubverse_sample_expected.csv` is produced by the routine sample converter.
- `r_hubverse_quantile_expected.csv` is produced by the routine quantile
  converter using its default Hubverse quantile levels.
- `generate_r_hubverse_parity.R` regenerates all three CSV files.
- `r_hubverse_parity.md` records the R repository commit, package versions,
  normalization boundary, regeneration command, and SHA-256 hashes.

The parity tests pass the R source names through the Python alias parameters.
They normalize the R value `epiweekly` to the Python input-contract value
`weekly`, then compare Python output with the committed R output. Expected
dates and data types are normalized only when the CSV files are loaded; task
keys, horizons, sample IDs, quantile levels, and forecast values are compared
without recalculation.

R and the routine-forecasting checkout are regeneration dependencies only.
They are not Python package dependencies and are not invoked by pytest.

When regenerating a fixture, update its provenance document and hashes in the
same change. Do not create expected output with the Python function being
tested.
