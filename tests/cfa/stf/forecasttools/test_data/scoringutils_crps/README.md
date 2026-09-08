# scoringutils CRPS parity fixture

The committed expected values were originally generated with:

- R 4.3.1;
- scoringutils 2.1.1; and
- scoringRules 1.1.3.

From the repository root, regenerate the expected files with:

```sh
Rscript tests/cfa/stf/forecasttools/test_data/scoringutils_crps/generate_expected.R
```

The generator uses the installed R package versions. R and these packages are
reference-generation tools only; Python tests read the committed CSV files and
do not invoke R. Review any resulting fixture changes before committing them.
