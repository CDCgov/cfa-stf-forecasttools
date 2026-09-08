fixture_dir <- file.path(
  "tests", "cfa", "stf", "forecasttools", "test_data", "scoringutils_crps"
)
input <- read.csv(file.path(fixture_dir, "input.csv"), check.names = FALSE)
forecast_unit <- c(
  "model",
  "reference_date",
  "target_end_date",
  "location",
  "target",
  "horizon"
)

scores <- input |>
  scoringutils::as_forecast_sample() |>
  scoringutils::transform_forecasts(fun = log1p, label = "log1p") |>
  scoringutils::score(metrics = list(crps = scoringutils::crps_sample)) |>
  as.data.frame()
scores <- scores[do.call(order, scores[forecast_unit]), ]

write.csv(
  scores[scores$scale == "natural", ],
  file.path(fixture_dir, "expected_natural.csv"),
  row.names = FALSE
)
write.csv(
  scores[scores$scale == "log1p", ],
  file.path(fixture_dir, "expected_log1p.csv"),
  row.names = FALSE
)
