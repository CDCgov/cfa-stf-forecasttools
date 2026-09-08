expected_r_version <- "4.3.1"
expected_scoringutils_version <- "2.1.1"
expected_scoringrules_version <- "1.1.3"

stopifnot(as.character(getRversion()) == expected_r_version)
stopifnot(
  as.character(packageVersion("scoringutils")) == expected_scoringutils_version
)
stopifnot(
  as.character(packageVersion("scoringRules")) == expected_scoringrules_version
)

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

score_scale <- function(input, scale) {
  "Calculate CRPS for every forecast unit on one scale."
  scored_input <- input
  if (scale == "log1p") {
    scored_input$predicted <- log1p(scored_input$predicted)
    scored_input$observed <- log1p(scored_input$observed)
  }

  units <- unique(scored_input[forecast_unit])
  units$scale <- scale
  units$crps <- vapply(seq_len(nrow(units)), function(index) {
    matches <- rep(TRUE, nrow(scored_input))
    for (column in forecast_unit) {
      matches <- matches & scored_input[[column]] == units[[column]][index]
    }
    observations <- unique(scored_input$observed[matches])
    stopifnot(length(observations) == 1L)
    scoringutils::crps_sample(
      observations,
      scored_input$predicted[matches]
    )
  }, numeric(1))

  units[do.call(order, units[forecast_unit]), ]
}

write.csv(
  score_scale(input, "natural"),
  file.path(fixture_dir, "expected_natural.csv"),
  row.names = FALSE
)
write.csv(
  score_scale(input, "log1p"),
  file.path(fixture_dir, "expected_log1p.csv"),
  row.names = FALSE
)
