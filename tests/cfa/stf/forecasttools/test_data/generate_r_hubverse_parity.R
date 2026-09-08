args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2L) {
  stop(paste(
    "usage: Rscript generate_r_hubverse_parity.R",
    "OUTPUT_DIRECTORY ROUTINE_CONVERTER"
  ))
}

output_dir <- normalizePath(args[[1]], mustWork = TRUE)
converter_path <- normalizePath(args[[2]], mustWork = TRUE)
source(converter_path)

reference_date <- as.Date("2025-01-28")

samples <- data.frame(
  ".draw" = rep(1:3, each = 4),
  date = rep(
    as.Date(c("2025-01-29", "2025-01-30", "2025-02-01", "2025-02-08")),
    times = 3
  ),
  geo_value = "CA",
  ".variable" = rep(
    c(
      "observed_ed_visits",
      "observed_ed_visits",
      "observed_hospital_admissions",
      "observed_hospital_admissions"
    ),
    times = 3
  ),
  ".value" = c(
    10, 13, 100, 130,
    20, 23, 200, 230,
    50, 53, 500, 530
  ),
  resolution = rep(c("daily", "daily", "epiweekly", "epiweekly"), times = 3),
  check.names = FALSE
)

utils::write.csv(
  samples,
  file.path(output_dir, "r_hubverse_parity_input.csv"),
  row.names = FALSE
)

prelim_samples <- samples |>
  tibble::as_tibble() |>
  dplyr::mutate(
    disease = "covid",
    target_prefix = dplyr::if_else(.data$resolution == "epiweekly", "wk ", ""),
    target_core = var_to_target(.data$.variable, .data$disease),
    target = stringr::str_c(.data$target_prefix, .data$target_core),
    reference_date = reference_date,
    horizon_timescale = "days",
    horizon = forecasttools::horizons_from_target_end_dates(
      reference_date = .data$reference_date,
      horizon_timescale = .data$horizon_timescale,
      target_end_dates = .data$date
    ),
    model_id = "r-parity-model"
  )

output_columns <- c(
  "reference_date",
  "target",
  "horizon",
  "target_end_date",
  "location",
  "output_type",
  "output_type_id",
  "value"
)
sample_expected <- prelim_to_hub_samples(prelim_samples) |>
  dplyr::select(dplyr::all_of(output_columns)) |>
  as.data.frame()

sort_columns <- c(
  "reference_date",
  "target",
  "horizon",
  "target_end_date",
  "location",
  "output_type",
  "output_type_id"
)
sample_expected <- sample_expected[
  do.call(order, sample_expected[sort_columns]),
]
utils::write.csv(
  sample_expected,
  file.path(output_dir, "r_hubverse_sample_expected.csv"),
  row.names = FALSE
)

quantile_expected <- prelim_to_hub_quantiles(prelim_samples) |>
  dplyr::select(dplyr::all_of(output_columns)) |>
  as.data.frame()
quantile_expected <- quantile_expected[
  do.call(order, quantile_expected[sort_columns]),
]
utils::write.csv(
  quantile_expected,
  file.path(output_dir, "r_hubverse_quantile_expected.csv"),
  row.names = FALSE
)
