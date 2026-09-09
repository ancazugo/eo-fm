# prepare_city_data.R ─ Build data/so2sat_city_summary.csv.
#
#     Rscript R/prepare_city_data.R
#
# Reads the 51 per-city GeoPackages under $DATA_DIR/.../So2Sat-LCZ42/v4/cities,
# takes each city's mean patch centroid and its per-split patch counts, and
# caches the result as a 52-row CSV so the plotting script never has to touch
# the data tree. Run once; re-run only if the So2Sat reference data changes.
#
# Why 52 rows from 51 directories: GUPPD merges Guangzhou and Shenzhen into one
# SMOD entity, so cities/Guangzhou holds both. Its `training` patches are
# Shenzhen and its validation/testing patches are Guangzhou -- see CITY_META in
# R/constants.R. This script is the only place that split is performed.

source("R/constants.R")

suppressPackageStartupMessages({
  library(sf)
  library(tidyr)
  library(readr)
  library(purrr)
})

# The merged GUPPD entity, and how its two patch clusters map to real cities.
MERGED_CITY_DIR   <- "Guangzhou"
MERGED_TRAIN_CITY <- "Shenzhen"   # the `training` patches
MERGED_TEST_CITY  <- "Guangzhou"  # the `validation` + `testing` patches

# Expected totals, asserted at the end so a silent data change is caught rather
# than quietly plotted.
EXPECT_CITIES  <- 52L
EXPECT_TRAIN   <- 42L
EXPECT_VALTEST <- 10L
EXPECT_PATCHES <- 395732L

#' Per-patch centroid + split for one city directory.
read_city_patches <- function(city_dir) {
  files <- list.files(city_dir, pattern = "^patches_reference_.*\\.gpkg$",
                      full.names = TRUE)
  files <- files[!grepl("_split\\.gpkg$", files)]
  if (length(files) == 0) return(NULL)

  gpkg <- files[[1]]
  # Read the layer name rather than constructing it: Dongying's layer is
  # `patches_reference_30_2149`, keyed by SMOD_ID rather than by city name.
  layer <- sf::st_layers(gpkg)$name[[1]]

  patches <- sf::st_read(gpkg, layer = layer, quiet = TRUE)
  xy <- suppressWarnings(sf::st_coordinates(sf::st_centroid(patches)))

  tibble(city_dir = basename(city_dir), dataset = patches$dataset,
         LCZ_class = patches$LCZ_class, lon = xy[, 1], lat = xy[, 2])
}

#' Collapse patches to one row per (city, role) with a mean centroid + counts.
summarise_city <- function(patches) {
  patches |>
    mutate(role = if_else(dataset == "training", "train", "val_test")) |>
    group_by(city_dir, role) |>
    summarise(lon = mean(lon), lat = mean(lat),
              n_train = sum(dataset == "training"),
              n_val   = sum(dataset == "validation"),
              n_test  = sum(dataset == "testing"),
              .groups = "drop")
}

message("Reading per-city patches from ", SO2SAT_CITIES_DIR, " ...")
city_dirs <- list.dirs(SO2SAT_CITIES_DIR, recursive = FALSE)
message("  ", length(city_dirs), " city directories")

all_patches <- map(city_dirs, function(d) {
  out <- read_city_patches(d)
  if (!is.null(out)) message("  ", basename(d), ": ", nrow(out), " patches")
  out
}) |> list_rbind()

# One row per city, except the merged GUPPD entity which yields two: every
# other city has patches from a single role, so grouping by role is a no-op
# for them and splits Guangzhou/Shenzhen for free.
summary_tbl <- summarise_city(all_patches) |>
  mutate(city_dir = case_when(
    city_dir == MERGED_CITY_DIR & role == "train"    ~ MERGED_TRAIN_CITY,
    city_dir == MERGED_CITY_DIR & role == "val_test" ~ MERGED_TEST_CITY,
    TRUE ~ city_dir
  )) |>
  mutate(n_total = n_train + n_val + n_test) |>
  left_join(CITY_META, by = "city_dir") |>
  select(city_dir, city_label, country, continent, lon, lat, role,
         n_train, n_val, n_test, n_total) |>
  arrange(city_label)

# ── Assertions ────────────────────────────────────────────────────────────────

missing_meta <- summary_tbl$city_dir[is.na(summary_tbl$city_label)]
if (length(missing_meta) > 0) {
  stop("No CITY_META entry for: ", paste(missing_meta, collapse = ", "),
       ". Add them to R/constants.R.", call. = FALSE)
}

stopifnot(
  "expected 52 cities"           = nrow(summary_tbl) == EXPECT_CITIES,
  "expected 42 training cities"  = sum(summary_tbl$role == "train") == EXPECT_TRAIN,
  "expected 10 val/test cities"  = sum(summary_tbl$role == "val_test") == EXPECT_VALTEST,
  "expected 395,732 patches"     = sum(summary_tbl$n_total) == EXPECT_PATCHES,
  "coordinates must be finite"   = all(is.finite(summary_tbl$lon) & is.finite(summary_tbl$lat))
)

# The 10 val/test cities must be exactly the So2Sat culture cities.
val_test_cities <- sort(summary_tbl$city_label[summary_tbl$role == "val_test"])
stopifnot("val/test cities must match SO2SAT_CULTURE_CITIES" =
            identical(val_test_cities, sort(SO2SAT_CULTURE_CITIES)))

dir.create(dirname(CITY_SUMMARY_CSV), showWarnings = FALSE, recursive = TRUE)
readr::write_csv(summary_tbl, CITY_SUMMARY_CSV)

# ── Per-city class composition (for the pie map) ──────────────────────────────

# Same Guangzhou/Shenzhen split, applied to the per-class counts.
class_tbl <- all_patches |>
  mutate(role = if_else(dataset == "training", "train", "val_test"),
         city_dir = case_when(
           city_dir == MERGED_CITY_DIR & role == "train"    ~ MERGED_TRAIN_CITY,
           city_dir == MERGED_CITY_DIR & role == "val_test" ~ MERGED_TEST_CITY,
           TRUE ~ city_dir)) |>
  count(city_dir, LCZ_class, name = "n") |>
  arrange(city_dir, LCZ_class)

stopifnot("class counts must match the city totals" =
            sum(class_tbl$n) == EXPECT_PATCHES)

readr::write_csv(class_tbl, CITY_CLASS_CSV)
message("Wrote ", CITY_CLASS_CSV, " (", nrow(class_tbl), " city-class rows)")

message("\nWrote ", CITY_SUMMARY_CSV, " (", nrow(summary_tbl), " cities, ",
        format(sum(summary_tbl$n_total), big.mark = ","), " patches)")
message("  training: ", sum(summary_tbl$role == "train"), " cities, ",
        format(sum(summary_tbl$n_train), big.mark = ","), " patches")
message("  val/test: ", sum(summary_tbl$role == "val_test"), " cities, ",
        format(sum(summary_tbl$n_val + summary_tbl$n_test), big.mark = ","), " patches")
print(summary_tbl |> filter(role == "val_test") |> as.data.frame())
