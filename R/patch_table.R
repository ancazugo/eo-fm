# patch_table.R ─ One row per So2Sat patch, everything joined onto it.
#
#     Rscript R/patch_table.R                      # -> data/patch_master_table.parquet
#     Rscript R/patch_table.R --format csv         # ...or CSV (~6x larger)
#     Rscript R/patch_table.R --runs proj_A proj_B # pick the projection runs
#
# Merges, keyed on `uid` (= "<dataset>/<patch_id>", the only unique patch key --
# `patch_id` restarts at 000000 in each of training/validation/testing):
#
#   identity   uid, dataset, patch_id, split, city
#   location   lon, lat  (patch centroid, EPSG:4326)
#   label      lcz_class (1-17), lcz_code (1-10/A-G), lcz_name
#   climate    koppen_code, koppen_desc, water
#   territory  un_country, iso_a2, iso_a3, m49_code, un_subregion, un_region
#   embedding  <run>_pca_1..3 and <run>_umap_1..3, one block per projection run
#
# ONE THING THAT WILL BURN YOU: the embedding blocks DO NOT SHARE A FRAME.
# `tesserav2_umap_1` and `alphaearthcoop_umap_1` are separate fits of separate
# feature spaces; no rotation relates them. Comparing a patch's position between
# two blocks, or computing a distance across them, is meaningless. What IS valid
# is per-patch comparison of *structure* -- neighbourhoods, cluster membership,
# how a class or a region scatters -- which is exactly why the rows are joined.
# PCA blocks are per-run fits too, on top of a per-run StandardScaler.
#
# t-SNE is deliberately not included: it is fitted on a balanced per-class
# subsample, so only 34,000 of 397,415 rows (8.6%) would carry a value. Add
# "tsne" to PROJ_DIMS if you want it, and expect the NAs.

source("R/constants.R")
source("R/embedding_projection.R")
source("R/patch_geo_context.R")

suppressPackageStartupMessages({
  library(arrow)
  library(dplyr)
})

PATCH_TABLE_STEM <- file.path("data", "patch_master_table")

# Which coordinate blocks to carry per run, and how many components of each.
PROJ_DIMS <- c(pca = 3L, umap = 3L)

# Runs merged when --runs is not given: the two current-format global runs.
# The pre-2026-09 exports have no `uid` and no third component, so they cannot
# be joined at all -- listing them here would fail, not degrade.
DEFAULT_RUNS <- c("proj_GeoTessera_v2_global_gap", "proj_AlphaEarthCoop_global_gap")

#' Short, syntactic prefix for a run's coordinate columns.
#'
#' "proj_GeoTessera_v2_global_gap" -> "geotessera_v2". The run directory name
#' carries the pooling and split, which are constant across the merge and would
#' only make every column name longer.
run_prefix <- function(run) {
  x <- sub("^proj_", "", run)
  x <- sub("_global_gap$", "", x)
  x <- sub("_global$", "", x)
  tolower(gsub("[^A-Za-z0-9]+", "_", x))
}

#' Read one run's identity, location and coordinate columns.
read_run_block <- function(run, first = FALSE) {
  runs <- list_projections()
  r <- resolve_run(runs, run)
  dims <- unlist(lapply(names(PROJ_DIMS),
                        function(m) paste0(m, "_", seq_len(PROJ_DIMS[[m]]))))
  base <- c("uid", "dataset", "patch_id", "split", "city", "lon", "lat",
            "LCZ_class")
  have <- names(arrow::open_dataset(r$full))
  miss <- setdiff(c("uid", dims), have)
  if (length(miss)) {
    stop("run '", r$run, "' has no ", paste(miss, collapse = ", "),
         ". It predates the current exporter; re-export it with ",
         "src/embedding_projection.py before merging.", call. = FALSE)
  }
  d <- read_projection(r$full, columns = intersect(c(base, dims), have))
  message("  ", r$run, ": ", format(nrow(d), big.mark = ","), " rows")

  pre <- run_prefix(r$run)
  names(d)[match(dims, names(d))] <- paste0(pre, "_", dims)
  # Identity and location are properties of the patch, not of the projection,
  # so they come from the first run only and every later run contributes
  # coordinates alone. Carrying them twice would invite a silent disagreement.
  if (first) d else d[, c("uid", paste0(pre, "_", dims))]
}

#' Build the merged table.
build_patch_table <- function(runs = DEFAULT_RUNS) {
  runs <- vapply(runs, function(r) resolve_run(list_projections(), r)$run, "")
  message("Merging ", length(runs), " projection run(s)")
  out <- read_run_block(runs[[1]], first = TRUE)
  for (r in runs[-1]) {
    add <- read_run_block(r)
    # inner join, and checked: a patch present in one run but not another would
    # otherwise become a row of silent NAs in half the columns.
    before <- nrow(out)
    out <- inner_join(out, add, by = "uid")
    if (nrow(out) != before) {
      warning(r, " shares only ", nrow(out), " of ", before, " uids; ",
              before - nrow(out), " patch(es) dropped", call. = FALSE)
    }
  }

  out <- geo_context(out)

  lcz <- LCZ_TABLE[, c("code", "alt_code", "name")]
  names(lcz) <- c("LCZ_class", "lcz_code", "lcz_name")
  out <- left_join(out, lcz, by = "LCZ_class")

  kg <- read_koppen_legend()[, c("code", "desc")]
  names(kg) <- c("koppen_code", "koppen_desc")
  out$koppen_code <- as.character(out$koppen)
  out <- left_join(out, kg, by = "koppen_code")

  out <- out |>
    rename(lcz_class = LCZ_class, un_country = country,
           un_subregion = subregion, un_region = region) |>
    select(uid, dataset, patch_id, split, city, lon, lat,
           lcz_class, lcz_code, lcz_name,
           koppen_code, koppen_desc, water,
           un_country, iso_a2, iso_a3, m49_code, un_subregion, un_region,
           everything())
  # `read_projection()` adds an `lcz` factor as a convenience for plotting; here
  # it only duplicates lcz_code and lands between the two embedding blocks.
  out <- out[, setdiff(names(out), c("lcz", "koppen"))]

  # Blocks contiguous and in run order, so the file reads as identity ->
  # context -> one block per embedding rather than an interleaving.
  dims <- unlist(lapply(names(PROJ_DIMS),
                        function(m) paste0(m, "_", seq_len(PROJ_DIMS[[m]]))))
  blocks <- unlist(lapply(vapply(runs, run_prefix, ""),
                          function(pre) paste0(pre, "_", dims)))
  out[, c(setdiff(names(out), blocks), intersect(blocks, names(out)))]
}

main <- function(args = commandArgs(trailingOnly = TRUE)) {
  fmt  <- .flag(args, "--format", "parquet")
  i    <- match("--runs", args)
  runs <- if (is.na(i)) DEFAULT_RUNS else {
    v <- args[(i + 1):length(args)]
    v[seq_len(match(TRUE, startsWith(v, "--"), nomatch = length(v) + 1L) - 1L)]
  }

  d <- build_patch_table(runs)
  path <- paste0(PATCH_TABLE_STEM, ".", fmt)
  dir.create(dirname(path), showWarnings = FALSE, recursive = TRUE)
  if (fmt == "parquet") arrow::write_parquet(d, path) else
    readr::write_csv(d, path)

  message("Wrote ", path, ": ", format(nrow(d), big.mark = ","), " rows x ",
          ncol(d), " cols (", round(file.size(path) / 1024^2, 1), " MB)")
  message("  columns: ", paste(names(d), collapse = ", "))
  invisible(d)
}

if (sys.nframe() == 0) main()
