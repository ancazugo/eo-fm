# lcz_composition.R ─ LCZ class composition of a raster, as a pie and a bar.
#
#     Rscript R/lcz_composition.R --input <file.tif> --name <stem> [--bbox W,S,E,N]
#
# The same two marks R/plotting.R builds for the So2Sat patches, but counted
# from a GeoTIFF's cells instead: the pie from the city map, and the stacked
# composition bar that serves as its legend -- drawn horizontally here. Both
# come from R/composition.R, so a change to either shows up in both figures.
#
# Both are drawn **without text** by default: they are graphical elements meant
# to sit beside or on an LCZ map, which carries the class legend already.
# `--labels` puts the class and share back on the bar for a standalone figure.
# R/lcz_raster.R can place either of them into the map itself.
#
# Counting is by cell, so a share is a share of *mapped area*, not of patches:
# a prediction raster and the patch table will not agree, and should not. The
# raster's own pixels are counted as they are stored -- no reprojection to an
# equal-area grid -- so on a lon/lat raster the higher-latitude cells are
# smaller on the ground. Over a city ROI that bias is under a tenth of a
# percent; over a continent it would not be, and this is a city-ROI tool.

source("R/constants.R")
source("R/composition.R")

suppressPackageStartupMessages({
  library(terra)
  library(sf)
  library(dplyr)
  library(patchwork)
})

# ── Counting ──────────────────────────────────────────────────────────────────

#' Per-class cell counts and shares for an LCZ raster.
#'
#' terra::freq() tabulates without materialising a data frame of every cell, so
#' this is cheap on the 10 m city rasters (Milan is 96M cells) that
#' R/lcz_raster.R has to decimate before it can plot them. Nodata is 0 on disk
#' and NA after the read, and freq() skips it, so the shares are over classified
#' cells only.
#'
#' @param bbox optional c(west, south, east, north) in degrees. The ROI is
#'   projected into the raster's CRS rather than the other way round, exactly as
#'   read_lcz_roi() does, so no resampling touches the counts.
lcz_class_counts <- function(path, bbox = NULL) {
  if (!file.exists(path)) stop("No such raster: ", path, call. = FALSE)
  r <- terra::rast(path)
  if (!is.null(bbox)) {
    if (length(bbox) != 4 || anyNA(bbox)) {
      stop("bbox must be four numbers: west, south, east, north.", call. = FALSE)
    }
    roi <- sf::st_bbox(c(xmin = bbox[1], ymin = bbox[2],
                         xmax = bbox[3], ymax = bbox[4]), crs = 4326) |>
      sf::st_as_sfc() |> sf::st_transform(terra::crs(r)) |> terra::vect()
    if (!terra::relate(terra::ext(r), terra::ext(roi), "intersects")) {
      stop("The bbox does not overlap ", basename(path), ".", call. = FALSE)
    }
    r <- terra::crop(r, roi)
  }

  lcz_counts(r)
}

#' The same, for a SpatRaster already in hand -- what R/lcz_raster.R calls after
#' it has cropped (and possibly aggregated) the raster it is about to draw, so
#' the inset pie describes exactly the pixels on the page.
lcz_counts <- function(r) {
  f <- terra::freq(r)
  f <- f[!is.na(f$value) & f$value != 0, ]
  if (!nrow(f)) stop("Every cell is nodata.", call. = FALSE)

  code <- as.integer(f$value)
  bad <- setdiff(code, LCZ_TABLE$code)
  if (length(bad)) {
    stop("Raster holds values outside LCZ 1-17: ", paste(bad, collapse = ", "),
         ". Model outputs are 0-indexed; on-disk rasters should already be ",
         "1-17 (see infer_roi.py).", call. = FALSE)
  }
  tibble::tibble(code = code, n = as.numeric(f$count)) |>
    arrange(match(code, LCZ_TABLE$code)) |>
    mutate(key = factor(LCZ_TABLE$alt_code[match(code, LCZ_TABLE$code)],
                        levels = LCZ_TABLE$alt_code),
           share = n / sum(n)) |>
    select(key, share, n)
}

# ── Figures ───────────────────────────────────────────────────────────────────

#' A share, to one decimal, but two more for anything under 0.1%.
#'
#' A prediction raster puts a handful of cells in most classes, and at
#' `accuracy = 0.1` nine of the seventeen LCZ labels here read "0.0%" -- which
#' says "absent" when the truth is "present and negligible".
fmt_share_fine <- function(share, n) {
  ifelse(share < 0.001, scales::percent(share, accuracy = 0.01),
         scales::percent(share, accuracy = 0.1))
}

#' Pie + horizontal bar for one raster's class mix.
#'
#' Returns the pie alone, the bar alone, the two stacked as one figure, and the
#' heights to save them at. The bar is the pie's legend, so the combined panel
#' is the one that stands on its own.
#'
#' Both the label spacing and the bar's height are computed from the type at
#' `width`, not guessed: the rotated labels have to clear one line height along
#' the bar, and the panel has to be as deep as the longest of them.
lcz_composition_plots <- function(df, width = 7, title = NULL, labels = FALSE) {
  pie <- pie_chart(df, LCZ_COLOURS)
  labs <- sprintf("%s (%s)", as.character(df$key), fmt_share_fine(df$share, df$n))
  # Unlabelled, the strip's height follows its own coordinate aspect so the
  # rectangle fills the figure exactly; labelled, it follows the type.
  bar_h <- if (labels) horizontal_height(labs) else width * BAR_T
  bar <- composition_bar(df, LCZ_COLOURS, side = "bottom", horizontal = TRUE,
                         fmt = fmt_share_fine, label_room = 1, labels = labels,
                         gap = horizontal_gap(width), title = title)
  list(pie = pie, bar = bar, bar_height = bar_h,
       panel = (pie / bar) + plot_layout(heights = c(width, bar_h)) +
         transparent_patchwork())
}

# ── CLI ───────────────────────────────────────────────────────────────────────

if (sys.nframe() == 0L && !interactive()) {
  suppressPackageStartupMessages(library(argparse))
  parser <- ArgumentParser(
    description = "LCZ class composition of a raster, as a pie and a bar.")
  parser$add_argument("--input", required = TRUE,
                      help = "LCZ GeoTIFF (classes 1-17, nodata 0)")
  parser$add_argument("--name", required = TRUE, help = "output stem under plots/")
  parser$add_argument("--bbox", default = NULL,
                      help = "restrict to west,south,east,north in degrees")
  parser$add_argument("--title", default = NULL,
                      help = "caption under the bar (default: none)")
  parser$add_argument("--labels", action = "store_true",
                      help = paste("label each bar segment with its class and",
                                   "share. Off by default: these are graphical",
                                   "elements for a map, and the map carries the",
                                   "class legend"))
  parser$add_argument("--width", type = "double", default = 7)
  parser$add_argument("--dpi", type = "integer", default = 400)
  # As in R/lcz_raster.R: argparse reads a value starting with "-" as another
  # flag, so a western bbox has to be glued onto its own flag first.
  argv <- commandArgs(trailingOnly = TRUE)
  glue <- which(argv %in% c("--bbox", "--width", "--dpi"))
  glue <- glue[glue < length(argv) & grepl("^-", argv[pmin(glue + 1L, length(argv))])]
  if (length(glue)) {
    argv[glue] <- paste0(argv[glue], "=", argv[glue + 1L])
    argv <- argv[-(glue + 1L)]
  }
  args <- parser$parse_args(argv)

  bbox <- if (is.null(args$bbox)) NULL else
    as.numeric(strsplit(args$bbox, "[, ]+")[[1]])

  df <- lcz_class_counts(args$input, bbox)
  message("  ", nrow(df), " classes over ",
          format(sum(df$n), big.mark = ","), " classified cells")
  w <- args$width
  p <- lcz_composition_plots(df, width = w, title = args$title,
                             labels = args$labels)

  save_plot(p$pie, paste0(args$name, "_pie"), width = w, height = w,
            dpi = args$dpi, subdir = PLOT_DIR_MAPS)
  save_plot(p$bar, paste0(args$name, "_bar"), width = w, height = p$bar_height,
            dpi = args$dpi, subdir = PLOT_DIR_MAPS)
  save_plot(p$panel, paste0(args$name, "_composition"), width = w,
            height = w + p$bar_height, dpi = args$dpi, subdir = PLOT_DIR_MAPS)
}
