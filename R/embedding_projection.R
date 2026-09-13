# embedding_projection.R ─ Read and plot the embedding-projection exports.
#
# As a CLI:
#
#     Rscript R/embedding_projection.R --list
#     Rscript R/embedding_projection.R --run GeoTessera --all-colours
#     Rscript R/embedding_projection.R --run AlphaEarth --method pca \
#         --colour city --depth asc --max-points 60000 --legend
#
#   -> plots/projection_<run>_<method>_<colour>.png
#
# As a library:
#
#     source("R/embedding_projection.R")
#     runs <- list_projections()
#     df   <- read_projection_sample(runs[1, ])
#     p    <- plot_projection(df, method = "pca", colour = "lcz_name")
#
# THE THIRD COORDINATE IS THE DEPTH CHANNEL. Points are sorted by `<method>_3`
# before drawing, so one range of depth values lands under the other -- which is
# the only way a 40k-point cloud shows anything but its topmost layer. Runs
# written before 2026-09 carry only `<method>_x/_y`; those still plot, in file
# order, with a warning. Re-export with src/embedding_projection.py to get
# `pca_1..3` (--pca-keep, default 3; UMAP/t-SNE additionally need --fit-3d,
# which is a SECOND 3-component fit, not a slice of the 2-D one).
#
# The Python side (src/embedding_projection.py) writes three parquets per run:
#
#   projection_<key>.parquet          every patch, every coordinate + metadata
#   projection_<key>_sample.parquet   a subsample chosen by a STABLE HASH of uid
#   projection_<key>_density.parquet  a pre-binned 2-D grid, long format
#
# Read the *sample* for scatter plots and the *density* grid for anything
# showing the whole cloud. This environment has arrow, nanoparquet, terra and
# data.table but NOT scattermore, ggrastr or hexbin, so 400k geom_point rows
# will not render in reasonable time -- that is exactly why the density grid
# exists. Reach for the full parquet only when you need per-patch joins.
#
# TWO THINGS THAT WILL BURN YOU IF YOU IGNORE THEM
#
#   1. `patch_id` is NOT unique. It restarts at 000000 in each of training /
#      validation / testing. Always key on `uid` (= "<dataset>/<patch_id>").
#   2. UMAP and t-SNE coordinates from SEPARATE RUNS share no frame. There is no
#      rotation aligning one run's UMAP to another's, so row-binding two runs is
#      valid for faceting ("show me each embedding's own cloud") and meaningless
#      for anything that compares coordinates or computes a distance across
#      runs. PCA is likewise per-run. `read_projection_long()` therefore adds an
#      `embedding` column and nothing more; it does not pretend the spaces align.
#
# Because the subsample is hash-stable, the SAME patches appear in every run's
# sample, so joining two runs on `uid` gives genuine paired rows -- which is the
# one honest way to compare embeddings point by point.

source("R/constants.R")
source("R/patch_geo_context.R")

suppressPackageStartupMessages({
  library(arrow)
  library(dplyr)
  library(digest)
  library(ggplot2)
  # Attached, not just namespaced: `ggplot + inset_element(...)` dispatches on
  # patchwork's own `+` method, which is invisible unless the package is on the
  # search path -- the inset silently does nothing otherwise.
  library(patchwork)
  library(tidyr)
})

PROJECTION_DIR <- file.path(DATA_DIR, "output", "lcz-classification", "embedding_viz")

# Coordinate columns, by method.
PROJ_METHODS <- c("pca", "umap", "tsne")

# ── Discovery ─────────────────────────────────────────────────────────────────

#' All projection runs under `dir`, one row per run.
#'
#' @return tibble with `run`, `full`, `sample`, `density` path columns. Columns
#'   are NA where that artefact was not written (older runs predate the sample
#'   and density exports).
list_projections <- function(dir = PROJECTION_DIR) {
  full <- list.files(dir, pattern = "^projection_.*[^ey]\\.parquet$",
                     recursive = TRUE, full.names = TRUE)
  full <- full[!grepl("_(sample|density)\\.parquet$", full)]
  if (length(full) == 0) {
    stop("No projection_*.parquet found under ", dir, call. = FALSE)
  }
  tibble::tibble(
    run     = basename(dirname(full)),
    full    = full,
    sample  = .sibling(full, "_sample"),
    density = .sibling(full, "_density")
  )
}

.sibling <- function(paths, suffix) {
  out <- sub("\\.parquet$", paste0(suffix, ".parquet"), paths)
  ifelse(file.exists(out), out, NA_character_)
}

# ── Reading ───────────────────────────────────────────────────────────────────

#' Read one projection parquet.
#'
#' @param path Parquet path (full, sample, or density -- any of them).
#' @param columns Optional character vector to read only those columns. Worth
#'   using on the full 400k-row files.
#' @return tibble, with `lcz_name` and `split` as ordered factors so the LCZ
#'   palette in constants.R lines up without further work.
read_projection <- function(path, columns = NULL) {
  # col_select must be omitted entirely when no subset is asked for: passing
  # all_of(NULL) selects zero columns rather than all of them.
  df <- if (is.null(columns)) {
    tibble::as_tibble(arrow::read_parquet(path))
  } else {
    tibble::as_tibble(arrow::read_parquet(path, col_select = dplyr::all_of(columns)))
  }

  if ("LCZ_class" %in% names(df)) {
    df$lcz <- lcz_factor(df$LCZ_class)
  }
  if ("lcz_name" %in% names(df)) {
    lvl <- paste0("LCZ ", LCZ_TABLE$code, ": ", LCZ_TABLE$name)
    df$lcz_name <- factor(df$lcz_name, levels = lvl[lvl %in% df$lcz_name])
  }
  if ("split" %in% names(df)) {
    df$split <- factor(df$split, levels = c("train", "val", "test"))
  }
  df
}

#' Row-bind several runs into one long frame for FACETING ONLY.
#'
#' See the header note: coordinates from separate runs do not share a frame.
#' This is here so you can draw one panel per embedding, not so you can compare
#' coordinates across them.
read_projection_long <- function(paths, columns = NULL) {
  purrr::map_dfr(paths, function(p) {
    d <- read_projection(p, columns)
    if (!"embedding" %in% names(d)) d$embedding <- basename(dirname(p))
    d
  })
}

#' Join two runs on `uid` into paired columns.
#'
#' Only meaningful because the subsample is hash-stable: both runs sampled the
#' same patches. Coordinate columns are suffixed `_a` / `_b`.
pair_projections <- function(path_a, path_b, columns = NULL) {
  a <- read_projection(path_a, columns)
  b <- read_projection(path_b, columns)
  inner_join(a, b, by = "uid", suffix = c("_a", "_b"))
}

#' Read the pre-binned density grid, optionally for one facet.
read_density_grid <- function(path, method = NULL, facet = NULL) {
  df <- tibble::as_tibble(arrow::read_parquet(path))
  if (!is.null(method)) df <- filter(df, method == !!method)
  if (!is.null(facet))  df <- filter(df, facet == !!facet)
  df
}

# ── Axes, subsampling, depth and colour ───────────────────────────────────────

# Number of levels a colour variable may show before the tail is collapsed into
# "Other". City is 51 and country 30 in the global runs; a legend that size is
# taller than the panel and its hues stop being distinguishable well before it.
COLOUR_TOP_N <- 12L
# Patches whose coordinate falls on the Köppen raster's ocean. Slate rather than
# a true blue: Köppen's own Af/Am are saturated blues and LCZ 17 is blue too, so
# a blue "Water" swatch would read as one of those.
WATER_LABEL  <- "Water"
WATER_COLOUR <- "#93a9b8"
OTHER_LABEL  <- "Other"
OTHER_COLOUR <- "grey75"

# Points drawn by default. This env has NO scattermore and NO ggrastr, so these
# are plain geom_point marks through ragg: 60k is a few seconds, 400k is not.
# Use plot_density() when you want the whole cloud.
MAX_POINTS <- 60000L

# Axis labels, by method. PCA components are PCs; the other two are just axes.
METHOD_AXIS <- c(pca = "PC", umap = "UMAP ", tsne = "t-SNE ")

# Quantile the axes are framed on: c(1-q, q), so 0.995 frames the central 99%.
#
# This is not cosmetic. PCA on these embeddings grows a long thin arm -- on
# AlphaEarth coop, 1,639 patches (0.41%) sit below PC1 = -10 while the other
# 99.6% of the cloud lives inside +-3 -- and under coord_fixed() that arm
# squashes the whole structure into a sliver a few pixels wide. The arm is real
# signal, not corruption: 1,440 of those 1,639 are LCZ 17 (Water), in coastal
# cities (Cape Town 896, New York 340, Mumbai 109). So it is framed out rather
# than removed -- the points are still in the data, the frame just does not
# chase them, and the caption reports how many fell outside. 0.999 is NOT enough
# to clear it; use --clip 1 to frame on the full range and see the arm.
CLIP_Q <- 0.995

#' Resolve the coordinate columns for one method.
#'
#' Two file generations exist. The current exporter writes `<method>_1/_2/_3`;
#' everything written before 2026-09 has only `<method>_x/_y` and no third
#' coordinate at all. This is the ONLY place that knows about both, so nothing
#' downstream has to branch on the file's age.
#'
#' @return list(x, y, z) of column names; `z` is NA_character_ when the run has
#'   no third coordinate.
proj_axes <- function(df, method = "pca") {
  num <- paste0(method, "_", 1:3)
  leg <- paste0(method, c("_x", "_y"))
  if (all(num[1:2] %in% names(df))) {
    return(list(x = num[1], y = num[2],
                z = if (num[3] %in% names(df)) num[3] else NA_character_))
  }
  if (all(leg %in% names(df))) {
    return(list(x = leg[1], y = leg[2], z = NA_character_))
  }
  have <- PROJ_METHODS[vapply(PROJ_METHODS, function(m)
    any(paste0(m, c("_1", "_x")) %in% names(df)), logical(1))]
  stop("no ", method, " coordinates in this run (has: ",
       paste(have, collapse = ", "), ")", call. = FALSE)
}

#' Reorder rows so the third coordinate reads as depth.
#'
#' ggplot2 draws rows in data order, so sorting the frame IS the depth encoding
#' -- there is no `order` aesthetic any more, and modulating size or alpha would
#' confound depth with a second channel. `"asc"` puts the largest z last, i.e.
#' on top. Rows with NA z go to the bottom of the pile in both directions.
#'
#' On a run with no third coordinate this warns ONCE per session and returns
#' `df` untouched, so the pre-2026-09 parquets still plot -- just in file order.
#' Once, not per call: --all-colours plots the same frame five times and five
#' copies of the same warning buries every other message.
.depth_warned <- new.env(parent = emptyenv())

depth_sort <- function(df, z, order = c("asc", "desc", "none")) {
  order <- match.arg(order)
  if (order == "none") return(df)
  if (is.na(z)) {
    if (is.null(.depth_warned$done)) {
      .depth_warned$done <- TRUE
      warning("no third coordinate in this run; drawing in file order. Re-export ",
              "with src/embedding_projection.py to get pca_3.", call. = FALSE)
    }
    return(df)
  }
  v <- df[[z]]
  df[order(v, decreasing = (order == "desc"), na.last = FALSE), , drop = FALSE]
}

`%||%` <- function(a, b) if (is.null(a)) b else a

#' Width in inches of the widest string, as it will actually be drawn.
#'
#' Same measurement `R/metrics_table.R` uses to self-size the table: ask the
#' shaper ragg will use, rather than guessing from character counts.
.str_in <- function(s, pt) {
  s <- s[!is.na(s)]
  if (!length(s)) return(0)
  if (requireNamespace("systemfonts", quietly = TRUE)) {
    return(max(systemfonts::string_width(s, size = pt, res = 72)) / 72)
  }
  max(nchar(s)) * pt * 0.52 / 72
}

# ── Geographic colour ─────────────────────────────────────────────────────────

# Longitude and latitude get DIVERGING ramps on purpose: both have a meaningful
# middle (the prime meridian, the equator) and two opposed directions, which is
# exactly what a diverging scale encodes and a sequential one throws away.
LON_RAMP <- "RdYlBu"   # west warm -> east cool
LAT_RAMP <- "BrBG"     # south brown -> north teal

# Bivariate corners, the Stevens scheme: neutral where both are low, one hue per
# axis, and their mix in the far corner. Named by (lon, lat) level.
BIVAR_CORNERS <- c(lo_lo = "#e8e8e8", hi_lo = "#5ac8c8",
                   lo_hi = "#be64ac", hi_hi = "#3b4994")

#' Rescale to [0, 1] on the observed range.
.unit <- function(v) {
  r <- range(v, na.rm = TRUE)
  if (!all(is.finite(r)) || diff(r) == 0) return(rep(0.5, length(v)))
  (v - r[1]) / diff(r)
}

#' Symmetric half-range about zero, rounded out to a tidy number.
#'
#' Longitude and latitude are signed coordinates whose zero is a real place --
#' the prime meridian, the equator. Both scales are therefore pinned so that 0
#' sits at the exact centre of the ramp and equal distances either side get
#' equal colour, which is the only reading under which "warmer = further west"
#' means anything. The cost is that the longer hemisphere sets the range, so the
#' shorter one does not reach the end of the ramp -- that asymmetry is the data.
.sym_limit <- function(v, step = 10) {
  m <- max(abs(range(v, na.rm = TRUE)))
  ceiling(m / step) * step
}

#' Rescale to [0, 1] with zero pinned to the middle.
.unit_mid <- function(v) {
  m <- max(abs(range(v, na.rm = TRUE)))
  if (!is.finite(m) || m == 0) return(rep(0.5, length(v)))
  0.5 + v / (2 * m)
}

#' Bilinear blend of the four bivariate corners, in CIE Lab.
#'
#' Lab, not sRGB: blending two saturated hues in sRGB runs through a muddy dark
#' middle, which is the whole quadrant where both coordinates are mid-range.
bivariate_colour <- function(x, y, corners = BIVAR_CORNERS, centre = FALSE) {
  scl <- if (centre) .unit_mid else .unit
  u <- scl(x); v <- scl(y)
  # Corner order must match BIVAR_CORNERS: lo_lo, hi_lo, lo_hi, hi_hi.
  lab <- farver::convert_colour(t(grDevices::col2rgb(corners)), "rgb", "lab")
  w   <- cbind((1 - u) * (1 - v), u * (1 - v), (1 - u) * v, u * v)
  out <- matrix(0, nrow = length(u), ncol = 3)
  for (k in 1:3) out[, k] <- w %*% lab[, k]
  rgb <- farver::convert_colour(out, "lab", "rgb")
  grDevices::rgb(pmin(pmax(rgb, 0), 255), maxColorValue = 255)
}

#' The 2-D key for a bivariate scale, as a small standalone plot.
#'
#' A bivariate scale has no ggplot guide -- there is no one-dimensional bar that
#' can show it -- so the key is built as its own square of tiles and inset into
#' an empty corner of the panel with patchwork, the way R/plotting.R already
#' insets its map panels.
bivariate_key <- function(n = 5, xlab = "Longitude", ylab = "Latitude",
                          corners = BIVAR_CORNERS) {
  g <- expand.grid(i = seq_len(n), j = seq_len(n))
  g$fill <- bivariate_colour(g$i, g$j, corners)
  # The tile grid spans the symmetric range, so its own centre is 0 -- the
  # meridian/equator cross, marked so the origin is readable rather than implied.
  mid <- (n + 1) / 2
  pad <- 0.95
  lab <- data.frame(
    x = c(0.5 - pad, n + 0.5 + pad, mid,          mid),
    y = c(mid,       mid,           0.5 - pad,    n + 0.5 + pad),
    t = c("W",       "E",           "S",          "N"))
  ggplot(g, aes(i, j, fill = fill)) +
    geom_tile() +
    geom_text(data = lab, aes(x, y, label = t), inherit.aes = FALSE,
              size = 7 / .pt, fontface = "bold", colour = "grey25") +
    annotate("segment", x = 0.5, xend = n + 0.5, y = mid, yend = mid,
             colour = "grey35", linewidth = 0.2, linetype = "22") +
    annotate("segment", x = mid, xend = mid, y = 0.5, yend = n + 0.5,
             colour = "grey35", linewidth = 0.2, linetype = "22") +
    scale_fill_identity() +
    coord_fixed(xlim = c(0.5 - 2 * pad, n + 0.5 + 2 * pad),
                ylim = c(0.5 - 2 * pad, n + 0.5 + 2 * pad), expand = FALSE) +
    labs(x = xlab, y = ylab) +
    theme_void(base_size = 9) +
    theme(
      axis.title.x = element_text(colour = "grey25", size = 7,
                                  margin = margin(t = 1)),
      axis.title.y = element_text(colour = "grey25", size = 7, angle = 90,
                                  margin = margin(r = 1)),
      plot.background = element_rect(fill = "transparent", colour = NA)
    )
}

#' Symmetric quantile range of `v`, padded, for framing an axis.
#'
#' Returns NULL when `q` is 1 (frame on everything) or the trimmed range would
#' be degenerate.
robust_limits <- function(v, q = CLIP_Q, pad = 0.04) {
  if (q >= 1) return(NULL)
  lim <- unname(stats::quantile(v, c(1 - q, q), na.rm = TRUE))
  if (!all(is.finite(lim)) || diff(lim) <= 0) return(NULL)
  lim + c(-1, 1) * pad * diff(lim)
}

# One hue family per M49 region, shaded within it, so the legend reads as five
# blocks rather than 22 unrelated colours -- a subregion's continent is the
# first thing you want off this plot, its exact identity the second.
REGION_HUE <- c(Africa = 60, Americas = 300, Asia = 10, Europe = 250,
                Oceania = 150)

#' Palette for the M49 sub-regions, grouped by region.
#'
#' Returns colours named by sub-region, ordered region-major so the legend's
#' reading order is Africa, Americas, Asia, Europe, Oceania.
subregion_palette <- function(keys, meta) {
  meta <- meta[!duplicated(meta$subregion), ]
  meta <- meta[meta$subregion %in% keys, ]
  meta <- meta[order(match(meta$region, names(REGION_HUE)), meta$subregion), ]
  out <- character(0)
  for (rg in unique(meta$region)) {
    sub <- meta$subregion[meta$region == rg]
    hue <- REGION_HUE[[rg]] %||% 0
    # Vary lightness and chroma together: at one lightness, four shades of the
    # same hue are hard to tell apart at this mark size.
    cols <- colorspace::sequential_hcl(
      max(length(sub), 2), h = hue, c = c(80, 35), l = c(32, 78), power = 1)
    out <- c(out, setNames(cols[seq_along(sub)], sub))
  }
  out
}

# Rows a grouped bottom key is laid out over. Four is what the Koppen codes were
# asked for; it is also the largest M49 region (Asia and Europe have four
# sub-regions each), so both modes land on it for different reasons.
LEGEND_NROW <- 4L

#' Pad a grouped set of legend levels so groups line up on the grid.
#'
#' `guide_legend` fills a plain rectangle, so consecutive groups run into each
#' other mid-row. Inserting invisible spacer levels after each group -- unique
#' whitespace strings, transparent swatch, blank label -- pushes the next group
#' to a fresh row or column. The scale must then be `drop = FALSE`, since the
#' spacers appear in no data.
#'
#' @param by "column" pads each group to a multiple of `nrow` so a group owns
#'   whole columns (short labels); "row" pads every group to the largest group
#'   so each group owns exactly one row (long labels).
#' @return list(levels, labels, colours, nrow, ncol, byrow)
pad_legend_groups <- function(groups, colours, by = c("column", "row"),
                              nrow = LEGEND_NROW) {
  by  <- match.arg(by)
  g   <- factor(groups, levels = unique(groups))
  siz <- table(g)
  cell <- if (by == "column") {
    ceiling(as.integer(siz) / nrow) * nrow
  } else {
    rep(max(as.integer(siz)), length(siz))
  }
  lv <- character(0); lb <- character(0); cl <- character(0); pad <- 0L
  for (i in seq_along(siz)) {
    keys <- names(groups)[g == levels(g)[i]]
    lv <- c(lv, keys); lb <- c(lb, keys); cl <- c(cl, colours[keys])
    n_pad <- cell[i] - length(keys)
    if (n_pad > 0) {
      sp <- strrep(" ", pad + seq_len(n_pad))   # unique, and invisible
      pad <- pad + n_pad
      lv <- c(lv, sp); lb <- c(lb, rep("", n_pad)); cl <- c(cl, rep("transparent", n_pad))
    }
  }
  list(levels = lv, labels = lb, colours = setNames(cl, lv),
       nrow = if (by == "column") nrow else length(siz),
       ncol = if (by == "column") length(lv) %/% nrow else cell[1],
       byrow = (by == "row"))
}

#' Resolve a colour variable to data + a scale.
#'
#' `lcz_name` and `split` have canonical palettes in constants.R. Everything
#' else is generic: keep the `top_n` largest levels, in descending size order,
#' and collapse the tail to a grey "Other" pinned last in the legend. That keeps
#' the legend at a fixed <= top_n + 1 entries whatever column you hand it.
#'
#' @return list(data, scale, title)
colour_spec <- function(df, colour, top_n = COLOUR_TOP_N) {
  # Geographic modes come first: they are derived from lon/lat, so the plain
  # "is this a column" check below would reject `lonlat` outright.
  geo <- c(lon = "Longitude", lat = "Latitude", lonlat = "Longitude / latitude")
  if (colour %in% names(geo)) {
    need <- if (colour == "lonlat") c("lon", "lat") else colour
    miss <- setdiff(need, names(df))
    if (length(miss)) {
      stop("this run has no ", paste(miss, collapse = "/"), " column, so it ",
           "cannot be coloured by geography. Re-export with ",
           "src/embedding_projection.py, which writes lon/lat.", call. = FALSE)
    }
    if (colour == "lonlat") {
      df$.bivar <- bivariate_colour(df$lon, df$lat, centre = TRUE)
      return(list(data = df, column = ".bivar", title = geo[[colour]],
                  kind = "identity", scale = scale_colour_identity(),
                  key = bivariate_key()))
    }
    pal  <- if (colour == "lon") LON_RAMP else LAT_RAMP
    ends <- if (colour == "lon") c("W", "E") else c("S", "N")
    # The scale spans the coordinate's FULL possible range -- +-180 lon, +-90
    # lat -- rather than the data's own. That is what lets the interior breaks
    # land on the round 60/120 and 30/60 graticule values with the compass
    # letters clear of them at the ends, and it makes two runs' colours mean the
    # same thing. The cost is contrast: So2Sat spans -123..151 and -38..56, so
    # neither ramp is driven to its extremes.
    lim  <- if (colour == "lon") 180 else 90
    step <- if (colour == "lon") 60  else 30
    inner <- seq(-2 * step, 2 * step, by = step)
    brk  <- c(-lim, inner, lim)
    # Bare degrees inside; the compass letter only at the ends of the band.
    labs <- c(ends[1], paste0(abs(inner), "\u00b0"), ends[2])
    return(list(data = df, column = colour, title = geo[[colour]],
                kind = "continuous",
                scale = scale_colour_gradientn(
                  colours = RColorBrewer::brewer.pal(11, pal),
                  limits = c(-lim, lim), breaks = brk, labels = labs,
                  oob = scales::squish)))
  }

  # Derived from lon/lat by a spatial lookup, and cached -- see
  # R/patch_geo_context.R. Neither is collapsed to top-N: the whole point of
  # both is the full classification, and 22 or 30 classes is what they are.
  if (colour %in% c("subregion", "koppen")) {
    if (!colour %in% names(df)) df <- geo_context(df)
    if (colour == "subregion") {
      # No Water level here, unlike `koppen`. A sub-region is a territorial
      # label, not a physical one: a patch of harbour belongs to the country
      # whose harbour it is, and the lookup has already given every patch the
      # containing -- or, for the 7.9% the 1:110m coastline misses, the nearest
      # -- country. Water only breaks a *climate* reading, which is why the
      # Koppen mode keeps its Water class and this one does not.
      meta <- read_m49()
      v    <- df$subregion
      keys <- sort(unique(v[!is.na(v)]))
      pal  <- subregion_palette(keys, meta)
      grp  <- setNames(meta$region[match(names(pal), meta$subregion)], names(pal))
      df$subregion <- factor(v, levels = names(pal))
      lay <- pad_legend_groups(grp, pal, by = "row")
      return(list(data = df, column = colour, title = "UN M49 sub-region",
                  kind = "discrete", layout = lay,
                  scale = scale_colour_manual(values = lay$colours,
                                              labels = lay$labels,
                                              limits = lay$levels,
                                              drop = FALSE, na.translate = FALSE)))
    }
    leg  <- read_koppen_legend()
    leg  <- leg[leg$code %in% levels(droplevels(df$koppen)), ]
    code <- as.character(df$koppen)
    lv   <- leg$code
    cols <- setNames(leg$colour, leg$code)
    # Code only. The full names run to 45 characters, which forces the key three
    # columns wide and taller than the panel; the codes are the standard
    # shorthand and their first letter is the grouping the layout uses.
    grp  <- setNames(substr(leg$code, 1, 1), leg$code)
    if (anyNA(code)) {
      message("  koppen: ", sum(is.na(code)), " patch(es) on water")
      code[is.na(code)] <- WATER_LABEL
      lv   <- c(lv, WATER_LABEL)
      cols <- c(cols, setNames(WATER_COLOUR, WATER_LABEL))
      grp  <- c(grp, setNames(WATER_LABEL, WATER_LABEL))
    }
    df$koppen <- factor(code, levels = lv)
    lay <- pad_legend_groups(grp, cols, by = "column")
    return(list(data = df, column = colour, title = "K\u00f6ppen-Geiger class",
                kind = "discrete", layout = lay,
                scale = scale_colour_manual(values = lay$colours,
                                            labels = lay$labels,
                                            limits = lay$levels,
                                            drop = FALSE, na.translate = FALSE)))
  }

  if (!colour %in% names(df)) {
    stop("no column '", colour, "' in this run (has: ",
         paste(names(df), collapse = ", "), ")", call. = FALSE)
  }

  if (colour %in% c("lcz_name", "lcz")) {
    pal <- if (colour == "lcz_name") {
      setNames(LCZ_TABLE$colour, paste0("LCZ ", LCZ_TABLE$code, ": ", LCZ_TABLE$name))
    } else {
      lcz_palette("alt")
    }
    return(list(data = df, column = colour, title = "LCZ class",
                kind = "discrete",
                scale = scale_colour_manual(values = pal, drop = TRUE,
                                            na.value = "grey80")))
  }

  if (colour == "split") {
    lab <- c(train = "Training", val = "Validation", test = "Testing")
    df[[colour]] <- factor(unname(lab[as.character(df[[colour]])]),
                           levels = unname(lab))
    return(list(data = df, column = colour, title = "Split", kind = "discrete",
                scale = scale_colour_manual(values = DATASET_COLOURS,
                                            drop = TRUE, na.value = "grey80")))
  }

  v      <- as.character(df[[colour]])
  counts <- sort(table(v), decreasing = TRUE)
  keep   <- names(counts)[seq_len(min(top_n, length(counts)))]
  n_drop <- length(counts) - length(keep)

  if (n_drop > 0) {
    frac <- sum(counts[!names(counts) %in% keep]) / sum(counts)
    message(sprintf("  %s: %d levels, showing the %d largest; %d collapsed to '%s' (%.1f%% of points)",
                    colour, length(counts), length(keep), n_drop, OTHER_LABEL, 100 * frac))
    v <- ifelse(v %in% keep, v, OTHER_LABEL)
    lvl <- c(keep, OTHER_LABEL)
  } else {
    lvl <- keep
  }
  df[[colour]] <- factor(v, levels = lvl)

  # Dark2 (8) + Set2 (8) are the two qualitative Brewer sets that stay distinct
  # at this mark size; past 16 levels there is no honest qualitative palette, so
  # interpolate rather than recycle -- recycling would give two levels one hue.
  base <- c(RColorBrewer::brewer.pal(8, "Dark2"), RColorBrewer::brewer.pal(8, "Set2"))
  cols <- if (length(keep) <= length(base)) base[seq_along(keep)] else
    grDevices::colorRampPalette(base)(length(keep))
  pal  <- setNames(cols, keep)
  if (n_drop > 0) pal[[OTHER_LABEL]] <- OTHER_COLOUR

  list(data = df, column = colour, title = tools::toTitleCase(colour),
       kind = "discrete",
       scale = scale_colour_manual(values = pal, drop = TRUE, na.value = "grey80"))
}

#' Python's `crc32(uid) %% m`, reproduced in R.
#'
#' `stable_subsample_mask` in src/embedding_projection.py selects the R sample by
#' `zlib.crc32(uid.encode()) %% 1000`, so the SAME patches land in every run's
#' sample and two runs can be joined point for point. Matching it here means a
#' subsample taken in R is the same kind of object.
#'
#' `strtoi(h, 16L)` cannot be used directly: a CRC is 32-bit unsigned and every
#' value above 2^31-1 comes back NA. Combining four bytes as doubles keeps it
#' exact (< 2^53) and was checked against Python on ten uids.
crc32_mod <- function(x, m = 1000L) {
  h <- vapply(x, function(u) digest::digest(u, algo = "crc32", serialize = FALSE),
              "", USE.NAMES = FALSE)
  h <- formatC(h, width = 8, flag = "0")
  b <- strtoi(substring(rep(h, each = 4L), c(1, 3, 5, 7), c(2, 4, 6, 8)), 16L)
  as.integer(colSums(matrix(b, nrow = 4L) * c(16777216, 65536, 256, 1)) %% m)
}

#' `uid` for a projection frame: "<dataset>/<patch_id>", as Python defines it.
#'
#' `patch_id` alone is NOT unique -- it restarts at 000000 in each of training /
#' validation / testing -- so it must never be used as a key on its own.
projection_uid <- function(df) {
  if ("uid" %in% names(df)) return(as.character(df$uid))
  if (all(c("dataset", "patch_id") %in% names(df))) {
    return(paste0(df$dataset, "/", df$patch_id))
  }
  NULL
}

#' Read one run's points, subsampled for plotting.
#'
#' Prefers the `_sample.parquet` the exporter writes (already ~10% by uid hash).
#' Falling back to the full file, only the columns needed are read -- that is
#' what makes a 400k-row parquet cheap -- and any excess is trimmed by the same
#' uid hash, so re-running gives the same points rather than a fresh RNG draw.
#'
#' @param run One row of `list_projections()`.
#' @param columns Columns to keep, or NULL for all.
#' @param max_points Cap on rows returned; Inf keeps everything.
#' @param full Read the full parquet even when a `_sample` sibling exists. Every
#'   patch, no subsample -- slow to draw, but it is the honest whole cloud.
read_projection_sample <- function(run, columns = NULL, max_points = MAX_POINTS,
                                   full = FALSE) {
  path <- if (!full && !is.na(run$sample)) run$sample else run$full
  if (identical(path, run$full) && !is.null(columns)) {
    # Ask only for columns the file actually has: a run written before a given
    # column existed must still read, not error on the missing name.
    have    <- names(arrow::open_dataset(path))
    columns <- intersect(columns, have)
  }
  df <- read_projection(path, columns)
  message("  read ", format(nrow(df), big.mark = ","), " rows from ", basename(path))

  if (nrow(df) <= max_points) return(df)

  uid <- projection_uid(df)
  if (is.null(uid)) {
    warning("no uid / dataset+patch_id in this run; falling back to a seeded ",
            "RNG subsample (not stable across runs)", call. = FALSE)
    set.seed(42L)
    return(df[sort(sample.int(nrow(df), max_points)), , drop = FALSE])
  }
  # Take the whole hash buckets that fit, so the kept set stays a hash-defined
  # subset -- the same property the Python sample has, and the reason two runs
  # subsampled here can still be joined on uid.
  keep_frac <- max_points / nrow(df)
  cutoff    <- floor(keep_frac * 1000)
  h         <- crc32_mod(uid)
  df        <- df[h < max(cutoff, 1L), , drop = FALSE]
  message("  subsampled to ", nrow(df), " rows (uid hash < ", max(cutoff, 1L), "/1000)")
  df
}

# ── Plotting ──────────────────────────────────────────────────────────────────

#' Scatter of one projection's first two dimensions, coloured by a variable.
#'
#' @param df From `read_projection_sample()` -- pass a SAMPLE, not 400k rows.
#' @param method One of pca / umap / tsne.
#' @param colour Column to colour by. `lcz_name` uses the canonical LCZ palette,
#'   `split` the dataset palette; anything else collapses to top-N + "Other".
#' @param depth "asc" (largest third coordinate on top), "desc", or "none".
#' @param top_n Levels to keep before collapsing, for a generic colour variable.
#' @param clip Central quantile the axes are framed on; 1 frames on everything.
#' @param legend Where the colour key goes: "none" (default), "bottom" or
#'   "right". Off by default -- these scatters are panelled beside figures that
#'   already carry the LCZ key. "bottom" keeps the panel square and full-width,
#'   which "right" cannot: a 13- or 17-entry key is wider than the cloud it
#'   explains. TRUE is accepted and means "bottom".
#' @param legend_ncol Columns in a bottom key; NULL picks from label length.
#' @param caption Draw the provenance caption (run, method, n, colour variable,
#'   depth column, points outside the frame)? FALSE by default -- the figure is
#'   a panel in a larger layout that carries its own titling. The same facts go
#'   to the console on every call, so nothing is lost by leaving it off.
#' @param label Run label for the caption; NULL omits it.
plot_projection <- function(df, method = "pca", colour = "lcz_name",
                            depth = "asc", top_n = COLOUR_TOP_N, clip = CLIP_Q,
                            legend = "none", caption = FALSE, legend_ncol = NULL,
                            size = NULL, alpha = NULL, label = NULL) {
  ax <- proj_axes(df, method)

  # t-SNE is fitted on a balanced per-class subsample, so tsne_x is NA for most
  # rows of a global run. Dropping them is load-bearing, not tidying.
  n_in <- nrow(df)
  df   <- df[!is.na(df[[ax$x]]) & !is.na(df[[ax$y]]), , drop = FALSE]
  if (nrow(df) == 0) {
    stop("every ", method, " coordinate is NA in this run", call. = FALSE)
  }
  if (nrow(df) < n_in) {
    message("  ", method, ": ", nrow(df), " of ", n_in,
            " rows have coordinates (the rest are NA by construction)")
  }

  df <- depth_sort(df, ax$z, depth)
  cs <- colour_spec(df, colour, top_n)
  df <- cs$data
  ccol <- cs$column %||% colour

  # Scale the mark to the cloud: one setting cannot serve both a 500-point
  # single-city sample and a 40k-point global one -- fixed small-and-faint
  # defaults make the former invisible and fixed large ones make the latter a
  # solid blob.
  n <- nrow(df)
  if (is.null(size))  size  <- max(0.25, min(2.0, 45 / sqrt(max(n, 1))))
  if (is.null(alpha)) alpha <- max(0.35, min(0.9, 300 / max(n, 1)))

  xlim <- robust_limits(df[[ax$x]], clip)
  ylim <- robust_limits(df[[ax$y]], clip)
  off  <- if (is.null(xlim)) 0L else sum(
    df[[ax$x]] < xlim[1] | df[[ax$x]] > xlim[2] |
    df[[ax$y]] < ylim[1] | df[[ax$y]] > ylim[2], na.rm = TRUE)
  if (off > 0) {
    message(sprintf("  framed on the central %.1f%%; %d point%s outside the frame",
                    100 * (2 * clip - 1), off, if (off == 1) "" else "s"))
  }

  pre <- METHOD_AXIS[[method]]
  cap <- if (!caption) NULL else {
    txt <- paste0(
      if (!is.null(label)) paste0(label, " · ") else "",
      toupper(method), " · ", format(n, big.mark = ","), " patches · ",
      "coloured by ", tolower(cs$title), " · ",
      if (depth != "none" && !is.na(ax$z))
        paste0("depth ", ax$z, " (", if (depth == "asc") "high on top" else "low on top", ")")
      else "no depth ordering",
      if (off > 0) paste0(" · ", off, " outside the frame") else "")
    # Wrap it: without a legend the figure is only 6.2 in wide and ggplot
    # silently CLIPS a caption that overruns rather than shrinking or wrapping.
    paste(strwrap(txt, width = 72), collapse = "\n")
  }

  p <- ggplot(df, aes(.data[[ax$x]], .data[[ax$y]], colour = .data[[ccol]])) +
    geom_point(size = size, alpha = alpha, stroke = 0) +
    cs$scale +
    coord_fixed(xlim = xlim, ylim = ylim) +
    labs(x = paste0(pre, 1), y = paste0(pre, 2), colour = cs$title, caption = cap) +
    theme_eofm()

  if (isTRUE(legend)) legend <- "bottom"
  if (isFALSE(legend)) legend <- "none"
  legend <- match.arg(legend, c("none", "bottom", "right"))

  # The bivariate key is checked BEFORE `legend == "none"`, and deliberately.
  # An identity scale has no guide ggplot can draw, so this 2-D square is the
  # only way to read the figure at all -- it is not decoration that "none"
  # should suppress, and it costs no layout because it sits inside the panel.
  if (identical(cs$kind, "identity")) {
    p <- p + theme(legend.position = "none")
    if (!is.null(cs$key)) {
      p <- p + patchwork::inset_element(cs$key, left = 0.015, bottom = 0.70,
                                        right = 0.235, top = 0.985,
                                        align_to = "panel")
    }
    attr(p, "legend_rows") <- 0L
    return(p)
  }

  if (legend == "none") {
    # position = "none" rather than guides(colour = "none"): it also reclaims the
    # strip ggplot reserves for the key, which is the point.
    p <- p + theme(legend.position = "none")
    attr(p, "legend_rows") <- 0L
    return(p)
  }

  if (identical(cs$kind, "continuous")) {
    bar <- if (legend == "bottom") {
      guide_colourbar(direction = "horizontal", title.position = "top",
                      barwidth = grid::unit(3.4, "in"),
                      barheight = grid::unit(0.14, "in"))
    } else guide_colourbar(barheight = grid::unit(1.6, "in"))
    p <- p + guides(colour = bar) +
      theme(legend.position = legend,
            legend.justification = if (legend == "bottom") "center" else "top",
            # Seven labels on one bar: at the inherited legend size the outer
            # pairs collide and ggplot drops them silently rather than shrinking.
            legend.text = element_text(size = LEGEND_TEXT_PT * 0.78))
    attr(p, "legend_rows") <- if (legend == "bottom") 1L else 0L
    return(p)
  }

  lv <- levels(cs$data[[ccol]])
  if (legend == "right") {
    p <- p + guides(colour = guide_legend(override.aes = list(size = 2.5, alpha = 1))) +
      theme(legend.position = "right", legend.justification = "top")
    attr(p, "legend_rows") <- 0L
    return(p)
  }

  # A bottom key is laid out by hand: ggplot fills to the panel width and will
  # run the last entries off the figure. Both the column count and the text size
  # come from the longest label.
  byrow <- TRUE
  if (!is.null(cs$layout)) {
    # Grouped modes keep like with like: Koppen's A/B/C/D/E down each column,
    # the M49 regions one per row. The padded scale was built in colour_spec, so
    # nothing is added here -- a second scale would only draw a "replacing the
    # existing scale" warning for no gain.
    lay   <- cs$layout
    lv    <- lay$levels
    ncol  <- lay$ncol
    byrow <- lay$byrow
    wid   <- max(nchar(lay$labels))
    pt    <- LEGEND_TEXT_PT * if (wid > 30) 0.68 else if (wid > 16) 0.85 else 1
    rows  <- lay$nrow
  } else {
    wid  <- max(nchar(cs$labels %||% lv))
    ncol <- min(legend_ncol %||%
                  if (wid > 16) 3L else if (wid > 10) 4L else 6L,
                length(lv))
    pt   <- LEGEND_TEXT_PT * if (wid > 30) 0.68 else if (wid > 16) 0.85 else 1
    rows <- as.integer(ceiling(length(lv) / ncol))
  }

  # The figure has to be wide enough for the key, not the other way round: at
  # three columns the Koppen names need ~8 in and silently ran off a 6.2 in
  # figure, taking the right-hand third of the labels with them.
  # The 1.08 is slack, not decoration: measured exactly, the key lands 7 px from
  # the right edge, which reads as clipped even when it is not.
  key_w <- (ncol * (.str_in(if (!is.null(cs$layout)) lay$labels else
                              cs$labels %||% lv, pt) + 0.34) + 0.2) * 1.08

  p <- p +
    guides(colour = guide_legend(override.aes = list(size = 2.5, alpha = 1),
                                 ncol = ncol, byrow = byrow,
                                 title.position = "top")) +
    theme(legend.position = "bottom", legend.justification = "center",
          legend.margin = margin(t = 2),
          legend.text = element_text(size = pt),
          legend.key.size = unit(0.7, "lines"))

  # The caller sizes the figure, so it has to be told how big the key will be.
  attr(p, "legend_rows")  <- rows
  attr(p, "legend_width") <- key_w
  p
}

#' Whole-cloud density map from the pre-binned grid.
#'
#' The right way to show all ~400k patches here: one geom_raster over a few
#' thousand bins instead of 400k points.
plot_density <- function(grid, facet_value = NULL, trans = "log10") {
  if (!is.null(facet_value)) grid <- filter(grid, facet_value == !!facet_value)
  # geom_tile, not geom_raster: empty bins are dropped from the export, so the
  # surviving bin centres are unevenly spaced and geom_raster would shift them.
  ggplot(grid, aes(bin_x, bin_y, fill = n)) +
    geom_tile() +
    coord_fixed() +
    scale_fill_viridis_c(trans = trans, option = "magma") +
    labs(x = NULL, y = NULL, fill = "patches") +
    theme_eofm()
}

#' Small multiples: the same projection, one panel per facet value.
plot_density_facets <- function(grid, ncol = 4, trans = "log10") {
  plot_density(grid, trans = trans) + facet_wrap(~ facet_value, ncol = ncol)
}

#' Colour every point by its embedding-space RGB, if the run carries it.
#'
#' These columns come from the PIXEL colour model in src/embedding_rgb.py, so
#' the colours here are the same quantity as the RGB rasters -- a patch that
#' looks teal in this scatter sits in the teal part of the map.
#'
#' Expect muted colours on a SINGLE-CITY run, for the same reason the rasters
#' are muted there: the colour model spans every city, so one city occupies a
#' narrow slice of it. On a global run the colours use the full range. That
#' muting is the cross-city signal, not a defect -- do not "fix" it by
#' rescaling these columns, which would break the correspondence with the maps.
plot_projection_rgb <- function(df, method = "pca", rgb_method = method,
                                size = NULL, alpha = NULL) {
  cols <- paste0("rgb_", rgb_method, "_", c("r", "g", "b"))
  if (!all(cols %in% names(df))) {
    stop("run has no ", rgb_method, " RGB columns; write them with ",
         "src/embedding_rgb.py --annotate-parquet", call. = FALSE)
  }
  xy <- paste0(method, c("_x", "_y"))
  df <- df[!is.na(df[[xy[1]]]), ]
  n <- nrow(df)
  if (is.null(size))  size  <- max(0.25, min(2.0, 45 / sqrt(max(n, 1))))
  if (is.null(alpha)) alpha <- max(0.35, min(0.9, 300 / max(n, 1)))
  df$.rgb <- grDevices::rgb(df[[cols[1]]], df[[cols[2]]], df[[cols[3]]], maxColorValue = 255)

  ggplot(df, aes(.data[[xy[1]]], .data[[xy[2]]])) +
    geom_point(colour = df$.rgb, size = size, alpha = alpha, stroke = 0) +
    coord_fixed() +
    labs(x = NULL, y = NULL) +
    theme_eofm()
}

# ── CLI ───────────────────────────────────────────────────────────────────────

# Colour variables --all-colours loops. Every one of these exists in the global
# runs; a run missing one errors from colour_spec() with the columns it does have.
CLI_COLOURS <- c("lcz_name", "city", "country", "continent", "split")

# Key columns that must survive the `columns =` restriction whatever is being
# coloured: the uid parts (for the hash subsample) and the split label.
KEY_COLUMNS <- c("uid", "dataset", "patch_id", "split")

#' Resolve `--run` against the runs on disk.
#'
#' A substring, not an exact name, because the run directories are long
#' (`proj_GeoTessera_v1.1_global_global_gap`). Ambiguous or unknown matches
#' error with the available names, so a typo never silently picks a neighbour.
resolve_run <- function(runs, pattern = NULL) {
  if (is.null(pattern)) {
    if (nrow(runs) == 1) return(runs[1, ])
    stop("--run is required; ", nrow(runs), " projection runs available:\n  ",
         paste(runs$run, collapse = "\n  "), call. = FALSE)
  }
  # Exact name wins outright. Without this, a run cannot be selected whenever
  # another run's name merely contains it -- which is exactly what happens once
  # a run is kept alongside a backup of itself
  # ("proj_X" vs "proj_X.june2026-backup").
  if (pattern %in% runs$run) return(runs[runs$run == pattern, ])
  hit <- grepl(pattern, runs$run, fixed = TRUE)
  if (sum(hit) == 1) return(runs[hit, ])
  stop(if (sum(hit) == 0) "no run matches '" else "'",
       pattern, if (sum(hit) == 0) "'. " else "' is ambiguous. ",
       "Available runs:\n  ", paste(runs$run, collapse = "\n  "), call. = FALSE)
}

.flag <- function(args, name, default = NULL) {
  i <- match(name, args)
  if (is.na(i)) return(default)
  if (i == length(args)) stop(name, " needs a value", call. = FALSE)
  args[[i + 1]]
}

main <- function(args = commandArgs(trailingOnly = TRUE)) {
  runs <- list_projections()

  if ("--list" %in% args) {
    print(as.data.frame(runs |> mutate(across(c(full, sample, density), basename))))
    return(invisible(runs))
  }

  run     <- resolve_run(runs, .flag(args, "--run"))
  method  <- match.arg(.flag(args, "--method", "pca"), PROJ_METHODS)
  depth   <- match.arg(.flag(args, "--depth", "asc"), c("asc", "desc", "none"))
  top_n   <- as.integer(.flag(args, "--top-n", COLOUR_TOP_N))
  clip    <- as.numeric(.flag(args, "--clip", CLIP_Q))
  # --legend takes an OPTIONAL value: bare it means "bottom". .flag would happily
  # swallow the next switch as the value, so only a non-switch token counts.
  legend  <- if (!"--legend" %in% args) "none" else {
    i <- match("--legend", args)
    v <- if (i < length(args)) args[[i + 1]] else NA_character_
    if (is.na(v) || startsWith(v, "--")) "bottom" else v
  }
  caption <- "--caption" %in% args
  # The stack saves on a transparent canvas; a scatter this dense sometimes has
  # to stand on its own instead, where the points' own colours need a ground.
  background <- .flag(args, "--background", "transparent")
  # Appended to the output stem, so a variant of a figure that already exists
  # (a white-ground copy of a transparent one) does not overwrite it.
  suffix <- .flag(args, "--suffix", "")
  # --full: every patch, not the ~10% sample. Slow to draw, which is the whole
  # reason for the 60k default, so it has to be asked for explicitly. It lifts
  # the cap too unless --max-points says otherwise.
  full    <- "--full" %in% args
  max_pts <- if (is.null(.flag(args, "--max-points"))) {
    if (full) Inf else MAX_POINTS
  } else as.numeric(.flag(args, "--max-points"))
  colours <- if ("--all-colours" %in% args) CLI_COLOURS else
    .flag(args, "--colour", "lcz_name")

  message("Run: ", run$run)
  # One read serves every colour variable, so --all-colours costs one pass over
  # the parquet rather than five.
  # `lonlat` is derived, not stored: ask for its two source columns or the
  # restricted read drops them and colour_spec then reports them missing.
  src <- unique(unlist(lapply(colours, function(c)
    if (c %in% c("lonlat", "subregion", "koppen")) c("lon", "lat") else c)))
  cols <- unique(c(KEY_COLUMNS, src, "LCZ_class", "lcz_name",
                   paste0(method, c("_1", "_2", "_3", "_x", "_y"))))
  df <- read_projection_sample(run, columns = cols, max_points = max_pts,
                               full = full)

  for (colour in colours) {
    p <- plot_projection(df, method = method, colour = colour, depth = depth,
                         top_n = top_n, clip = clip, legend = legend,
                         caption = caption, label = run$run)
    # PNG only: tens of thousands of vector marks make an enormous PDF that no
    # viewer opens quickly, and this is a raster-dense figure either way.
    # A right-hand key takes width from the panel; a bottom one takes height, and
    # how much depends on how many rows it wrapped to -- hence the attribute.
    # theme_eofm() paints plot.background transparent, which wins over the
    # device's canvas colour: without this the file saves transparent whatever
    # --background says. `&` reaches inside a patchwork, `+` does not.
    if (background != "transparent") {
      bg_theme <- theme(plot.background = element_rect(fill = background,
                                                       colour = NA))
      p <- if (inherits(p, "patchwork")) p & bg_theme else p + bg_theme
    }
    rows <- attr(p, "legend_rows") %||% 0L
    base <- if (legend == "right") 7.5 else 6.2
    save_plot(p, paste0("projection_", run$run, "_", method, "_", colour, suffix),
              width  = max(base, attr(p, "legend_width") %||% 0),
              height = 6.2 + rows * 0.26 + if (rows > 0) 0.18 else 0,
              formats = "png", bg = background,
            subdir = PLOT_DIR_EMBEDDINGS)
  }
  invisible(NULL)
}

if (sys.nframe() == 0) main()
