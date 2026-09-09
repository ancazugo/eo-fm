# split_maps.R ─ Two-panel city maps explaining how the dataset split works.
#
#     Rscript R/split_maps.R
#
# Run from the repo root so .Renviron (DATA_DIR) and the relative plots/ path
# resolve. Shared constants, the theme and the save helper live in
# R/constants.R.
#
# These are diagrams, not maps of a place: no basemap, no coordinates, no scale,
# no labels and no key. They answer one question -- *which patches does a model
# see, and when?* -- so patches are coloured by their role in the split,
# Training black, Validation white with a hatch, Testing white. That is exactly
# the greyscale of the `dataset_composition_vertical` bar in R/plotting.R, which
# is the intended legend for these panels; LCZ class is deliberately not shown.
#
# Each panel is a whole city: the real So2Sat patch polygons over the full
# labelled extent, so the scatter of labelled blobs is the shape of the figure.
#
# Two pairs, over the same two cities so the reader sees one geography
# recoloured:
#
#   split_map_global    - the published culture-10 split: a whole city is either
#                         training or held out. London is all black; Nairobi is
#                         entirely validation + test.
#   split_map_orig_test - the `--orig-test` hybrid: the So2Sat *testing* patches
#                         stay test, everything else is re-split by grid cell.
#                         London becomes train + val; Nairobi's validation half
#                         becomes train + val, and only its testing half is held
#                         out. This is the picture behind the ~+22 kappa of
#                         same-city leakage measured for that mode.
#
# Figures written to plots/ (PNG).

source("R/constants.R")

suppressPackageStartupMessages({
  library(sf)
  library(dplyr)
  library(purrr)
  library(patchwork)
})

# ── The two panels ────────────────────────────────────────────────────────────
#
# A window is a `size` in metres about a centre, given either as `lonlat`
# (c(lon, lat), EPSG:4326) or as `centre` in the city's own UTM -- the CRS the
# GeoPackages are already in. `size = NULL` frames the city's entire labelled
# extent instead, squared and padded.
#
# Both are 12 km squares, so the two panels are at the same scale and directly
# comparable:
#
#   London   the City and the Thames: 2,129 patches, all `dataset == training`;
#            the grid split puts 80% train / 11% val / 8% test.
#   Nairobi  1,570 patches, 59% validation / 41% testing, straddling the line
#            where So2Sat's validation patches (west) give way to its testing
#            patches (east).

WINDOWS <- list(
  list(city = "London",  lonlat = c(-0.0513, 51.518),  size = 7000),
  list(city = "Nairobi", lonlat = c(36.873,  -1.3437), size = 7000)
)

# Fraction of the extent left as margin when a panel frames a whole city.
EXTENT_PAD <- 0.03

SPLIT_LEVELS <- c("Training", "Validation", "Testing")

#' Role of each patch under one split mode.
#'
#' "global" is the published culture-10 design: So2Sat's own `dataset` column is
#' the answer, and a city is wholly training or wholly held out.
#'
#' "orig_test" is `patch_classification.py --orig-test`, transcribed from the
#' mapping in datasets/so2sat.py: testing patches stay test, and *everything
#' else* -- including the validation patches of a held-out city -- is re-split by
#' the per-city grid, with the grid's own test fold folded into train.
patch_role <- function(dataset, split, mode = c("global", "orig_test")) {
  mode <- match.arg(mode)
  role <- if (mode == "global") {
    unname(DATASET_LABELS[dataset])
  } else {
    ifelse(dataset == "testing", "Testing",
           ifelse(split == "val", "Validation", "Training"))
  }
  factor(role, levels = SPLIT_LEVELS)
}

#' All the So2Sat patches of one city.
#'
#' The geometry is the real 320 m patch polygon, not a stand-in: the squares are
#' rotated a degree or two off the UTM axes (a patch's bounding box measures
#' 328.8 x 320.0 m), so they are drawn as polygons rather than rectangles.
read_patches <- function(city) {
  gpkg <- file.path(SO2SAT_CITIES_DIR, city,
                    paste0("patches_reference_", city, "_split.gpkg"))
  if (!file.exists(gpkg)) {
    stop("No split GeoPackage for ", city, ": ", gpkg, call. = FALSE)
  }
  st_read(gpkg, quiet = TRUE)
}

#' A window's centre in the patches' own CRS.
#'
#' `lonlat` is projected here rather than in the window list because the target
#' CRS is whatever UTM zone the city's GeoPackage happens to use.
window_centre <- function(win, crs) {
  if (!is.null(win$lonlat)) {
    xy <- st_coordinates(st_transform(
      st_sfc(st_point(win$lonlat), crs = 4326), crs))
    # Unnamed: panel_window() builds a c(xmin = , xmax = , ...) vector from these
    # and inherited names would mangle every key into "xmin.X".
    return(unname(c(xy[1, 1], xy[1, 2])))
  }
  unname(win$centre)
}

#' The patches whose sample point falls inside one window.
crop_patches <- function(g, centre, size) {
  if (is.null(size)) return(g)
  xy <- st_coordinates(st_centroid(st_geometry(g)))
  h <- size / 2
  keep <- xy[, 1] >= centre[1] - h & xy[, 1] < centre[1] + h &
          xy[, 2] >= centre[2] - h & xy[, 2] < centre[2] + h
  if (!any(keep)) stop("No patches in that window.", call. = FALSE)
  g[keep, ]
}

#' The square window a panel is drawn in, in the city's own CRS.
panel_window <- function(g, centre = NULL, size = NULL, pad = EXTENT_PAD) {
  if (is.null(size)) {
    bb <- st_bbox(g)
    centre <- c((bb[["xmin"]] + bb[["xmax"]]) / 2,
                (bb[["ymin"]] + bb[["ymax"]]) / 2)
    size <- (1 + 2 * pad) * max(bb[["xmax"]] - bb[["xmin"]],
                                bb[["ymax"]] - bb[["ymin"]])
  }
  h <- size / 2
  c(xmin = centre[1] - h, xmax = centre[1] + h,
    ymin = centre[2] - h, ymax = centre[2] + h, size = size)
}

# ── The split grid ────────────────────────────────────────────────────────────
#
# `--orig-test` re-splits a city by *grid cell*, not by patch, so the cells are
# what the second figure is really about. They live in their own file,
# <City>_grid.gpkg: 1,280 m axis-aligned squares carrying the `split` each cell
# was assigned. That file is the authority -- a patch's `split` column is exactly
# its cell's, verified on Nairobi at 1.000 agreement -- so the overlay is read
# from it rather than reconstructed from the patches.
#
# The whole lattice is drawn, not only the cells holding patches (267 of
# Nairobi's 1,485 do): the split is assigned to every cell regardless, and a
# ragged outline of the occupied ones would not read as a grid.

# Mid grey and a touch heavier than the patch outlines: the grid is what decides
# the split under --orig-test, so it has to be legible as structure in its own
# right, while staying light enough not to read as another patch boundary.
GRID_COL <- "grey45"
GRID_LW  <- 0.35

#' The split grid over one window, as a `geom_rect` layer. NULL if the city has
#' no grid file.
grid_layer <- function(city, bb) {
  gpkg <- file.path(SO2SAT_CITIES_DIR, city, paste0(city, "_grid.gpkg"))
  if (!file.exists(gpkg)) {
    warning("No grid GeoPackage for ", city, "; drawing without the grid.",
            call. = FALSE)
    return(NULL)
  }
  win <- st_as_text(st_as_sfc(st_bbox(c(xmin = bb[["xmin"]], ymin = bb[["ymin"]],
                                        xmax = bb[["xmax"]], ymax = bb[["ymax"]]))))
  g <- st_read(gpkg, wkt_filter = win, quiet = TRUE)
  if (!nrow(g)) return(NULL)
  # The cells are axis-aligned squares, so their bounding boxes are the cells.
  cb <- as.data.frame(t(vapply(st_geometry(g), st_bbox, numeric(4))))
  geom_rect(data = cb,
            aes(xmin = xmin, xmax = xmax, ymin = ymin, ymax = ymax),
            inherit.aes = FALSE, fill = NA, colour = GRID_COL,
            linewidth = GRID_LW)
}

# ── Hatching ──────────────────────────────────────────────────────────────────
#
# Validation and Test are both white -- the greyscale of the composition bar --
# so the hatch is the only thing separating them. It is drawn as 45-degree lines
# clipped to the union of the validation patches, the same hand-built approach
# R/plotting.R uses for its bar segment (no pattern fill is available here
# either).

#' Line spacing for a window `size` metres across: wide enough that the hatch
#' reads as distinct lines rather than a grey wash at figure scale (about two
#' lines across a 320 m patch), still bounded so a zoomed-out window does not
#' lose the pattern altogether.
hatch_spacing <- function(size) max(200, size / 110)

hatch_segments <- function(geom, spacing) {
  if (!length(geom) || all(st_is_empty(geom))) return(NULL)
  u  <- st_union(geom)
  bb <- st_bbox(u)
  # Lines of slope 1, indexed by their intercept c in y = x + c. The panel is
  # square and drawn with a fixed 1:1 aspect, so slope 1 reads as 45 degrees on
  # the page. The intercepts have to be taken in y - x, not in y: these are UTM
  # coordinates, so an intercept range built from y alone puts every line
  # millions of metres above the window.
  cs <- seq(bb[["ymin"]] - bb[["xmax"]], bb[["ymax"]] - bb[["xmin"]],
            by = spacing)
  lines <- st_sfc(lapply(cs, function(cc) {
    st_linestring(cbind(c(bb[["xmin"]], bb[["xmax"]]),
                        c(bb[["xmin"]] + cc, bb[["xmax"]] + cc)))
  }), crs = st_crs(geom))
  clipped <- suppressWarnings(st_intersection(lines, u))
  clipped <- clipped[!st_is_empty(clipped)]
  if (!length(clipped)) return(NULL)
  xy <- st_coordinates(st_cast(clipped, "MULTILINESTRING"))
  grp <- if ("L2" %in% colnames(xy)) paste(xy[, "L1"], xy[, "L2"]) else xy[, "L1"]
  # One segment per clipped run: take each run's endpoints. The source columns
  # are deliberately not called x/y -- summarise() evaluates left to right, so
  # `xend = last(x)` after `x = first(x)` would read back the scalar just
  # assigned and collapse every segment to a point.
  data.frame(px = xy[, "X"], py = xy[, "Y"], grp = grp) |>
    group_by(grp) |>
    summarise(x = first(px), y = first(py), xend = last(px), yend = last(py),
              .groups = "drop")
}

# ── Panels ────────────────────────────────────────────────────────────────────

SPLIT_CELL_FILL <- DATASET_COLOURS_BW            # black / white / white

# Outlines are per role, not one colour: on the black training fill a dark
# stroke is invisible and the overlapping patches merge into a single blob, so
# training is outlined in white and the white roles in grey.
PATCH_OUTLINE <- c(Training = "white", Validation = "grey30", Testing = "grey30")
HATCH_COL       <- "grey20"
HATCH_LW        <- 0.18   # heavier to match the wider spacing

#' Outline weight for a window `size` metres across. So2Sat patches are 320 m
#' squares sampled on a *100 m* stride, so they overlap about six deep (New
#' York's polygons sum to 2,085 km2 over a 358 km2 union). Zoomed out that is a
#' virtue -- the outlines read as texture inside each labelled blob -- but the
#' stroke has to thin out as the patches shrink or the blobs go solid grey.
outline_lw <- function(size) max(0.06, min(0.14, 0.14 * 3500 / size))

#' One city panel: So2Sat patches coloured by role, validation hatched.
#'
#' `grid = TRUE` overlays the 1,280 m cells of the per-city grid split -- the
#' thing that decides who is train and who is val under `--orig-test`. It is off
#' by default and means nothing under `mode = "global"`, where the split is by
#' city.
split_panel <- function(win, mode, grid = FALSE) {
  g <- read_patches(win$city)
  centre <- window_centre(win, st_crs(g))
  g <- crop_patches(g, centre, win$size)
  g$role <- patch_role(g$dataset, g$split, mode)
  bb <- panel_window(g, centre, win$size)

  # Patches overlap, so draw order decides what survives: Training first, then
  # Testing, then Validation. The held-out roles are the point of the figure and
  # are the minority everywhere, and a 100 m stride makes every boundary fuzzy
  # by a patch-width anyway.
  ord <- order(as.integer(g$role))
  g   <- g[ord, ]

  xy <- st_coordinates(st_geometry(g))
  d  <- data.frame(x = xy[, "X"], y = xy[, "Y"], grp = xy[, "L2"])
  role_chr <- as.character(g$role)
  d$fill    <- unname(SPLIT_CELL_FILL[role_chr])[d$grp]
  d$outline <- unname(PATCH_OUTLINE[role_chr])[d$grp]
  # geom_polygon draws groups in the order their rows appear, and L2 already
  # follows the reordered geometry, so the sort above is what reaches the page.
  d <- d[order(d$grp), ]

  p <- ggplot() +
    geom_polygon(data = d,
                 aes(x = x, y = y, group = grp, fill = fill, colour = outline),
                 linewidth = outline_lw(bb[["size"]]))

  hs <- hatch_segments(st_geometry(g)[g$role == "Validation"],
                       hatch_spacing(bb[["size"]]))
  if (!is.null(hs)) {
    p <- p + geom_segment(data = hs, aes(x = x, y = y, xend = xend, yend = yend),
                          colour = HATCH_COL, linewidth = HATCH_LW)
  }

  # Over the patches, not under: the point is to show which cell each patch fell
  # in, and a grid drawn underneath would vanish beneath the black ones.
  if (grid) p <- p + grid_layer(win$city, bb)

  # coord_fixed, not coord_sf: everything is already in one projected CRS whose
  # units are metres, the window is square, and coord_sf on this stack is both
  # slow and awkward about hand-placed breaks (see the note in R/lcz_raster.R).
  p +
    scale_fill_identity() +
    scale_colour_identity() +
    coord_fixed(ratio = 1, expand = FALSE,
                xlim = c(bb[["xmin"]], bb[["xmax"]]),
                ylim = c(bb[["ymin"]], bb[["ymax"]])) +
    theme_void(base_size = 11) +
    theme(
      panel.background = element_rect(fill = "white", colour = NA),
      # coord_fixed makes the panel square; the border is what shows it, so it
      # is drawn heavier than a hairline.
      panel.border     = element_rect(fill = NA, colour = "grey20",
                                      linewidth = 0.8),
      plot.background  = element_rect(fill = "transparent", colour = NA),
      plot.margin      = margin(0, 0, 0, 0)
    )
}

# Gutter between the two panels, as a fraction of one panel's width. A spacer
# column rather than plot.margin: the panels are aspect-constrained by
# coord_fixed, so margins are absorbed by the letterboxing instead of pushing
# the panels apart.
PANEL_GAP <- 0.14

#' Two city panels side by side, with nothing else on the page.
split_pair <- function(mode, windows = WINDOWS, gap = PANEL_GAP, grid = FALSE) {
  panels <- lapply(windows, split_panel, mode = mode, grid = grid)
  gaps   <- rep(list(plot_spacer()), length(panels) - 1)
  parts  <- c(rbind(panels, c(gaps, list(NULL))))
  parts  <- parts[!vapply(parts, is.null, logical(1))]
  Reduce(`+`, parts) +
    plot_layout(nrow = 1,
                widths = head(rep(c(1, gap), length(panels)), -1)) &
    theme(plot.background = element_rect(fill = "transparent", colour = NA))
}

# ── Figures ───────────────────────────────────────────────────────────────────

if (sys.nframe() == 0L && !interactive()) {
  message("Figure: dataset split, published culture-10 design")
  save_plot(split_pair("global"), "split_map_global", width = 8, height = 3.9,
            subdir = PLOT_DIR_DATASET)

  message("Figure: dataset split, --orig-test hybrid")
  save_plot(split_pair("orig_test"), "split_map_orig_test",
            width = 8, height = 3.9,
            subdir = PLOT_DIR_DATASET)

  message("Figure: dataset split, --orig-test hybrid, with the split grid")
  save_plot(split_pair("orig_test", grid = TRUE), "split_map_orig_test_grid",
            width = 8, height = 3.9,
            subdir = PLOT_DIR_DATASET)

  message("Done.")
}
