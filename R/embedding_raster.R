# embedding_raster.R ─ An embedding as a picture, with nothing else on the page.
#
#     Rscript R/embedding_raster.R --input <rgb.tif> --name <stem>
#     Rscript R/embedding_raster.R --input <rgb.tif> --window Nairobi --name <stem>
#     Rscript R/embedding_raster.R --mosaic --run GeoTessera_v2 --city Nairobi --name <stem>
#
# or, as a library:
#
#     source("R/embedding_raster.R")
#     p <- embedding_raster_plot("Nairobi_v2_pca.tif", window = "Nairobi")
#
# The pixels are coloured in Python -- R cannot read the embedding sources at
# all (Tessera is int8 .npy plus a separate scales array, AlphaEarth is zarr;
# only src/datasets/tiles.py opens them), so `src/embedding_rgb.py apply` paints
# an ROI into a 4-band uint8 GeoTIFF and this script is the figure end of that
# pipeline. What it adds over the sidecar PNG is the R stack: the split-map
# window, the patch and grid overlays, save_plot(), and the same typography as
# every other figure.
#
# Bare by default -- no axes, ticks, grid, legend, title or margin, just the
# image. `--axes` and `--scalebar` put the map furniture back, both borrowed
# from R/lcz_raster.R rather than rewritten.
#
# TWO WAYS TO COLOUR A PATCH, AND THEY ARE NOT THE SAME PICTURE. `--mosaic`
# draws patch polygons rather than pixels, coloured from a projection parquet:
#   * `--colour rgb_pca` runs each patch's *pooled* vector through the same
#     per-pixel colour model the raster uses, so the two agree -- this is a
#     patch's mean colour, the flat version of the image.
#   * `--colour pca|umap|tsne` uses the parquet's own patch-level basis, a
#     different fit of a different space. Its axes are per-run and comparable to
#     nothing else, including the raster beside it.
# Neither is wrong; drawing one and reading it as the other is.

source("R/constants.R")
# For lcz_scale_info(), degree_labeller() and scalebar_layers(). Sourcing is
# safe: the CLIs of all three are guarded by sys.nframe().
source("R/lcz_raster.R")
# For WINDOWS, read_patches(), window_centre(), panel_window(), grid_layer() --
# the geometry of the black-and-white split maps, so a window here is the same
# window there.
source("R/split_maps.R")
# For list_projections(), resolve_run(), read_projection() and .flag().
source("R/embedding_projection.R")

suppressPackageStartupMessages({
  library(terra)
  library(sf)
  library(ggplot2)
})

# Cell-count budget before the raster is decimated, as in read_lcz_roi().
MAX_CELLS <- 4e6

# Percentiles for the patch-level stretch in mosaic mode. The same pair
# src/embedding_rgb.py fixes its persisted stretch on, so the two modes clip
# alike even though only one of them is comparable across runs.
STRETCH_PCT <- c(2, 98)

# ── Reading ───────────────────────────────────────────────────────────────────

#' Read an RGB(A) raster and crop it to an ROI.
#'
#' The raster is never reprojected -- the ROI is projected into the raster's own
#' CRS instead, exactly as read_lcz_roi() does, so the pixels reach the page as
#' they were written. The one real departure from that function is the reducer:
#' decimation averages, because these are colour channels. `modal` is right for
#' class codes and would be meaningless here.
#'
#' @param roi an `sfc` in any CRS, or NULL for the raster's full extent.
read_rgb_roi <- function(path, roi = NULL, max_cells = MAX_CELLS) {
  if (!file.exists(path)) stop("No such raster: ", path, call. = FALSE)
  r <- terra::rast(path)
  if (terra::nlyr(r) < 3) {
    stop(basename(path), " has ", terra::nlyr(r), " band(s); an RGB raster ",
         "needs 3, or 4 with alpha. Paint one with ",
         "`python src/embedding_rgb.py apply`.", call. = FALSE)
  }

  if (!is.null(roi)) {
    v <- terra::vect(sf::st_transform(roi, terra::crs(r)))
    if (!terra::relate(terra::ext(r), terra::ext(v), "intersects")) {
      stop("The ROI does not overlap ", basename(path), ".", call. = FALSE)
    }
    r <- terra::crop(r, v)
  }
  if (terra::ncell(r) > max_cells) {
    fact <- ceiling(sqrt(terra::ncell(r) / max_cells))
    message("  ", terra::ncell(r), " cells exceeds the ", format(max_cells),
            " budget; aggregating by ", fact, "x (mean)")
    r <- terra::aggregate(r, fact = fact, fun = "mean", na.rm = TRUE)
  }
  r
}

#' A cropped RGB(A) raster as one hex colour per cell.
#'
#' Fully transparent cells are dropped rather than drawn: the panel background
#' shows through, which is both what an alpha of 0 means and far cheaper than
#' materialising them. Bands beyond the fourth are ignored.
rgb_cells <- function(rc) {
  d <- terra::as.data.frame(rc, xy = TRUE, na.rm = FALSE)
  ch <- as.matrix(d[, 3:min(ncol(d), 6), drop = FALSE])
  # A missing channel is missing data, whatever the alpha band says.
  a <- if (ncol(ch) >= 4) ch[, 4] else rep(255, nrow(ch))
  a[!is.finite(a) | !stats::complete.cases(ch[, 1:3, drop = FALSE])] <- 0
  ch[!is.finite(ch)] <- 0
  keep <- a > 0
  if (!any(keep)) stop("Every cell of this ROI is transparent.", call. = FALSE)
  data.frame(
    x = d$x[keep], y = d$y[keep],
    .col = grDevices::rgb(ch[keep, 1], ch[keep, 2], ch[keep, 3], a[keep],
                          maxColorValue = 255)
  )
}

# ── Windows ───────────────────────────────────────────────────────────────────

#' A city's frame: either the split-map window or its whole labelled extent.
#'
#' With `window = TRUE` this is the same square R/split_maps.R draws, so a
#' figure here and the black-and-white split panel show the same ground.
#' Otherwise the city's entire labelled extent, squared and padded, which is
#' what that script's own city-wide panels use.
#'
#' Returns the frame as an `sfc` in the city's own UTM, plus the patches (read
#' for their CRS anyway, and what --patches draws) and the `bb` vector
#' panel_window() returns.
split_window <- function(city, window = TRUE) {
  g <- read_patches(city)
  if (window) {
    i <- which(vapply(WINDOWS, function(w) w$city, "") == city)
    if (!length(i)) {
      stop("No split-map window for '", city, "'. Available: ",
           paste(vapply(WINDOWS, function(w) w$city, ""), collapse = ", "),
           call. = FALSE)
    }
    win <- WINDOWS[[i]]
    centre <- window_centre(win, sf::st_crs(g))
    g <- crop_patches(g, centre, win$size)
    bb <- panel_window(g, centre, win$size)
  } else {
    win <- NULL
    bb <- panel_window(g)
  }
  box <- sf::st_as_sfc(sf::st_bbox(
    c(xmin = bb[["xmin"]], ymin = bb[["ymin"]],
      xmax = bb[["xmax"]], ymax = bb[["ymax"]]), crs = sf::st_crs(g)))
  list(win = win, patches = g, bb = bb, sfc = box, city = city)
}

#' An ROI from --bbox: four numbers, west/south/east/north, in degrees.
bbox_roi <- function(bbox) {
  if (length(bbox) != 4 || anyNA(bbox)) {
    stop("--bbox must be four numbers: west,south,east,north", call. = FALSE)
  }
  sf::st_as_sfc(sf::st_bbox(c(xmin = bbox[1], ymin = bbox[2],
                              xmax = bbox[3], ymax = bbox[4]), crs = 4326))
}

# ── Overlays ──────────────────────────────────────────────────────────────────

# Drawn over the image, in a colour that survives both a dark and a light one.
PATCH_OUTLINE_COL <- "#ffffffcc"
PATCH_OUTLINE_LW  <- 0.3

#' The patches of `city` that fall in an arbitrary frame, for --bbox overlays.
#'
#' --window carries its own patches (cropped by centroid, as the split maps do);
#' this is the other case, where the frame is a bbox and the city has to be named
#' separately. Selection is by envelope overlap rather than by centroid, so a
#' patch straddling the edge is drawn and clipped instead of vanishing.
#'
#' The city GeoPackages and the RGB rasters are all in the city's UTM zone, but
#' that is a fact about these files, not a guarantee: a mismatch is an error
#' here rather than an overlay quietly drawn in the wrong place, because
#' grid_layer() reads its own file and cannot be reprojected after the fact.
frame_patches <- function(city, e, crs) {
  g <- read_patches(city)
  if (sf::st_crs(g) != sf::st_crs(crs)) {
    stop("The patches of ", city, " are in ", sf::st_crs(g)$input,
         " but the raster is in ", sf::st_crs(crs)$input,
         "; reproject the raster to match, or drop --patches/--grid.",
         call. = FALSE)
  }
  bb <- sf::st_bbox(c(xmin = e[["xmin"]], ymin = e[["ymin"]],
                      xmax = e[["xmax"]], ymax = e[["ymax"]]), crs = sf::st_crs(g))
  keep <- lengths(sf::st_intersects(sf::st_geometry(g), sf::st_as_sfc(bb))) > 0
  if (!any(keep)) stop("No ", city, " patch falls in this frame.", call. = FALSE)
  g[keep, ]
}

#' The So2Sat patch outlines over a window, as a `geom_polygon` layer.
#'
#' geom_polygon on extracted coordinates rather than geom_sf, which insists on
#' coord_sf -- and coord_sf is exactly what this figure cannot use (the note in
#' R/lcz_raster.R). The same route R/split_maps.R takes for the patches
#' themselves, so the two figures put their squares in the same places.
patch_layer <- function(g, colour = PATCH_OUTLINE_COL, linewidth = PATCH_OUTLINE_LW) {
  if (!nrow(g)) return(NULL)
  xy <- sf::st_coordinates(sf::st_geometry(g))
  d <- data.frame(x = xy[, "X"], y = xy[, "Y"], grp = xy[, "L2"])
  geom_polygon(data = d, aes(x = x, y = y, group = grp),
               inherit.aes = FALSE, fill = NA, colour = colour,
               linewidth = linewidth)
}

# ── The panel ─────────────────────────────────────────────────────────────────

#' One bare panel: the marks, the fixed aspect, and no furniture.
#'
#' `rc` is only needed for the aspect and the axis labels, so mosaic mode passes
#' the window's CRS and extent instead of a raster.
bare_panel <- function(layers, e, ratio, crs_for_labels = NULL, rc = NULL,
                       axes = FALSE, scalebar = FALSE, digits = 1,
                       panel_in = 6.5, resolution = TRUE) {
  brk <- function(lo, hi) c(lo, (lo + hi) / 2, hi)
  p <- ggplot()
  for (l in layers) if (!is.null(l)) p <- p + l

  if (scalebar) {
    if (is.null(rc)) {
      warning("--scalebar needs a raster; mosaic mode has no cell size.",
              call. = FALSE)
    } else {
      cell <- terra::res(rc) * lcz_scale_info(rc)$m_per_x
      p <- p + scalebar_layers(rc, corner = "br", panel_in = panel_in,
                               resolution = if (resolution) cell else NULL)
    }
  }

  p <- p +
    scale_fill_identity() +
    scale_colour_identity() +
    coord_fixed(ratio = ratio, expand = FALSE,
                xlim = e[c("xmin", "xmax")], ylim = e[c("ymin", "ymax")])

  p <- if (axes) {
    p +
      scale_x_continuous(breaks = brk(e[["xmin"]], e[["xmax"]]),
                         labels = degree_labeller(rc, "x", digits)) +
      scale_y_continuous(breaks = brk(e[["ymin"]], e[["ymax"]]),
                         labels = degree_labeller(rc, "y", digits)) +
      theme_eofm() +
      theme(axis.title = element_blank(), panel.grid = element_blank(),
            plot.margin = margin(6, 22, 6, 8),
            panel.background = element_rect(fill = "transparent", colour = NA),
            legend.position = "none")
  } else {
    # Just the image: theme_void leaves nothing but the panel, and the zero
    # margin means the figure IS the panel. Transparent rather than white, so
    # an alpha-0 region reads as absent instead of as a colour.
    p +
      theme_void() +
      theme(plot.margin = margin(0, 0, 0, 0),
            panel.background = element_rect(fill = "transparent", colour = NA),
            plot.background = element_rect(fill = "transparent", colour = NA),
            legend.position = "none")
  }

  attr(p, "eo_aspect") <- ratio * (e[["ymax"]] - e[["ymin"]]) /
                                  (e[["xmax"]] - e[["xmin"]])
  p
}

#' Draw an RGB GeoTIFF over an ROI.
#'
#' @param window a city name from R/split_maps.R's WINDOWS, framing the figure
#'   exactly as the split map's panel for that city. Overrides `bbox`.
#' @param patches,grid overlay the So2Sat patch polygons and the 1280 m split
#'   grid. Both need a `window` (or a `city`), since both are read per city.
embedding_raster_plot <- function(path, bbox = NULL, window = NULL, city = NULL,
                                  patches = FALSE, grid = FALSE,
                                  axes = FALSE, scalebar = FALSE, digits = 1,
                                  panel_in = 6.5, max_cells = MAX_CELLS) {
  w <- if (!is.null(window)) split_window(window) else NULL
  roi <- if (!is.null(w)) w$sfc else if (!is.null(bbox)) bbox_roi(bbox) else NULL
  if (patches || grid) {
    if (is.null(w) && is.null(city)) {
      stop("--patches and --grid need --window <city> or --city <city>: the ",
           "patch and grid outlines are read per city.", call. = FALSE)
    }
  }

  rc <- read_rgb_roi(path, roi, max_cells = max_cells)
  # The window is authoritative when there is one: the raster may not reach all
  # of it (the shipped Nairobi tif stops short of the window's southern edge),
  # and cropping the frame to the data would silently move the panel.
  e <- if (!is.null(w)) w$bb[c("xmin", "xmax", "ymin", "ymax")]
       else as.vector(terra::ext(rc))
  inf <- lcz_scale_info(rc)

  # Overlay frame and source: --window brings both, --bbox names the city and
  # takes the frame from the raster.
  ov_city <- if (!is.null(w)) w$city else city
  ov_bb <- if (!is.null(w)) w$bb else e
  ov_patches <- if (!(patches || grid)) NULL
                else if (!is.null(w)) w$patches
                else frame_patches(city, e, terra::crs(rc))

  layers <- list(
    geom_raster(data = rgb_cells(rc), aes(x = x, y = y, fill = .col)),
    if (grid) grid_layer(ov_city, ov_bb) else NULL,
    if (patches) patch_layer(ov_patches) else NULL
  )
  code <- terra::crs(rc, describe = TRUE)$code
  message("  ", basename(path), ": ", terra::nlyr(rc), " bands, ",
          terra::nrow(rc), "x", terra::ncol(rc), " cells, EPSG:",
          if (is.na(code)) "?" else code)
  bare_panel(layers, e, inf$ratio, rc = rc, axes = axes, scalebar = scalebar,
             digits = digits, panel_in = panel_in)
}

# ── Mosaic mode ───────────────────────────────────────────────────────────────

#' Colour columns for one `--colour` choice, and whether they are already 0-255.
mosaic_columns <- function(colour) {
  if (grepl("^rgb_", colour)) {
    list(cols = paste0(colour, "_", c("r", "g", "b")), scaled = TRUE)
  } else {
    list(cols = paste0(colour, "_", 1:3), scaled = FALSE)
  }
}

#' So2Sat patches of one city, coloured from a projection run.
#'
#' The join is on `uid` -- "<dataset>/<patch_id>" -- because `patch_id` restarts
#' at 000000 in each of training/validation/testing and is not unique on its own.
mosaic_plot <- function(run = NULL, city, colour = "rgb_pca", window = NULL,
                        axes = FALSE, digits = 1) {
  r <- resolve_run(list_projections(), run)
  spec <- mosaic_columns(colour)
  have <- names(arrow::open_dataset(r$full))
  miss <- setdiff(spec$cols, have)
  if (length(miss)) {
    stop("run '", r$run, "' has no ", paste(miss, collapse = ", "),
         if (spec$scaled) paste0(". Write them with `python src/embedding_rgb.py ",
                                 "annotate-parquet`.") else ".", call. = FALSE)
  }
  d <- read_projection(r$full, columns = c("uid", "city", spec$cols))
  d <- d[d$city == city, ]
  if (!nrow(d)) stop("No rows for city '", city, "' in ", r$run, call. = FALSE)

  m <- as.matrix(d[, spec$cols])
  if (spec$scaled) {
    col <- grDevices::rgb(m[, 1], m[, 2], m[, 3], maxColorValue = 255)
  } else {
    # A per-figure stretch over the rows drawn, so these colours mean nothing
    # outside this figure -- said again in the message, because the two --colour
    # modes look alike on the page and are not alike at all.
    q <- apply(m, 2, stats::quantile, probs = STRETCH_PCT / 100, na.rm = TRUE)
    z <- sweep(sweep(m, 2, q[1, ], "-"), 2, pmax(q[2, ] - q[1, ], 1e-12), "/")
    z[!is.finite(z)] <- 0
    z <- pmin(pmax(z, 0), 1)
    col <- grDevices::rgb(z[, 1], z[, 2], z[, 3])
    message("  ", colour, ": patch-level basis, stretched over these ",
            format(nrow(d), big.mark = ","), " patches only -- ",
            "not comparable to the raster or to another run")
  }
  d$.col <- col

  # --city alone frames the whole city; --window narrows to the split-map square.
  w <- split_window(if (is.null(window)) city else window,
                    window = !is.null(window))
  g <- w$patches
  g$.col <- d$.col[match(paste0(g$dataset, "/", g$patch_id), d$uid)]
  n_na <- sum(is.na(g$.col))
  if (n_na) {
    message("  ", n_na, " of ", nrow(g), " patches are not in the run; ",
            "left undrawn")
    g <- g[!is.na(g$.col), ]
  }
  if (!nrow(g)) stop("No patch in the window is in run ", r$run, call. = FALSE)

  xy <- sf::st_coordinates(sf::st_geometry(g))
  dd <- data.frame(x = xy[, "X"], y = xy[, "Y"], grp = xy[, "L2"])
  dd$fill <- g$.col[dd$grp]

  e <- w$bb[c("xmin", "xmax", "ymin", "ymax")]
  bare_panel(list(geom_polygon(data = dd,
                               aes(x = x, y = y, group = grp, fill = fill))),
             e, ratio = 1, axes = axes, digits = digits)
}

# ── Saving ────────────────────────────────────────────────────────────────────

#' Render to plots/embeddings/<name>.png, at the panel's own aspect.
save_embedding_raster <- function(p, name, width = 6, dpi = 400) {
  save_plot(p, name, width = width, height = width * attr(p, "eo_aspect"),
            dpi = dpi, subdir = PLOT_DIR_EMBEDDINGS)
}

# ── CLI ───────────────────────────────────────────────────────────────────────

if (sys.nframe() == 0L && !interactive()) {
  suppressPackageStartupMessages(library(argparse))
  parser <- ArgumentParser(description = "Draw an embedding as a bare image.")
  parser$add_argument("--input", default = NULL,
                      help = "RGB(A) GeoTIFF from `embedding_rgb.py apply`")
  parser$add_argument("--name", required = TRUE, help = "output stem under plots/")
  parser$add_argument("--bbox", default = NULL,
                      help = "ROI as west,south,east,north in degrees")
  parser$add_argument("--window", default = NULL,
                      help = "frame on a split-map city window (London, Nairobi)")
  parser$add_argument("--patches", action = "store_true",
                      help = "outline the So2Sat patches (needs --window)")
  parser$add_argument("--grid", action = "store_true",
                      help = "outline the 1280 m split grid (needs --window)")
  parser$add_argument("--axes", action = "store_true",
                      help = "draw coordinate labels instead of a bare image")
  parser$add_argument("--scalebar", action = "store_true")
  parser$add_argument("--mosaic", action = "store_true",
                      help = "draw patch polygons from a projection run, not pixels")
  parser$add_argument("--run", default = NULL, help = "projection run (substring)")
  parser$add_argument("--city", default = NULL,
                      help = "city supplying --patches/--grid with --bbox; also the city for --mosaic")
  parser$add_argument("--colour", default = "rgb_pca",
                      help = "rgb_pca (pixel model on pooled vectors) or pca|umap|tsne")
  parser$add_argument("--width", type = "double", default = 6)
  parser$add_argument("--dpi", type = "integer", default = 400)
  parser$add_argument("--digits", type = "integer", default = 1)
  parser$add_argument("--max-cells", type = "double", default = MAX_CELLS,
                      dest = "max_cells")
  # argparse reads a value starting with "-" as another flag, so a western bbox
  # would be rejected; glue such a value onto its flag first (as R/lcz_raster.R).
  argv <- commandArgs(trailingOnly = TRUE)
  glue <- which(argv %in% c("--bbox", "--width", "--dpi", "--max-cells"))
  glue <- glue[glue < length(argv) & grepl("^-", argv[pmin(glue + 1L, length(argv))])]
  if (length(glue)) {
    argv[glue] <- paste0(argv[glue], "=", argv[glue + 1L])
    argv <- argv[-(glue + 1L)]
  }
  args <- parser$parse_args(argv)

  p <- if (args$mosaic) {
    if (is.null(args$city)) stop("--mosaic needs --city", call. = FALSE)
    mosaic_plot(args$run, args$city, colour = args$colour,
                window = args$window, axes = args$axes, digits = args$digits)
  } else {
    if (is.null(args$input)) stop("--input is required (or use --mosaic)", call. = FALSE)
    embedding_raster_plot(
      args$input,
      bbox = if (is.null(args$bbox)) NULL
             else as.numeric(strsplit(args$bbox, "[, ]+")[[1]]),
      window = args$window, city = args$city,
      patches = args$patches, grid = args$grid,
      axes = args$axes, scalebar = args$scalebar, digits = args$digits,
      panel_in = max(1, args$width - if (args$axes) 0.81 else 0),
      max_cells = args$max_cells)
  }
  save_embedding_raster(p, args$name, width = args$width, dpi = args$dpi)
}
