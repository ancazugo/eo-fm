# lcz_raster.R ─ Plot an LCZ GeoTIFF over a lon/lat ROI, on the canonical palette.
#
#     Rscript R/lcz_raster.R --input <file.tif> --bbox <W,S,E,N> --name <stem>
#
# or, as a library:
#
#     source("R/lcz_raster.R")
#     p <- lcz_raster_plot("prediction_London.tif", c(-0.30, 51.40, 0.10, 51.62))
#
# An R port of src/utils/plot_lcz.py, which takes a numpy array rather than a
# path and so can only be called from inside an inference run. Behaviour follows
# that module: the class encoding on disk is 1-17 with 0 = nodata drawn white,
# the legend lists only the classes actually present, and the axes carry exactly
# three lon and three lat labels at the ROI's west/centre/east and
# south/centre/north. Departures from it, all deliberate:
#
#   * a scale bar inside the panel, carrying a cell-size key alongside the
#     distance bar;
#   * an optional GUPPD settlement overlay, in which the settlement the figure
#     is about is drawn dark and the rest recede;
#   * the aspect ratio is geographic (1/cos(lat)) rather than equal-pixel, which
#     matters for the EPSG:4326 reference tifs whose pixels are not square;
#   * no title, caption or axis titles by default -- figures are captioned in
#     the paper, as everywhere else in this stack.

source("R/constants.R")
# The composition marks, and the cell counter that feeds them. Sourcing the
# script is safe: its CLI is guarded by sys.nframe().
source("R/lcz_composition.R")

suppressPackageStartupMessages({
  library(terra)
  library(sf)
  library(ggplot2)
  library(systemfonts)
  # Attached, not namespaced: `ggplot / bar` dispatches on patchwork's own `+`.
  library(patchwork)
})

# Legend labels in the Python style: numeric code, colon, name ("11: Dense
# Trees"). Kept local so constants.R kepes its single label vocabulary
# (alt_code / name / alt_name), which the other figures share.
LCZ_LABELS_CODE  <- paste0(LCZ_TABLE$code, ": ", LCZ_TABLE$name)
LCZ_COLOURS_PY   <- setNames(LCZ_TABLE$colour, LCZ_LABELS_CODE)

# GUPPD settlement footprints, the same file the city map in R/plotting.R draws
# its urban layer from.
GUPPD_GPKG <- file.path(DATA_DIR, "input", "NASA", "GUPPD",
                        "urbanspatial-guppd-v1-gpkg.gpkg")

# Inset-pie diameter, as a fraction of the panel width. A mark on the map, at
# about the scale bar's own size -- not a second figure sitting on it.
PIE_SIZE <- 0.10

# Composition-strip thickness, as a fraction of the map's width. A rule beside
# the map, not a second figure: at the default 7 in width this is about 3.5 mm.
DIST_THICKNESS <- 0.02

# Fill level for nodata. Never shown in the legend; just needs to be a string no
# class label can collide with.
NODATA_KEY <- "0: nodata"

# ── Reading ───────────────────────────────────────────────────────────────────

#' Read an LCZ raster and crop it to a lon/lat ROI.
#'
#' The raster is never reprojected: the ROI is projected into the raster's own
#' CRS instead, so the pixels are drawn exactly as they were written. Prediction
#' tifs are in a local UTM zone and the So2Sat reference tifs in EPSG:4326; both
#' carry `nodata = 0`, which terra turns into NA on read, so nodata needs no
#' special handling here.
#'
#' @param bbox      c(west, south, east, north) in degrees (EPSG:4326).
#' @param max_cells decimation budget. Some rasters are 10 m city-wide (Milan is
#'   11187x8600 = 96M cells), far more than can go through as.data.frame, and
#'   more than any figure can resolve. Above the budget the raster is aggregated
#'   by an integer factor with `modal`, the reducer that suits a categorical
#'   layer -- a nearest-neighbour subsample would drop thin classes entirely.
read_lcz_roi <- function(path, bbox, max_cells = 4e6) {
  if (!file.exists(path)) stop("No such raster: ", path, call. = FALSE)
  if (length(bbox) != 4 || anyNA(bbox)) {
    stop("bbox must be four numbers: west, south, east, north.", call. = FALSE)
  }
  r <- terra::rast(path)

  roi <- sf::st_bbox(c(xmin = bbox[1], ymin = bbox[2],
                       xmax = bbox[3], ymax = bbox[4]), crs = 4326) |>
    sf::st_as_sfc() |>
    sf::st_transform(terra::crs(r)) |>
    terra::vect()

  if (!terra::relate(terra::ext(r), terra::ext(roi), "intersects")) {
    stop("The bbox does not overlap ", basename(path), ". Raster bounds in ",
         "lon/lat are approximately: ",
         paste(round(as.vector(terra::ext(terra::project(
           terra::rast(terra::ext(r), crs = terra::crs(r)), "EPSG:4326"))), 4),
           collapse = ", "), call. = FALSE)
  }
  rc <- terra::crop(r, roi)

  if (terra::ncell(rc) > max_cells) {
    fact <- ceiling(sqrt(terra::ncell(rc) / max_cells))
    message("  ", terra::ncell(rc), " cells exceeds the ", format(max_cells),
            " budget; aggregating by ", fact, "x (modal)")
    rc <- terra::aggregate(rc, fact = fact, fun = "modal", na.rm = TRUE)
  }
  rc
}

# ── Geometry helpers ──────────────────────────────────────────────────────────

#' Metres of ground per unit of the raster's x axis, and the y/x aspect ratio.
#'
#' A projected CRS is already in metres and square, so both are trivial. A
#' geographic one is neither: a degree of longitude shrinks with latitude, so
#' one degree of *latitude* is 1/cos(lat) times longer on the ground than one of
#' longitude, which is exactly the aspect coord_fixed needs. The same mean
#' latitude feeds both numbers, so the scale bar and the aspect cannot disagree.
lcz_scale_info <- function(rc) {
  e <- as.vector(terra::ext(rc))
  if (terra::is.lonlat(rc)) {
    lat <- mean(c(e[["ymin"]], e[["ymax"]]))
    m_per_x <- 111320 * cos(lat * pi / 180)
    list(m_per_x = m_per_x, ratio = 1 / cos(lat * pi / 180))
  } else {
    list(m_per_x = 1, ratio = 1)
  }
}

#' Round `x` down to the nearest 1, 2 or 5 times a power of ten.
nice_down <- function(x) {
  if (!is.finite(x) || x <= 0) return(NA_real_)
  p <- 10^floor(log10(x))
  m <- x / p
  p * if (m >= 5) 5 else if (m >= 2) 2 else 1
}

#' A ground distance as a short label: "100 m", "2 km", "12.5 m".
distance_label <- function(m) {
  if (!is.finite(m) || m <= 0) return(NA_character_)
  v <- if (m >= 1000) m / 1000 else m
  unit <- if (m >= 1000) " km" else " m"
  # signif() then a trailing-zero trim, so 9.999999 reads "10" and 12.53 reads
  # "12.5" -- pixel sizes come out of a projection and are rarely round.
  paste0(format(signif(v, 3), trim = TRUE, scientific = FALSE), unit)
}

#' Label for a cell size given as c(ground x, ground y) in metres.
#'
#' The So2Sat reference tifs are EPSG:4326 with a 320 m *latitude* step and a
#' longitude step chosen independently, so their cells are 414 x 321 m on the
#' ground at Nairobi. Collapsing that to one number would print "414 m" for what
#' everyone calls a 320 m grid, so anisotropic cells are labelled as both sides.
resolution_label <- function(res_m) {
  if (length(res_m) == 1L) return(distance_label(res_m))
  if (isTRUE(all.equal(res_m[[1]], res_m[[2]], tolerance = 0.02))) {
    return(distance_label(mean(res_m)))
  }
  # One unit for the pair, taken from the longer side.
  big <- max(res_m)
  v <- if (big >= 1000) res_m / 1000 else res_m
  unit <- if (big >= 1000) " km" else " m"
  paste0(format(signif(v[[1]], 3), trim = TRUE, scientific = FALSE), "\u00d7",
         format(signif(v[[2]], 3), trim = TRUE, scientific = FALSE), unit)
}

#' Width of a string in x-axis units, for laying out the scale-bar backing box.
#'
#' systemfonts is the shaper ragg draws with, so this is the width the label will
#' actually occupy. `panel_in` is the panel's physical width; save_lcz_raster()
#' derives it from the figure width, and the default matches its own default.
str_x_units <- function(s, pt, w, panel_in) {
  in_ <- systemfonts::string_width(s, size = pt, res = 72) / 72
  in_ * w / panel_in
}

#' Scale bar layers, drawn inside the panel.
#'
#' ggspatial is not installed in this environment, so the bar is built from
#' primitives -- the same route R/plotting.R takes for its pie wedges. It sits
#' on a translucent white backing because the LCZ palette runs to pure black
#' (class E) and dark red (class 1), over which a dark bar and its label would
#' otherwise disappear.
#'
#' Two rows share the backing box: the distance bar with its label above it, and
#' underneath, a resolution key -- a small square and the raster's cell size. The
#' square is a *key*, drawn at a legible fixed size, not a to-scale cell: a 10 m
#' pixel over a city ROI is a quarter of a screen pixel wide and could not be
#' seen. It says "one cell of this raster is 10 m", which is the thing a reader
#' cannot otherwise recover from the figure.
#'
#' @param frac target bar length as a fraction of the panel width, before it is
#'   rounded down to a readable number.
#' @param resolution ground size of one cell as c(x, y) in metres, or NULL for
#'   no key. The key is drawn with that aspect, so an anisotropic grid looks
#'   anisotropic.
#' @param panel_in physical width of the panel in inches, used to convert label
#'   widths into data units so the backing box is sized to its contents.
scalebar_layers <- function(rc, corner = c("br", "bl", "tr", "tl"),
                            frac = 0.25, resolution = NULL, panel_in = 6.5) {
  corner <- match.arg(corner)
  e   <- as.vector(terra::ext(rc))
  inf <- lcz_scale_info(rc)
  w   <- e[["xmax"]] - e[["xmin"]]
  h   <- e[["ymax"]] - e[["ymin"]]
  pt  <- LEGEND_TEXT_PT * 0.9

  len_m <- nice_down(frac * w * inf$m_per_x)
  # A sub-100 m ROI can round down to a bar wider than the panel; no bar beats a
  # wrong one.
  if (is.na(len_m) || len_m / inf$m_per_x > 0.9 * w) return(NULL)
  len_x <- len_m / inf$m_per_x
  label <- distance_label(len_m)

  res_lab <- if (is.null(resolution)) NA_character_ else resolution_label(resolution)

  # ── Box geometry ───────────────────────────────────────────────────────────
  # Inner margins, previously 0.03 w on each side and a 0.105 h box. Tightened
  # to roughly half that; the box is now sized to its contents rather than to a
  # guessed fraction of the panel.
  padx <- 0.012 * w
  pady <- 0.014 * h
  gap  <- 0.006 * h                   # vertical gap between the two rows
  barh <- 0.010 * h                   # bar thickness
  txt  <- 0.030 * h                   # one line of type, in y units

  # The key is sized in x and converted to y through the coord_fixed ratio, so
  # it is square on the page whatever the latitude -- and rectangular, at the
  # true aspect, when the cells themselves are.
  aspect <- if (is.null(resolution) || length(resolution) == 1L) 1 else
    resolution[[2]] / resolution[[1]]
  sq_x <- 0.013 * w / max(1, aspect)
  sq_y <- sq_x * aspect / inf$ratio
  sq_gap_x <- 0.010 * w

  row_h <- if (is.na(res_lab)) 0 else max(sq_y, txt)
  res_w <- if (is.na(res_lab)) 0 else
    sq_x + sq_gap_x + str_x_units(res_lab, pt, w, panel_in)

  # A short bar can be narrower than its own label, and narrower still than the
  # resolution row, so take the widest of the three.
  boxw <- max(len_x, str_x_units(label, pt, w, panel_in), res_w) + 2 * padx
  boxh <- pady * 2 + txt + 0.003 * h + barh +
    if (is.na(res_lab)) 0 else gap + row_h

  right <- corner %in% c("br", "tr")
  top   <- corner %in% c("tr", "tl")
  pad   <- 0.03                       # inset from the panel edge, as a fraction
  box_x <- if (right) e[["xmax"]] - pad * w - boxw else e[["xmin"]] + pad * w
  box_y <- if (top)   e[["ymax"]] - pad * h - boxh else e[["ymin"]] + pad * h

  y <- box_y + pady
  if (!is.na(res_lab)) {
    res_x0 <- box_x + (boxw - res_w) / 2
    sq_y0  <- y + (row_h - sq_y) / 2
    y <- y + row_h + gap
  }
  bar_x <- box_x + (boxw - len_x) / 2
  bar_y <- y

  rect <- function(xmin, xmax, ymin, ymax, ...) {
    geom_rect(data = data.frame(xmin = xmin, xmax = xmax,
                                ymin = ymin, ymax = ymax),
              aes(xmin = xmin, xmax = xmax, ymin = ymin, ymax = ymax),
              inherit.aes = FALSE, ...)
  }
  txt_layer <- function(x, y, lab, hjust = 0.5, vjust = 0) {
    geom_text(data = data.frame(x = x, y = y, lab = lab),
              aes(x = x, y = y, label = lab), inherit.aes = FALSE,
              hjust = hjust, vjust = vjust, colour = "grey15", size = pt / .pt)
  }

  layers <- list(
    rect(box_x, box_x + boxw, box_y, box_y + boxh,
         fill = "white", colour = NA, alpha = 0.75),
    rect(bar_x, bar_x + len_x, bar_y, bar_y + barh,
         fill = "grey15", colour = "white", linewidth = 0.2),
    txt_layer(bar_x + len_x / 2, bar_y + barh + 0.003 * h, label)
  )
  if (!is.na(res_lab)) {
    layers <- c(layers, list(
      rect(res_x0, res_x0 + sq_x, sq_y0, sq_y0 + sq_y,
           fill = "grey15", colour = "white", linewidth = 0.2),
      txt_layer(res_x0 + sq_x + sq_gap_x, box_y + pady + row_h / 2, res_lab,
                hjust = 0, vjust = 0.5)
    ))
  }
  layers
}

# ── GUPPD overlay ─────────────────────────────────────────────────────────────

#' Normalise a settlement name for matching: lowercase, ASCII, no punctuation.
norm_name <- function(x) {
  x <- iconv(x, from = "UTF-8", to = "ASCII//TRANSLIT", sub = " ")
  x <- tolower(x)
  x <- gsub("[^a-z0-9]+", " ", x)
  trimws(x)
}

#' Which GUPPD rows correspond to the settlement named `highlight`?
#'
#' GUPPD carries several name fields and they disagree: over the Nairobi ROI the
#' main agglomeration is JRC_NAME_MAIN "Nairobi", while three smaller entities
#' ("Jomo Kenyatta", "Njiru Town", "Kamulu") carry CIESIN_NAME "Nairobi". All
#' four are Nairobi, so every name field is searched -- including the
#' comma-separated JRC_NAME_LIST -- and every match is highlighted.
guppd_match <- function(g, highlight) {
  target <- norm_name(highlight)
  fields <- intersect(c("JRC_NAME_MAIN", "CIESIN_NAME", "CIESIN_NAME_ADJ"),
                      names(g))
  hit <- Reduce(`|`, lapply(fields, function(f) norm_name(g[[f]]) %in% target),
                init = rep(FALSE, nrow(g)))
  if ("JRC_NAME_LIST" %in% names(g)) {
    hit <- hit | vapply(strsplit(g[["JRC_NAME_LIST"]], ","),
                        function(v) any(norm_name(v) %in% target), logical(1))
  }
  hit
}

#' GUPPD settlement outlines over the ROI, as plain paths.
#'
#' Drawn with geom_path on coordinates pulled out of the geometry rather than
#' with geom_sf, which insists on coord_sf -- and coord_sf is exactly what this
#' figure cannot use (see the note on coord_fixed above). Splitting on L1/L2/L3
#' keeps every ring and every part of a multipolygon a separate path, so holes
#' and detached suburbs are not joined up by a stray segment.
#'
#' A city-sized ROI contains many settlements -- 24 over Nairobi -- and the one
#' the figure is about should not be one outline among 24. The subject is drawn
#' dark and thick; everything else stays visible but recedes to a thin, pale,
#' half-transparent line.
#'
#' @param highlight settlement to emphasise, matched on the GUPPD name fields.
#'   `NULL` emphasises the largest settlement intersecting the ROI, which on a
#'   city ROI is the city; `NA` draws every outline alike.
guppd_layer <- function(rc, highlight = NULL, colour = "grey20",
                        linewidth = 0.45, alpha = 0.9,
                        dim_colour = "grey45", dim_linewidth = 0.22,
                        dim_alpha = 0.5) {
  if (!file.exists(GUPPD_GPKG)) {
    warning("GUPPD gpkg not found at ", GUPPD_GPKG, "; skipping the overlay.",
            call. = FALSE)
    return(NULL)
  }
  e <- as.vector(terra::ext(rc))
  # Filter in the file's own CRS so only the handful of settlements over the ROI
  # are read, not all 123k.
  roi_ll <- sf::st_bbox(c(xmin = e[["xmin"]], ymin = e[["ymin"]],
                          xmax = e[["xmax"]], ymax = e[["ymax"]]),
                        crs = sf::st_crs(terra::crs(rc))) |>
    sf::st_as_sfc() |>
    sf::st_transform(4326)
  g <- sf::st_read(GUPPD_GPKG, layer = "urbanspatial_guppd_v1_polygons",
                   wkt_filter = sf::st_as_text(roi_ll), quiet = TRUE)
  if (!nrow(g)) return(NULL)

  focus <- rep(FALSE, nrow(g))
  if (is.null(highlight)) {
    # No name given: the largest settlement over the ROI. On the city ROIs this
    # figure is made for that is the city itself (Nairobi 335 km2, the next
    # 28 km2), and it is reported so a surprising pick is visible.
    if ("AREA_SQKM" %in% names(g)) {
      focus[which.max(g[["AREA_SQKM"]])] <- TRUE
      message("  guppd: highlighting the largest settlement over the ROI, ",
              g[["JRC_NAME_MAIN"]][which(focus)], " (",
              round(max(g[["AREA_SQKM"]])), " km2); pass --guppd-highlight ",
              "to choose another, or 'none' to draw them alike")
    }
  } else if (!is.na(highlight)) {
    focus <- guppd_match(g, highlight)
    if (!any(focus)) {
      warning("No GUPPD settlement over the ROI is named '", highlight,
              "'; drawing every outline alike. Names present: ",
              paste(unique(g[["JRC_NAME_MAIN"]]), collapse = ", "), call. = FALSE)
    }
  }

  paths <- function(rows, ...) {
    if (!any(rows)) return(NULL)
    xy <- sf::st_geometry(g[rows, ]) |>
      sf::st_transform(terra::crs(rc)) |>
      sf::st_coordinates() |>
      as.data.frame()
    grp <- interaction(xy$L1, xy$L2,
                       xy[[if ("L3" %in% names(xy)) "L3" else "L2"]], drop = TRUE)
    geom_path(data = data.frame(x = xy$X, y = xy$Y, grp = grp),
              aes(x = x, y = y, group = grp), inherit.aes = FALSE, ...)
  }
  # Dimmed first, so the subject's outline wins wherever two settlements touch.
  list(
    paths(!focus, colour = dim_colour, linewidth = dim_linewidth,
          alpha = dim_alpha),
    paths(focus, colour = colour, linewidth = linewidth, alpha = alpha)
  )
}

#' Format a projected coordinate as a lon/lat degree label, Python-style.
#'
#' The breaks are in the raster's own units, so each one is turned back into a
#' point on the panel's mid-line and projected to EPSG:4326 -- taking the middle
#' of the other axis keeps the answer meaningful on a UTM grid, where a meridian
#' is not quite a vertical line.
#'
#' @param digits decimal places. One is enough to separate the three labels on a
#'   city-sized ROI spanning more than ~0.2 degrees; below that two of them round
#'   together and `digits` has to be raised.
degree_labeller <- function(rc, axis = c("x", "y"), digits = 1) {
  axis <- match.arg(axis)
  e <- as.vector(terra::ext(rc))
  function(v) {
    keep <- is.finite(v)
    out  <- rep(NA_character_, length(v))
    if (!any(keep)) return(out)
    xy <- if (axis == "x") {
      cbind(v[keep], mean(c(e[["ymin"]], e[["ymax"]])))
    } else {
      cbind(mean(c(e[["xmin"]], e[["xmax"]])), v[keep])
    }
    ll <- sf::st_coordinates(sf::st_transform(
      sf::st_as_sf(as.data.frame(xy), coords = 1:2, crs = terra::crs(rc)), 4326))
    d <- ll[, if (axis == "x") 1 else 2]
    suffix <- if (axis == "x") ifelse(d >= 0, "E", "W") else ifelse(d >= 0, "N", "S")
    out[keep] <- sprintf("%.*f°%s", digits, abs(d), suffix)
    out
  }
}

# ── Class-distribution marks ──────────────────────────────────────────────────

#' The class mix of the raster being drawn, keyed to the map's own fill scale.
#'
#' The pie has to share the map's single `fill` scale -- ggplot allows one per
#' aesthetic and neither ggnewscale nor a second scale is available here -- so
#' its levels are the map's long "3: Compact Low-Rise" keys, not the short alt
#' codes the standalone figures use. The two always agree: both are counted from
#' the same cropped raster, so the class sets are identical by construction.
map_class_shares <- function(rc, levels_) {
  df <- lcz_counts(rc)
  key <- LCZ_LABELS_CODE[match(LCZ_TABLE$code[match(as.character(df$key),
                                                    LCZ_TABLE$alt_code)],
                               LCZ_TABLE$code)]
  df$key <- factor(key, levels = levels_)
  df
}

#' A pie of the class mix, drawn inside the panel on the scale bar's backing.
#'
#' Placed like the scale bar and painted on the same translucent white box, so
#' the two read as one furniture set rather than two conventions. The wedges are
#' `wedge_arc()` -- the same geometry as the standalone pie and as the 52 pies on
#' the city map -- scaled by the coord_fixed ratio on y so the circle is round on
#' the page whatever the latitude.
#'
#' @param size pie diameter as a fraction of the panel width.
pie_layers <- function(rc, df, corner = c("bl", "br", "tl", "tr"),
                       size = PIE_SIZE) {
  corner <- match.arg(corner)
  e   <- as.vector(terra::ext(rc))
  inf <- lcz_scale_info(rc)
  w   <- e[["xmax"]] - e[["xmin"]]
  h   <- e[["ymax"]] - e[["ymin"]]

  r_x <- size / 2 * w
  r_y <- r_x / inf$ratio
  box <- 1.12                        # backing box, a little wider than the pie
  pad <- 0.03                        # inset from the panel edge, as a fraction

  right <- corner %in% c("br", "tr")
  top   <- corner %in% c("tr", "tl")
  cx <- if (right) e[["xmax"]] - pad * w - box * r_x else e[["xmin"]] + pad * w + box * r_x
  cy <- if (top)   e[["ymax"]] - pad * h - box * r_y else e[["ymin"]] + pad * h + box * r_y

  d <- df |> mutate(a1 = cumsum(share) * 2 * pi, a0 = a1 - share * 2 * pi)
  wedges <- purrr::pmap(d, function(key, a0, a1, ...) {
    wedge_arc(0, 0, 1, a0, a1, 720) |>
      mutate(lcz = key, x = cx + x * r_x, y = cy + y * r_y)
  }) |> purrr::list_rbind() |>
    mutate(grp = as.integer(lcz))

  list(
    geom_rect(data = data.frame(xmin = cx - box * r_x, xmax = cx + box * r_x,
                                ymin = cy - box * r_y, ymax = cy + box * r_y),
              aes(xmin = xmin, xmax = xmax, ymin = ymin, ymax = ymax),
              inherit.aes = FALSE, fill = "white", colour = NA, alpha = 0.75),
    geom_polygon(data = wedges, aes(x = x, y = y, group = grp, fill = lcz),
                 inherit.aes = FALSE, colour = "grey25", linewidth = 0.25)
  )
}

#' Attach the composition strip to one side of a finished map.
#'
#' The strip is a separate plot rather than another layer, because it has to
#' span the map's full width (or height) and nothing in the panel's data space
#' does. patchwork aligns the two panel regions, so the strip lines up with the
#' map rather than with the figure -- which is why the y tick labels do not
#' push it out of register. `BAR_T` is the strip's thickness as a fraction of
#' its length, so its size follows the map's.
#'
#' Returns the combined plot with a `dist_extra` attribute: the inches to add to
#' the figure on the strip's axis.
#'
#' @param thickness strip thickness as a fraction of the map's width, in either
#'   orientation.
attach_dist_bar <- function(p, df, side = c("bottom", "top", "left", "right"),
                            map_w, map_h, thickness = DIST_THICKNESS) {
  side <- match.arg(side)
  horizontal <- side %in% c("bottom", "top")
  # LCZ_COLOURS_PY, not LCZ_COLOURS: `df` is keyed on the map's long class
  # labels, because the inset pie has to share the map's one fill scale.
  bar <- composition_bar(df, LCZ_COLOURS_PY, horizontal = horizontal,
                         labels = FALSE,
                         side = if (horizontal) "bottom" else "left") +
    # Match the map's side margins so the two panels start and end together on
    # the shared axis; patchwork aligns panels, and equal margins keep the
    # strip's own extent from drifting inside its cell.
    theme(plot.margin = if (horizontal) margin(4, 22, 0, 8) else margin(6, 0, 6, 4))
  # An unlabelled strip fills its panel whatever the aspect, so its thickness is
  # simply the inches it is given. Measured against the map's *width* in both
  # orientations, deliberately: a fraction of the edge it happens to span would
  # make the strip on a landscape map's left side thinner than the same strip
  # along its bottom, and the two should look alike.
  thick <- thickness * map_w

  out <- switch(side,
    bottom = p / bar + plot_layout(heights = c(map_h, thick)),
    top    = bar / p + plot_layout(heights = c(thick, map_h)),
    left   = bar | p,
    right  = p | bar
  )
  if (!horizontal) {
    out <- out + plot_layout(widths = if (side == "left") c(thick, map_w)
                             else c(map_w, thick))
  }
  out <- out + transparent_patchwork()
  attr(out, "dist_extra") <- thick
  attr(out, "dist_side") <- side
  out
}

# ── Plot ──────────────────────────────────────────────────────────────────────

#' Plot an LCZ raster over a lon/lat ROI.
#'
#' @param path      GeoTIFF with LCZ classes 1-17 and nodata 0.
#' @param bbox      c(west, south, east, north) in degrees.
#' @param legend    show the class legend. Off by default; when on it lists only
#'                  the classes present, as src/utils/plot_lcz.py does.
#' @param digits    decimal places on the coordinate labels.
#' @param legend_ncol columns in the legend. save_lcz_raster() derives this from
#'                  the figure width, since a class name is about 1.9 in wide and
#'                  four columns overflow anything narrower than ~8 in.
#' @param scalebar  draw the scale bar inside the panel.
#' @param resolution add the cell-size key to the scale bar. The size is read
#'                  off the raster, so it is the source tif's own resolution
#'                  unless read_lcz_roi() had to aggregate, in which case it is
#'                  the aggregated cell actually drawn and that is reported.
#' @param guppd     overlay the GUPPD settlement outlines. Off by default.
#' @param guppd_highlight settlement to draw prominently while the rest are
#'                  dimmed. NULL picks the largest over the ROI; NA draws them
#'                  all alike.
#' @param panel_in  physical width of the panel in inches; only the scale bar's
#'                  backing box depends on it. save_lcz_raster() supplies it.
#' @param title     optional plot title; there is none by default.
#' @param distribution add the raster's class mix to the figure: "pie" insets it
#'   in the panel on the scale bar's backing, "bar" butts a bare composition
#'   strip against one edge at the map's own size, "none" omits it. Neither
#'   carries text -- the map's legend names the classes.
#' @param dist_side edge for the "bar": bottom/top (horizontal) or left/right
#'   (vertical).
#' @param pie_corner corner for the "pie", as scalebar_corner.
#' @param pie_size pie diameter as a fraction of the panel width.
lcz_raster_plot <- function(path, bbox, legend = FALSE, scalebar = TRUE,
                            scalebar_corner = "br", title = NULL,
                            legend_ncol = 4, digits = 1, guppd = FALSE,
                            guppd_highlight = NULL, resolution = TRUE,
                            distribution = c("none", "pie", "bar"),
                            dist_side = "bottom", pie_corner = "bl",
                            pie_size = PIE_SIZE,
                            panel_in = 6.5, max_cells = 4e6) {
  distribution <- match.arg(distribution)
  rc <- read_lcz_roi(path, bbox, max_cells = max_cells)

  # The value column is named after the file, so take it positionally rather
  # than reconstructing a name that would be wrong for every other raster.
  # Nodata cells are kept rather than dropped. They are drawn by the scale's
  # `na.value`, which is the same white the panel is painted, so the result
  # looks identical -- but the grid stays complete. Dropping them leaves whole
  # columns missing on a sparse raster (the Nairobi reference tif is 94% nodata),
  # and geom_raster then warns that the pixels sit at uneven intervals and
  # shifts them.
  d <- terra::as.data.frame(rc, xy = TRUE, na.rm = FALSE)
  names(d)[3] <- "lcz"
  if (all(is.na(d$lcz))) stop("Every cell in the ROI is nodata.", call. = FALSE)

  present <- sort(unique(as.integer(d$lcz)))
  bad <- setdiff(present, LCZ_TABLE$code)
  if (length(bad)) {
    stop("Raster holds values outside LCZ 1-17: ", paste(bad, collapse = ", "),
         ". Model outputs are 0-indexed; on-disk rasters should already be ",
         "1-17 (see infer_roi.py).", call. = FALSE)
  }
  idx <- match(present, LCZ_TABLE$code)
  keys <- LCZ_LABELS_CODE[idx]
  # Nodata becomes a real level rather than staying NA. Both alternatives are
  # worse: na.translate = FALSE drops the cells (warning, and holes in the grid
  # that geom_raster then complains are unevenly spaced), while leaving them NA
  # adds an "NA" key to the legend. A level excluded from the legend by `breaks`
  # draws white and stays out of the key, as plot_lcz.py does.
  d$lcz <- LCZ_LABELS_CODE[match(as.integer(d$lcz), LCZ_TABLE$code)]
  d$lcz[is.na(d$lcz)] <- NODATA_KEY
  d$lcz <- factor(d$lcz, levels = c(keys, NODATA_KEY))

  e   <- as.vector(terra::ext(rc))
  inf <- lcz_scale_info(rc)
  brk <- function(lo, hi) c(lo, (lo + hi) / 2, hi)

  p <- ggplot(d, aes(x = x, y = y, fill = lcz)) +
    geom_raster() +
    scale_fill_manual(values = c(LCZ_COLOURS_PY[idx],
                                 setNames(LCZ_NODATA_COLOUR, NODATA_KEY)),
                      breaks = keys, name = NULL, drop = TRUE)

  # Ground size of one drawn cell. terra::res is in the raster's own units, so
  # it needs the same degrees-to-metres factor the scale bar uses.
  # A degree of longitude is m_per_x metres and a degree of latitude is
  # m_per_x * ratio, which is why the two sides need different factors.
  cell_m <- terra::res(rc) * inf$m_per_x * c(1, inf$ratio)
  if (resolution) {
    native <- terra::res(terra::rast(path)) * inf$m_per_x * c(1, inf$ratio)
    if (!isTRUE(all.equal(native, cell_m))) {
      message("  resolution key shows the aggregated cell (",
              resolution_label(cell_m), "), not the tif's native ",
              resolution_label(native), "; raise --max-cells to draw it natively")
    }
  }

  shares <- if (distribution == "none") NULL else
    map_class_shares(rc, levels(d$lcz))

  # Under the scale bar, so the bar's backing panel still masks it.
  if (guppd)    p <- p + guppd_layer(rc, highlight = guppd_highlight)
  if (distribution == "pie") {
    p <- p + pie_layers(rc, shares, corner = pie_corner, size = pie_size)
  }
  if (scalebar) {
    p <- p + scalebar_layers(rc, corner = scalebar_corner, panel_in = panel_in,
                             resolution = if (resolution) cell_m else NULL)
  }

  p <- p +
    coord_fixed(ratio = inf$ratio, expand = FALSE,
                xlim = e[c("xmin", "xmax")], ylim = e[c("ymin", "ymax")]) +
    # `labels` must be a function, not a character vector: ggplot2 4.x drops
    # breaks that land exactly on the limits before pairing them with a vector,
    # and then rejects the mismatched lengths. A function is applied to whatever
    # breaks survive, so the edge labels come through.
    scale_x_continuous(breaks = brk(e[["xmin"]], e[["xmax"]]),
                       labels = degree_labeller(rc, "x", digits)) +
    scale_y_continuous(breaks = brk(e[["ymin"]], e[["ymax"]]),
                       labels = degree_labeller(rc, "y", digits)) +
    theme_eofm() +
    theme(
      axis.title       = element_blank(),
      panel.grid       = element_blank(),
      # The outer tick labels are centred on the panel edges and so hang past
      # them; without the side margins the easternmost one is cut off by the
      # figure boundary. Python gets away with centring them because it saves
      # with bbox_inches="tight", which grows the canvas instead.
      plot.margin      = margin(6, 22, 6, 8),
      # Painting the panel is what makes nodata white, and it also fills any
      # corner of the ROI the raster does not reach -- far cheaper than
      # materialising millions of white cells.
      panel.background = element_rect(fill = LCZ_NODATA_COLOUR, colour = NA),
      legend.position  = if (legend) "bottom" else "none",
      legend.title     = element_blank(),
      legend.text      = element_text(size = LEGEND_TEXT_PT)
    ) +
    guides(fill = guide_legend(ncol = legend_ncol, byrow = TRUE))

  if (!is.null(title)) p <- p + ggtitle(title)
  aspect <- inf$ratio * (e[["ymax"]] - e[["ymin"]]) / (e[["xmax"]] - e[["xmin"]])
  # Attributes are attached in their own statements: `|>` binds tighter than
  # `+`, and patchwork's `+` would drop them anyway.
  attr(p, "lcz_aspect") <- aspect
  attr(p, "lcz_nclass") <- length(present)
  attr(p, "dist_shares") <- shares
  attr(p, "dist_bar") <- if (distribution == "bar") dist_side else NA_character_
  p
}

#' Render an LCZ ROI to plots/<name>.png.
#'
#' The height follows the fixed panel aspect, so the map fills the figure
#' instead of being letterboxed, with an allowance for the legend rows when
#' those are drawn.
save_lcz_raster <- function(path, bbox, name, width = 7, legend = FALSE,
                            dpi = 400, legend_ncol = NULL,
                            dist_thickness = DIST_THICKNESS, ...) {
  # One legend entry runs to about 1.9 in ("7: Lightweight Low-Rise" at 9.35 pt
  # plus its key), and Python caps the layout at four columns.
  if (is.null(legend_ncol)) legend_ncol <- max(1, min(4, floor(width / 2.0)))
  # The panel is the figure less plot.margin (6 + 22 pt) and the y tick labels
  # and their ticks (~0.42 in for "1.4 deg S"); the scale bar's backing box is
  # sized in inches through this.
  p <- lcz_raster_plot(path, bbox, legend = legend, legend_ncol = legend_ncol,
                       panel_in = max(1, width - 0.81), ...)
  height <- width * attr(p, "lcz_aspect")
  if (legend) {
    rows   <- ceiling(attr(p, "lcz_nclass") / legend_ncol)
    height <- height + rows * 0.18 + 0.1
  }
  # The strip is added last, after the legend allowance, so the map keeps its
  # aspect and the strip only ever grows the figure on its own axis.
  side <- attr(p, "dist_bar")
  if (!is.na(side)) {
    p <- attach_dist_bar(p, attr(p, "dist_shares"), side, width, height,
                         thickness = dist_thickness)
    extra <- attr(p, "dist_extra")
    if (side %in% c("bottom", "top")) height <- height + extra
    else width <- width + extra
  }
  save_plot(p, name, width = width, height = height, dpi = dpi,
            subdir = PLOT_DIR_MAPS)
}

# ── CLI ───────────────────────────────────────────────────────────────────────

if (sys.nframe() == 0L && !interactive()) {
  suppressPackageStartupMessages(library(argparse))
  parser <- ArgumentParser(description = "Plot an LCZ GeoTIFF over a lon/lat ROI.")
  parser$add_argument("--input", required = TRUE, help = "LCZ GeoTIFF (classes 1-17, nodata 0)")
  parser$add_argument("--bbox", required = TRUE,
                      help = "ROI as west,south,east,north in degrees")
  parser$add_argument("--name", required = TRUE, help = "output stem under plots/")
  parser$add_argument("--legend", action = "store_true", help = "show the class legend")
  parser$add_argument("--no-scalebar", action = "store_true", dest = "no_scalebar",
                      help = "omit the scale bar")
  parser$add_argument("--scalebar-corner", default = "br", dest = "scalebar_corner",
                      choices = c("br", "bl", "tr", "tl"))
  parser$add_argument("--title", default = NULL)
  parser$add_argument("--width", type = "double", default = 7)
  parser$add_argument("--dpi", type = "integer", default = 400)
  parser$add_argument("--legend-ncol", type = "integer", default = NULL,
                      dest = "legend_ncol",
                      help = "legend columns (default: derived from --width)")
  parser$add_argument("--guppd", action = "store_true",
                      help = "overlay the GUPPD settlement outlines")
  parser$add_argument("--guppd-highlight", default = NULL, dest = "guppd_highlight",
                      help = paste("settlement to draw prominently while the",
                                   "others are dimmed (default: the largest over",
                                   "the ROI; 'none' draws them all alike)"))
  parser$add_argument("--no-resolution", action = "store_true", dest = "no_resolution",
                      help = "omit the cell-size key from the scale bar")
  parser$add_argument("--distribution", default = "none",
                      choices = c("none", "pie", "bar"),
                      help = "add the raster's class mix to the figure")
  parser$add_argument("--dist-side", default = "bottom", dest = "dist_side",
                      choices = c("bottom", "top", "left", "right"),
                      help = "edge for --distribution bar")
  parser$add_argument("--pie-corner", default = "bl", dest = "pie_corner",
                      choices = c("bl", "br", "tl", "tr"),
                      help = "corner for --distribution pie")
  parser$add_argument("--dist-thickness", type = "double", default = DIST_THICKNESS,
                      dest = "dist_thickness",
                      help = "bar thickness as a fraction of the map's width")
  parser$add_argument("--pie-size", type = "double", default = PIE_SIZE,
                      dest = "pie_size",
                      help = "pie diameter as a fraction of the panel width")
  parser$add_argument("--digits", type = "integer", default = 1,
                      help = "decimal places on the coordinate labels")
  parser$add_argument("--max-cells", type = "double", default = 4e6, dest = "max_cells")
  # argparse reads a value starting with "-" as another flag, so a western
  # bbox ("--bbox -0.30,51.4,...") would be rejected. Glue such a value onto its
  # flag as "--bbox=..." first, which argparse does accept, so both spellings
  # work from the shell.
  argv <- commandArgs(trailingOnly = TRUE)
  glue <- which(argv %in% c("--bbox", "--width", "--dpi", "--max-cells"))
  glue <- glue[glue < length(argv) & grepl("^-", argv[pmin(glue + 1L, length(argv))])]
  if (length(glue)) {
    argv[glue] <- paste0(argv[glue], "=", argv[glue + 1L])
    argv <- argv[-(glue + 1L)]
  }
  args <- parser$parse_args(argv)

  bbox <- as.numeric(strsplit(args$bbox, "[, ]+")[[1]])
  if (length(bbox) != 4 || anyNA(bbox)) {
    stop("--bbox must be four numbers: west,south,east,north", call. = FALSE)
  }
  save_lcz_raster(args$input, bbox, args$name, width = args$width,
                  legend = args$legend, dpi = args$dpi,
                  legend_ncol = args$legend_ncol, digits = args$digits,
                  guppd = args$guppd,
                  guppd_highlight = if (is.null(args$guppd_highlight)) NULL
                                    else if (tolower(args$guppd_highlight) == "none") NA
                                    else args$guppd_highlight,
                  resolution = !args$no_resolution,
                  distribution = args$distribution, dist_side = args$dist_side,
                  dist_thickness = args$dist_thickness,
                  pie_corner = args$pie_corner, pie_size = args$pie_size,
                  scalebar = !args$no_scalebar,
                  scalebar_corner = args$scalebar_corner,
                  title = args$title, max_cells = args$max_cells)
}
