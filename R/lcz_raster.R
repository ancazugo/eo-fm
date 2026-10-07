# lcz_raster.R ─ Plot an LCZ GeoTIFF over a lon/lat ROI, on the canonical palette.
#
#     Rscript R/lcz_raster.R --input <file.tif> --bbox <W,S,E,N> --name <stem>
#         [--distribution bar --dist-side bottom --x-axis top]
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

# Where maptiles keeps the basemap tiles it downloads. A session-local default,
# so nothing is written into the repo or into $DATA_DIR (which has run full
# before); set EOFM_TILE_CACHE to a real directory to keep them between runs.
TILE_CACHE <- Sys.getenv("EOFM_TILE_CACHE",
                         unset = file.path(tempdir(), "maptiles"))

# Tile servers maptiles does not ship. Google publishes no documented tile API,
# so these are the `mt{1..3}.google.com` endpoints its own web client uses --
# they work without a key, and their terms of use are Google's, not OSM's, so
# check them before a figure goes anywhere public. `lyrs` picks the layer:
# s = satellite, y = hybrid (satellite + labels), m = roads, p = terrain.
EXTRA_PROVIDERS <- list(
  "Google.Satellite" = list(lyrs = "s", cit = "Imagery \u00a9 Google"),
  "Google.Hybrid"    = list(lyrs = "y", cit = "Imagery \u00a9 Google"),
  "Google.Roads"     = list(lyrs = "m", cit = "Map data \u00a9 Google"),
  "Google.Terrain"   = list(lyrs = "p", cit = "Map data \u00a9 Google")
)

# Providers whose URL carries an {apikey}. Stadia hosts the Stamen designs
# (Toner, Terrain, Watercolor) since Stamen retired its own servers in 2023 and
# answers 401 without a key; a free key covers this kind of use. Set
# STADIA_API_KEY (or pass --basemap-apikey) and the Stamen styles work like any
# other provider.
# A list, not a named vector: `[[` on an absent name must return NULL (most
# providers need no key), and a character vector errors instead.
# The first name in each entry is this repo's; the rest are the ones maptiles
# reads by itself, so an environment already set up for it keeps working.
API_KEY_ENV <- list(
  Stadia           = c("STADIA_API_KEY", "STADIA_MAPS_API_KEY", "STADIA_MAPS"),
  Thunderforest    = c("THUNDERFOREST_API_KEY", "THUNDERFOREST_MAPS"),
  Jawg             = "JAWG_API_KEY",
  MapBox           = "MAPBOX_API_KEY",
  OpenWeatherMap   = "OPENWEATHERMAP_API_KEY",
  HERE             = "HERE_API_KEY",
  GeoportailFrance = "GEOPORTAIL_API_KEY")

#' Resolve a provider name to whatever get_tiles() needs.
#'
#' A maptiles name passes straight through; one of the EXTRA_PROVIDERS is built
#' on the spot with create_provider().
resolve_provider <- function(name) {
  if (!is.null(EXTRA_PROVIDERS[[name]])) {
    e <- EXTRA_PROVIDERS[[name]]
    return(maptiles::create_provider(
      name = name,
      # The trailing "#.jpg" is a URL fragment -- never sent to the server --
      # and is there only so maptiles can name the cache file. Its
      # get_extension() greps the URL for an image suffix and, finding none,
      # falls through without assigning, so `return(ext)` picks up terra::ext
      # and the download dies with "cannot coerce type 'closure'". Google's
      # tile endpoint carries no suffix of its own.
      url = paste0("https://mt{s}.google.com/vt/lyrs=", e$lyrs,
                   "&x={x}&y={y}&z={z}#.jpg"),
      sub = c("1", "2", "3"), citation = e$cit))
  }
  name
}

#' The API key for a provider, from the argument or the environment.
#'
#' Returns "" when none is needed or none is set -- get_tiles() only reads it
#' for a provider whose URL carries {apikey}, so an empty string is harmless
#' everywhere else.
resolve_apikey <- function(name, apikey = NULL) {
  if (!is.null(apikey) && nzchar(apikey)) return(apikey)
  vars <- API_KEY_ENV[[sub("\\..*$", "", name)]]
  if (is.null(vars)) return("")
  set <- Sys.getenv(vars, unset = "")
  set <- set[nzchar(set)]
  if (length(set)) return(set[[1]])
  # Nothing in the environment, so try the repo's .env. R reads .Renviron and
  # never .env, but .env is where this project's secrets already live (the
  # Python side reads it), and a key kept in one place cannot drift from the
  # other.
  for (v in vars) {
    val <- dotenv_value(v)
    if (nzchar(val)) return(val)
  }
  ""
}

#' One value from the repo's .env, or "" if it is not there.
#'
#' A deliberately small parser: KEY=VALUE lines, `#` comments, optional
#' surrounding quotes, no interpolation and no export of anything into the
#' session. Nothing else in this stack reads .env, so it stays local rather
#' than becoming a second environment.
dotenv_value <- function(name, path = ".env") {
  if (!file.exists(path)) return("")
  lines <- readLines(path, warn = FALSE)
  lines <- lines[!grepl("^\\s*(#|$)", lines)]
  hit <- grep(paste0("^\\s*(export\\s+)?", name, "\\s*="), lines, value = TRUE)
  if (!length(hit)) return("")
  val <- sub("^[^=]*=", "", hit[[1]])
  trimws(gsub("^['\"]|['\"]$", "", trimws(val)))
}

# Composition-strip thickness, as a fraction of the map's width. A rule beside
# the map, not a second figure: at the default 7 in width this is about 3.5 mm.
DIST_THICKNESS <- 0.02

# Fill level for nodata. Never shown in the legend; just needs to be a string no
# class label can collide with.
NODATA_KEY <- "0: nodata"

# ── Reading ───────────────────────────────────────────────────────────────────

#' Merge several LCZ tiles that share a CRS but not a grid.
#'
#' The Demuzere et al. global map ships as 0.5-degree tiles in per-region UTM
#' zones, and adjacent tiles are NOT on a common grid: `lcz_36.5_-1.5` and
#' `lcz_37.0_-1.5` are both EPSG:32737 at 100 m but their origins differ by
#' 62 m in x and 43 m in y. `merge()` and `mosaic()` both require alignment, so
#' each tile is resampled onto one template first -- nearest neighbour, the only
#' resampling a class raster admits -- built on the FIRST tile's grid so that
#' tile survives untouched and the shift lands on its neighbours.
#'
#' Tiles from different UTM zones are refused rather than silently reprojected:
#' one of them would have to be resampled twice, and a class map does not
#' survive that. Crop to a ROI inside one zone, or pass a pre-built mosaic.
mosaic_tiles <- function(paths) {
  rs <- lapply(paths, terra::rast)
  crss <- vapply(rs, function(x) terra::crs(x, describe = TRUE)$code, character(1))
  if (length(unique(crss)) > 1L) {
    stop("These tiles are in different CRSs (", paste(unique(crss), collapse = ", "),
         "); merging them would mean resampling a class raster twice.",
         call. = FALSE)
  }
  e <- Reduce(terra::union, lapply(rs, terra::ext))
  template <- terra::rast(terra::align(e, rs[[1]]),
                          resolution = terra::res(rs[[1]]),
                          crs = terra::crs(rs[[1]]))
  parts <- lapply(rs, function(x) terra::resample(x, template, method = "near"))
  out <- Reduce(function(a, b) terra::merge(a, b), parts)
  message("  merged ", length(paths), " tiles into ", terra::ncell(out),
          " cells on ", basename(paths[[1]]), "'s grid")
  seam_fill(out)
}

#' Fill the hairline of nodata left along a tile join.
#'
#' Adjacent Demuzere tiles do not abut: 36.5/-1.5 ends at x = 277418.5 and
#' 37.0/-1.5 begins at 277480.4, a 62 m gap, so a merged mosaic carries a
#' one-cell nodata line straight down the join -- a white scratch across the
#' map that is an artefact of the tiling, not a statement about the ground.
#'
#' Only cells with at least six of their eight neighbours classified are
#' filled, and with the neighbourhood's modal class. A seam cell has seven or
#' eight; a cell on the mosaic's outer edge has five at most, and real nodata
#' inside a tile comes in blocks. So this repairs the join and cannot spread
#' into either kind of genuine gap.
seam_fill <- function(r, min_neighbours = 6) {
  gap <- is.na(r)
  if (!any(terra::values(gap), na.rm = TRUE)) return(r)
  n <- terra::focal(!gap, w = 3, fun = "sum", na.rm = TRUE)
  fill <- gap & n >= min_neighbours
  k <- sum(terra::values(fill), na.rm = TRUE)
  if (!k) return(r)
  out <- terra::ifel(fill, terra::focal(r, w = 3, fun = "modal", na.rm = TRUE), r)
  message("  filled ", k, " nodata cells on the tile seam (modal of 3x3)")
  out
}

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
  missing_ <- path[!file.exists(path)]
  if (length(missing_)) {
    stop("No such raster: ", paste(missing_, collapse = ", "), call. = FALSE)
  }
  if (length(bbox) != 4 || anyNA(bbox)) {
    stop("bbox must be four numbers: west, south, east, north.", call. = FALSE)
  }
  r <- if (length(path) == 1L) terra::rast(path) else mosaic_tiles(path)

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

# ── Basemap ─────────────────────────────────────────────────────────────

#' Web-map tiles under the panel, as a single annotation raster.
#'
#' `annotation_raster`, not `geom_raster`: a ggplot has exactly one `fill` scale
#' and the map has already spent it on the LCZ classes (the same constraint the
#' inset pie runs into). An annotation carries its own colours and takes no
#' scale at all, which is what lets a full-colour basemap sit under a
#' categorical layer here.
#'
#' `get_tiles()` returns the mosaic in the CRS of whatever it is handed, so
#' passing the cropped raster puts the tiles on the map's own coordinates -- no
#' reprojection of either, and the annotation's corners are the tile mosaic's
#' own extent rather than the ROI's (they differ by up to one tile pixel).
#'
#' Requires network access. A failed fetch is a warning and a basemap-less map,
#' never an error: the map is the figure, the tiles are backdrop.
#'
#' @param provider any maptiles provider name ("OpenStreetMap",
#'   "CartoDB.Positron", "Esri.WorldImagery", ...).
#' @param zoom tile zoom level, or NULL to let maptiles pick from the extent.
#' @param alpha opacity of the backdrop. Below 1 the panel background shows
#'   through and lightens it, which is one way to keep the class colours on top
#'   reading as the subject.
basemap_layer <- function(rc, provider = "OpenStreetMap", zoom = NULL,
                          alpha = 1, cachedir = TILE_CACHE, apikey = NULL) {
  if (!requireNamespace("maptiles", quietly = TRUE)) {
    warning("maptiles is not installed; skipping the basemap.", call. = FALSE)
    return(NULL)
  }
  key <- resolve_apikey(provider, apikey)
  dir.create(cachedir, recursive = TRUE, showWarnings = FALSE)
  # `zoom` is passed only when it was asked for: get_tiles() derives it from the
  # extent when the argument is MISSING, and an explicit NULL is not missing --
  # it reaches the tile arithmetic and dies with "argument of length 0".
  call_args <- list(x = rc, provider = resolve_provider(provider), crop = TRUE,
                    cachedir = cachedir, apikey = key)
  if (!is.null(zoom)) call_args$zoom <- zoom
  tl <- try(do.call(maptiles::get_tiles, call_args), silent = TRUE)
  if (inherits(tl, "try-error") || is.null(tl)) {
    needs_key <- !nzchar(key) && !is.null(API_KEY_ENV[[sub("\\..*$", "", provider)]])
    warning("Could not fetch ", provider, " tiles; drawing without a basemap. ",
            if (needs_key) paste0(provider, " needs an API key: set $",
                                  API_KEY_ENV[[sub("\\..*$", "", provider)]][[1]],
                                  " or pass --basemap-apikey. "),
            if (inherits(tl, "try-error")) conditionMessage(attr(tl, "condition")),
            call. = FALSE)
    return(NULL)
  }
  a <- terra::as.array(tl)
  if (dim(a)[[3]] < 3L) {
    warning("Basemap came back with ", dim(a)[[3]], " band(s); expected RGB.",
            call. = FALSE)
    return(NULL)
  }
  # Tiles are uint8 and should not contain NA, but a mosaic that reaches past
  # the provider's coverage can; white is the colour of an absent tile.
  a[!is.finite(a)] <- 255
  cols <- grDevices::rgb(a[, , 1], a[, , 2], a[, , 3],
                         alpha = round(255 * alpha), maxColorValue = 255)
  m <- matrix(cols, nrow = nrow(a), ncol = ncol(a))
  e <- as.vector(terra::ext(tl))
  annotation_raster(m, xmin = e[["xmin"]], xmax = e[["xmax"]],
                    ymin = e[["ymin"]], ymax = e[["ymax"]], interpolate = TRUE)
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

#' LCZ polygons over the ROI (e.g. the So2Sat patch GeoPackages), for drawing
#' in place of the raster.
#'
#' A rasterised label file puts the patches on a grid of its own, which is not
#' the patches' grid: the So2Sat reference tifs have 0.0037 deg cells for
#' 0.0029 deg patches, so every drawn patch is resampled. The polygons are the
#' labels as published. They are read with a bbox filter, moved into the
#' raster's CRS and split into rings exactly as guppd_layer does (geom_polygon,
#' not geom_sf, because the figure is on coord_fixed).
#'
#' @return list(df = ring coordinates with an `lcz` code per ring,
#'   shares = the class mix in lcz_counts' format, weighted by polygon area).
read_lcz_vector <- function(path, rc, field = "LCZ_class") {
  if (!file.exists(path)) stop("No such vector file: ", path, call. = FALSE)
  e <- as.vector(terra::ext(rc))
  roi <- sf::st_bbox(c(xmin = e[["xmin"]], ymin = e[["ymin"]],
                       xmax = e[["xmax"]], ymax = e[["ymax"]]),
                     crs = sf::st_crs(terra::crs(rc))) |>
    sf::st_as_sfc()
  roi_ll <- sf::st_transform(roi, 4326)
  v <- sf::st_read(path, wkt_filter = sf::st_as_text(roi_ll), quiet = TRUE)
  if (!nrow(v)) stop("No polygons in ", basename(path), " over the bbox.",
                     call. = FALSE)
  if (!field %in% names(v)) {
    stop("Field '", field, "' not in ", basename(path), "; columns are ",
         paste(setdiff(names(v), attr(v, "sf_column")), collapse = ", "),
         call. = FALSE)
  }
  v <- sf::st_transform(v, sf::st_crs(terra::crs(rc)))
  code <- as.integer(v[[field]])
  bad <- setdiff(code, LCZ_TABLE$code)
  if (length(bad)) {
    stop("Vector holds values outside LCZ 1-17: ", paste(bad, collapse = ", "),
         call. = FALSE)
  }

  xy <- as.data.frame(sf::st_coordinates(sf::st_geometry(v)))
  # The last L column indexes the feature; the ones before it the ring/part.
  lcols <- grep("^L[0-9]$", names(xy), value = TRUE)
  feat <- xy[[lcols[length(lcols)]]]
  df <- data.frame(x = xy$X, y = xy$Y, lcz = code[feat],
                   grp = interaction(xy[lcols], drop = TRUE))

  # Area-weighted, clipped to the ROI, so the class-mix bar describes what the
  # panel shows (a patch half outside the bbox counts half).
  area <- as.numeric(sf::st_area(sf::st_intersection(
    sf::st_geometry(v), sf::st_geometry(roi))))
  if (length(area) != nrow(v)) {
    # st_intersection drops polygons that only touch the ROI edge.
    area <- vapply(seq_len(nrow(v)), function(i) {
      a <- sf::st_area(sf::st_intersection(sf::st_geometry(v)[i], roi))
      if (length(a)) as.numeric(a) else 0
    }, numeric(1))
  }
  agg <- tapply(area, code, sum)
  shares <- tibble::tibble(code = as.integer(names(agg)), n = as.numeric(agg)) |>
    dplyr::filter(n > 0) |>
    dplyr::arrange(match(code, LCZ_TABLE$code)) |>
    dplyr::mutate(key = factor(LCZ_TABLE$alt_code[match(code, LCZ_TABLE$code)],
                               levels = LCZ_TABLE$alt_code),
                  share = n / sum(n)) |>
    dplyr::select(key, share, n)
  message("  vector: ", nrow(v), " polygons from ", basename(path))
  list(df = df, shares = shares)
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
    # Breaks closer together than 10^-digits degrees would print the same
    # label twice (a 0.15 deg ROI gave 36.8E, 36.8E, 36.9E), so add decimals
    # until the labels are as distinct as the breaks are.
    dg <- digits
    repeat {
      labs <- sprintf("%.*f°%s", dg, abs(d), suffix)
      if (!anyDuplicated(labs) || dg >= 4) break
      dg <- dg + 1
    }
    out[keep] <- labs
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
map_class_shares <- function(rc, levels_, counts = NULL) {
  df <- if (is.null(counts)) lcz_counts(rc) else counts
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
#' @param basemap   draw web-map tiles under the raster. Off by default; when
#'                  on, nodata cells and the panel background are transparent
#'                  instead of white, so the backdrop shows through wherever the
#'                  raster has nothing to say. Needs network access.
#' @param basemap_provider maptiles provider name.
#' @param basemap_zoom tile zoom level, or NULL to derive it from the extent.
#' @param basemap_alpha opacity of the backdrop.
#' @param basemap_apikey key for a provider that needs one (the Stadia-hosted
#'                  Stamen styles, Thunderforest, Jawg, ...); defaults to the
#'                  provider family's environment variable.
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
#' @param x_axis edge carrying the longitude labels, "bottom" or "top". Worth
#'   moving when the class-mix bar is at the bottom: the two then sit at
#'   opposite edges instead of the coordinates being read across a strip of
#'   colour that has nothing to do with them.
lcz_raster_plot <- function(path, bbox, legend = FALSE, scalebar = TRUE,
                            scalebar_corner = "br", title = NULL,
                            legend_ncol = 4, digits = 1, guppd = FALSE,
                            basemap = FALSE,
                            basemap_provider = "OpenStreetMap",
                            basemap_zoom = NULL, basemap_alpha = 1,
                            basemap_apikey = NULL,
                            guppd_highlight = NULL, resolution = TRUE,
                            distribution = c("none", "pie", "bar"),
                            dist_side = "bottom", pie_corner = "bl",
                            pie_size = PIE_SIZE, x_axis = c("bottom", "top"),
                            panel_in = 6.5, max_cells = 4e6,
                            vector = NULL, vector_field = "LCZ_class") {
  distribution <- match.arg(distribution)
  x_axis <- match.arg(x_axis)
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
  # With a vector file the raster only sets the extent, CRS and scale; the
  # classes, the drawn layer and the class mix all come from the polygons.
  vec <- if (is.null(vector)) NULL else read_lcz_vector(vector, rc, vector_field)

  present <- if (is.null(vec)) sort(unique(as.integer(d$lcz)))
             else sort(unique(vec$df$lcz))
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
  if (!is.null(vec)) {
    vec$df$lcz <- factor(LCZ_LABELS_CODE[match(vec$df$lcz, LCZ_TABLE$code)],
                         levels = levels(d$lcz))
  }

  e   <- as.vector(terra::ext(rc))
  inf <- lcz_scale_info(rc)
  brk <- function(lo, hi) c(lo, (lo + hi) / 2, hi)

  # With a basemap the two whites have to go transparent -- the nodata fill and
  # the painted panel -- or the backdrop is covered by the 94% of a reference
  # tif that is nodata. NA is ggplot's transparent fill.
  # "transparent", not NA: an NA fill makes ggplot treat the cells as missing
  # data and drop them with a warning, which is a lie about the raster.
  nodata_fill <- if (basemap) "transparent" else LCZ_NODATA_COLOUR

  p <- ggplot(d, aes(x = x, y = y, fill = lcz))
  # Added before the class layer, so it draws under it.
  if (basemap) {
    p <- p + basemap_layer(rc, provider = basemap_provider,
                           zoom = basemap_zoom, alpha = basemap_alpha,
                           apikey = basemap_apikey)
  }
  p <- p +
    (if (is.null(vec)) geom_raster()
     else geom_polygon(data = vec$df, aes(x = x, y = y, group = grp, fill = lcz),
                       inherit.aes = FALSE, colour = NA)) +
    scale_fill_manual(values = c(LCZ_COLOURS_PY[idx],
                                 setNames(nodata_fill, NODATA_KEY)),
                      breaks = keys, name = NULL, drop = TRUE,
                      na.value = nodata_fill)

  # Ground size of one drawn cell. terra::res is in the raster's own units, so
  # it needs the same degrees-to-metres factor the scale bar uses.
  # A degree of longitude is m_per_x metres and a degree of latitude is
  # m_per_x * ratio, which is why the two sides need different factors.
  cell_m <- terra::res(rc) * inf$m_per_x * c(1, inf$ratio)
  if (resolution && !is.null(vec)) {
    # Polygons are drawn, not cells, so a key giving the raster's cell size
    # would describe nothing on the map.
    message("  resolution key omitted: --vector draws polygons, not raster cells")
    resolution <- FALSE
  }
  if (resolution) {
    native <- terra::res(terra::rast(path[[1]])) * inf$m_per_x * c(1, inf$ratio)
    if (!isTRUE(all.equal(native, cell_m))) {
      message("  resolution key shows the aggregated cell (",
              resolution_label(cell_m), "), not the tif's native ",
              resolution_label(native), "; raise --max-cells to draw it natively")
    }
  }

  shares <- if (distribution == "none") NULL else
    map_class_shares(rc, levels(d$lcz), counts = vec$shares)

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
                       labels = degree_labeller(rc, "x", digits),
                       position = x_axis) +
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
      panel.background = element_rect(fill = nodata_fill, colour = NA),
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
  parser$add_argument("--input", required = TRUE, nargs = "+",
                      help = paste("LCZ GeoTIFF(s) with classes 1-17 and nodata 0.",
                                   "Several are merged onto the first one's grid."))
  parser$add_argument("--vector", default = NULL,
                      help = paste("draw this LCZ polygon file (e.g. a So2Sat",
                                   "patches_reference_<city>.gpkg) instead of",
                                   "the raster, which then only sets the extent"))
  parser$add_argument("--vector-field", default = "LCZ_class", dest = "vector_field",
                      help = "class column (1-17) in --vector")
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
  parser$add_argument("--basemap", action = "store_true",
                      help = "draw web-map tiles under the raster (needs network)")
  parser$add_argument("--basemap-provider", default = "OpenStreetMap",
                      dest = "basemap_provider",
                      help = "maptiles provider name (default: OpenStreetMap)")
  parser$add_argument("--basemap-zoom", type = "integer", default = NULL,
                      dest = "basemap_zoom",
                      help = "tile zoom level (default: derived from the extent)")
  parser$add_argument("--basemap-alpha", type = "double", default = 1,
                      dest = "basemap_alpha", help = "opacity of the backdrop")
  parser$add_argument("--basemap-apikey", default = NULL, dest = "basemap_apikey",
                      help = paste("key for a provider that needs one; defaults",
                                   "to $STADIA_API_KEY and friends"))
  parser$add_argument("--list-basemaps", action = "store_true", dest = "list_basemaps",
                      help = "print the available provider names and exit")
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
  parser$add_argument("--x-axis", default = "bottom", dest = "x_axis",
                      choices = c("bottom", "top"),
                      help = paste("edge carrying the longitude labels. Pair",
                                   "--x-axis top with --dist-side bottom so the",
                                   "coordinates and the class-mix bar sit at",
                                   "opposite edges"))
  parser$add_argument("--digits", type = "integer", default = 1,
                      help = "decimal places on the coordinate labels")
  parser$add_argument("--max-cells", type = "double", default = 4e6, dest = "max_cells")
  # argparse reads a value starting with "-" as another flag, so a western
  # bbox ("--bbox -0.30,51.4,...") would be rejected. Glue such a value onto its
  # flag as "--bbox=..." first, which argparse does accept, so both spellings
  # work from the shell.
  argv <- commandArgs(trailingOnly = TRUE)
  # Handled before parsing: --input and --bbox are required, and listing the
  # providers should not need a raster to list them against.
  if ("--list-basemaps" %in% argv) {
    keyed <- names(API_KEY_ENV)
    for (n in c(names(EXTRA_PROVIDERS), names(maptiles::get_providers()))) {
      var <- API_KEY_ENV[[sub("\\..*$", "", n)]]
      cat(n, if (!is.null(var)) paste0("   [needs $", var[[1]], "]") else "",
          "\n", sep = "")
    }
    quit(save = "no")
  }
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
                  basemap = args$basemap,
                  basemap_provider = args$basemap_provider,
                  basemap_zoom = args$basemap_zoom,
                  basemap_alpha = args$basemap_alpha,
                  basemap_apikey = args$basemap_apikey,
                  guppd = args$guppd,
                  guppd_highlight = if (is.null(args$guppd_highlight)) NULL
                                    else if (tolower(args$guppd_highlight) == "none") NA
                                    else args$guppd_highlight,
                  resolution = !args$no_resolution,
                  distribution = args$distribution, dist_side = args$dist_side,
                  x_axis = args$x_axis,
                  dist_thickness = args$dist_thickness,
                  pie_corner = args$pie_corner, pie_size = args$pie_size,
                  scalebar = !args$no_scalebar,
                  scalebar_corner = args$scalebar_corner,
                  title = args$title, max_cells = args$max_cells,
                  vector = args$vector, vector_field = args$vector_field)
}
