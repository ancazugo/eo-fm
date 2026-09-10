# constants.R ─ Shared constants and helpers for the eo-fm R figures.
#
# Source this from any plotting script:
#
#     source("R/constants.R")
#
# The LCZ palette and class names are a transcription of `lcz_dict` in
# src/utils/constants.py, which remains the single source of truth. If that
# dictionary changes, mirror the change here (LCZ_TABLE below).

suppressPackageStartupMessages({
  library(tibble)
  library(dplyr)
  library(ggplot2)
})

# ── Environment ───────────────────────────────────────────────────────────────

# The conda R environment ships a valid share/proj (PROJ 9.7.1) but does not
# point PROJ at it, so authority codes ("EPSG:3857", "ESRI:54030") fail with
# `proj_create: Open of .../share/proj failed` while proj4 strings still work.
# Setting PROJ_DATA fixes it. Derived from R.home() so this holds on any conda
# env, and applied only when it is unset and proj.db is actually there.
local({
  if (!nzchar(Sys.getenv("PROJ_DATA"))) {
    candidate <- file.path(dirname(dirname(R.home())), "share", "proj")
    if (file.exists(file.path(candidate, "proj.db"))) {
      Sys.setenv(PROJ_DATA = candidate)
    }
  }
})

DATA_DIR <- Sys.getenv("DATA_DIR")
if (!nzchar(DATA_DIR)) {
  stop("DATA_DIR is unset. Run from the repo root so .Renviron is picked up, ",
       "or export DATA_DIR before starting R.", call. = FALSE)
}

SO2SAT_DIR        <- file.path(DATA_DIR, "input", "So2Sat-LCZ42", "v4")
SO2SAT_CITIES_DIR <- file.path(SO2SAT_DIR, "cities")
SO2SAT_PATCHES_GPKG <- file.path(SO2SAT_DIR, "patches_reference_rxr.gpkg")
PLOTS_DIR         <- "plots"

# Thematic subfolders under plots/. Every figure script declares which one it
# writes to, so `plots/` stays navigable as the figure set grows -- it reached
# 45 files in one flat directory before this was introduced.
PLOT_DIR_DATASET    <- "dataset"      # what the data is: classes, splits, cities
PLOT_DIR_EMBEDDINGS <- "embeddings"   # projection scatters, density grids
PLOT_DIR_MODELS     <- "models"       # results table, confusion matrices
PLOT_DIR_MAPS       <- "maps"         # LCZ rasters for an ROI
PLOT_DIR_RASTERS    <- "rasters"      # one place as a picture: embedding RGB,
                                      #   its labels, the imagery under it
CITY_SUMMARY_CSV  <- file.path("data", "so2sat_city_summary.csv")
CITY_CLASS_CSV    <- file.path("data", "so2sat_city_class_counts.csv")

# ── LCZ classes ───────────────────────────────────────────────────────────────

# Transcribed from src/utils/constants.py::lcz_dict. `alt_code` is the standard
# LCZ short label (1-10, A-G) and matches LCZ_ORDER in src/training/lcz_metrics.py;
# `group` mirrors that module's URBAN_INDICES (1-10) / NATURAL_INDICES (A-G).
LCZ_TABLE <- tibble::tribble(
  ~code, ~alt_code, ~name,                  ~colour,     ~group,
  1L,    "1",       "Compact High-Rise",    "#8c0000",   "Built",
  2L,    "2",       "Compact Mid-Rise",     "#d10000",   "Built",
  3L,    "3",       "Compact Low-Rise",     "#ff0000",   "Built",
  4L,    "4",       "Open High-Rise",       "#bf4d00",   "Built",
  5L,    "5",       "Open Mid-Rise",        "#ff6600",   "Built",
  6L,    "6",       "Open Low-Rise",        "#ff9955",   "Built",
  7L,    "7",       "Lightweight Low-Rise", "#faee05",   "Built",
  8L,    "8",       "Large Low-Rise",       "#bcbcbc",   "Built",
  9L,    "9",       "Sparsely Built",       "#ffccaa",   "Built",
  10L,   "10",      "Heavy Industry",       "#555555",   "Built",
  11L,   "A",       "Dense Trees",          "#006a00",   "Natural",
  12L,   "B",       "Scattered Trees",      "#00aa00",   "Natural",
  13L,   "C",       "Bush, Scrub",          "#648525",   "Natural",
  14L,   "D",       "Low Plants",           "#b9db79",   "Natural",
  15L,   "E",       "Bare Rock or Paved",   "#000000",   "Natural",
  16L,   "F",       "Bare Soil or Sand",    "#fbf7ae",   "Natural",
  17L,   "G",       "Water",                "#6a6aff",   "Natural"
) |>
  mutate(alt_name = paste0(alt_code, " — ", name))

# Named palettes. Naming them means scale_*_manual matches by value rather than
# by position, so a subset or a reordered factor can never silently mis-colour.
LCZ_COLOURS      <- setNames(LCZ_TABLE$colour, LCZ_TABLE$alt_code)
LCZ_COLOURS_NAME <- setNames(LCZ_TABLE$colour, LCZ_TABLE$name)
LCZ_COLOURS_FULL <- setNames(LCZ_TABLE$colour, LCZ_TABLE$alt_name)
LCZ_COLOURS_CODE <- setNames(LCZ_TABLE$colour, as.character(LCZ_TABLE$code))

# Rasters use 0 = nodata, drawn white (see src/utils/plot_lcz.py::lcz_colormap).
LCZ_NODATA_COLOUR <- "#ffffff"

#' Turn LCZ class codes into a factor with canonical ordering and nice labels.
#'
#' @param x       integer vector of class codes.
#' @param labels  "alt" (1-10/A-G), "name", "alt_name" ("A - Dense Trees"), "code".
#' @param offset  added to `x` before lookup. Model outputs use 0-16 rather than
#'                the 1-17 of LCZ_class, so pass `offset = 1` for those.
#' @param rev     reverse the level order. ggplot draws the first factor level at
#'                the bottom of a discrete y axis, so `rev = TRUE` puts LCZ 1 on top.
lcz_factor <- function(x, labels = c("alt", "name", "alt_name", "code"),
                       offset = 0L, rev = FALSE) {
  labels <- match.arg(labels)
  column <- switch(labels, alt = "alt_code", name = "name",
                   alt_name = "alt_name", code = "code")
  labels_vec <- as.character(LCZ_TABLE[[column]])
  idx    <- match(as.integer(x) + offset, LCZ_TABLE$code)
  # Index the canonical order, then reverse only the level order. Reversing
  # before the lookup would silently pair each class with its mirror label.
  values <- labels_vec[idx]
  lvls   <- if (rev) base::rev(labels_vec) else labels_vec
  factor(values, levels = lvls)
}

#' Palette matching the label style used by `lcz_factor()`.
lcz_palette <- function(labels = c("alt", "name", "alt_name", "code")) {
  switch(match.arg(labels),
         alt = LCZ_COLOURS, name = LCZ_COLOURS_NAME,
         alt_name = LCZ_COLOURS_FULL, code = LCZ_COLOURS_CODE)
}

scale_fill_lcz <- function(labels = "alt_name", ...) {
  scale_fill_manual(values = lcz_palette(labels), ...)
}

scale_colour_lcz <- function(labels = "alt_name", ...) {
  scale_colour_manual(values = lcz_palette(labels), ...)
}

# ── Splits ────────────────────────────────────────────────────────────────────

# The 10 So2Sat "culture" cities: the only ones carrying validation/testing
# patches. Ported from src/utils/city_split.py::SO2SAT_CULTURE_CITIES, but with
# the display spellings used in the figures rather than directory names.
SO2SAT_CULTURE_CITIES <- c(
  "Guangzhou", "Jakarta", "Moscow", "Mumbai", "Munich",
  "Nairobi", "San Jose", "Santiago", "Sydney", "Tehran"
)

# Okabe-Ito colourblind-safe pair, deliberately outside the LCZ palette so a
# split legend can never be confused with a class legend.
SPLIT_COLOURS <- c("Training" = "#0072B2", "Validation + Test" = "#D55E00")

ROLE_LABELS <- c("train" = "Training", "val_test" = "Validation + Test")

# So2Sat's own split names -> figure labels (cf. city_split.py::DATASET_TO_ROLE).
DATASET_LABELS <- c("training" = "Training",
                    "validation" = "Validation",
                    "testing" = "Testing")

# Three-way split palette (Okabe-Ito), for figures that separate val from test.
DATASET_COLOURS <- c("Training" = "#0072B2", "Validation" = "#009E73",
                     "Testing" = "#D55E00")

# Greyscale variant matching the pie map's centre dots exactly: black =
# training, white = held out. Validation and Test are both white, so the
# boundary between them is carried by the segment border alone.
DATASET_COLOURS_BW <- c("Training" = "#000000", "Validation" = "#ffffff",
                        "Testing" = "#ffffff")

# Centre-dot fill on the pie map: black = training, white = held out.
ROLE_DOT_FILL <- c("Training" = "#000000", "Validation + Test" = "#ffffff")

# Glyphs echoing those centre dots, so a split label carries the same mark the
# map uses: filled circle = training, hollow circle = held out.
DATASET_GLYPHS <- c("Training" = "\u25CF", "Validation" = "\u25CB",
                    "Testing" = "\u25CB")

#' Black or white, whichever reads better on `hex`.
#'
#' The LCZ palette spans pure black (#000000, class E) to near-white (#fbf7ae,
#' class F), so in-bar text needs a per-colour decision rather than one choice.
contrast_text <- function(hex) {
  rgb <- grDevices::col2rgb(hex) / 255
  # Relative luminance, ITU-R BT.709 coefficients.
  lum <- 0.2126 * rgb[1, ] + 0.7152 * rgb[2, ] + 0.0722 * rgb[3, ]
  ifelse(lum > 0.55, "grey10", "white")
}

# ── City metadata ─────────────────────────────────────────────────────────────

# Ported from src/utils/geo_lookup.py (CITY_TO_COUNTRY / CITY_TO_CONTINENT),
# keyed by the city *directory* name under $DATA_DIR/.../v4/cities.
#
# Two departures from the Python lookups, both deliberate:
#
#  1. Shenzhen is added. GUPPD merges the Guangzhou and Shenzhen agglomerations
#     into a single SMOD entity, so every file in the repo counts 51 cities. The
#     patches disagree: within cities/Guangzhou, the 5,583 `training` patches sit
#     at 113.99E 22.58N (Shenzhen) and the 4,816 validation/testing patches at
#     113.25E 23.08N (Guangzhou), ~85 km apart and not interleaved. Splitting
#     them recovers the true 42 training + 10 val/test = 52 cities.
#  2. `city_label` carries display names: bracketed alternates are trimmed to the
#     primary city and Dongying replaces the CJK JRC name.
CITY_META <- tibble::tribble(
  ~city_dir,                ~city_label,      ~country,         ~continent,
  "Amsterdam",              "Amsterdam",      "Netherlands",    "Europe",
  "Beijing",                "Beijing",        "China",          "Asia",
  "Berlin",                 "Berlin",         "Germany",        "Europe",
  "Bogota",                 "Bogota",         "Colombia",       "South America",
  "Buenos_Aires",           "Buenos Aires",   "Argentina",      "South America",
  "Cairo",                  "Cairo",          "Egypt",          "Africa",
  "Cape_Town",              "Cape Town",      "South Africa",   "Africa",
  "Caracas",                "Caracas",        "Venezuela",      "South America",
  "Changsha",               "Changsha",       "China",          "Asia",
  "Chicago",                "Chicago",        "United States",  "North America",
  "Cologne",                "Cologne",        "Germany",        "Europe",
  "Dhaka",                  "Dhaka",          "Bangladesh",     "Asia",
  "Dongying",               "Dongying",       "China",          "Asia",
  "Guangzhou",              "Guangzhou",      "China",          "Asia",
  "Hong_Kong",              "Hong Kong",      "China",          "Asia",
  "Istanbul",               "Istanbul",       "Turkey",         "Asia",
  "Jakarta",                "Jakarta",        "Indonesia",      "Asia",
  "Karachi",                "Karachi",        "Pakistan",       "Asia",
  "Lima",                   "Lima",           "Peru",           "South America",
  "Lisbon",                 "Lisbon",         "Portugal",       "Europe",
  "London",                 "London",         "United Kingdom", "Europe",
  "Los_Angeles",            "Los Angeles",    "United States",  "North America",
  "Madrid",                 "Madrid",         "Spain",          "Europe",
  "Melbourne",              "Melbourne",      "Australia",      "Oceania",
  "Milan",                  "Milan",          "Italy",          "Europe",
  "Moscow",                 "Moscow",         "Russia",         "Europe",
  "Mumbai",                 "Mumbai",         "India",          "Asia",
  "Munich",                 "Munich",         "Germany",        "Europe",
  "Nairobi",                "Nairobi",        "Kenya",          "Africa",
  "Nanjing",                "Nanjing",        "China",          "Asia",
  "New_York",               "New York",       "United States",  "North America",
  "Osaka_[Kyoto]",          "Osaka",          "Japan",          "Asia",
  "Paris",                  "Paris",          "France",         "Europe",
  "Philadelphia",           "Philadelphia",   "United States",  "North America",
  "Qingdao",                "Qingdao",        "China",          "Asia",
  "Quezon_City_[Manila]",   "Quezon City",    "Philippines",    "Asia",
  "Rawalpindi_[Islamabad]", "Rawalpindi",     "Pakistan",       "Asia",
  "Rio_De_Janeiro",         "Rio de Janeiro", "Brazil",         "South America",
  "Rome",                   "Rome",           "Italy",          "Europe",
  "Salvador",               "Salvador",       "Brazil",         "South America",
  "San_Jose",               "San Jose",       "United States",  "North America",
  "Santiago",               "Santiago",       "Chile",          "South America",
  "Sao_Paulo",              "São Paulo",      "Brazil",         "South America",
  "Shanghai",               "Shanghai",       "China",          "Asia",
  "Shenzhen",               "Shenzhen",       "China",          "Asia",
  "Sydney",                 "Sydney",         "Australia",      "Oceania",
  "Tehran",                 "Tehran",         "Iran",           "Asia",
  "Tokyo",                  "Tokyo",          "Japan",          "Asia",
  "Vancouver",              "Vancouver",      "Canada",         "North America",
  "Washington_D.C.",        "Washington D.C.","United States",  "North America",
  "Wuhan",                  "Wuhan",          "China",          "Asia",
  "Zurich",                 "Zurich",         "Switzerland",    "Europe"
)

# ── Theme and output ──────────────────────────────────────────────────────────

#' Publication theme: readable in a single paper column and when blown up for a
#' poster. Keep base_size at 11 for the paper, raise it to ~16 for poster prints.
theme_eofm <- function(base_size = 11, base_family = "") {
  theme_minimal(base_size = base_size, base_family = base_family) +
    theme(
      plot.title      = element_text(face = "bold", size = rel(1.15),
                                     margin = margin(b = 4)),
      plot.subtitle   = element_text(colour = "grey30", size = rel(0.95),
                                     margin = margin(b = 8)),
      plot.caption    = element_text(colour = "grey40", size = rel(0.75),
                                     hjust = 0, margin = margin(t = 10)),
      plot.title.position   = "plot",
      plot.caption.position = "plot",
      axis.title      = element_text(colour = "grey25", size = rel(0.95)),
      axis.text       = element_text(colour = "grey25"),
      strip.text      = element_text(face = "bold", colour = "grey15",
                                     size = rel(0.95), margin = margin(4, 4, 4, 4)),
      panel.grid.minor = element_blank(),
      panel.grid.major = element_line(colour = "grey92", linewidth = 0.3),
      plot.background  = element_rect(fill = "transparent", colour = NA),
      panel.background = element_rect(fill = "transparent", colour = NA),
      legend.background = element_rect(fill = "transparent", colour = NA),
      legend.key       = element_rect(fill = "transparent", colour = NA),
      legend.title    = element_text(face = "bold", size = rel(0.9)),
      legend.text     = element_text(size = rel(0.85)),
      legend.key.size = unit(0.9, "lines")
    )
}

#' Map variant: no axis text or grid furniture, just the graticule.
theme_eofm_map <- function(base_size = 11, base_family = "") {
  theme_eofm(base_size, base_family) +
    theme(
      plot.margin = margin(0, 0, 0, 0),
      axis.title  = element_blank(),
      axis.text   = element_blank(),
      axis.ticks  = element_blank(),
      panel.grid.major = element_line(colour = "grey93", linewidth = 0.25),
      panel.border = element_blank()
    )
}

# Legend/title type, shared by the map legend and the bar titles so the three
# panels of the merged figure agree. Sizes are in points; LABEL_SIZE_MM is the
# geom_text equivalent of LEGEND_TEXT_PT (ggplot text sizes are mm).
LEGEND_TITLE_PT <- 11 * 0.9
LEGEND_TEXT_PT  <- 11 * 0.85
LABEL_SIZE_MM   <- LEGEND_TEXT_PT / .pt

#' Directory a figure of this theme belongs in, created on demand.
#'
#' `subdir = NULL` means the top of plots/, which nothing should use any more --
#' it is kept so an ad-hoc `save_plot()` at the console still works.
plot_path <- function(subdir = NULL, ...) {
  if (is.null(subdir)) file.path(PLOTS_DIR, ...) else
    file.path(PLOTS_DIR, subdir, ...)
}

#' Save one plot as PNG.
#'
#' Uses ragg for the PNG: better text rendering than the default device, and it
#' is already installed in this environment.
save_plot <- function(plot, name, width = 7, height = 5, dpi = 400,
                      formats = "png", subdir = NULL) {
  dir  <- plot_path(subdir)
  dir.create(dir, showWarnings = FALSE, recursive = TRUE)
  paths <- character(0)
  for (fmt in formats) {
    path <- file.path(dir, paste0(name, ".", fmt))
    if (fmt == "png") {
      ggsave(path, plot, width = width, height = height, dpi = dpi,
             device = ragg::agg_png, bg = "transparent")
    } else {
      ggsave(path, plot, width = width, height = height, device = cairo_pdf,
             bg = "transparent")
    }
    paths <- c(paths, path)
  }
  message("  wrote ", paste(paths, collapse = ", "))
  invisible(paths)
}
