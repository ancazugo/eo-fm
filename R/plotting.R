# plotting.R ─ Publication figures for the eo-fm paper and poster.
#
#     Rscript R/plotting.R
#
# Run from the repo root so .Renviron (DATA_DIR) and the relative data/ and
# plots/ paths resolve. Shared constants, the LCZ palette, the theme and the
# save helper all live in R/constants.R. Figures carry no titles, subtitles or
# captions: they are captioned in the paper and on the poster.
#
# Figures written to plots/ (PNG):
#   class_distribution           - LCZ shares per split, faceted
#   class_distribution_overall   - LCZ shares over the whole dataset
#   class_composition_vertical   - LCZ class mix as one thin stacked bar
#   dataset_composition_vertical - split mix as one thin stacked bar
#   so2sat_city_map_pies_regional - the 52 cities as class-composition pies,
#                                  with five regional insets
#   *_urban_points               - variant drawing GUPPD urban areas as points
#   so2sat_overview_panel        - the three above merged side by side
#
# Prediction maps from the neural networks are produced in Python and are not
# part of this script.

source("R/constants.R")

suppressPackageStartupMessages({
  library(sf)
  library(readr)
  library(dplyr)
  library(tidyr)
  library(purrr)
  library(forcats)
  library(stringr)
  library(scales)
  library(spData)
  library(patchwork)
  library(units)
})

EQUAL_EARTH <- "+proj=eqearth"   # equal-area; proj4 string, not an authority code

# ── Shared data ───────────────────────────────────────────────────────────────

if (!file.exists(CITY_SUMMARY_CSV) || !file.exists(CITY_CLASS_CSV)) {
  stop("Missing cached city data. Run: Rscript R/prepare_city_data.R", call. = FALSE)
}

cities <- read_csv(CITY_SUMMARY_CSV, show_col_types = FALSE) |>
  mutate(split = factor(ROLE_LABELS[role], levels = names(SPLIT_COLOURS)))

city_classes <- read_csv(CITY_CLASS_CSV, show_col_types = FALSE)

# Class counts for the whole dataset, straight out of the GeoPackage: the figure
# needs 51 numbers, not 400,673 polygons, so aggregate in SQL.
class_counts <- st_read(
  SO2SAT_PATCHES_GPKG,
  query = paste("SELECT dataset, LCZ_class, COUNT(*) AS n",
                "FROM patches_reference_rxr GROUP BY dataset, LCZ_class"),
  quiet = TRUE
) |> as_tibble()

world_ll <- spData::world |> filter(name_long != "Antarctica")
world <- st_transform(world_ll, EQUAL_EARTH)

# Latitude of the map's northern and southern extremes (Antarctica is dropped,
# so the south stops near 56S). The meridian labels are anchored on the outline
# at whichever of the two is at the top of the panel.
WORLD_LAT_MIN <- as.numeric(st_bbox(world_ll))[2]
WORLD_LAT_MAX <- as.numeric(st_bbox(world_ll))[4]

# Country borders carry the political division, so they are drawn as a visible
# line rather than the usual hairline knock-out: a mid grey, darker than both
# the land and the GUPPD urban overlay (grey66) so the three read apart.
LAND_FILL  <- "grey93"
BORDER_COL <- "grey55"
BORDER_LW  <- 0.22

# ── South-up variant ──────────────────────────────────────────────────────────
#
# The upside-down map is produced by negating y in projected space rather than
# by reprojecting or rotating the device. Equal Earth is symmetric about the
# equator, so a y-negation maps its graticule exactly onto itself: coord_sf goes
# on drawing the same meridians and parallels, and all text stays upright. Only
# the things placed by hand -- countries, urban areas, city dots, pie centres,
# the tropics, the latitude labels and the inset windows -- have to be mirrored.

#' Mirror an sf/sfc object vertically, keeping the CRS label.
mirror_sf <- function(x) {
  g <- st_set_crs(st_geometry(x) * matrix(c(1, 0, 0, -1), 2, 2), EQUAL_EARTH)
  if (inherits(x, "sf")) {
    st_geometry(x) <- g
    x
  } else {
    g
  }
}

#' Mirror a pair of projected y limits (negating reverses their order).
mirror_ylim <- function(ylim) if (is.null(ylim)) NULL else -rev(ylim)

world_flipped <- mirror_sf(world)

# Patch counts span 1 (Salvador) to 30,657 (Vancouver). scale_size_area makes
# area strictly proportional, which renders the single-digit cities invisible,
# so use a square-root scale with a floor: still monotonic in count, but every
# city keeps a visible dot.
SIZE_BREAKS <- c(10, 1000, 10000, 30000)
scale_city_size <- function(range = c(1.5, 8), ...) {
  scale_size(range = range, transform = "sqrt", breaks = SIZE_BREAKS,
             labels = label_number(big.mark = ","), ...)
}

# ══════════════════════════════════════════════════════════════════════════════
# Figure 1 ─ LCZ class distribution per split
# ══════════════════════════════════════════════════════════════════════════════

message("Figure 1: class distribution (faceted)")

split_totals <- class_counts |>
  group_by(dataset) |>
  summarise(total = sum(n), .groups = "drop") |>
  mutate(facet = sprintf("%s (n = %s)", DATASET_LABELS[dataset],
                         format(total, big.mark = ",", trim = TRUE)))

class_dist <- class_counts |>
  left_join(split_totals, by = "dataset") |>
  mutate(
    share = n / total,
    lcz   = lcz_factor(LCZ_class, "alt", rev = TRUE),
    facet = factor(facet, levels = split_totals$facet[
      match(names(DATASET_LABELS), split_totals$dataset)])
  )

# Built classes 1-10 sit above the natural ones A-G; with reversed levels the
# 7 natural classes occupy the bottom 7 slots.
n_natural <- sum(LCZ_TABLE$group == "Natural")

class_dist_plot <- ggplot(class_dist) +
  aes(x = share, y = lcz, fill = lcz) +
  geom_col(colour = "grey30", linewidth = 0.2, width = 0.78) +
  geom_hline(yintercept = n_natural + 0.5, colour = "grey55",
             linetype = "dashed", linewidth = 0.35) +
  facet_wrap(~facet, nrow = 1) +
  scale_fill_manual(values = LCZ_COLOURS, guide = "none") +
  scale_x_continuous(labels = percent_format(accuracy = 1),
                     expand = expansion(mult = c(0, 0.08))) +
  labs(x = "Share of patches within split", y = "LCZ class") +
  theme_eofm() +
  theme(panel.grid.major.y = element_blank(),
        panel.grid.major.x = element_line(colour = "grey92", linewidth = 0.3),
        panel.spacing = unit(1.1, "lines"))

save_plot(class_dist_plot, "class_distribution", width = 9, height = 5,
            subdir = PLOT_DIR_DATASET)

# ══════════════════════════════════════════════════════════════════════════════
# Figure 2 ─ LCZ class distribution over the whole dataset
# ══════════════════════════════════════════════════════════════════════════════

message("Figure 2: class distribution (pooled)")

overall <- class_counts |>
  group_by(LCZ_class) |>
  summarise(n = sum(n), .groups = "drop") |>
  mutate(share = n / sum(n),
         lcz   = lcz_factor(LCZ_class, "alt", rev = TRUE))

class_dist_overall <- ggplot(overall) +
  aes(x = share, y = lcz, fill = lcz) +
  geom_col(colour = "grey30", linewidth = 0.2, width = 0.78) +
  geom_hline(yintercept = n_natural + 0.5, colour = "grey55",
             linetype = "dashed", linewidth = 0.35) +
  geom_text(aes(label = percent(share, accuracy = 0.1)),
            hjust = -0.15, size = 3, colour = "grey25") +
  scale_fill_manual(values = LCZ_COLOURS, guide = "none") +
  scale_x_continuous(labels = percent_format(accuracy = 1),
                     expand = expansion(mult = c(0, 0.13))) +
  labs(x = "Share of all patches", y = "LCZ class") +
  theme_eofm() +
  theme(panel.grid.major.y = element_blank(),
        panel.grid.major.x = element_line(colour = "grey92", linewidth = 0.3))

save_plot(class_dist_overall, "class_distribution_overall", width = 5.5, height = 5,
            subdir = PLOT_DIR_DATASET)

# ══════════════════════════════════════════════════════════════════════════════
# Figure 3 ─ Composition bars: LCZ classes and data split
# ══════════════════════════════════════════════════════════════════════════════
#
# Four one-bar figures. They double as the legend for the pie map, which is why
# that map carries no class or split legend of its own. Segments are drawn as
# explicit rectangles rather than via position_stack so the order is controlled
# outright, and every label sits outside its segment with the share appended.

message("Figure 3: composition bars")

source("R/composition.R")

# LCZ classes, canonical order (bars read 1, 2, 3 ... G downwards).
class_comp <- overall |>
  arrange(LCZ_class) |>
  transmute(key = factor(as.character(lcz_factor(LCZ_class, "alt")),
                         levels = LCZ_TABLE$alt_code),
            share, n)

# Splits, Training first (so it lands at the top of the bar).
split_comp <- split_totals |>
  mutate(key = factor(DATASET_LABELS[dataset], levels = names(DATASET_COLOURS))) |>
  arrange(key) |>
  transmute(key, share = total / sum(total), n = total,
            glyph = DATASET_GLYPHS[as.character(key)])

# LCZ: labels to the right of the bar. Split: greyscale matching the map's
# centre dots, labels to the left.
class_bar_plot <- composition_bar(class_comp, LCZ_COLOURS, side = "right",
                                  title = "LCZ Class", label_room = 0.58)
split_bar_plot <- composition_bar(split_comp, DATASET_COLOURS_BW, side = "left",
                                  title = "Data Split", label_room = 1.05,
                                  border_lw = 0.5, fmt = fmt_count,
                                  hatch_keys = "Validation")

save_plot(class_bar_plot, "class_composition_vertical", width = 1.6, height = 7,
            subdir = PLOT_DIR_DATASET)
save_plot(split_bar_plot, "dataset_composition_vertical", width = 1.6, height = 7,
            subdir = PLOT_DIR_DATASET)

# ══════════════════════════════════════════════════════════════════════════════
# GUPPD urban areas ─ a subtle worldwide urban layer under the city markers
# ══════════════════════════════════════════════════════════════════════════════
#
# The GUPPD gpkg ships both a polygon and a point representation of the same
# 123,034 settlements, so the "points" variant of each map is the point layer
# rather than centroids computed here. Geometry only: the attribute table is
# large and nothing here needs it.

message("Loading GUPPD urban areas ...")

GUPPD_GPKG <- file.path(DATA_DIR, "input", "NASA", "GUPPD",
                        "urbanspatial-guppd-v1-gpkg.gpkg")

urban_poly <- st_read(GUPPD_GPKG, layer = "urbanspatial_guppd_v1_polygons",
                      quiet = TRUE) |> st_geometry() |> st_transform(EQUAL_EARTH)
urban_pts  <- st_read(GUPPD_GPKG, layer = "urbanspatial_guppd_v1_points",
                      quiet = TRUE) |> st_geometry() |> st_transform(EQUAL_EARTH)

# Most settlements are a few km across -- sub-pixel at world scale -- so the
# world panel uses a simplified copy. Insets get the full geometry, cropped.
urban_poly_world <- st_simplify(urban_poly, dTolerance = 3000)

# South-up copies (see mirror_sf), built once because mirroring 123k geometries
# is not free.
urban_poly_flipped       <- mirror_sf(urban_poly)
urban_pts_flipped        <- mirror_sf(urban_pts)
urban_poly_world_flipped <- mirror_sf(urban_poly_world)

message("  ", format(length(urban_poly), big.mark = ","), " urban areas")

# Deliberately close to the land grey: the overlay should read as texture under
# the city markers, not compete with them.
URBAN_COL <- "grey66"

bbox_sfc <- function(xlim, ylim) {
  st_as_sfc(st_bbox(c(xmin = xlim[1], xmax = xlim[2],
                      ymin = ylim[1], ymax = ylim[2]), crs = EQUAL_EARTH))
}

#' The urban layer for one panel. `zoomed = FALSE` means the whole world: the
#' simplified copy, uncropped. The world panel passes explicit `xlim`/`ylim`
#' (padded, so the map outline is not clipped) but is still not a zoom, which is
#' why `zoomed` is a parameter rather than derived from `xlim`.
urban_layer <- function(urban = c("polygons", "points", "none"),
                        xlim = NULL, ylim = NULL, zoomed = !is.null(xlim),
                        flip = FALSE) {
  urban <- match.arg(urban)
  if (urban == "none") return(NULL)
  if (urban == "points") {
    all_pts <- if (flip) urban_pts_flipped else urban_pts
    g <- if (zoomed) all_pts[bbox_sfc(xlim, ylim)] else all_pts
    return(geom_sf(data = g, colour = URBAN_COL, shape = 16,
                   size = if (zoomed) 0.26 else 0.045))
  }
  g <- if (zoomed) {
    (if (flip) urban_poly_flipped else urban_poly)[bbox_sfc(xlim, ylim)]
  } else if (flip) urban_poly_world_flipped else urban_poly_world
  # Stroke in the same colour as the fill: most settlements are a few km across
  # and would be sub-pixel at world scale, so fill alone leaves them invisible.
  geom_sf(data = g, fill = URBAN_COL, colour = URBAN_COL, linewidth = 0.16)
}

#' A one-key legend for the GUPPD urban layer.
#'
#' The layer itself is a geom_sf with a fixed colour, so it contributes no
#' guide. The key therefore comes from an empty dummy layer mapped to `shape`,
#' the only aesthetic the map does not already spend (fill = LCZ class,
#' colour = split, size = patch count). `order` puts it on the same row as, and
#' after, the patch-count legend.
urban_key <- function(urban = "polygons") {
  if (urban == "none") return(NULL)
  list(
    geom_point(data = tibble(X = NA_real_, Y = NA_real_),
               aes(x = X, y = Y, shape = "Urban areas (GUPPD)"), na.rm = TRUE),
    scale_shape_manual(
      name = NULL, values = c("Urban areas (GUPPD)" = 16),
      guide = guide_legend(
        order = 2, override.aes = list(colour = URBAN_COL, size = 1.8))),
    guides(size = guide_legend(order = 1))
  )
}

# City points, shared by the pie panels below.

cities_sf <- st_as_sf(cities, coords = c("lon", "lat"), crs = 4326) |>
  st_transform(EQUAL_EARTH)

# ══════════════════════════════════════════════════════════════════════════════
# Pie-map machinery ─ wedges, panels and inset assembly
# ══════════════════════════════════════════════════════════════════════════════
#
# No class or split legend here by design: the Figure 3 bars serve as the legend.

# Pie radius is constant within a panel so the class mix stays legible for every
# city; patch count is carried by the centre dot instead. Projected metres.
PIE_RADIUS <- 3.7e5

#' Turn per-city class shares into wedge polygons around each city centre.
#'
#' The wedge geometry itself lives in R/composition.R, so the standalone pie and
#' the 52 pies on the map are the same mark.
build_wedges <- function(df, radius, n_seg = 96) {
  pmap(df, function(city_dir, X, Y, lcz, a0, a1, ...) {
    wedge_arc(X, Y, radius, a0, a1, n_seg) |>
      mutate(wedge_id = paste(city_dir, lcz, sep = "|"), lcz = lcz)
  }) |> list_rbind()
}

city_xy <- cities_sf |>
  mutate(X = st_coordinates(geometry)[, 1],
         Y = st_coordinates(geometry)[, 2]) |>
  st_drop_geometry() |>
  select(city_dir, city_label, X, Y, role, n_total, split)

wedge_input <- city_classes |>
  inner_join(select(city_xy, city_dir, X, Y), by = "city_dir") |>
  group_by(city_dir) |>
  arrange(LCZ_class, .by_group = TRUE) |>
  mutate(frac = n / sum(n),
         a1 = cumsum(frac) * 2 * pi,
         a0 = a1 - frac * 2 * pi,
         lcz = as.character(lcz_factor(LCZ_class, "alt"))) |>
  ungroup() |>
  select(city_dir, X, Y, lcz, a0, a1)

#' One pie-map panel. `xlim`/`ylim` are projected extents; NULL means whole world.
pie_panel <- function(radius, xlim = NULL, ylim = NULL, dot_range = c(0.7, 3.4),
                      wedge_lw = 0.08, tag = NULL,
                      tag_corner = c("tl", "tr", "bl", "br"),
                      tag_size = 3.4, extents = NULL, urban = "polygons",
                      axes = FALSE, zoomed = !is.null(xlim), flip = FALSE) {
  tag_corner <- match.arg(tag_corner)
  # Mirroring the *centres* rather than the finished wedges keeps each pie
  # upright and clockwise: only the position of the pie moves, not its drawing.
  wedges <- build_wedges(
    if (flip) mutate(wedge_input, Y = -Y) else wedge_input, radius) |>
    mutate(lcz = factor(lcz, levels = LCZ_TABLE$alt_code))
  dots <- if (flip) mutate(city_xy, Y = -Y) else city_xy
  base_world <- if (flip) world_flipped else world

  p <- ggplot() +
    geom_sf(data = base_world, fill = LAND_FILL, colour = BORDER_COL,
            linewidth = BORDER_LW) +
    urban_layer(urban, xlim, ylim, zoomed = zoomed, flip = flip)

  if (!is.null(extents)) {
    p <- p + geom_sf(data = extents, fill = NA, colour = "grey25",
                     linewidth = 0.4, linetype = "22")
  }

  p <- p +
    geom_polygon(data = wedges,
                 aes(x = x, y = y, group = wedge_id, fill = lcz),
                 colour = "grey35", linewidth = wedge_lw) +
    # Centre dot in two layers so the split uses the *colour* scale, leaving
    # `fill` for the LCZ classes: a solid body under a thin dark ring that keeps
    # the white (held-out) dots legible on pale wedges.
    geom_point(data = dots,
               aes(x = X, y = Y, size = n_total, colour = split), shape = 19) +
    geom_point(data = dots, aes(x = X, y = Y, size = n_total),
               shape = 21, fill = NA, colour = "grey20", stroke = 0.3) +
    scale_colour_manual(values = ROLE_DOT_FILL, guide = "none") +
    scale_fill_manual(values = LCZ_COLOURS, guide = "none", drop = FALSE) +
    scale_city_size(range = dot_range, name = "# of Patches")

  # Panel label in data coordinates: plot.tag escapes the panel under patchwork,
  # which would put it in the margin instead of on the map.
  if (!is.null(tag) && !is.null(xlim) && !is.null(ylim)) {
    at_left <- tag_corner %in% c("tl", "bl")
    at_top  <- tag_corner %in% c("tl", "tr")
    p <- p + annotate(
      "text",
      x = if (at_left) xlim[1] + 0.035 * diff(xlim) else xlim[2] - 0.035 * diff(xlim),
      y = if (at_top)  ylim[2] - 0.045 * diff(ylim) else ylim[1] + 0.045 * diff(ylim),
      label = tag, hjust = if (at_left) 0 else 1, vjust = if (at_top) 1 else 0,
      fontface = "bold", size = tag_size, colour = "grey20")
  }

  p <- p + coord_sf(crs = EQUAL_EARTH, xlim = xlim, ylim = ylim,
                    expand = FALSE) +
    theme_eofm_map()

  # Meridian and parallel labels are both drawn inside the panel with geom_text
  # rather than on the axes. coord_sf can only label a graticule line where it
  # meets the panel edge, and on this projection neither family reliably does:
  # parallels stop well short of the left and right edges (the panel is only as
  # wide as the equator), and the meridians converge so hard towards the Arctic
  # that only the central one is found at the top. Placing the labels by hand
  # also means the south-up variant needs no special case -- the anchors mirror
  # with everything else.
  if (axes) {
    # Parallels are drawn inside the map rather than on the y axis. In an
    # equal-area pseudocylindrical projection they stop well short of the panel
    # edge -- the panel is only as wide as the equator -- so coord_sf can label
    # the equator and nothing else. Meridians do reach the top edge, so they
    # stay as ordinary axis labels.
    lat_breaks <- c(-60, -30, 0, 30, 60)
    # Placed on the eastern side: the western half of the map is where the
    # Europe and US West Coast insets sit, and they cover the 30N/30S labels.
    lat_pts <- st_as_sf(data.frame(lon = 172, lat = lat_breaks),
                        coords = c("lon", "lat"), crs = 4326) |>
      st_transform(EQUAL_EARTH)
    if (flip) lat_pts <- mirror_sf(lat_pts)
    lat_lab <- tibble(
      X = st_coordinates(lat_pts)[, 1], Y = st_coordinates(lat_pts)[, 2],
      lab = ifelse(lat_breaks == 0, "0\u00b0",
                   paste0(abs(lat_breaks), "\u00b0",
                          ifelse(lat_breaks > 0, "N", "S"))))
    # Antarctica is dropped, so the panel stops near 55S and the 60S label would
    # be drawn half-clipped on the bottom edge. Keep only parallels with room.
    if (!is.null(ylim)) {
      pad <- 0.03 * diff(range(ylim))
      lat_lab <- filter(lat_lab, Y > min(ylim) + pad, Y < max(ylim) - pad)
    }
    # Meridians, labelled at the top of the panel. The anchor is the point where
    # each meridian meets the outline at the panel's topmost latitude, so the
    # labels sit in the white space above the map and still line up with the
    # graticule lines they name. Nudged off the antimeridian so both ends of the
    # map get a "180" rather than one label landing on the seam.
    lon_breaks <- seq(-180, 180, 60)
    lon_pts <- st_as_sf(
      data.frame(lon = lon_breaks * (1 - 1e-6),
                 lat = if (flip) WORLD_LAT_MIN else WORLD_LAT_MAX),
      coords = c("lon", "lat"), crs = 4326) |>
      st_transform(EQUAL_EARTH)
    if (flip) lon_pts <- mirror_sf(lon_pts)
    lon_lab <- tibble(
      X = st_coordinates(lon_pts)[, 1], Y = st_coordinates(lon_pts)[, 2],
      lab = paste0(abs(lon_breaks), "\u00b0",
                   ifelse(lon_breaks == 0 | abs(lon_breaks) == 180, "",
                          ifelse(lon_breaks > 0, "E", "W"))))

    p <- p +
      # Tropics, drawn but deliberately unlabelled. Equal Earth is
      # pseudocylindrical, so every parallel is a straight horizontal line and a
      # projected-y hline is exact -- no need to densify a lon/lat line.
      geom_segment(data = TROPIC_LINES,
                   aes(x = x, xend = xend, y = y, yend = yend),
                   inherit.aes = FALSE, colour = "grey78",
                   linetype = "22", linewidth = 0.3) +
      geom_text(data = lat_lab, aes(x = X, y = Y, label = lab),
                inherit.aes = FALSE, hjust = 1, vjust = -0.35,
                size = 11 * 0.62 / .pt, colour = "grey45") +
      geom_text(data = lon_lab, aes(x = X, y = Y, label = lab),
                inherit.aes = FALSE, hjust = 0.5, vjust = -0.5,
                size = 11 * 0.62 / .pt, colour = "grey45") +
      # Graticule every 30 degrees, labelled every 60.
      scale_x_continuous(breaks = seq(-180, 180, 30)) +
      scale_y_continuous(breaks = seq(-90, 90, 30)) +
      theme(plot.margin = margin(1, 3, 1, 1))
  }
  p
}

#' Projected bbox of a lon/lat window, densified so the edges curve correctly.
latlon_window <- function(xmin, ymin, xmax, ymax) {
  st_sf(geometry = st_sfc(st_polygon(list(rbind(
    c(xmin, ymin), c(xmax, ymin), c(xmax, ymax), c(xmin, ymax), c(xmin, ymin)
  ))), crs = 4326)) |>
    st_segmentize(units::set_units(50, "km")) |>
    st_transform(EQUAL_EARTH)
}

bx <- function(w) as.numeric(st_bbox(w))[c(1, 3)]
by <- function(w) as.numeric(st_bbox(w))[c(2, 4)]

# The tropics, as segments rather than hlines. Parallels are straight lines in
# Equal Earth, but they are also *shorter* than the equator, and the panel is
# padded past the projection outline (see WORLD_XLIM) -- an hline would run the
# full panel width and stick out either side of the map. Ending each segment at
# the projection edge for its own latitude keeps it inside the outline.
TROPIC_LAT <- 23.43663
TROPIC_LINES <- local({
  ends <- expand.grid(lon = c(-179.999, 179.999), lat = c(TROPIC_LAT, -TROPIC_LAT))
  xy <- st_coordinates(st_transform(
    st_as_sf(ends, coords = c("lon", "lat"), crs = 4326), EQUAL_EARTH))
  tibble(x = xy[c(1, 3), 1], xend = xy[c(2, 4), 1],
         y = xy[c(1, 3), 2], yend = xy[c(2, 4), 2])
})

# The world panel is drawn with explicit limits rather than coord_sf's automatic
# ones: `expand = FALSE` clips exactly to the data bbox, which cuts the outline
# stroke in half at the widest point of the projection (the equator) and at the
# poleward extremes. Pad both axes by a small fraction of the extent.
WORLD_BBOX <- as.numeric(st_bbox(world))
WORLD_PAD_X <- 0.018   # fraction of the world's projected width, per side
WORLD_PAD_Y <- 0.032   # also the headroom the meridian labels sit in
WORLD_XLIM <- WORLD_BBOX[c(1, 3)] +
  c(-1, 1) * WORLD_PAD_X * diff(WORLD_BBOX[c(1, 3)])
WORLD_YLIM <- WORLD_BBOX[c(2, 4)] +
  c(-1, 1) * WORLD_PAD_Y * diff(WORLD_BBOX[c(2, 4)])
# Panel width / height, needed to size the inset boxes (see build_pie_map).
WORLD_ASPECT <- diff(WORLD_XLIM) / diff(WORLD_YLIM)

EUROPE_WIN    <- latlon_window(-11, 35, 21, 56)
# East edge stops just past Tokyo (139.8E): far enough to clear its pie, close
# enough to keep the inset box off Australia.
EAST_ASIA_WIN <- latlon_window(107, 18, 140, 45)

# Insets sit on empty ocean inside the map. Opaque, so the world beneath does
# not show through.
inset_style <- theme(
  panel.background = element_rect(fill = "white", colour = NA),
  plot.background  = element_rect(fill = "white", colour = "grey25",
                                  linewidth = 0.5),
  plot.margin = margin(1, 1, 1, 1)
)

#' Shrink one inset box to its window's aspect, anchored at its bottom-left.
#'
#' An inset panel is aspect-constrained by coord_sf, so patchwork collapses its
#' gtable to that fixed shape and *centres* it in the allotted viewport --
#' background and border included. A box whose aspect does not match its
#' window's therefore floats inside `pos`, which is why East Asia's bottom
#' border sat above Europe's and Brazil's despite all three sharing `bottom`.
#'
#' Doing the fit here instead removes the slack: the box keeps `pos` as its
#' bound and is shrunk on whichever axis has room, anchored at the bottom-left,
#' so boxes sharing a `bottom` always align regardless of window shape.
#'
#' `anchor = "top"` fixes the top edge instead, which is what the south-up map
#' needs: mirroring `pos` turns a shared bottom edge into a shared top one.
inset_box <- function(pos, win, anchor = c("bottom", "top")) {
  anchor <- match.arg(anchor)
  w <- pos[3] - pos[1]
  h <- pos[4] - pos[2]
  # Box height per unit box width, for this window, in panel fractions.
  ratio <- WORLD_ASPECT * (diff(by(win)) / diff(bx(win)))
  if (w * ratio <= h) h <- w * ratio else w <- h / ratio
  if (anchor == "bottom") c(pos[1], pos[2], pos[1] + w, pos[2] + h)
  else                    c(pos[1], pos[4] - h, pos[1] + w, pos[4])
}

#' Mirror an inset box vertically within the panel, so a box tuned against a
#' patch of empty ocean lands on that same patch once the map is flipped.
mirror_pos <- function(pos) c(pos[1], 1 - pos[4], pos[3], 1 - pos[2])

#' Assemble a pie map from a world panel plus a list of inset region specs.
build_pie_map <- function(regions, urban = "polygons", flip = FALSE) {
  extents <- do.call(rbind, lapply(regions, `[[`, "win"))
  if (flip) extents <- mirror_sf(extents)
  main <- pie_panel(PIE_RADIUS, xlim = WORLD_XLIM,
                    ylim = if (flip) mirror_ylim(WORLD_YLIM) else WORLD_YLIM,
                    zoomed = FALSE, extents = extents, urban = urban,
                    axes = TRUE, flip = flip) +
    urban_key(urban) +
    theme(legend.position = "bottom",
          legend.box = "horizontal",
          legend.margin = margin(t = 0, b = 0),
          legend.box.spacing = unit(4, "pt"))

  Reduce(function(acc, r) {
    pos <- inset_box(if (flip) mirror_pos(r$pos) else r$pos, r$win,
                     anchor = if (flip) "top" else "bottom")
    panel <- pie_panel(diff(bx(r$win)) / r$div,
                       xlim = bx(r$win),
                       ylim = if (flip) mirror_ylim(by(r$win)) else by(r$win),
                       dot_range = c(0.7, 3.0), wedge_lw = 0.12,
                       tag = r$name, tag_size = r$tag_size,
                       tag_corner = if (is.null(r$corner)) "tl" else r$corner,
                       urban = urban, flip = flip) +
      guides(size = "none") +
      # No panel.border: the panel is aspect-constrained and sits letterboxed
      # inside its allotted box, so a panel border would not line up between
      # insets. The box edge (inset_style's plot.background) is the border, and
      # boxes sharing a `bottom` in `pos` therefore align exactly.
      inset_style
    acc + patchwork::inset_element(panel, left = pos[1], bottom = pos[2],
                                   right = pos[3], top = pos[4],
                                   align_to = "panel")
  }, regions, main)
}

# ══════════════════════════════════════════════════════════════════════════════
# Figure 6 ─ Pie map with five regional insets
# ══════════════════════════════════════════════════════════════════════════════
#
# Every cluster that
# overlaps at world scale gets a zoom, each parked on empty ocean near its own
# region. `pos` is left/bottom/right/top in panel fractions, chosen so each box
# matches its window's projected aspect (otherwise the zoom letterboxes inside
# the frame); `div` sets the pie radius as a fraction of the window width, tuned
# so pies render at a similar physical size across boxes of different widths.

REGIONS <- list(
  # Window reaches to 58N and 130W so there is empty sea above Vancouver and
  # west of the coast for the tag to sit in; without that headroom the label
  # collides with Vancouver at the top or Los Angeles at the bottom.
  list(name = "US West Coast", tag_size = 3.1, corner = "tl",
       win = latlon_window(-128, 29, -115, 58),
       div = 12, pos = c(0.072, 0.470, 0.172, 0.970)),
  # Wedged between three pies: Caracas ends at ~0.335 npc, its own cities
  # (Washington, Philadelphia, New York) bottom out at ~0.723, and Lisbon
  # begins at ~0.467. Up and right off Caracas, splitting the other two gaps.
  list(name = "US East Coast", tag_size = 3.1,
       win = latlon_window(-80, 36, -70, 44),
       div = 15, pos = c(0.339, 0.504, 0.457, 0.764)),
  list(name = "Europe", tag_size = 3.1, win = EUROPE_WIN,
       div = 25, pos = c(0.082, 0.035, 0.272, 0.395)),
  # Kept clear of Cape Town, whose pie starts at ~0.536 npc, without backing
  # into Rio de Janeiro's, which ends at ~0.396.
  list(name = "Southern Brazil", tag_size = 3.1,
       win = latlon_window(-50, -27, -40, -20),
       div = 17, pos = c(0.403, 0.035, 0.542, 0.275)),
  # Boxed in on three sides: Madagascar ends at ~0.645 npc, Australia begins at
  # ~0.85, and Jakarta's pie reaches down to ~0.406.
  list(name = "East Asia", tag_size = 3.1, win = EAST_ASIA_WIN,
       div = 27, pos = c(0.648, 0.035, 0.818, 0.383))
)

message("Figure 6: city pie map, regional insets")
pie_map_poly <- build_pie_map(REGIONS, "polygons")
pie_map_points <- build_pie_map(REGIONS, "points")
save_plot(pie_map_poly, "so2sat_city_map_pies_regional", width = 12, height = 6.3,
            subdir = PLOT_DIR_DATASET)
save_plot(pie_map_points, "so2sat_city_map_pies_regional_urban_points", width = 12, height = 6.3,
            subdir = PLOT_DIR_DATASET)

# South-up variants. Same regions and same tuned inset boxes: mirror_pos maps
# each box onto the mirror image of the ocean it was placed in.
message("Figure 6b: city pie map, south up")
pie_map_poly_flipped   <- build_pie_map(REGIONS, "polygons", flip = TRUE)
pie_map_points_flipped <- build_pie_map(REGIONS, "points",   flip = TRUE)
save_plot(pie_map_poly_flipped, "so2sat_city_map_pies_regional_south_up",
          width = 12, height = 6.3,
            subdir = PLOT_DIR_DATASET)
save_plot(pie_map_points_flipped,
          "so2sat_city_map_pies_regional_urban_points_south_up",
          width = 12, height = 6.3,
            subdir = PLOT_DIR_DATASET)

# ══════════════════════════════════════════════════════════════════════════════
# Figure 7 ─ Merged panel: LCZ classes | map | data split
# ══════════════════════════════════════════════════════════════════════════════
#
# The two bars double as the map's legend, which is why the map itself carries
# no class or split key.
#
# Rather than three columns side by side, the bars are added as insets over the
# map so they can overlap it: the ocean at the far left and right of an Equal
# Earth world is empty, and letting the bars sit in it buys back roughly a fifth
# of the figure width. The two constants below are the only things to tune --
# left/bottom/right/top in fractions of the *map panel*, so values below 0 or
# above 1 push a bar off the map and into the margin.

message("Figure 7: merged panel")

# Each bar's own panel puts the bar at one end and the labels at the other
# (`side` in composition_bar), so these boxes are mostly label column: the LCZ
# bar itself lands at ~20% across its box, the split bar at ~88%. That is what
# keeps the two bars near the edges of the map while their labels reach inwards
# over the empty Pacific and Indian ocean.
#
# The whole box is also dropped below the panel by CAPTION_DROP so that each
# bar's caption lands on the map's size legend, which sits *outside* the panel:
# a caption is the last row of its bar's own layout, so aligning the two means
# pushing the bar boxes down past the panel edge by the legend's own offset.
# Panel fractions. Re-measure this whenever the figure height, the map's
# vertical padding (WORLD_PAD_Y) or legend.box.spacing changes: all three move
# the size legend relative to the panel, and the captions have to follow it.
CAPTION_DROP  <- 0.052
CLASS_BAR_POS <- c(-0.030, 0.010, 0.085, 0.995) - c(0, 1, 0, 1) * CAPTION_DROP
SPLIT_BAR_POS <- c( 0.875, 0.010, 1.030, 0.995) - c(0, 1, 0, 1) * CAPTION_DROP

# Room for the parts of the bars that now overhang the panel left and right.
OVERVIEW_MARGIN <- margin(2, 30, 2, 30)

#' Lay the two composition bars over a pie map as insets.
overview_panel <- function(map) {
  map +
    patchwork::inset_element(
      class_bar_plot, left = CLASS_BAR_POS[1], bottom = CLASS_BAR_POS[2],
      right = CLASS_BAR_POS[3], top = CLASS_BAR_POS[4],
      align_to = "panel", on_top = TRUE, clip = FALSE) +
    patchwork::inset_element(
      split_bar_plot, left = SPLIT_BAR_POS[1], bottom = SPLIT_BAR_POS[2],
      right = SPLIT_BAR_POS[3], top = SPLIT_BAR_POS[4],
      align_to = "panel", on_top = TRUE, clip = FALSE) +
    # Sets only the outer composition's canvas; using `&` here would also strip
    # the white backing the insets need to sit on top of the map.
    patchwork::plot_annotation(
      theme = theme(plot.background = element_rect(fill = "transparent",
                                                   colour = NA),
                    plot.margin = OVERVIEW_MARGIN))
}

save_plot(overview_panel(pie_map_poly), "so2sat_overview_panel",
          width = 13, height = 6.3,
            subdir = PLOT_DIR_DATASET)
save_plot(overview_panel(pie_map_points), "so2sat_overview_panel_points",
          width = 13, height = 6.3,
            subdir = PLOT_DIR_DATASET)

# The bars are deliberately NOT mirrored: only the map is south-up, so the LCZ
# and split keys keep reading top-to-bottom as everywhere else in the paper.
save_plot(overview_panel(pie_map_poly_flipped), "so2sat_overview_panel_south_up",
          width = 13, height = 6.3,
            subdir = PLOT_DIR_DATASET)
save_plot(overview_panel(pie_map_points_flipped),
          "so2sat_overview_panel_points_south_up", width = 13, height = 6.3,
            subdir = PLOT_DIR_DATASET)

message("Done.")
