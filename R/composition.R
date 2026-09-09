# composition.R ─ One stacked composition bar, and one pie, from a table of shares.
#
# Shared by R/plotting.R (the dataset figures) and R/lcz_composition.R (the same
# two marks built from a raster). Not a script: source it.
#
# Every function takes a data frame of `key` (a factor, drawn in level order),
# `share` and `n`, and a named palette keyed by the same levels. `key` order is
# the drawing order, so the caller decides whether the bar reads 1..G or by
# descending share.

source("R/constants.R")

suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
  library(purrr)
  library(scales)
  library(tibble)
  library(systemfonts)
  library(patchwork)
})

#' Lay out one stacked bar's segments along 0-1 and place a label per segment.
#'
#' `flip = FALSE` stacks downwards from the top (first key at the top of a
#' vertical bar); `flip = TRUE` stacks left to right.
#' Share as a percentage, e.g. "13.6%".
fmt_share <- function(share, n) percent(share, accuracy = 0.1)

#' Rounded absolute count, e.g. "~352k". Rounded to the nearest thousand (the
#' nearest 0.1M above a million), so 352,366 reads as ~352k rather than losing
#' the leading digits to a coarser rounding.
fmt_count <- function(share, n) {
  dplyr::case_when(
    n >= 1e6 ~ paste0("~", round(n / 1e6, 1), "M"),
    n >= 1e3 ~ paste0("~", round(n / 1e3), "k"),
    TRUE     ~ paste0("~", n)
  )
}

#' Lay out one stacked bar's segments along 0-1 and label each.
#'
#' `lo`/`hi` are positions *along* the bar and `pos` is where the label sits on
#' that same axis, so the caller decides which screen axis that is. Vertical
#' stacks downwards from the top (the first key at the top); horizontal stacks
#' left to right, which is the reading order for the same list.
bar_segments <- function(df, fmt = fmt_share, horizontal = FALSE,
                         gap = if (horizontal) LABEL_GAP_H else LABEL_GAP) {
  # A `glyph` column, when present, prefixes each label with the marker the map
  # uses for that category.
  key_lab <- if ("glyph" %in% names(df)) paste(df$glyph, as.character(df$key))
             else as.character(df$key)
  df |>
    mutate(key_lab = key_lab, value = fmt(share, n),
           hi = if (horizontal) cumsum(share) else 1 - cumsum(share) + share,
           lo = if (horizontal) cumsum(share) - share else 1 - cumsum(share),
           mid = (hi + lo) / 2,
           pos = spread_labels(mid, gap),
           label = sprintf("%s (%s)", key_lab, value))
}

BAR_T   <- 0.16   # bar thickness on the non-share axis (thin by design)
LAB_PAD <- 0.05   # gap between the bar edge and its label column

# The same, for a horizontal bar. Its labels are rotated 90 degrees, so what
# has to clear along the bar is the *line height*, not the line length -- which
# is the only reason 17 LCZ labels fit along one bar at all. Unlike the vertical
# bar, whose height is fixed by the figures it sits in, a horizontal bar's
# width varies, so the gap is computed from it: horizontal_gap(width).
LABEL_GAP_H <- 0.030

#' Minimum label separation along a horizontal bar `width` inches wide, as a
#' fraction of that width -- one line of type plus a fifth for air.
horizontal_gap <- function(width, pt = LEGEND_TEXT_PT) {
  (pt / 72) * 1.2 / width
}

#' Across-axis extent a horizontal bar needs, in inches, to fit its own rotated
#' labels: the label column is `label_room` units and the bar plus its pad make
#' up the rest, so the whole panel scales off the longest label.
horizontal_height <- function(labels, label_room = 1, pt = LEGEND_TEXT_PT) {
  lab_in <- max(systemfonts::string_width(labels, size = pt, res = 72)) / 72
  (BAR_T + LAB_PAD + label_room) / label_room * lab_in
}

# One line of label type as a fraction of the bar's height. The bars are ~6.2 in
# tall as insets in the overview and 7 in standalone, so this is sized for the
# shorter of the two: too large only costs a little extra spread in the crowded
# runs, too small lets lines touch.
LABEL_GAP <- 0.026

#' Pool adjacent violators: the L2-optimal non-decreasing fit to `v`.
pava <- function(v) {
  vals <- numeric(0)
  wts  <- integer(0)
  for (x in v) {
    vals <- c(vals, x)
    wts  <- c(wts, 1L)
    while (length(vals) > 1 && vals[length(vals) - 1] > vals[length(vals)]) {
      k <- length(vals)
      w <- wts[k - 1] + wts[k]
      vals[k - 1] <- (vals[k - 1] * wts[k - 1] + vals[k] * wts[k]) / w
      wts[k - 1] <- w
      vals <- vals[-k]
      wts  <- wts[-k]
    }
  }
  rep(vals, wts)
}

#' Place stacked-bar labels at their segment centres, spread only where needed.
#'
#' Every label wants to sit at the vertical centre of the segment it names, but
#' the thin classes (E at 0.7%, 7 at 1.1%) are closer together than one line of
#' type. That is the one-dimensional label placement problem: minimise the total
#' squared displacement from the wanted centres subject to a minimum gap, which
#' pool-adjacent-violators solves exactly. So every label with room stays
#' *exactly* centred and only the crowded runs are pushed apart -- where ggrepel
#' pushes on everything, drifting labels that had no reason to move.
#'
#' The run is also kept inside [0, 1] with half a line to spare at each end:
#' the topmost class is a 1.4% sliver, so its centre alone would put half the
#' line off the top of the figure. Clamping the fitted values (rather than the
#' final positions) preserves both the ordering and the gap, and is still the
#' L2-optimal answer under the added box constraint.
spread_labels <- function(mid, gap = LABEL_GAP) {
  o <- order(mid)
  i <- seq_along(mid)
  n <- length(mid)
  m <- pava(mid[o] - i * gap)
  m <- pmin(pmax(m, gap / 2 - gap), 1 - gap / 2 - n * gap)
  (m + i * gap)[order(o)]
}

#' A single thin vertical stacked bar: no share axis, one bold label per segment
#' sitting outside the bar, hugging its near edge and centred on its own segment.
#'
#' `side` is where the labels go relative to the bar. Type sizes come from the
#' shared LEGEND_* constants so these read as one system with the map legend.
#' Diagonal hatch segments clipped to a rectangle, for texturing one bar segment.
#'
#' `m` is the slope in data units; the panel is far from square, so the value
#' that reads as 45 degrees on screen is not 1.
hatch_lines <- function(x0, x1, y0, y1, m = 0.4, n = 9) {
  cs <- seq(y0 - m * (x1 - x0), y1, length.out = n + 2)
  cs <- cs[-c(1, length(cs))]
  map(cs, function(cc) {
    xa <- max(x0, x0 + (y0 - cc) / m)
    xb <- min(x1, x0 + (y1 - cc) / m)
    if (xb <= xa) return(NULL)
    tibble(x = xa, xend = xb, y = m * (xa - x0) + cc, yend = m * (xb - x0) + cc)
  }) |> list_rbind()
}

composition_bar <- function(df, palette, side = NULL, title = NULL,
                            label_size = LABEL_SIZE_MM, label_room = 1.0,
                            border_lw = 0.25, fmt = fmt_share,
                            hatch_keys = character(0), horizontal = FALSE,
                            labels = TRUE,
                            gap = if (horizontal) LABEL_GAP_H else LABEL_GAP) {
  if (is.null(side)) side <- if (horizontal) "bottom" else "left"
  sides <- if (horizontal) c("bottom", "top") else c("left", "right")
  if (!side %in% sides) {
    stop("side must be one of ", paste(sides, collapse = "/"),
         " for a ", if (horizontal) "horizontal" else "vertical", " bar.",
         call. = FALSE)
  }
  if (horizontal && length(hatch_keys)) {
    stop("hatch_keys is only implemented for the vertical bar.", call. = FALSE)
  }
  seg <- bar_segments(df, fmt = fmt, horizontal = horizontal, gap = gap)
  # Unlabelled, the bar is a bare strip filling its panel edge to edge: no
  # label column and no pad, so its saved aspect ratio is exactly BAR_T and it
  # can be butted against a map of matching size.
  if (!labels) { label_room <- 0; pad <- 0 } else pad <- LAB_PAD
  # `near` = the labels are on the low side of the across-axis, so the bar is
  # pushed up/right to leave them room.
  near <- side %in% c("left", "bottom")
  bar_lo <- if (near) label_room + pad else 0
  lab_at <- if (near) bar_lo - pad else BAR_T + pad
  across_max <- if (near) bar_lo + BAR_T else BAR_T + pad + label_room

  # Texture for any segment named in `hatch_keys`, drawn over its fill.
  hatched <- seg |> filter(as.character(key) %in% hatch_keys)
  hatch <- if (nrow(hatched) > 0) {
    map(seq_len(nrow(hatched)), function(i) {
      hatch_lines(bar_lo, bar_lo + BAR_T, hatched$lo[i], hatched$hi[i])
    }) |> list_rbind()
  } else NULL

  # The only orientation-dependent part: which screen axis carries the stack.
  rect <- if (horizontal) {
    geom_rect(aes(xmin = lo, xmax = hi, ymin = bar_lo, ymax = bar_lo + BAR_T,
                  fill = key), colour = "grey25", linewidth = border_lw)
  } else {
    geom_rect(aes(xmin = bar_lo, xmax = bar_lo + BAR_T, ymin = lo, ymax = hi,
                  fill = key), colour = "grey25", linewidth = border_lw)
  }
  # Rotated a quarter turn on a horizontal bar; `hjust` still measures along the
  # text, so it points away from the bar in both orientations.
  text <- if (!labels) NULL else if (horizontal) {
    geom_text(aes(x = pos, y = lab_at, label = label), angle = 90,
              hjust = if (near) 1 else 0, vjust = 0.5, size = label_size)
  } else {
    geom_text(aes(x = lab_at, y = pos, label = label),
              hjust = if (near) 1 else 0, vjust = 0.5, size = label_size)
  }

  ggplot(seg) +
    rect +
    (if (!is.null(hatch))
       geom_segment(data = hatch, aes(x = x, xend = xend, y = y, yend = yend),
                    colour = "grey25", linewidth = 0.3, inherit.aes = FALSE)) +
    text +
    scale_fill_manual(values = palette, guide = "none") +
    # expand = FALSE matters: coord_cartesian otherwise pads the range by 5% at
    # each end, so the bar fills only ~91% of the panel and the leftover shows
    # up as a gap between the bar and the caption.
    coord_cartesian(xlim = if (horizontal) c(0, 1) else c(0, across_max),
                    ylim = if (horizontal) c(0, across_max) else c(0, 1),
                    expand = FALSE, clip = "off") +
    labs(caption = title) +
    theme_void(base_size = 11) +
    theme(
      plot.caption = element_text(face = "bold", size = LEGEND_TITLE_PT,
                                  hjust = 0.5, margin = margin(t = 4, b = 0)),
      plot.caption.position = "panel",
      plot.background  = element_rect(fill = "transparent", colour = NA),
      panel.background = element_rect(fill = "transparent", colour = NA),
      plot.margin = margin(0, 0, 0, 0)
    )
}

#' Make an assembled patchwork figure transparent.
#'
#' `save_plot()` writes with `bg = "transparent"` and every theme here paints a
#' transparent background, but patchwork adds a plot.background of its own from
#' `theme_get()` -- white -- over the assembled figure, so a combined plot comes
#' out opaque unless this is added. R/plotting.R does the same thing for its
#' overview panels.
transparent_patchwork <- function() {
  patchwork::plot_annotation(
    theme = theme(plot.background = element_rect(fill = "transparent",
                                                 colour = NA)))
}

# ── Pie ───────────────────────────────────────────────────────────────────────

#' One wedge as a closed polygon: centre, then the arc.
#'
#' A wedge is centre + arc, so geom_polygon strokes the two radii that close it.
#' When the wedge is the whole circle those two radii coincide and draw a
#' spurious line at 12 o'clock, so the centre vertex is dropped there: the
#' polygon becomes a plain ring closing arc-end to arc-start at the same point,
#' and only the circle outline is stroked. Angle 0 is north, sweeping clockwise.
wedge_arc <- function(x0, y0, r, a0, a1, n_seg = 96) {
  k <- max(3L, ceiling(n_seg * (a1 - a0) / (2 * pi)))
  a <- seq(a0, a1, length.out = k)
  full <- (a1 - a0) > 2 * pi - 1e-9
  tibble(x = c(if (full) NULL else x0, x0 + r * sin(a)),
         y = c(if (full) NULL else y0, y0 + r * cos(a)))
}

#' A single pie of one composition, on the same palette as its bar.
#'
#' No labels and no key: this is meant to sit beside `composition_bar()`, which
#' is the legend -- the same division of labour the city pie map uses.
pie_chart <- function(df, palette, border_lw = 0.35, title = NULL,
                      n_seg = 720) {
  d <- df |>
    mutate(a1 = cumsum(share) * 2 * pi, a0 = a1 - share * 2 * pi)
  wedges <- pmap(d, function(key, a0, a1, ...) {
    wedge_arc(0, 0, 1, a0, a1, n_seg) |> mutate(key = key)
  }) |> list_rbind() |>
    mutate(key = factor(as.character(key), levels = levels(df$key)))

  ggplot(wedges, aes(x = x, y = y, group = key, fill = key)) +
    geom_polygon(colour = "grey25", linewidth = border_lw) +
    scale_fill_manual(values = palette, guide = "none") +
    coord_fixed(xlim = c(-1, 1), ylim = c(-1, 1), expand = TRUE) +
    labs(caption = title) +
    theme_void(base_size = 11) +
    theme(
      plot.caption = element_text(face = "bold", size = LEGEND_TITLE_PT,
                                  hjust = 0.5, margin = margin(t = 4, b = 0)),
      plot.caption.position = "panel",
      plot.background  = element_rect(fill = "transparent", colour = NA),
      panel.background = element_rect(fill = "transparent", colour = NA),
      plot.margin = margin(0, 0, 0, 0)
    )
}

