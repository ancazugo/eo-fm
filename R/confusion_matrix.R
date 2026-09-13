# confusion_matrix.R ─ LCZ confusion matrix for the paper and poster.
#
#     Rscript R/confusion_matrix.R [--run <run_name>] [--normalize true|none]
#                                  [--all] [--input <csv>]
#
# Reads data/confusion_matrices.csv (written by src/export_confusion_matrix.py —
# R has no .npy reader in this env, so the run artefacts are cached first, the
# same way R/metrics_table.R reads data/model_metrics.csv). Run from the repo root.
#
# Written to plots/:
#   confusion_matrix_<run>_proportions.png / .pdf   - rows sum to 1
#   confusion_matrix_<run>_counts.png / .pdf        - raw patch counts
#
# This is an R port of src/training/evaluate.py::save_confusion_matrix +
# style_lcz_ticklabels, with two deliberate departures:
#
#  1. NOT viridis. The matrix shares a poster with the LCZ maps and the metrics
#     table, and viridis' green midtones are CIELAB dE 1.0 from LCZ A (Dense
#     Trees) under tritanopia — indistinguishable. See CM_RAMP.
#  2. The axis chips are drawn as geometry, not as styled tick labels. `ggtext`,
#     `gridtext` and `marquee` are all absent from this conda env, so there is no
#     element_markdown() to hang a background box on. Hand-drawing is the route
#     R/metrics_table.R takes for its cells and R/split_maps.R for its hatching,
#     and it buys exact control of chip size and gap besides.
#
# The Python plot uses sklearn's *observed* labels for its axes; this one always
# draws the full 17x17, because the exported matrix is dense over all classes and
# a fixed axis makes two runs comparable cell-for-cell.

source("R/constants.R")

suppressPackageStartupMessages({
  library(readr)
  library(dplyr)
  library(scales)
})

CM_CSV      <- file.path("data", "confusion_matrices.csv")
METRICS_CSV <- file.path("data", "model_metrics.csv")

# ── Colour ────────────────────────────────────────────────────────────────────

# White -> magenta. Chosen by simulating protan/deutan/tritan (Vienot-Brettel-
# Mollon) and measuring CIELAB dE from the ramp's *saturated half* to all 17 LCZ
# colours and to the metrics table's three hues (#0072B2 blue, #946C51
# terracotta, grey45). Measuring only the saturated half is the meaningful test:
# a white-ended sequential ramp must contain a pale sample near LCZ F (#fbf7ae
# cream), so a min-over-the-whole-ramp score rejects every candidate.
#
#   ramp             worst dE vs LCZ         worst dE vs table
#   white->magenta   12.4 (LCZ 10, deutan)   12.7 (blue, protan)   <- chosen
#   white->violet    18.7                     2.9 (terracotta)
#   white->purple     7.2                     4.1
#   white->pink-red   3.4                     4.0
#   white->teal       0.9 (LCZ G, tritan)     5.6
#   viridis           1.0 (LCZ A, tritan)     3.2                  <- what this replaces
#
# It is the only candidate in double digits on both counts. L* runs 100 -> 15.6
# strictly monotonically (steps -5.9 to -11.1), so it survives greyscale, and its
# dark end is a deep plum rather than black — LCZ E *is* pure black, and the ramp
# must not converge on a class colour.
CM_RAMP <- c("#ffffff", "#f2d5ea", "#d986c4", "#b23f95", "#7b1560", "#4a0c3a")

#' Map values in [0, 1] onto CM_RAMP, returning hex.
cm_fill <- function(t, n = 256) {
  cols <- grDevices::colorRampPalette(CM_RAMP)(n)
  out  <- rep(NA_character_, length(t))
  ok   <- !is.na(t)
  out[ok] <- cols[1 + round(pmin(pmax(t[ok], 0), 1) * (n - 1))]
  out
}

# ── Geometry (inches, like R/metrics_table.R) ─────────────────────────────────

CELL_MIN   <- 0.30                   # matrix cell floor, square
NUM_PAD    <- 0.05                   # clear space inside a cell, each side
CHIP_IN    <- 0.24                   # axis chip, square
CHIP_GAP   <- 0.07                   # panel edge -> chip
TITLE_GAP  <- 0.10                   # chip -> axis title
BAR_GAP    <- 0.34                   # panel right edge -> colourbar
BAR_W      <- 0.17                   # colourbar width
BAR_LAB    <- 0.06                   # colourbar -> its tick labels
BODY_PT    <- LEGEND_TEXT_PT
# In-cell type, relative to BODY_PT. Above 1 by design: the number in the cell
# is what the figure is FOR, and it was set smaller than the body size of every
# other figure in the stack while being the thing a reader leans in to read. The
# cell sizes to its own widest label, so raising this grows the matrix with the
# type rather than crowding it.
CELL_TYPE  <- 1.10
CHIP_PT    <- LEGEND_TEXT_PT * 0.92
TITLE_PT   <- 11 * 0.95              # theme_eofm()'s axis.title size
PAD_OUT    <- 0.08
MARGIN_PT  <- 4
W_SLACK    <- 1.04
TEXT_COL   <- "grey15"
TITLE_COL  <- "grey25"
GRID_COL   <- "grey85"               # hairline between cells

#' Width of the widest string in inches, through the shaper ragg actually uses.
str_w <- function(s, pt = BODY_PT, bold = FALSE) {
  s <- s[!is.na(s)]
  if (!length(s)) return(0)
  if (requireNamespace("systemfonts", quietly = TRUE)) {
    w <- systemfonts::string_width(s, size = pt, res = 72,
                                   weight = if (bold) "bold" else "normal")
    return(max(w) / 72 * W_SLACK)
  }
  max(nchar(s)) * pt * 0.68 * (if (bold) 1.08 else 1) / 72
}

# ── Data ──────────────────────────────────────────────────────────────────────

#' Read the cached matrices, with the LCZ codes as ordered factors.
read_confusion <- function(path = CM_CSV) {
  if (!file.exists(path)) {
    stop("Missing ", path, ". Run: python src/export_confusion_matrix.py",
         call. = FALSE)
  }
  read_csv(path, show_col_types = FALSE)
}

#' The run to draw when none is named: best test_kappa on the culture-10 split.
#'
#' Culture-10 because the grid split is leakage-inflated and not the number the
#' paper argues from (see docs/global_lcz_campaign_2026-07.md).
default_run <- function(cm, metrics_path = METRICS_CSV) {
  if (file.exists(metrics_path)) {
    m <- read_csv(metrics_path, show_col_types = FALSE) |>
      filter(run_name %in% cm$run_name)
    cultural <- filter(m, split_source == "global_so2sat")
    if (nrow(cultural)) m <- cultural
    if (nrow(m)) return(m$run_name[which.max(m$test_kappa)])
  }
  sort(unique(cm$run_name))[1]
}

#' One run's matrix as a tidy frame carrying `value`, `fill` and `label`.
prepare_matrix <- function(cm, run, normalize = c("true", "none")) {
  normalize <- match.arg(normalize)
  d <- filter(cm, run_name == run)
  if (!nrow(d)) {
    stop("Unknown run '", run, "'. Available:\n  ",
         paste(sort(unique(cm$run_name)), collapse = "\n  "), call. = FALSE)
  }

  d <- d |>
    group_by(true_code) |>
    mutate(support = sum(count)) |>
    ungroup() |>
    mutate(
      # Row-normalised: the proportion of each TRUE class, so a row sums to 1.
      # A class with no test samples would divide by zero; it stays NA and is
      # painted as an empty cell rather than a misleading 0.00.
      value = if (normalize == "true") ifelse(support > 0, count / support, NA_real_)
              else count
    )

  # The fill domain is the whole matrix, not the row: the off-diagonal structure
  # is only readable if a cell means the same thing wherever it sits. Proportions
  # are pinned to [0, 1] so two runs are directly comparable; counts cannot be,
  # so they stretch to their own maximum.
  hi <- if (normalize == "true") 1 else max(d$value, na.rm = TRUE)
  d$fill <- cm_fill(d$value / hi)
  d$fill[is.na(d$fill)] <- "#ffffff"
  d$text <- unname(contrast_text(d$fill))

  fmt <- if (normalize == "true") function(v) sprintf("%.2f", v)
         else label_number(big.mark = ",", accuracy = 1)
  d$label <- ifelse(is.na(d$value), "",
                    # An exact zero is drawn as a dot: seventeen columns of
                    # "0.00" bury the diagonal in noise.
                    ifelse(d$value == 0, "·", fmt(d$value)))

  attr(d, "hi") <- hi
  attr(d, "normalize") <- normalize
  d
}

# ── The figure ────────────────────────────────────────────────────────────────

#' Confusion matrix for one run.
#'
#' Everything is positioned in cell units on both axes and the figure size is
#' derived from them, so the cells are square on the page without coord_fixed().
confusion_matrix_plot <- function(cm, run, normalize = "true", title = NULL) {
  d  <- prepare_matrix(cm, run, normalize)
  hi <- attr(d, "hi")

  lcz <- LCZ_TABLE |> arrange(code)
  n   <- nrow(lcz)                       # 17

  # The cell sizes to its widest label, never below CELL_MIN. Proportions are
  # always four characters and stay at the floor; counts are not ("3,150" is
  # half again as wide as "0.90"), and a fixed cell would let them overrun the
  # tile edges. Square cells, so this sets both axes.
  cell   <- max(CELL_MIN, str_w(d$label, BODY_PT * CELL_TYPE) + 2 * NUM_PAD)
  chip_u <- CHIP_IN / cell               # chip size, in cell units
  gap_u  <- CHIP_GAP / cell

  # x = predicted (1..n, left to right), y = true (n..1, top to bottom).
  d$px <- d$pred_code
  d$py <- n + 1 - d$true_code

  chips <- lcz |>
    transmute(code, alt_code, colour,
              text = unname(contrast_text(colour)),
              pos  = code)

  # Chip strips sit just outside the panel; clip = "off" lets them draw there.
  chip_x_y <- n + 0.5 + gap_u + chip_u / 2      # above the top row
  chip_y_x <- 0.5 - gap_u - chip_u / 2          # left of the first column

  p <- ggplot() +
    # Matrix body. A hairline border keeps the pale cells from merging into one
    # another and into the transparent background.
    geom_tile(data = d, aes(px, py, fill = fill),
              width = 1, height = 1, colour = GRID_COL, linewidth = 0.2) +
    geom_text(data = d, aes(px, py, label = label, colour = text),
              size = BODY_PT * CELL_TYPE / .pt) +
    # Predicted chips, along the top.
    geom_tile(data = chips, aes(pos, chip_x_y, fill = colour),
              width = chip_u, height = chip_u, colour = NA) +
    geom_text(data = chips, aes(pos, chip_x_y, label = alt_code, colour = text),
              size = CHIP_PT / .pt, fontface = "bold") +
    # True chips, down the left.
    geom_tile(data = chips, aes(chip_y_x, n + 1 - pos, fill = colour),
              width = chip_u, height = chip_u, colour = NA) +
    geom_text(data = chips, aes(chip_y_x, n + 1 - pos, label = alt_code,
                                colour = text),
              size = CHIP_PT / .pt, fontface = "bold") +
    scale_fill_identity() +
    scale_colour_identity()

  # ── Colourbar ───────────────────────────────────────────────────────────────
  # Drawn by hand: scale_fill_identity() cannot emit a guide, and this keeps the
  # bar's type identical to the rest of the figure.
  bar_x <- n + 0.5 + BAR_GAP / cell
  bar_w <- BAR_W / cell
  steps <- 128
  bar <- data.frame(t = (seq_len(steps) - 0.5) / steps) |>
    mutate(fill = cm_fill(t),
           y    = 0.5 + t * n)
  ticks <- if (normalize == "true") seq(0, 1, 0.25) else pretty(c(0, hi), 4)
  ticks <- ticks[ticks <= hi]
  tick_lab <- if (normalize == "true") sprintf("%.2f", ticks)
              else label_number(big.mark = ",", accuracy = 1)(ticks)

  p <- p +
    geom_tile(data = bar, aes(bar_x + bar_w / 2, y, fill = fill),
              width = bar_w, height = n / steps) +
    annotate("rect", xmin = bar_x, xmax = bar_x + bar_w,
             ymin = 0.5, ymax = 0.5 + n, fill = NA, colour = GRID_COL,
             linewidth = 0.25) +
    annotate("text", x = bar_x + bar_w + BAR_LAB / cell,
             y = 0.5 + ticks / hi * n, label = tick_lab,
             hjust = 0, size = CHIP_PT / .pt, colour = TITLE_COL)

  # ── Axis titles ─────────────────────────────────────────────────────────────
  # Rotated 90 degrees, the y title occupies its LINE HEIGHT across the page and
  # its string length down it — so the left margin is sized by the former. Using
  # the string width here (the obvious mistake) reserves ~0.8 in of empty space
  # and still lets the glyphs collide with the chips.
  title_off <- (CHIP_GAP + CHIP_IN + TITLE_GAP) / cell
  line_u    <- (TITLE_PT * 1.45 / 72) / cell
  p <- p +
    annotate("text", x = (n + 1) / 2, y = n + 0.5 + title_off,
             label = "Predicted LCZ class", size = TITLE_PT / .pt,
             colour = TITLE_COL, vjust = 0) +
    annotate("text", x = 0.5 - title_off - line_u / 2, y = (n + 1) / 2,
             label = "True LCZ class", size = TITLE_PT / .pt,
             colour = TITLE_COL, angle = 90, vjust = 0.5)

  if (!is.null(title)) {
    p <- p + annotate("text", x = 0.5 - title_off, y = n + 0.5 + title_off,
                      label = title, size = TITLE_PT / .pt * 1.05,
                      colour = TEXT_COL, fontface = "bold",
                      hjust = 0, vjust = 0)
  }

  # Panel extents in cell units, then the same numbers in inches.
  title_h  <- if (is.null(title)) 0 else (TITLE_PT * 1.05 * 1.5 / 72) / cell
  left_u   <- title_off + line_u
  right_u  <- (BAR_GAP + BAR_W + BAR_LAB) / cell +
              str_w(tick_lab, CHIP_PT) / cell
  top_u    <- title_off + line_u + title_h

  x_lim <- c(0.5 - left_u, n + 0.5 + right_u)
  y_lim <- c(0.5 - CHIP_GAP / cell, n + 0.5 + top_u)

  p <- p +
    scale_x_continuous(expand = expansion(0), limits = x_lim) +
    scale_y_continuous(expand = expansion(0), limits = y_lim) +
    coord_cartesian(clip = "off") +
    theme_eofm() +
    theme(
      axis.title = element_blank(), axis.text = element_blank(),
      axis.ticks = element_blank(),
      # A child setting beats a later parent, and theme_eofm() sets
      # panel.grid.major explicitly — so both must be blanked by name.
      panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
      plot.margin = margin(MARGIN_PT, MARGIN_PT, MARGIN_PT, MARGIN_PT)
    )

  # `|>` binds tighter than `+`, so the attributes cannot be piped off the end of
  # the layer sum — that would attach them to the theme, not the plot.
  margin_in <- 2 * MARGIN_PT / 72
  attr(p, "fig_width")  <- diff(x_lim) * cell + 2 * PAD_OUT + margin_in
  attr(p, "fig_height") <- diff(y_lim) * cell + 2 * PAD_OUT + margin_in
  p
}

# PNG only: this is a raster heat map either way, and 17x17 cells of type make a
# heavy PDF for no gain in a poster. Pass formats = c("png", "pdf") if a vector
# version is ever wanted for print.
save_confusion_matrix <- function(cm, run, normalize = "true",
                                  title = NULL, formats = "png") {
  p <- confusion_matrix_plot(cm, run, normalize, title)
  suffix <- if (normalize == "true") "proportions" else "counts"
  save_plot(p, paste0("confusion_matrix_", run, "_", suffix),
            width = attr(p, "fig_width"), height = attr(p, "fig_height"),
            formats = formats,
            subdir = PLOT_DIR_MODELS)
}

# ── CLI ───────────────────────────────────────────────────────────────────────

main <- function(args = commandArgs(trailingOnly = TRUE)) {
  run <- NULL; normalize <- "true"; all_runs <- FALSE; input <- CM_CSV
  i <- 1
  while (i <= length(args)) {
    switch(args[i],
      "--run"       = { run <- args[i + 1]; i <- i + 1 },
      "--input"     = { input <- args[i + 1]; i <- i + 1 },
      "--normalize" = { normalize <- args[i + 1]; i <- i + 1 },
      "--all"       = { all_runs <- TRUE },
      stop("Unknown argument '", args[i], "'", call. = FALSE))
    i <- i + 1
  }
  if (!normalize %in% c("true", "none")) {
    stop("--normalize must be 'true' or 'none'", call. = FALSE)
  }

  cm <- read_confusion(input)
  runs <- if (all_runs) sort(unique(cm$run_name))
          else if (!is.null(run)) run
          else default_run(cm)

  message("Confusion matrix: ", length(runs), " run(s), normalize = ", normalize)
  for (r in runs) {
    message("  ", r)
    save_confusion_matrix(cm, r, normalize = normalize)
  }
}

if (sys.nframe() == 0) main()
