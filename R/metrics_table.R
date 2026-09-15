# metrics_table.R ─ Results table for the paper and poster.
#
#     Rscript R/metrics_table.R [--task classification|segmentation|
#                                       classification_val|segmentation_val]
#                               [--highlight dash|ring|halo|chip|bar|none]
#
# Reads data/model_metrics.csv (written by src/export_run_metrics.py — R has no
# W&B client, so the numbers are cached first, exactly as R/prepare_city_data.R
# caches the city summary for R/plotting.R). Run from the repo root.
#
# Written to plots/:
#   model_metrics_table.png / .pdf  - the poster/paper asset
#   model_metrics_table.html        - a sortable reactable companion
#
# `--task segmentation` draws the SAME table for the segmentation campaign, from
# data/seg_metrics.csv. It is one table with two inputs rather than two scripts:
# the hues, ramps, geometry, highlight and caption machinery are what make the
# two comparable at a glance, and a fork would let them drift apart. A task
# supplies only what actually differs -- its CSV, its metric columns and the
# name it is saved under -- through TASK_PROFILES.
#
# The `*_val` tasks are the validation counterparts, from the `--stage val`
# exports. They are the same figure with two fewer certainties: neither campaign
# logs OAu during validation, and the segmentation one never aggregates to So2Sat
# patches while validating, so its numbers are per-pixel rather than patch-exact.
# Each says so in its note rather than quietly looking like the test table.
#
# The static table is hand-drawn from geom_tile + geom_text. `gt`, `gtExtras`,
# `formattable`, `flextable`, `webshot2` and `chromote` are all absent from this
# conda env and there is no headless Chrome, so an htmlwidget cannot be
# rasterised here: reactable can only ever be a companion, never the deliverable.
# Hand-building the graphic is the same route R/plotting.R takes for its pie
# wedges and R/split_maps.R for its hatching, for the same reason.
#
# THE TABLE IS EXPECTED TO GROW. Nothing here is hardcoded to the current run
# set: columns, column widths, figure size, split hues and the caption are all
# derived from the CSV and from METRIC_COLS at draw time. Adding a metric is one
# entry in METRIC_COLS + one in METRIC_GLOSS; adding runs, architectures, splits
# or embeddings needs no edit at all. See `table_layout()` for the geometry.

source("R/constants.R")

suppressPackageStartupMessages({
  library(readr)
  library(dplyr)
  library(scales)
})

METRICS_CSV <- file.path("data", "model_metrics.csv")   # the default task's cache

# ── What the table shows ──────────────────────────────────────────────────────

# Metric columns in table order: CSV column -> header. Add one here and it gets a
# column, a ramp, a best-value highlight, a caption entry and the extra figure
# width automatically.
METRIC_COLS <- c("test_acc" = "OA", "test_oau" = "OAu",
                 "test_f1" = "Macro F1", "test_kappa" = "κ")

# Expansions for the caption under the table. A metric with no entry here is
# simply left out of the caption (with a warning), never mislabelled.
METRIC_GLOSS <- c("test_acc"   = "Overall accuracy",
                  "test_oau"   = "Urban-class OA (LCZ 1-10)",
                  "test_f1"    = "Class-mean F1",
                  "test_kappa" = "Cohen's kappa",
                  "test_oabu"  = "Built-up-class OA",
                  "test_oaw"   = "Weighted OA",
                  "test_kappa_w"   = "Weighted kappa",
                  "test_acc_macro" = "Class-mean accuracy",
                  "test_f1_micro"  = "Micro F1",
                  "test_acc_patch_exact"   = "Overall accuracy",
                  "test_oau_patch_exact"   = "Urban-class OA (LCZ 1-10)",
                  "test_f1_patch_exact"    = "Class-mean F1",
                  "test_kappa_patch_exact" = "Cohen's kappa",
                  "test_miou"              = "Mean IoU",
                  "val_acc"        = "Overall accuracy",
                  "val_f1"         = "Class-mean F1",
                  "val_kappa"      = "Cohen's kappa",
                  "val_miou"       = "Mean IoU",
                  "val_acc_macro"  = "Class-mean accuracy",
                  "val_f1_micro"   = "Micro F1")

# A metric whose name says nothing the reader needs; the note below the caption
# carries it instead. Set per task, so only the segmentation table pays for it.
METRIC_NOTE <- NULL

# What differs between the two campaigns, and nothing else.
#
# A segmentation model predicts pixels and a classifier predicts patches, so the
# only honest comparison is the segmentation run's `_patch_exact` evaluation --
# its predictions scored on the So2Sat patches it covers exactly. Those columns
# therefore carry the SAME headers as the classification table: read across the
# two figures and "OA" means one thing. mIoU is the exception and is labelled as
# itself, because it has no classification counterpart at all.
TASK_PROFILES <- list(
  classification = list(
    csv     = file.path("data", "model_metrics.csv"),
    name    = "model_metrics_table",
    note    = NULL,
    metrics = c("test_acc" = "OA", "test_oau" = "OAu",
                "test_f1" = "Macro F1", "test_kappa" = "κ")),
  segmentation = list(
    csv     = file.path("data", "seg_metrics.csv"),
    name    = "seg_metrics_table",
    note    = paste("Patch-exact evaluation: predictions scored on the So2Sat",
                    "patches they cover exactly.  mIoU is per-pixel at 10 m."),
    metrics = c("test_acc_patch_exact" = "OA", "test_oau_patch_exact" = "OAu",
                "test_f1_patch_exact" = "Macro F1",
                "test_kappa_patch_exact" = "κ", "test_miou" = "mIoU")),

  # The validation counterparts. Same rows, same hues, same headers -- but two
  # columns of the test tables cannot be drawn, and saying why is the note's job:
  #
  #   * OAu is missing from BOTH, because the urban-class split of OA is computed
  #     only in the test evaluator. The column is dropped rather than blanked: a
  #     column of dashes would suggest the number exists and was not measured.
  #   * the segmentation numbers are per-pixel at the native 10 m grid. Validation
  #     runs every epoch on the training machine and never aggregates to So2Sat
  #     patches, so there is no `_patch_exact` validation at all -- which is
  #     exactly the footing the test tables' star claims, and cannot be claimed
  #     here.
  #
  # Both read the epoch that maximised the run's monitor, so the row describes
  # the checkpoint the test row describes and the two tables can be read together.
  classification_val = list(
    csv     = file.path("data", "model_metrics_val.csv"),
    name    = "model_metrics_table_val",
    note    = paste("Validation at the best-monitor epoch -- the checkpoint the",
                    "test table reports.  No OAu: the urban-class split of OA is",
                    "computed only in the test evaluator."),
    metrics = c("val_acc" = "OA", "val_f1" = "Macro F1", "val_kappa" = "κ")),
  segmentation_val = list(
    csv     = file.path("data", "seg_metrics_val.csv"),
    name    = "seg_metrics_table_val",
    note    = paste("Validation at the best-monitor epoch, per-pixel at 10 m:",
                    "validation never aggregates to So2Sat patches, so these are",
                    "not the patch-exact quantities the test table reports."),
    metrics = c("val_acc" = "OA", "val_f1" = "Macro F1",
                "val_kappa" = "κ", "val_miou" = "mIoU"))
)

# ── Presentation vocabulary ───────────────────────────────────────────────────
#
# Every lookup below is a *preference*, not a filter: an unlisted value keeps its
# raw string and sorts after the listed ones, so a new embedding, split or
# architecture appears in the table rather than vanishing from it.

EMBEDDING_LABELS <- c("GeoTessera_v2"          = "Tessera v2",
                      "AlphaEarthCoop"         = "AlphaEarth",
                      "GeoTessera_v1.1_global" = "Tessera v1.1",
                      "GeoTessera_v1.1"        = "Tessera v1.1 (city)",
                      "EmbeddedSeamless"       = "Seamless")

SPLIT_LABELS <- c("global_so2sat"  = "Culture-10",
                  "grid_orig_test" = "Gridded",
                  "grid"           = "Per-city grid")

# The Model column names the architecture, not the preset: "ResNet34" says what
# was trained, "ResNet (small)" only says where it sits in this repo's ladder.
ARCH_LABELS <- c(
  "scnn_16-32"             = "ShallowCNN-16/32",
  "scnn_32-64"             = "ShallowCNN-32/64",
  "mobilenetv3_small_050"  = "MobileNetV3-S 0.5x",
  "mobilenetv3_small_100"  = "MobileNetV3-S 1.0x",
  "mobilenetv3_large_150d" = "MobileNetV3-L 1.5x",
  "mobilenetv4_conv_large" = "MobileNetV4-L",
  "resnet18" = "ResNet18", "resnet34" = "ResNet34",
  "resnet50" = "ResNet50", "resnet101" = "ResNet101", "resnet152" = "ResNet152",
  "efficientnet_b1" = "EfficientNet-B1", "efficientnet_b5" = "EfficientNet-B5",
  "efficientnet_b7" = "EfficientNet-B7",
  "convnext_tiny" = "ConvNeXt-T", "convnext_base" = "ConvNeXt-B",
  "convnext_large" = "ConvNeXt-L", "gap" = "GAP",
  # Segmentation has no timm name to quote: the family IS the architecture and
  # the preset is its capacity, so the label names the capacity the way the
  # ShallowCNN rows do -- depth/base-width, straight out of UNET_PRESETS and
  # FCN8_PRESETS in src/models/.
  "fcn8_nano"  = "FCN-8s-2/8",  "fcn8_small"  = "FCN-8s-3/16",
  "fcn8_base"  = "FCN-8s-3/32", "fcn8_medium" = "FCN-8s-3/48",
  "fcn8_large" = "FCN-8s-3/64",
  "unet_nano"  = "U-Net-2/8",   "unet_small"  = "U-Net-3/32",
  "unet_base"  = "U-Net-3/48",  "unet_medium" = "U-Net-4/32",
  "unet_large" = "U-Net-4/48")

# Acronyms are NOT expanded in the table. "Global Average Pooling (GAP)" was
# spelled out on its first row until 2026-09-12, when it turned out to be the
# widest string in the Model column by half an inch -- and in the mirrored
# table, where the column is paid for twice, that set the width of the whole
# figure. The expansion belongs in the prose that carries the figure, not in a
# cell. Recover the first-use machinery from git history if a table ever wants
# it back.

# Families excluded from the figure, with the reason. This is a *display*
# decision, so it lives here rather than in src/export_run_metrics.py: the CSV
# stays a faithful cache of W&B and the rows come back by deleting a line.
#
#   densenet  dropped 2026-09-10 at the author's request -- the two surviving
#             DenseNet cells are coop-only and pre-date the current run set, so
#             they compared nothing.
DROP_FAMILIES <- c("densenet")

# Row order within an embedding block. The linear probe leads: it is the floor
# every other row is measured against -- pooled features and one linear layer,
# three orders of magnitude fewer parameters than the CNNs beneath it -- so a
# block reads as "this is what the embedding alone gives you, and this is what
# each architecture adds". After it, simplest family first. Families not listed
# sort alphabetically after these.
FAMILY_ORDER <- c("linear_probe", "shallow_cnn", "mobilenet", "resnet",
                  # Segmentation, same principle: the plain encoder-decoder with
                  # score fusion before the one with skip connections.
                  "fcn8", "unet")

#' Apply a label lookup, keeping unmatched values as themselves.
relabel <- function(x, lookup) {
  out <- unname(lookup[as.character(x)])
  ifelse(is.na(out), as.character(x), out)
}

#' Factor whose levels follow `lookup`'s order, with anything unlisted appended.
#'
#' `intersect()` alone would silently *drop* an unlisted level to NA, which is
#' how a new split would quietly disappear from the figure.
ordered_by <- function(x, lookup) {
  preferred <- intersect(unname(lookup), x)
  factor(x, levels = c(preferred, sort(setdiff(unique(x), preferred))))
}

# ── Colour ────────────────────────────────────────────────────────────────────

# One hue per split, assigned by the split's position in the table. The splits
# are not comparable to each other -- the grid split reuses So2Sat's own test
# cities and is leakage-inflated relative to culture-10 -- so a separate hue and
# a separate normalisation say "read these blocks apart", and let the best model
# in each stand out without one block's numbers swamping the other's ramp.
#
# Hues are audited two ways, by simulating protanopia, deuteranopia and
# tritanopia (Vienot-Brettel-Mollon) and measuring CIELAB dE: against all 17
# LCZ_TABLE colours, because this figure shares a poster with the LCZ maps and a
# reader arrives with "orange = open mid-rise" primed; and against each other,
# because telling the two blocks apart is the point.
#
#                          dE to nearest LCZ    vs blue   vs grey45
#   #0072B2 blue              20.3 (LCZ 11)          -         -
#   #946C51 terracotta        21.1 (LCZ 13)        46.1      18.8
#   #D55E00 Okabe-Ito orange   2.2 (LCZ 3)         83.9      52.3   <- rejected
#   #8E2A8B purple            30.8 (LCZ 10)        18.8      30.6   <- rejected
#
# dE 2.2 is indistinguishable, so a saturated orange reads as LCZ 3/4 for the ~6%
# of male readers with deuteranopia. Purple cleared the LCZ palette best but sits
# in the 255-315 degree band, the only one LCZ leaves free, which is close enough
# to blue that the two blocks barely separated under red-green deficiency.
# Terracotta resolves both: muting the orange is what buys the LCZ distance (a
# saturated orange at any lightness scores 2-7), while orange-vs-blue is the
# classic dichromat-safe axis, so block separation more than doubles. The cost is
# the last column: muted orange sits dE 18.8 from the neutral grey of # Params
# under protanopia -- distinguishable, but the tightest margin in the figure, so
# do not mute this hue further. The binding LCZ neighbour throughout is LCZ 13
# (Bush/Scrub olive), which a muted orange converges on under deuteranopia.
# Slots 3-4 are unaudited; check them the same way before relying on a third split.
SPLIT_HUES <- c("#0072B2",   # blue       - culture-10 (Okabe-Ito, SPLIT_COLOURS "Training")
                "#946C51",   # terracotta - grid split
                "#453A9E",   # indigo     - unaudited
                "#7A3D7A")   # plum       - unaudited

# Parameters get a *neutral* grey ramp: more parameters is not "better", and a
# metric hue would imply it is.
PARAM_HUE <- "grey45"

#' Mix `hex` toward white (`f` = 1 is white).
lighten <- function(hex, f) {
  m <- grDevices::col2rgb(hex) / 255
  grDevices::rgb(t(m + (1 - m) * f))
}

#' Mix `hex` toward black (`f` = 1 is black).
darken <- function(hex, f) {
  m <- grDevices::col2rgb(hex) / 255
  grDevices::rgb(t(m * (1 - f)))
}

#' A white -> hue ramp function.
hue_ramp <- function(hue) grDevices::colorRampPalette(c(lighten(hue, 0.94), hue))

#' One ramp per split label, keyed by label. Hues cycle if a table ever carries
#' more splits than SPLIT_HUES has entries, and says so rather than colliding
#' silently.
build_ramps <- function(split_levels) {
  if (length(split_levels) > length(SPLIT_HUES)) {
    warning("More splits (", length(split_levels), ") than distinct hues (",
            length(SPLIT_HUES), "); hues repeat.", call. = FALSE)
  }
  idx <- (seq_along(split_levels) - 1) %% length(SPLIT_HUES) + 1
  setNames(lapply(SPLIT_HUES[idx], hue_ramp), split_levels)
}

#' Map values onto a ramp by their position within `x`, low = pale.
#'
#' Constant or single-valued groups map to the pale end rather than dividing by
#' zero; NA stays NA so the caller can paint the cell transparent.
ramp_fill <- function(x, ramp, n = 256, lo = 0.06, hi = 0.92) {
  cols <- ramp(n)
  out  <- rep(NA_character_, length(x))
  ok   <- !is.na(x)
  if (!any(ok)) return(out)
  rng <- range(x[ok])
  t   <- if (diff(rng) == 0) rep(0, sum(ok)) else (x[ok] - rng[1]) / diff(rng)
  out[ok] <- cols[1 + round((lo + t * (hi - lo)) * (n - 1))]
  out
}

#' Fills for every shaded column, as a named list of character vectors.
#'
#' Shared by the PNG and the htmlwidget so the two can never disagree.
column_fills <- function(df, grp, ramps) {
  metric <- lapply(setNames(names(METRIC_COLS), names(METRIC_COLS)), function(col)
    unsplit(Map(function(vals, g) ramp_fill(vals, ramps[[g]]),
                split(df[[col]], grp), names(split(df[[col]], grp))), grp))
  # Model size is comparable across the whole table -- leakage is a property of a
  # split, not of a model's parameter count -- so one table-wide ramp, log-scaled
  # because the counts span four orders of magnitude.
  c(list(n_params = ramp_fill(log10(df$n_params), hue_ramp(PARAM_HUE))), metric)
}

#' The best value per metric column *within a split*.
#'
#' Always per split, whatever `scale_by` is doing to the ramps: the splits are
#' not one ranking, so there is no single overall best to mark.
column_bests <- function(df) {
  lapply(setNames(names(METRIC_COLS), names(METRIC_COLS)), function(col)
    ave(df[[col]], as.character(df$split_label), FUN = function(z)
      if (all(is.na(z))) NA_real_ else max(z, na.rm = TRUE)))
}

# ── Data ──────────────────────────────────────────────────────────────────────

fmt_params <- label_number(accuracy = 0.1, scale_cut = cut_short_scale())
fmt_metric <- function(v) ifelse(is.na(v), "—", sprintf("%.3f", v))

#' Read the cached export and put it in table order.
read_metrics <- function(path = METRICS_CSV) {
  if (!file.exists(path)) {
    stop("Missing ", path, ". Run: python src/export_run_metrics.py", call. = FALSE)
  }
  df <- read_csv(path, show_col_types = FALSE)

  missing <- setdiff(names(METRIC_COLS), names(df))
  if (length(missing)) {
    stop(path, " has no column(s): ", paste(missing, collapse = ", "),
         ". Re-run src/export_run_metrics.py, or drop them from METRIC_COLS.",
         call. = FALSE)
  }

  dropped <- df$family %in% DROP_FAMILIES
  if (any(dropped)) {
    message("  dropping ", sum(dropped), " row(s) from excluded famil(ies): ",
            paste(sort(unique(df$family[dropped])), collapse = ", "))
    df <- df[!dropped, , drop = FALSE]
  }

  df |>
    mutate(
      embedding_label = relabel(embedding, EMBEDDING_LABELS),
      split_label     = relabel(split_source, SPLIT_LABELS),
      model_label     = relabel(arch, ARCH_LABELS),
      params_label    = fmt_params(n_params),
      embedding_label = ordered_by(embedding_label, EMBEDDING_LABELS),
      split_label     = ordered_by(split_label, SPLIT_LABELS),
      family_rank     = match(family, FAMILY_ORDER),
      # Carried as a column, not read from `df` after the arrange(): the frame
      # is reordered below and a bare vector from the outer scope would then be
      # off by however far each row moved -- silently tagging the wrong run.
      run_state       = df_state(df)
    ) |>
    # Split leads: the splits are not comparable to each other, so they are
    # stacked tables that happen to share a header.
    arrange(split_label, embedding_label, is.na(family_rank), family_rank,
            family, n_params) |>
    mutate(model_label = mark_pending(model_label, run_state))
}

#' The `state` column if the export wrote one, else "finished" for every row.
#'
#' Older caches predate the column; a missing state is not evidence that a run
#' was unfinished, so it reads as finished rather than marking the whole table.
df_state <- function(df) {
  if ("state" %in% names(df)) as.character(df$state) else rep("finished", nrow(df))
}

#' Tag a row whose run has not finished.
#'
#' Without this the table says "—" in two different voices: a metric that does
#' not exist for that split (segmentation mIoU is not computed on the gridded
#' split) and a metric that does not exist YET. The cell is still worth drawing
#' -- it is in the ladder and its size is already known -- but the reader has to
#' be told which kind of blank it is.
mark_pending <- function(labels, state) {
  ifelse(state == "finished", labels, paste0(labels, "  (", state, ")"))
}

# ── Geometry ──────────────────────────────────────────────────────────────────
#
# The layout is computed in INCHES from the strings that will actually be drawn,
# so a longer architecture name or an extra metric widens the figure instead of
# colliding with the next column. The x scale is inches; the y scale is rows, at
# ROW_H inches each.

BODY_PT   <- LEGEND_TEXT_PT          # body type, shared with the rest of the stack
ROW_H     <- 0.28                    # row pitch, inches
TEXT_PAD  <- 0.12                    # clear space around a string, for column spacing
NUM_PAD   <- 0.07                    # clear space inside a shaded tile, each side
HDR_GAP   <- 0.10                    # minimum clear space between two headers
CELL_PAD  <- 0.09                    # gutter between one tile and the next, each side
RING_IN   <- 0.05                    # inner rule inset from the tile edge
RING_OUT  <- 0.10                    # highlight rule's height over the tile, row units

# Dash pattern for the "dash" highlight, in units of the line width. R's named
# "dashed" is too coarse at this box size: it lands one or two long dashes on
# each short side and breaks the corners, so the mark reads as a damaged box
# rather than a dashed one. A short on/off keeps the corners closed.
DASH_PATTERN <- "41"   # 4 on, 1 off, in line widths
CELL_H    <- 0.86                    # tile height, in row units
CAP_PT    <- 11 * 0.75               # theme_eofm()'s plot.caption size
CAP_PT_MIN <- 11 * 0.55              # floor when the caption is shrunk to fit
NOTE_DROP  <- 1.75                   # the note's baseline, rows below the last row
HEADER_Y  <- 1.15                    # header baseline, in row units above the top rule
Y_ABOVE   <- 1.75                    # panel top, row units above the first row
Y_BELOW   <- 1.95                    # panel bottom, row units below the last row
NOTE_ROOM <- 0.55                    # extra bottom room when a note is drawn

PAD_OUT   <- 0.06                    # breathing room outside the outermost column
MARGIN_PT <- 4                       # plot.margin, all four sides

# Fallback average glyph advance as a fraction of the point size, used only if
# systemfonts is missing. Deliberately generous: over-estimating a column costs
# whitespace, under-estimating it collides two columns.
CHAR_EM   <- 0.68
# Slack on every measured width. ragg shapes through systemfonts, so the two
# agree closely, but a hair of headroom keeps a glyph off the tile edge.
W_SLACK   <- 1.04

CAPTION_COL <- "grey40"
RULE_COL    <- "grey25"
SEP_COL     <- "grey85"
TEXT_COL    <- "grey15"

#' Width of the widest of `s` in inches at `pt`.
#'
#' Measured through systemfonts, which is what ragg shapes with, so the numbers
#' match what actually lands on the page. Falls back to a character-count
#' estimate only if systemfonts is unavailable.
str_w <- function(s, pt = BODY_PT, bold = FALSE, italic = FALSE) {
  s <- s[!is.na(s)]
  if (!length(s)) return(0)
  if (requireNamespace("systemfonts", quietly = TRUE)) {
    # `italic` matters: the note is drawn in the italic face, and measuring it
    # upright underestimates the width enough to let a shrunk line still run off
    # the page -- which is a silent truncation, not a visible overflow.
    w <- systemfonts::string_width(s, size = pt, res = 72, italic = italic,
                                   weight = if (bold) "bold" else "normal")
    return(max(w) / 72 * W_SLACK)
  }
  max(nchar(s)) * pt * CHAR_EM *
    (if (bold) 1.08 else 1) * (if (italic) 1.03 else 1) / 72
}

#' Column geometry for one data frame: one row per column, in draw order.
#'
#' @return a data.frame of `key, header, align, width, left, centre` (inches),
#'   carrying `total` (table width) as an attribute.
table_layout <- function(df) {
  text_cols <- list(
    list(key = "split",     header = "Split",     values = levels(droplevels(df$split_label))),
    list(key = "embedding", header = "Embedding", values = levels(droplevels(df$embedding_label))),
    list(key = "model",     header = "Model",     values = df$model_label)
  )
  num_cols <- c(
    list(list(key = "params", header = "# Params", values = df$params_label)),
    lapply(names(METRIC_COLS), function(m)
      list(key = m, header = unname(METRIC_COLS[m]),
           values = fmt_metric(df[[m]])))
  )

  spec <- c(lapply(text_cols, function(c) c(c, list(align = "left"))),
            lapply(num_cols,  function(c) c(c, list(align = "centre"))))

  keys   <- vapply(spec, `[[`, "", "key")
  align  <- vapply(spec, `[[`, "", "align")
  hdr_w  <- vapply(spec, function(c) str_w(c$header, bold = TRUE), numeric(1))
  vals_w <- vapply(spec, function(c) str_w(c$values), numeric(1))

  # A numeric column is sized by its *tile*, not by its header. "Macro F1" is far
  # wider than "0.000", and letting it set the spacing pushed the colour blocks
  # apart with nothing but empty gutter between them. Headers are centred, so a
  # wide one simply overhangs into the gutter its narrow neighbours leave; the
  # pass below then widens a column only where two headers would actually touch.
  tile  <- vals_w + 2 * NUM_PAD
  width <- ifelse(align == "left",
                  pmax(vals_w, hdr_w) + 2 * TEXT_PAD + 2 * CELL_PAD,
                  tile + 2 * CELL_PAD)

  # Every metric column gets the widest metric column's tile. They hold the same
  # kind of number and are read across, so ragged widths make the row look like a
  # bar chart it is not.
  is_metric <- keys %in% names(METRIC_COLS)
  if (any(is_metric)) {
    tile[is_metric]  <- max(tile[is_metric])
    width[is_metric] <- max(width[is_metric])
  }

  # Resolve header collisions by widening the offending column, one at a time and
  # by exactly the deficit, so the table stays as tight as the headers allow.
  n <- length(width)
  repeat {
    left   <- cumsum(c(0, head(width, -1)))
    anchor <- ifelse(align == "left", left, left + width / 2)
    hstart <- ifelse(align == "left", anchor, anchor - hdr_w / 2)
    over   <- (hstart[-n] + hdr_w[-n] + HDR_GAP) - hstart[-1]
    if (!length(over) || max(over) <= 1e-9) break
    i <- which.max(over)
    width[i] <- width[i] + over[i]
  }

  data.frame(key = keys, header = vapply(spec, `[[`, "", "header"), align = align,
             width = width, tile = tile, left = left, centre = left + width / 2,
             stringsAsFactors = FALSE) |>
    structure(total = sum(width))
}

# ── Highlighting the best value ───────────────────────────────────────────────
#
# "Best in this column, in this split" has to survive being printed small, on a
# poster, next to three other saturated cells. The styles below are the ones
# worth choosing between; `highlight = "..."` picks one.
#
#   dash  - heavy dashed rule in the gutter. The dashes read as a deliberate
#           marker rather than as another table rule, and the broken line stays
#           legible where a solid one would merge with a neighbouring cell.
#   ring  - thin solid rule in the gutter around the cell. Quietest.
#   halo  - solid rule plus a white inner rule: reads as a lifted chip.
#   chip  - cell repainted in a deep ink of its split's hue, text knocked out
#           white. Loudest, and independent of how dark the ramp got.
#   bar   - fill kept, with a heavy rule along the cell's bottom edge.
#   none  - bold type only.
#
# Every style keeps the bold type, so the mark survives a greyscale print.
HIGHLIGHT_STYLES <- c("dash", "ring", "halo", "chip", "bar", "none")

# ── Static table ──────────────────────────────────────────────────────────────

#' The results table as a ggplot.
#'
#' The returned plot carries `fig_width` / `fig_height` attributes (inches) sized
#' to its own content; `main()` passes them to save_plot().
#'
#' @param df        output of `read_metrics()`.
#' @param scale_by  "split" normalises each colour ramp within a split, "column"
#'   across the whole table. Split is the default and the honest one: a shared
#'   ramp would paint every culture-10 row pale and destroy the contrast the
#'   table exists to show.
#' @param highlight one of HIGHLIGHT_STYLES; see the block above.
metrics_table_plot <- function(df, scale_by = c("split", "column"),
                               highlight = "dash") {
  scale_by  <- match.arg(scale_by)
  highlight <- match.arg(highlight, HIGHLIGHT_STYLES)

  n     <- nrow(df)
  lay   <- table_layout(df)
  xof   <- setNames(lay$centre, lay$key)
  tof   <- setNames(lay$tile, lay$key)
  xleft <- setNames(lay$left, lay$key)
  wof   <- setNames(lay$width, lay$key)
  x_right <- attr(lay, "total")

  df$.row   <- seq_len(n)
  df$.split <- as.character(df$split_label)
  df$.block <- paste(df$.split, df$embedding_label)
  grp <- if (scale_by == "split") df$.split else rep("all", n)

  ramps <- build_ramps(unique(if (scale_by == "split") df$.split else "all"))
  fills <- column_fills(df, grp, ramps)
  bests <- column_bests(df)

  # One long frame of shaded cells: parameters plus every metric column.
  cells <- do.call(rbind, lapply(names(fills), function(col) {
    key <- if (col == "n_params") "params" else col
    v   <- df[[col]]
    data.frame(
      row = df$.row, x = unname(xof[key]), w = unname(tof[key]),
      fill = fills[[col]],
      label = if (col == "n_params") df$params_label else fmt_metric(v),
      best = if (col == "n_params") FALSE
             else !is.na(v) & !is.na(bests[[col]]) & v == bests[[col]],
      split = df$.split, stringsAsFactors = FALSE
    )
  }))
  cells$best[is.na(cells$fill)] <- FALSE
  cells$face <- ifelse(cells$best, "bold", "plain")

  # Style-dependent repaint, before the contrast decision so knocked-out text
  # follows whatever the cell ended up being.
  if (highlight == "chip" && any(cells$best)) {
    hue <- vapply(cells$split[cells$best],
                  function(s) tail(ramps[[if (scale_by == "split") s else "all"]](256), 1),
                  character(1))
    cells$fill[cells$best] <- darken(hue, 0.34)
  }
  cells$colour <- unname(ifelse(is.na(cells$fill), TEXT_COL, contrast_text(cells$fill)))
  cells$fill[is.na(cells$fill)] <- "transparent"

  best <- cells[cells$best, , drop = FALSE]

  # Split and Embedding are written once per block and centred vertically on it:
  # with a block six rows deep a top-anchored label reads as belonging to that
  # one row.
  centred <- function(key, label, x) {
    keep <- !duplicated(key)
    data.frame(row = ave(df$.row, key, FUN = mean)[keep],
               x = unname(x), label = label[keep], stringsAsFactors = FALSE)
  }
  texts <- rbind(
    centred(df$.split, as.character(df$split_label), xleft["split"]),
    centred(df$.block, as.character(df$embedding_label), xleft["embedding"]),
    data.frame(row = df$.row, x = unname(xleft["model"]), label = df$model_label,
               stringsAsFactors = FALSE)
  )

  # Two kinds of hairline. A split boundary spans the whole table; an embedding
  # boundary *inside* a split starts at the Embedding column, so nothing cuts
  # across the Split label standing in the middle of its block.
  brk <- function(k) df$.row[c(FALSE, k[-1] != k[-length(k)])]
  split_rows <- brk(df$.split)
  emb_rows   <- setdiff(brk(df$.block), split_rows)
  rules <- data.frame(
    x  = c(0, 0, rep(0, length(split_rows)), rep(unname(xleft["embedding"]), length(emb_rows))),
    y  = -c(-0.5, n + 0.5, split_rows - 0.5, emb_rows - 0.5),
    lw = c(0.6, 0.6, rep(0.35, length(split_rows)), rep(0.25, length(emb_rows))),
    colour = c(RULE_COL, RULE_COL,
               rep(SEP_COL, length(split_rows)), rep(SEP_COL, length(emb_rows))),
    stringsAsFactors = FALSE
  )

  headers <- data.frame(
    x = ifelse(lay$align == "left", lay$left, lay$centre),
    label = lay$header,
    hjust = ifelse(lay$align == "left", 0, 0.5),
    stringsAsFactors = FALSE)

  cap <- metric_caption(total_w = x_right + 2 * PAD_OUT)
  y_below <- Y_BELOW + if (is.null(METRIC_NOTE)) 0 else NOTE_ROOM

  p <- ggplot() +
    geom_tile(data = cells, aes(x = x, y = -row, fill = fill, width = w),
              height = CELL_H, colour = NA)

  # --- the highlight layers -------------------------------------------------
  if (nrow(best)) {
    if (highlight %in% c("dash", "ring", "halo")) {
      # The winning cell is always the darkest of its column-block (the ramp is
      # monotonic in the value), so a stroke laid on the fill would be
      # dark-on-dark. Drawn in the gutter instead, it has full contrast whatever
      # the hue underneath, and never crowds the number.
      p <- p + geom_tile(
        data = best, aes(x = x, y = -row, width = w + 2 * CELL_PAD * 0.7),
        height = CELL_H + RING_OUT, fill = NA, colour = "grey10",
        linewidth = if (highlight == "ring") 0.55 else 0.85,
        linetype = if (highlight == "dash") DASH_PATTERN else "solid")
    }
    if (highlight == "halo") {
      p <- p + geom_tile(data = best, aes(x = x, y = -row, width = w - 2 * RING_IN),
                         height = CELL_H - 2 * RING_IN, fill = NA,
                         colour = "white", linewidth = 0.7)
    }
    if (highlight == "bar") {
      p <- p + geom_segment(
        data = best,
        aes(x = x - w / 2, xend = x + w / 2,
            y = -row - CELL_H / 2, yend = -row - CELL_H / 2),
        colour = "grey10", linewidth = 1.4, inherit.aes = FALSE)
    }
  }

  p <- p +
    geom_text(data = cells,
              aes(x = x, y = -row, label = label, colour = colour, fontface = face),
              size = LABEL_SIZE_MM, hjust = 0.5) +
    geom_text(data = texts, aes(x = x, y = -row, label = label),
              size = LABEL_SIZE_MM, hjust = 0, colour = TEXT_COL) +
    geom_text(data = headers, aes(x = x, y = HEADER_Y, label = label, hjust = hjust),
              size = LABEL_SIZE_MM, fontface = "bold", colour = TEXT_COL) +
    geom_segment(data = rules,
                 aes(x = x, xend = x_right, y = y, yend = y,
                     linewidth = lw, colour = colour)) +
    scale_fill_identity() +
    scale_colour_identity() +
    scale_linewidth_identity() +
    # No expansion: the layout is computed in inches, and a 5% expansion would
    # silently shrink every column while the type stayed at its fixed point size
    # -- which is exactly how the columns end up too tight for their contents.
    scale_x_continuous(expand = expansion(0)) +
    scale_y_continuous(expand = expansion(0)) +
    annotate("text", x = x_right, y = -n - 1.1, label = cap$expr, parse = TRUE,
             hjust = 1, vjust = 1, size = cap$size_mm, colour = CAPTION_COL)

  # The note says what the numbers ARE, not what they are called, so it gets its
  # own line under the glossary rather than a slot inside it: joined on, it
  # forced the whole caption down to an unreadable size to fit the width.
  # ... and it is shrunk to the table's width on its own account. It used to
  # inherit the caption's size, which was safe only while every note happened to
  # be shorter than every glossary: the validation tables have one fewer metric
  # column, so the figure narrowed, and a note wider than the device is silently
  # TRUNCATED at both ends with no warning -- the same trap metric_caption()
  # already guards against.
  if (!is.null(METRIC_NOTE)) {
    note_pt <- CAP_PT
    have <- str_w(METRIC_NOTE, pt = note_pt, italic = TRUE)
    total_w <- x_right + 2 * PAD_OUT
    if (have > total_w) note_pt <- max(CAP_PT_MIN, note_pt * total_w / have)
    p <- p + annotate("text", x = x_right, y = -n - NOTE_DROP,
                      label = METRIC_NOTE, hjust = 1, vjust = 1,
                      fontface = "italic", size = min(cap$size_mm, note_pt / .pt),
                      colour = CAPTION_COL)
  }
  p <- p +
    coord_cartesian(xlim = c(-PAD_OUT, x_right + PAD_OUT),
                    ylim = c(-n - y_below, Y_ABOVE), clip = "off") +
    theme_eofm() +
    theme(
      axis.title = element_blank(), axis.text = element_blank(),
      axis.ticks = element_blank(),
      panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
      plot.margin = margin(4, 4, 4, 4)
    )

  # `|>` binds tighter than `+`, so these have to be attached in a statement of
  # their own rather than piped off the end of the layer sum.
  # Panel extent in inches, plus the plot margins: with no scale expansion this
  # makes one x unit exactly one inch and one row exactly ROW_H on the page, so
  # the measured column widths mean what they say.
  margin_in <- 2 * MARGIN_PT / 72
  structure(p,
            fig_width  = x_right + 2 * PAD_OUT + margin_in,
            fig_height = (n + Y_ABOVE + y_below) * ROW_H + margin_in)
}

#' The caption under the table, as a plotmath expression sized to fit.
#'
#' ggtext is not installed in this env, so `bold("OA:")` through `parse = TRUE`
#' is the only way to weight the acronym apart from its expansion inside one
#' label. Built from METRIC_COLS so a new metric captions itself.
metric_caption <- function(total_w = NULL) {
  keys <- intersect(names(METRIC_COLS), names(METRIC_GLOSS))
  skipped <- setdiff(names(METRIC_COLS), keys)
  if (length(skipped)) {
    warning("No METRIC_GLOSS entry for ", paste(skipped, collapse = ", "),
            "; left out of the caption.", call. = FALSE)
  }
  if (!length(keys)) return(list(expr = '""', size_mm = CAP_PT / .pt))

  head_expr <- vapply(keys, function(k) {
    h <- unname(METRIC_COLS[k])
    # Greek in plotmath is a symbol, not a literal: this way the caption's kappa
    # is the same glyph as the column head's, on any device.
    if (h == "κ") 'bold(kappa*":")' else sprintf('bold("%s:")', h)
  }, character(1))

  parts <- sprintf('%s*" %s"', head_expr, unname(METRIC_GLOSS[keys]))
  plain <- paste(sprintf("%s: %s", unname(METRIC_COLS[keys]),
                         unname(METRIC_GLOSS[keys])), collapse = "      ")
  # Shrink to fit rather than run off the page. The glossary grows with the
  # metric set while the figure's width is set by its columns, so the two can
  # disagree -- and a caption wider than the device is silently TRUNCATED,
  # losing the end of the line with no warning at all.
  pt <- CAP_PT
  if (!is.null(total_w)) {
    have <- str_w(plain, pt = pt)
    if (have > total_w) pt <- max(CAP_PT_MIN, pt * total_w / have)
  }
  list(expr = paste(parts, collapse = '*"      "*'),
       size_mm = pt / .pt, plain = plain)
}

# ── reactable companion ───────────────────────────────────────────────────────

#' The same table as a sortable htmlwidget. Not the poster asset — see the header.
metrics_table_reactable <- function(df, scale_by = c("split", "column")) {
  scale_by <- match.arg(scale_by)
  grp   <- if (scale_by == "split") as.character(df$split_label) else rep("all", nrow(df))
  ramps <- build_ramps(unique(grp))
  fills <- column_fills(df, grp, ramps)
  bests <- column_bests(df)

  cell_style <- function(col) function(value, index) {
    bg <- fills[[col]][index]
    if (is.na(bg)) return(list())
    st <- list(background = unname(bg), color = unname(contrast_text(bg)))
    if (!is.null(bests[[col]]) && !is.na(value) && !is.na(bests[[col]][index]) &&
        value == bests[[col]][index]) {
      st$fontWeight   <- "bold"
      st$outline      <- "2px dashed #1a1a1a"
      st$outlineOffset <- "-4px"
    }
    st
  }

  out <- df |>
    transmute(Split = as.character(split_label),
              Embedding = as.character(embedding_label),
              Model = model_label, Arch = arch, Run = run_name,
              n_params) |>
    bind_cols(df[names(METRIC_COLS)])

  # Match the PNG: every metric column the same width, and the best value marked
  # with a dashed rule rather than a solid one. `outline` is used instead of a
  # border so the rule does not take part in the cell's layout, and the negative
  # offset pulls it inside the cell the way the PNG's sits in its gutter.
  metric_px <- max(72, 13 * max(nchar(unname(METRIC_COLS))) + 42)
  cols <- list(
    Arch = reactable::colDef(show = FALSE),
    Run  = reactable::colDef(show = FALSE),
    n_params = reactable::colDef(name = "# Params", align = "right",
                                 style = cell_style("n_params"),
                                 cell = function(v) fmt_params(v)))
  for (m in names(METRIC_COLS)) {
    cols[[m]] <- reactable::colDef(name = unname(METRIC_COLS[m]), align = "center",
                                   width = metric_px,
                                   style = cell_style(m), cell = fmt_metric)
  }

  reactable::reactable(out, columns = cols, defaultPageSize = 50,
                       sortable = TRUE, striped = FALSE, highlight = TRUE,
                       compact = TRUE, borderless = TRUE,
                       columnGroups = list(
                         reactable::colGroup(name = "Test metrics",
                                             columns = names(METRIC_COLS))))
}

# ── Entry point ───────────────────────────────────────────────────────────────

#' Render and save. `name` lets a variant be written next to the canonical file.
save_metrics_table <- function(df, name = "model_metrics_table",
                               highlight = "dash", scale_by = "split",
                               formats = c("png", "pdf")) {
  p <- metrics_table_plot(df, scale_by = scale_by, highlight = highlight)
  save_plot(p, name, width = attr(p, "fig_width"), height = attr(p, "fig_height"),
            formats = formats,
            subdir = PLOT_DIR_MODELS)
}

main <- function(args = commandArgs(trailingOnly = TRUE)) {
  highlight <- "dash"; task <- "classification"
  i <- 1
  while (i <= length(args)) {
    switch(args[i],
      "--highlight" = { highlight <- args[i + 1]; i <- i + 1 },
      "--task"      = { task      <- args[i + 1]; i <- i + 1 },
      stop("Unknown argument '", args[i], "'", call. = FALSE))
    i <- i + 1
  }
  if (!task %in% names(TASK_PROFILES)) {
    stop("--task must be one of: ", paste(names(TASK_PROFILES), collapse = ", "),
         call. = FALSE)
  }
  profile <- TASK_PROFILES[[task]]

  # Rebound in the script's own environment, which every drawing function below
  # resolves through: the alternative is threading one argument through a dozen
  # signatures that exist only to pass it on.
  METRIC_COLS <<- profile$metrics
  METRIC_NOTE <<- profile$note

  df <- read_metrics(profile$csv)
  message("Metrics table (", task, "): ", nrow(df), " rows, ",
          length(METRIC_COLS), " metric columns, highlight = ", highlight)

  save_metrics_table(df, name = profile$name, highlight = highlight)

  if (requireNamespace("reactable", quietly = TRUE) &&
      requireNamespace("htmlwidgets", quietly = TRUE)) {
    html <- plot_path(PLOT_DIR_MODELS, paste0(profile$name, ".html"))
    htmlwidgets::saveWidget(metrics_table_reactable(df), file.path(getwd(), html),
                            selfcontained = TRUE, title = "eo-fm model metrics")
    # saveWidget stages the JS dependencies next to the output and does not clean
    # up after inlining them. The HTML carries no reference to the directory, so
    # leaving it behind would only be misleading clutter.
    libdir <- plot_path(PLOT_DIR_MODELS, paste0(profile$name, "_files"))
    if (dir.exists(libdir)) unlink(libdir, recursive = TRUE)
    message("  wrote ", html)
  } else {
    message("  reactable/htmlwidgets unavailable; skipping the HTML companion")
  }
}

if (sys.nframe() == 0) main()
