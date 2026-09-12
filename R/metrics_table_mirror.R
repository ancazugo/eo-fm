# metrics_table_mirror.R ─ Both campaigns in one table, mirrored about the split.
#
#     Rscript R/metrics_table_mirror.R [--highlight dash|ring|halo|chip|bar|none]
#
# Segmentation on the left, patch classification on the right, and between them
# the one column the two share: the split. Everything else is reflected about
# it -- embedding, then model, then # Params, then the metrics, outward in both
# directions -- so the axis of the figure is the thing the two halves have in
# common, and distance from that axis means the same on either side.
#
# Written to plots/:
#   metrics_table_mirror.png / .pdf
#
# THE TWO HALVES ARE NOT ONE RANKING. A segmentation run predicts pixels and is
# scored here on the So2Sat patches it covers exactly; a classifier predicts a
# patch outright. That is what makes the columns comparable enough to sit in one
# figure, and it is still not the same task, so each half keeps its own colour
# normalisation and its own best-value marks. The shared hue is the SPLIT's, not
# the task's: a row's colour says which split it belongs to, and the halves are
# told apart by which side of the axis they are on.
#
# Sources R/metrics_table.R for everything that is genuinely shared -- reading,
# ramps, fills, bests, string measurement, formatting, highlight styles, the
# task profiles. That file's main() is guarded, so sourcing it runs nothing.
# Only the geometry is new, because only the geometry is actually different.

source("R/metrics_table.R")

# Which task goes on which side. The halves are not titled: the Model column
# names them well enough -- U-Nets on one side, ResNets on the other -- and a
# banner over each half competed with the column headers it sat on top of.
SIDES <- list(
  left  = list(key = "L", task = "segmentation"),
  right = list(key = "R", task = "classification")
)

# What the asterisk on the segmentation half's headers means. It is a glossary
# term, so it goes on the caption's one line with the rest of them rather than
# into a note of its own.
STAR <- "*"
STAR_GLOSS <- "aggregated to the So2Sat patch"

# Metrics a half carries in its own table but not in this one. mIoU is the
# segmentation table's fifth column and belongs there; here it would be a column
# with nothing facing it across the axis, on a split where it is not even
# computed. Dropping it also makes the two halves four columns each, so the
# figure is a reflection rather than an approximation of one.
MIRROR_DROP <- c("test_miou")

#' Evaluate `expr` with METRIC_COLS temporarily bound to `cols`.
#'
#' read_metrics(), column_fills() and column_bests() all resolve METRIC_COLS
#' from this environment, which is exactly what lets one set of functions serve
#' two metric sets -- but only one at a time, so the binding is swapped around
#' each half rather than threaded through every signature.
with_metrics <- function(cols, expr) {
  old <- METRIC_COLS
  METRIC_COLS <<- cols
  on.exit(METRIC_COLS <<- old, add = TRUE)
  expr
}

# ── Rows ──────────────────────────────────────────────────────────────────────

#' Give both halves row numbers on one shared grid.
#'
#' Alignment happens at the EMBEDDING block, not at the split.
#'
#' Each (split, embedding) block is as deep as its deeper half and the
#' shallower one is centred in it. The alternative -- stretching the shorter
#' half's rows -- would put the two halves at different type sizes, which is
#' precisely what the figure must not do: the point of one table is that a
#' number on the left and a number on the right are read the same way. So the
#' leftover space has to go somewhere, and the question is only where.
#'
#' Centring at the split put all of it in one gap: four segmentation rows, then
#' six blank ones, and the reader had to count to work out which embedding the
#' rows opposite belonged to. Centring at the embedding spreads the same space
#' into a band above and below each block, and lands the Tessera and AlphaEarth
#' blocks opposite each other across the axis -- so the embedding hairlines line
#' up on both sides and the two halves can be read row-band by row-band.
#'
#' @return the input frames with a `.row` column, plus a split block table.
assign_rows <- function(dfs) {
  ordered_levels <- function(f, lookup) {
    levs <- unique(unlist(lapply(dfs, function(d) as.character(d[[f]]))))
    i <- match(levs, unname(lookup))
    levs[order(is.na(i), i, levs)]
  }
  splits <- ordered_levels("split_label", SPLIT_LABELS)
  embeddings <- ordered_levels("embedding_label", EMBEDDING_LABELS)

  offset <- 0
  blocks <- list()
  emb_tops <- integer(0)
  for (d in names(dfs)) dfs[[d]]$.row <- NA_integer_
  for (s in splits) {
    top <- offset + 1
    for (e in embeddings) {
      rows_of <- function(d) which(d$split_label == s & d$embedding_label == e)
      depth <- max(vapply(dfs, function(d) length(rows_of(d)), integer(1)))
      if (!depth) next
      if (offset + 1 > top) emb_tops <- c(emb_tops, offset + 1L)
      for (d in names(dfs)) {
        i <- rows_of(dfs[[d]])
        pad <- floor((depth - length(i)) / 2)
        if (length(i)) dfs[[d]]$.row[i] <- offset + pad + seq_along(i)
      }
      offset <- offset + depth
    }
    blocks[[s]] <- c(top = top, bottom = offset)
  }
  list(dfs = dfs, blocks = blocks, emb_tops = emb_tops, n = offset)
}

# ── Columns ───────────────────────────────────────────────────────────────────

#' Mark the metrics that are an aggregation, not a native measurement.
#'
#' A segmentation run predicts pixels; its OA here is those predictions pooled
#' onto the So2Sat patches they cover exactly. Under the same header as the
#' classification half's OA -- which is the point, they are meant to be read
#' across -- that difference is invisible, so the header carries a star. mIoU
#' has no star: it IS per-pixel, and is the one column with no counterpart.
star_headers <- function(metrics) {
  starred <- grepl("_patch_exact$", names(metrics))
  metrics[starred] <- paste0(metrics[starred], STAR)
  metrics
}

#' One side's columns, in OUTWARD order (nearest the axis first).
#'
#' Keys are side-prefixed: the two halves both have an "embedding" and a
#' "params" column, and they are different columns in different places.
side_spec <- function(df, metrics, key) {
  text <- list(
    list(key = "embedding", header = "Embedding",
         values = levels(droplevels(df$embedding_label)), align = "text"),
    list(key = "model", header = "Model", values = df$model_label, align = "text"))
  num <- c(
    list(list(key = "params", header = "# Params",
              values = df$params_label, align = "num")),
    lapply(names(metrics), function(m)
      list(key = m, header = unname(star_headers(metrics)[m]),
           values = fmt_metric(df[[m]]), align = "num")))
  lapply(c(text, num), function(c) { c$key <- paste0(key, ".", c$key); c })
}

#' Lay out both halves and the shared axis, left to right across the page.
#'
#' The collision pass is the one from table_layout(), run over the WHOLE page
#' rather than one half: the two innermost columns face each other across the
#' Split column, and a header wide enough to reach across it has to widen
#' something, not overprint.
mirror_layout <- function(specs, split_values) {
  # One tile width for every metric column on both sides, and one for the two
  # # Params columns. They hold the same kind of number and the figure is
  # symmetric, so a ragged pair either side of the axis would read as meaning.
  is_metric <- function(s) grepl("^[LR]\\.(?!embedding|model|params)", s$key, perl = TRUE)
  all_cols <- unlist(specs, recursive = FALSE)
  tile_of <- function(s) str_w(s$values) + 2 * NUM_PAD
  metric_tile <- max(vapply(Filter(is_metric, all_cols), tile_of, numeric(1)))
  params_tile <- max(vapply(Filter(function(s) grepl("\\.params$", s$key), all_cols),
                            tile_of, numeric(1)))

  axis <- list(key = "split", header = "Split", values = split_values,
               align = "axis")
  page <- c(rev(specs$left), list(axis), specs$right)

  hdr_w <- vapply(page, function(s) str_w(s$header, bold = TRUE), numeric(1))
  tile  <- vapply(page, function(s) {
    if (s$align != "num") 0
    else if (grepl("\\.params$", s$key)) params_tile else metric_tile
  }, numeric(1))
  vals_w <- vapply(page, function(s) str_w(s$values), numeric(1))
  align  <- vapply(page, `[[`, "", "align")
  keys   <- vapply(page, `[[`, "", "key")

  width <- ifelse(align == "num", tile + 2 * CELL_PAD,
                  pmax(vals_w, hdr_w) + 2 * TEXT_PAD + 2 * CELL_PAD)

  # Which edge a text column's strings hang from: away from the axis on the
  # right, towards it on the left, so the two halves read outward together.
  side <- substr(keys, 1, 1)
  hjust <- ifelse(align == "axis", 0.5, ifelse(align == "num", 0.5,
                  ifelse(side == "R", 0, 1)))

  n <- length(width)
  repeat {
    left   <- cumsum(c(0, head(width, -1)))
    anchor <- left + width * hjust
    hstart <- anchor - hdr_w * hjust
    over   <- (hstart[-n] + hdr_w[-n] + HDR_GAP) - hstart[-1]
    if (!length(over) || max(over) <= 1e-9) break
    i <- which.max(over)
    width[i] <- width[i] + over[i]
  }

  data.frame(key = keys, header = vapply(page, `[[`, "", "header"),
             align = align, hjust = hjust, width = width, tile = tile,
             left = left, centre = left + width / 2,
             anchor = left + width * hjust, stringsAsFactors = FALSE) |>
    structure(total = sum(width))
}

# ── The figure ────────────────────────────────────────────────────────────────

#' Shaded cells for one half, as the long frame the plot draws from.
side_cells <- function(df, metrics, key, lay, ramps) {
  grp   <- as.character(df$split_label)
  fills <- with_metrics(metrics, column_fills(df, grp, ramps))
  bests <- with_metrics(metrics, column_bests(df))
  xof <- setNames(lay$centre, lay$key)
  tof <- setNames(lay$tile, lay$key)

  do.call(rbind, lapply(names(fills), function(col) {
    k <- paste0(key, ".", if (col == "n_params") "params" else col)
    v <- df[[col]]
    data.frame(
      row = df$.row, x = unname(xof[k]), w = unname(tof[k]), fill = fills[[col]],
      label = if (col == "n_params") df$params_label else fmt_metric(v),
      best = if (col == "n_params") FALSE
             else !is.na(v) & !is.na(bests[[col]]) & v == bests[[col]],
      split = grp, stringsAsFactors = FALSE)
  }))
}

#' The mirrored table as a ggplot, carrying its own figure size in inches.
mirror_table_plot <- function(dfs, profiles, highlight = "dash") {
  highlight <- match.arg(highlight, HIGHLIGHT_STYLES)

  laid     <- assign_rows(dfs)
  dfs      <- laid$dfs
  n        <- laid$n
  blocks   <- laid$blocks
  emb_tops <- laid$emb_tops

  specs <- setNames(lapply(names(SIDES), function(s)
    side_spec(dfs[[s]], profiles[[s]]$metrics, SIDES[[s]]$key)), names(SIDES))
  split_values <- names(blocks)
  lay <- mirror_layout(specs, split_values)
  x_right <- attr(lay, "total")
  xleft   <- setNames(lay$left, lay$key)
  xanch   <- setNames(lay$anchor, lay$key)
  wof     <- setNames(lay$width, lay$key)

  ramps <- build_ramps(split_values)

  cells <- do.call(rbind, lapply(names(SIDES), function(s)
    side_cells(dfs[[s]], profiles[[s]]$metrics, SIDES[[s]]$key, lay, ramps)))
  cells <- cells[!is.na(cells$row), , drop = FALSE]
  cells$best[is.na(cells$fill)] <- FALSE
  cells$face <- ifelse(cells$best, "bold", "plain")

  if (highlight == "chip" && any(cells$best)) {
    hue <- vapply(cells$split[cells$best],
                  function(s) tail(ramps[[s]](256), 1), character(1))
    cells$fill[cells$best] <- darken(hue, 0.34)
  }
  cells$colour <- unname(ifelse(is.na(cells$fill), TEXT_COL,
                                contrast_text(cells$fill)))
  cells$fill[is.na(cells$fill)] <- "transparent"
  best <- cells[cells$best, , drop = FALSE]

  # --- text ------------------------------------------------------------------
  # The split label centres on the whole block, not on either half's rows: it
  # belongs to both, which is the reason it is where it is.
  axis_text <- data.frame(
    row = vapply(blocks, function(b) mean(b), numeric(1)),
    x = unname(xanch["split"]), label = names(blocks), hjust = 0.5,
    stringsAsFactors = FALSE)

  texts <- do.call(rbind, c(list(axis_text), lapply(names(SIDES), function(s) {
    d <- dfs[[s]]; k <- SIDES[[s]]$key
    hj <- unname(lay$hjust[match(paste0(k, ".model"), lay$key)])
    emb <- !duplicated(paste(d$split_label, d$embedding_label))
    rbind(
      data.frame(row = ave(d$.row, paste(d$split_label, d$embedding_label),
                           FUN = mean)[emb],
                 x = unname(xanch[paste0(k, ".embedding")]),
                 label = as.character(d$embedding_label)[emb], hjust = hj,
                 stringsAsFactors = FALSE),
      data.frame(row = d$.row, x = unname(xanch[paste0(k, ".model")]),
                 label = d$model_label, hjust = hj, stringsAsFactors = FALSE))
  })))

  headers <- data.frame(x = lay$anchor, label = lay$header, hjust = lay$hjust,
                        stringsAsFactors = FALSE)

  # --- rules -----------------------------------------------------------------
  # A split boundary spans the page; an embedding boundary inside a split spans
  # only its own half, and stops at the axis so it never cuts the Split label.
  split_rows <- unname(vapply(blocks, function(b) b["top"], numeric(1)))
  split_rows <- split_rows[split_rows > 1]

  # An embedding boundary is a position in the BLOCK grid, not a gap between
  # two of a half's own rows: the blocks are aligned across the axis now, so
  # both halves rule at the same place -- including a half whose rows are
  # centred away from that edge, or absent from the block entirely.
  emb_rules <- do.call(rbind, lapply(names(SIDES), function(s) {
    if (!length(emb_tops)) return(NULL)
    k <- SIDES[[s]]$key
    cols <- lay[substr(lay$key, 1, 1) == k, ]
    # Out to the page edge, in to the Embedding column's inner edge, so the
    # rule never cuts across the Split label standing between the halves.
    inner <- if (k == "R") unname(xleft[paste0(k, ".embedding")]) else
      unname(xleft[paste0(k, ".embedding")] + wof[paste0(k, ".embedding")])
    data.frame(x = if (k == "R") inner else min(cols$left),
               xend = if (k == "R") max(cols$left + cols$width) else inner,
               y = -(emb_tops - 0.5), lw = 0.25, colour = SEP_COL,
               stringsAsFactors = FALSE)
  }))

  rules <- rbind(
    data.frame(x = 0, xend = x_right, y = -c(-0.5, n + 0.5), lw = 0.6,
               colour = RULE_COL, stringsAsFactors = FALSE),
    data.frame(x = 0, xend = x_right, y = -(split_rows - 0.5), lw = 0.35,
               colour = SEP_COL, stringsAsFactors = FALSE),
    emb_rules)

  # One glossary, not two. The halves deliberately share column HEADERS -- that
  # is what makes them readable across the axis -- so a naive union would define
  # "OA" twice and say nothing the second time.
  gloss_cols <- c(profiles$right$metrics,
                  profiles$left$metrics[!profiles$left$metrics %in%
                                          profiles$right$metrics])
  star_term <- sprintf('bold("%s")*" %s"', STAR, STAR_GLOSS)
  # The star's own term has to be inside the width the glossary is fitted to,
  # or shrink-to-fit would fit the line and then push it off the page.
  star_w <- str_w(paste0(STAR, " ", STAR_GLOSS, "      "), pt = CAP_PT)
  cap <- with_metrics(gloss_cols,
                      metric_caption(total_w = x_right + 2 * PAD_OUT - star_w))
  cap$expr <- paste(cap$expr, star_term, sep = '*"      "*')

  p <- ggplot() +
    geom_tile(data = cells, aes(x = x, y = -row, fill = fill, width = w),
              height = CELL_H, colour = NA)

  if (nrow(best)) {
    if (highlight %in% c("dash", "ring", "halo")) {
      p <- p + geom_tile(
        data = best, aes(x = x, y = -row, width = w + 2 * CELL_PAD * 0.7),
        height = CELL_H + RING_OUT, fill = NA, colour = "grey10",
        linewidth = if (highlight == "ring") 0.55 else 0.85,
        linetype = if (highlight == "dash") DASH_PATTERN else "solid")
    }
    if (highlight == "halo") {
      p <- p + geom_tile(data = best, aes(x = x, y = -row, width = w - RING_IN),
                         height = CELL_H - RING_IN, fill = NA, colour = "white",
                         linewidth = 0.5)
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
              aes(x = x, y = -row, label = label, colour = colour,
                  fontface = face),
              size = LABEL_SIZE_MM, hjust = 0.5) +
    geom_text(data = texts, aes(x = x, y = -row, label = label, hjust = hjust),
              size = LABEL_SIZE_MM, colour = TEXT_COL) +
    geom_text(data = headers, aes(x = x, y = HEADER_Y, label = label,
                                  hjust = hjust),
              size = LABEL_SIZE_MM, fontface = "bold", colour = TEXT_COL) +
    geom_segment(data = rules,
                 aes(x = x, xend = xend, y = y, yend = y, linewidth = lw,
                     colour = colour)) +
    scale_fill_identity() +
    scale_colour_identity() +
    scale_linewidth_identity() +
    scale_x_continuous(expand = expansion(0)) +
    scale_y_continuous(expand = expansion(0)) +
    annotate("text", x = x_right, y = -n - 1.1, label = cap$expr, parse = TRUE,
             hjust = 1, vjust = 1, size = cap$size_mm, colour = CAPTION_COL) +
    coord_cartesian(xlim = c(-PAD_OUT, x_right + PAD_OUT),
                    ylim = c(-n - Y_BELOW, Y_ABOVE), clip = "off") +
    theme_eofm() +
    theme(
      axis.title = element_blank(), axis.text = element_blank(),
      axis.ticks = element_blank(),
      panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
      plot.margin = margin(MARGIN_PT, MARGIN_PT, MARGIN_PT, MARGIN_PT))

  margin_in <- 2 * MARGIN_PT / 72
  structure(p,
            fig_width  = x_right + 2 * PAD_OUT + margin_in,
            fig_height = (n + Y_ABOVE + Y_BELOW) * ROW_H + margin_in)
}

# ── Entry point ───────────────────────────────────────────────────────────────

main <- function(args = commandArgs(trailingOnly = TRUE)) {
  highlight <- "dash"
  i <- 1
  while (i <= length(args)) {
    switch(args[i],
      "--highlight" = { highlight <- args[i + 1]; i <- i + 1 },
      stop("Unknown argument '", args[i], "'", call. = FALSE))
    i <- i + 1
  }

  profiles <- lapply(SIDES, function(s) {
    pr <- TASK_PROFILES[[s$task]]
    pr$metrics <- pr$metrics[!names(pr$metrics) %in% MIRROR_DROP]
    pr
  })
  dfs <- lapply(profiles, function(p) with_metrics(p$metrics, read_metrics(p$csv)))

  message("Mirrored table: ",
          paste(vapply(names(dfs), function(s)
            sprintf("%s %d rows", s, nrow(dfs[[s]])), character(1)),
            collapse = ", "), ", highlight = ", highlight)

  p <- mirror_table_plot(dfs, profiles, highlight = highlight)
  save_plot(p, "metrics_table_mirror", width = attr(p, "fig_width"),
            height = attr(p, "fig_height"), formats = c("png", "pdf"),
            subdir = PLOT_DIR_MODELS)
}

if (sys.nframe() == 0) main()
