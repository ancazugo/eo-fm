# patch_geo_context.R ─ Per-patch geographic context, from lon/lat alone.
#
#     source("R/patch_geo_context.R")
#     ctx <- geo_context(df)          # df needs uid, lon, lat
#
# Two lookups, both done on the patch's OWN coordinates rather than on its city
# or country name — a city name is one label for a 320 m patch that may sit in a
# different climate zone from the city centre, and the So2Sat city field is not
# always a country's own name for the place anyway.
#
#   subregion  UN M49 sub-region (22 worldwide), by point-in-polygon against
#              country boundaries, then a join to the UN's own M49 table.
#   koppen     Köppen-Geiger class (Beck et al. 2023), sampled from the 1 km
#              1991-2020 raster.
#
# The result is cached to data/patch_geo_context.csv, keyed by `uid`, because
# both lookups are slow enough to be annoying and neither ever changes: a
# patch's coordinates are fixed. Rebuild by deleting the file.
#
# TWO COASTLINE FACTS THAT DRIVE THE DESIGN
#
#   1. `spData::world` is 1:110m, so **7.8% of patches fall outside every
#      country polygon** — all of them in coastal cities (Cape Town, New York,
#      Melbourne, Vancouver, Istanbul, Hong Kong...). They are not errors, the
#      coastline is just coarse, so they are resolved by nearest country
#      instead; the largest such distance measured is 37 km.
#   2. The Köppen raster is LAND ONLY, so **9.3% of patches sample NA**. Those
#      are water, and are reported as such (`water = TRUE`) rather than being
#      filled from the nearest land cell — a harbour patch is water, not its
#      city's climate. Both plots take this one raster as the definition, so
#      they agree on which patches are wet.

source("R/constants.R")

suppressPackageStartupMessages({
  library(dplyr)
  library(sf)
})

GEO_CONTEXT_CSV <- file.path("data", "patch_geo_context.csv")

M49_CSV    <- file.path(DATA_DIR, "input", "UN", "M49", "UNSD — Methodology.csv")
KOPPEN_TIF <- file.path(DATA_DIR, "input", "GloH2O", "koppen_geiger_tif",
                        "1991_2020", "koppen_geiger_0p00833333.tif")
KOPPEN_LEGEND <- file.path(DATA_DIR, "input", "GloH2O", "koppen_geiger_tif",
                           "legend.txt")

# ── Reference tables ──────────────────────────────────────────────────────────

#' The UN M49 table: ISO-alpha2 -> region / sub-region.
#'
#' Semicolon-separated with a UTF-8 BOM on the first header. Namibia's
#' ISO-alpha2 is "NA", which read.csv turns into a missing value unless NA
#' strings are switched off — that is the one row this has to get right.
#'
#' **The 22 sub-regions are not the `Sub-region Name` column.** That column
#' holds the 17-member tier, in which the whole of Africa below the Sahara is
#' one bucket ("Sub-Saharan Africa") and the Americas are two. The familiar 22
#' come from taking `Intermediate Region Name` wherever the UN defines one —
#' which is exactly Africa (Eastern/Middle/Southern/Western) and the Americas
#' (Caribbean/Central America/South America) — and the sub-region otherwise.
read_m49 <- function(path = M49_CSV) {
  if (!file.exists(path)) {
    stop("Missing the UN M49 table at ", path, call. = FALSE)
  }
  m <- utils::read.csv2(path, fileEncoding = "UTF-8-BOM", na.strings = character(0),
                        colClasses = "character")
  inter <- m[["Intermediate.Region.Name"]]
  tibble::tibble(
    iso_a2    = m[["ISO.alpha2.Code"]],
    iso_a3    = m[["ISO.alpha3.Code"]],
    m49_code  = m[["M49.Code"]],
    country   = m[["Country.or.Area"]],
    region    = m[["Region.Name"]],
    subregion = ifelse(nzchar(inter), inter, m[["Sub.region.Name"]])
  ) |> filter(nzchar(iso_a2), nzchar(subregion))
}

#' The Köppen-Geiger legend: value, code, description, official RGB.
#'
#' Parsed from the shipped legend.txt rather than retyped, so the colours are
#' exactly the ones Beck et al. (2023) publish.
read_koppen_legend <- function(path = KOPPEN_LEGEND) {
  if (!file.exists(path)) stop("Missing ", path, call. = FALSE)
  ln <- grep("^\\s*\\d+:", readLines(path, warn = FALSE), value = TRUE)
  m  <- regmatches(ln, regexec(
    "^\\s*(\\d+):\\s+(\\S+)\\s+(.*?)\\s*\\[\\s*(\\d+)\\s+(\\d+)\\s+(\\d+)\\s*\\]\\s*$", ln))
  parts <- do.call(rbind, lapply(m, function(x) x[-1]))
  tibble::tibble(
    value  = as.integer(parts[, 1]),
    code   = parts[, 2],
    desc   = trimws(parts[, 3]),
    colour = grDevices::rgb(as.integer(parts[, 4]), as.integer(parts[, 5]),
                            as.integer(parts[, 6]), maxColorValue = 255)
  )
}

# ── Lookups ───────────────────────────────────────────────────────────────────

#' M49 sub-region for each lon/lat.
lookup_subregion <- function(lon, lat) {
  w   <- spData::world[, c("iso_a2")]
  pts <- sf::st_as_sf(data.frame(lon = lon, lat = lat),
                      coords = c("lon", "lat"), crs = 4326)
  i <- vapply(sf::st_within(pts, w),
              function(z) if (length(z)) z[[1]] else NA_integer_, integer(1))
  na <- is.na(i)
  if (any(na)) {
    # Coastal, not wrong — see the header. Nearest country, no distance cap:
    # every one of these measured under 40 km, and a cap would only turn a
    # correct answer into a missing one.
    i[na] <- sf::st_nearest_feature(pts[na, ], w)
    message("  subregion: ", sum(na), " point(s) outside every country polygon, ",
            "resolved by nearest country")
  }
  iso <- sf::st_drop_geometry(w)$iso_a2[i]
  m49 <- read_m49()
  # The country name, ISO-alpha3 and numeric M49 code come from the UN table,
  # not from spData: spData supplies geometry and an ISO-alpha2 to key on, the
  # UN table is the authority for every label derived from it.
  out <- tibble::tibble(iso_a2 = iso) |> left_join(m49, by = "iso_a2")
  if (anyNA(out$subregion)) {
    message("  subregion: ", sum(is.na(out$subregion)),
            " point(s) with no M49 row for ISO code(s) ",
            paste(unique(out$iso_a2[is.na(out$subregion)]), collapse = ", "))
  }
  out
}

#' Köppen-Geiger class value for each lon/lat. NA means water.
#'
#' The 1 km raster is land-only, so a patch whose centre falls on ocean or a
#' large lake samples nothing. That NA is not a gap to be patched — it is the
#' answer, and it is the definition of "water" both plots use, so the two agree
#' on which patches are wet. An earlier version filled these from the nearest
#' land cell within ~28 km; that quietly gave a harbour patch its city's climate
#' and made 9% of the data look like confident land classifications.
lookup_koppen <- function(lon, lat) {
  if (!file.exists(KOPPEN_TIF)) stop("Missing ", KOPPEN_TIF, call. = FALSE)
  v <- terra::extract(terra::rast(KOPPEN_TIF), cbind(lon, lat))[, 1]
  v[v == 0] <- NA_integer_
  as.integer(v)
}

# ── Cache ─────────────────────────────────────────────────────────────────────

#' Geographic context for every uid in `df`, cached on disk.
#'
#' @param df Frame with `uid`, `lon`, `lat`.
#' @return `df` with `country`, `iso_a2`, `iso_a3`, `m49_code`, `subregion`,
#'   `region`, `water` and `koppen` (the Köppen code, a factor in the standard
#'   A->E order) joined on.
geo_context <- function(df, cache = GEO_CONTEXT_CSV) {
  stopifnot(all(c("uid", "lon", "lat") %in% names(df)))

  have <- if (file.exists(cache)) {
    readr::read_csv(cache, show_col_types = FALSE, progress = FALSE)
  } else {
    tibble::tibble(uid = character(), iso_a2 = character(),
                   iso_a3 = character(), m49_code = character(),
                   country = character(), subregion = character(),
                   region = character(), koppen_value = integer(),
                   water = logical())
  }

  need <- df[!df$uid %in% have$uid, c("uid", "lon", "lat")]
  need <- need[!duplicated(need$uid), ]
  if (nrow(need)) {
    message("Building geographic context for ", format(nrow(need), big.mark = ","),
            " patch(es) (cached: ", format(nrow(have), big.mark = ","), ")")
    sub <- lookup_subregion(need$lon, need$lat)
    kop <- lookup_koppen(need$lon, need$lat)
    message("  koppen: ", sum(is.na(kop)), " point(s) on water (",
            sprintf("%.1f%%", 100 * mean(is.na(kop))), ")")
    add <- tibble::tibble(uid = need$uid, iso_a2 = sub$iso_a2,
                          iso_a3 = sub$iso_a3, m49_code = sub$m49_code,
                          country = sub$country, subregion = sub$subregion,
                          region = sub$region, koppen_value = kop,
                          water = is.na(kop))
    have <- bind_rows(have, add)
    dir.create(dirname(cache), showWarnings = FALSE, recursive = TRUE)
    readr::write_csv(have, cache)
    message("  wrote ", cache, " (", format(nrow(have), big.mark = ","), " rows)")
  }

  leg <- read_koppen_legend()
  have$koppen <- factor(leg$code[match(have$koppen_value, leg$value)],
                        levels = leg$code)
  keep <- c("uid", "iso_a2", "iso_a3", "m49_code", "country", "subregion",
            "region", "koppen", "water")
  left_join(df, have[, intersect(keep, names(have))], by = "uid")
}

if (sys.nframe() == 0) {
  # Standalone: warm the cache from a projection run's parquet.
  args <- commandArgs(trailingOnly = TRUE)
  source("R/embedding_projection.R")
  runs <- list_projections()
  run  <- resolve_run(runs, if (length(args)) args[[1]] else NULL)
  d <- read_projection(run$full, columns = c("uid", "lon", "lat"))
  invisible(geo_context(d))
}
