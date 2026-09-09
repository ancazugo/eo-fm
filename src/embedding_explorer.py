"""Interactive Dash explorer for the embedding-projection parquet outputs.

Loads the ``projection_*.parquet`` files written by ``src/embedding_projection.py``
(400k-row UMAP/t-SNE/PCA coordinates + per-patch metadata) and serves a WebGL
scatter explorer so you can:

  * pick the projection method (umap / tsne / pca) — only the ones a given run
    actually contains are offered,
  * colour by any facet and independently set the marker **shape** by a second,
  * colour by the embedding's own **RGB** when a run carries ``rgb_*`` columns,
    which are the same quantity the RGB rasters use, so a patch that looks teal
    here sits in the teal part of the map,
  * filter the cloud by any combination of facet values,
  * switch between a single plot, **facet panels** (small multiples), or
    **linked dual plots** — either two methods of one run, or **two different
    runs** (v2 vs coop, gap vs ring), with a box/lasso selection in one ringing
    the same patches in the other.

Run it, then port-forward 8050::

    source /maps/acz25/envs/eo_fm-env/bin/activate
    python src/embedding_explorer.py \
        --data-dir $DATA_DIR/output/lcz-classification/embedding_viz

The run dropdown lists every ``projection_*.parquet`` under ``--data-dir``, so
new embeddings and poolings are switchable without restarting.

Two things this file is careful about:

* **``patch_id`` is not unique.** It restarts at 000000 in each of training /
  validation / testing, so linked selection keys on ``uid`` ("<dataset>/<patch_id>").
  Older parquets predate the column and get it synthesised on load.
* **Two runs need not share a population.** Cross-run selection therefore reports
  how many of the selected patches actually exist in the second run; without
  that, silently dropped patches read as signal.
"""

from __future__ import annotations

import argparse
import os
from functools import lru_cache
from pathlib import Path

import pandas as pd
import plotly.express as px
import pyarrow.parquet as pq
from dash import Dash, Input, Output, State, ctx, dcc, html

from utils.constants import lcz_dict

# ── Facet / coordinate metadata ───────────────────────────────────────────────
# Candidates only: the layout is built from the union actually present across
# the discovered runs, because Dash wires callback ids at layout time and a
# per-run facet list would raise "nonexistent object was used in an Input".
CANDIDATE_FACETS = ["lcz_name", "city", "country", "continent", "split",
                    "embedding", "pooling", "year"]
METHODS = ["umap", "tsne", "pca"]
SPLIT_ORDER = ["train", "val", "test"]
HOVER_COLS = ["uid", "city", "country", "continent", "lcz_name", "split"]
# scattergl symbol palette is small — shape-by is only meaningful for few groups.
LOW_CARD_SHAPE = {"continent", "split", "embedding", "pooling", "year"}
RGB_OPTION = "__rgb__"


def _lcz_color_map() -> dict[str, str]:
    """Map the parquet's ``lcz_name`` strings ("LCZ {n}: {label}") to colours."""
    out: dict[str, str] = {}
    for code, meta in lcz_dict.items():
        out[f"LCZ {code}: {meta['name']}"] = meta["color"]
    return out


LCZ_COLORS = _lcz_color_map()


# ── Data access (cached per parquet path) ─────────────────────────────────────
@lru_cache(maxsize=8)
def run_schema(path: str) -> tuple[str, ...]:
    """Column names only — a metadata read, no data pages touched."""
    return tuple(pq.read_schema(path).names)


@lru_cache(maxsize=8)
def load_run(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    # patch_id restarts at 000000 per split, so it is not a key. Synthesise the
    # real one for parquets written before the column existed.
    if "uid" not in df.columns and {"dataset", "patch_id"} <= set(df.columns):
        df["uid"] = df["dataset"].astype(str) + "/" + df["patch_id"].astype(str)
    # Stable categorical orders for the two ordinal-ish facets.
    if "lcz_name" in df:
        order = [n for n in LCZ_COLORS if n in set(df["lcz_name"].unique())]
        df["lcz_name"] = pd.Categorical(df["lcz_name"], categories=order)
    if "split" in df:
        order = [s for s in SPLIT_ORDER if s in set(df["split"].unique())]
        df["split"] = pd.Categorical(df["split"], categories=order)
    return df


def discover_runs(data_dir: Path) -> list[dict]:
    """Return [{label, value}] for every projection parquet under data_dir."""
    paths = sorted(data_dir.rglob("projection_*.parquet"))
    # The sample/density exports are companions, not runs of their own.
    paths = [p for p in paths if not p.stem.endswith(("_sample", "_density"))]
    opts = []
    for p in paths:
        label = p.parent.name if p.parent != data_dir else p.stem
        opts.append({"label": label, "value": str(p)})
    return opts


def available_methods(df_or_cols) -> list[str]:
    """Methods this run actually has coordinates for.

    A parquet written with ``--methods umap`` has no ``pca_x``; offering the PCA
    radio for it used to raise a KeyError deep in the render callback.
    """
    cols = set(df_or_cols if isinstance(df_or_cols, (tuple, list, set))
               else df_or_cols.columns)
    return [m for m in METHODS if f"{m}_x" in cols and f"{m}_y" in cols]


def available_facets(cols) -> list[str]:
    return [f for f in CANDIDATE_FACETS if f in set(cols)]


def rgb_methods(cols) -> list[str]:
    cols = set(cols)
    return [m for m in METHODS if f"rgb_{m}_r" in cols]


def _category_orders(df: pd.DataFrame) -> dict:
    orders = {}
    for f in ("lcz_name", "split"):
        if f in df and hasattr(df[f], "cat"):
            orders[f] = list(df[f].cat.categories)
    return orders


# ── Figure building ───────────────────────────────────────────────────────────
def _prepare(
    df: pd.DataFrame,
    method: str,
    filters: dict[str, list],
    max_points: int,
    seed: int = 0,
) -> pd.DataFrame:
    """Apply facet filters, drop missing coords for the method, sample to cap."""
    xcol, ycol = f"{method}_x", f"{method}_y"
    if xcol not in df.columns:
        return df.iloc[0:0]
    sub = df.dropna(subset=[xcol, ycol])
    for col, vals in filters.items():
        if vals and col in sub.columns:
            sub = sub[sub[col].astype(str).isin(vals)]
    if max_points and len(sub) > max_points:
        sub = sub.sample(max_points, random_state=seed)
    return sub


def build_figure(
    df: pd.DataFrame,
    *,
    method: str,
    color: str | None,
    symbol: str | None,
    facet_col: str | None,
    selected_ids: set | None,
    size: float,
    opacity: float,
    title: str | None = None,
    rgb_method: str | None = None,
) -> "px.scatter":
    xcol, ycol = f"{method}_x", f"{method}_y"
    if df.empty:
        fig = px.scatter(title="No points match the current filters")
        fig.update_layout(template="plotly_white")
        return fig

    hover = [c for c in HOVER_COLS if c in df]

    if rgb_method:
        # Colour each point by its own embedding-space RGB. These come from the
        # PIXEL colour model, so they match the rasters; the run's patch-level
        # pca_1..3 would be a different basis and would not.
        cols = [f"rgb_{rgb_method}_{c}" for c in "rgb"]
        rgb = df[cols].to_numpy()
        css = ["rgb(%d,%d,%d)" % tuple(v) for v in rgb]
        fig = px.scatter(df, x=xcol, y=ycol, custom_data=["uid"],
                         hover_data={c: True for c in hover},
                         render_mode="webgl", title=title)
        fig.update_traces(marker=dict(color=css, size=size, opacity=opacity))
    else:
        color_map = LCZ_COLORS if color == "lcz_name" else None
        kwargs = dict(
            x=xcol, y=ycol,
            color=color if color and color != "none" else None,
            symbol=symbol if symbol and symbol != "none" else None,
            facet_col=facet_col if facet_col and facet_col != "none" else None,
            facet_col_wrap=3,
            color_discrete_map=color_map,
            category_orders=_category_orders(df),
            custom_data=["uid"],      # customdata[0] == uid for linked select
            hover_data={c: True for c in hover},
            render_mode="webgl",
            title=title,
        )
        fig = px.scatter(df, **{k: v for k, v in kwargs.items() if v is not None})
        for tr in fig.data:
            tr.marker.size = size
            tr.marker.opacity = opacity

    # Linked highlight: fade the whole cloud and ring the selected patches.
    if selected_ids:
        for tr in fig.data:
            tr.marker.opacity = opacity * 0.25
        sel = df[df["uid"].isin(selected_ids)]
        if not sel.empty:
            fig.add_scatter(
                x=sel[xcol], y=sel[ycol], mode="markers",
                marker=dict(size=size + 3, color="rgba(0,0,0,0)",
                            line=dict(width=1.5, color="#111")),
                hoverinfo="skip", showlegend=False, name="selected",
            )

    fig.update_layout(
        template="plotly_white",
        legend=dict(itemsizing="constant", title_text=color or ""),
        margin=dict(l=10, r=10, t=40 if title else 10, b=10),
        uirevision="keep",  # preserve zoom across non-data updates
        dragmode="lasso",
    )
    fig.update_xaxes(title_text="")
    fig.update_yaxes(title_text="")
    return fig


# ── App factory ───────────────────────────────────────────────────────────────
def make_app(data_dir: Path) -> Dash:
    run_opts = discover_runs(data_dir)
    if not run_opts:
        raise SystemExit(f"No projection_*.parquet found under {data_dir}")
    default_run = run_opts[0]["value"]

    # Union of what any discovered run offers, read from parquet metadata only.
    # The layout is fixed at build time; per-run gaps are disabled, not removed.
    all_cols: set[str] = set()
    for o in run_opts:
        try:
            all_cols |= set(run_schema(o["value"]))
        except Exception:  # a corrupt or half-written parquet must not stop startup
            continue
    facets = available_facets(all_cols) or ["lcz_name"]

    app = Dash(__name__, title="Embedding Explorer")

    def dropdown(id_, options, value, **kw):
        return dcc.Dropdown(id=id_, options=options, value=value,
                            clearable=False, **kw)

    facet_opts = [{"label": "none", "value": "none"}] + [
        {"label": f, "value": f} for f in facets
    ]
    colour_opts = facet_opts + [{"label": "▦ embedding RGB", "value": RGB_OPTION}]

    controls = html.Div(
        [
            html.Label("Run"),
            dropdown("run", run_opts, default_run),
            html.Hr(),
            html.Label("Display mode"),
            dcc.RadioItems(
                id="mode",
                options=[
                    {"label": "Single", "value": "single"},
                    {"label": "Facet panels", "value": "facet"},
                    {"label": "Dual (linked)", "value": "dual"},
                ],
                value="single",
                inline=True,
            ),
            html.Hr(),
            html.Label("Method"),
            dcc.RadioItems(id="method", options=[], value=None, inline=True),
            html.Label("Colour by"),
            dropdown("color", colour_opts, "lcz_name"),
            html.Label("Shape by (few groups only)"),
            dropdown("symbol", facet_opts, "none"),
            html.Div(
                [
                    html.Label("Facet by (facet mode)"),
                    dropdown("facet_col", facet_opts, "split"),
                ],
                id="facet_box",
            ),
            html.Div(
                [
                    html.Label("2nd panel: run"),
                    dropdown("run2", [{"label": "— same run —", "value": "same"}]
                             + run_opts, "same"),
                    html.Label("2nd panel: method"),
                    dcc.RadioItems(id="method2", options=[], value=None, inline=True),
                    html.Label("2nd panel: colour"),
                    dropdown("color2", colour_opts, "continent"),
                ],
                id="dual_box",
            ),
            html.Hr(),
            html.Label("Filters"),
            html.Div(
                [
                    html.Div(
                        [
                            html.Small(f),
                            dcc.Dropdown(id=f"filter-{f}", multi=True,
                                         placeholder=f"all {f}"),
                        ]
                    )
                    for f in facets
                ]
            ),
            html.Hr(),
            html.Label("Point size"),
            dcc.Slider(2, 12, 1, value=4, id="size",
                       marks=None, tooltip={"placement": "bottom"}),
            html.Label("Opacity"),
            dcc.Slider(0.1, 1.0, 0.1, value=0.7, id="opacity",
                       marks=None, tooltip={"placement": "bottom"}),
            html.Label("Max points (sample)"),
            dcc.Slider(10_000, 400_000, 10_000, value=80_000, id="maxpts",
                       marks=None, tooltip={"placement": "bottom"}),
            html.Hr(),
            html.Button("Clear selection", id="clear_sel", n_clicks=0),
            html.Div(id="status", style={"fontSize": "12px", "marginTop": "8px",
                                         "color": "#555"}),
        ],
        style={"width": "300px", "padding": "12px", "overflowY": "auto",
               "height": "100vh", "boxSizing": "border-box",
               "borderRight": "1px solid #ddd", "flex": "0 0 300px"},
    )

    graphs = html.Div(
        [
            dcc.Graph(id="graph", style={"height": "100vh", "flex": "1 1 0"}),
            dcc.Graph(id="graph2", style={"height": "100vh", "flex": "1 1 0",
                                          "display": "none"}),
        ],
        style={"display": "flex", "flex": "1 1 auto", "minWidth": 0},
    )

    app.layout = html.Div(
        [dcc.Store(id="sel_store"), controls, graphs],
        style={"display": "flex", "height": "100vh", "fontFamily": "sans-serif"},
    )

    # Populate / reset filter options when the run changes. Facets this run
    # lacks are disabled rather than dropped, so the ids stay wired.
    @app.callback(
        [Output(f"filter-{f}", "options") for f in facets]
        + [Output(f"filter-{f}", "value") for f in facets]
        + [Output(f"filter-{f}", "disabled") for f in facets],
        Input("run", "value"),
    )
    def _refresh_filters(run):
        df = load_run(run)
        opts, disabled = [], []
        for f in facets:
            if f not in df.columns:
                opts.append([])
                disabled.append(True)
                continue
            cats = (list(df[f].cat.categories)
                    if hasattr(df[f], "cat") else sorted(df[f].dropna().unique()))
            opts.append([{"label": str(c), "value": str(c)} for c in cats])
            disabled.append(False)
        return opts + [[] for _ in facets] + disabled

    # Offer only the methods each run actually carries.
    @app.callback(
        Output("method", "options"), Output("method", "value"),
        Output("method2", "options"), Output("method2", "value"),
        Input("run", "value"), Input("run2", "value"),
        State("method", "value"), State("method2", "value"),
    )
    def _refresh_methods(run, run2, cur, cur2):
        avail = available_methods(run_schema(run))
        opts = [{"label": m.upper(), "value": m} for m in avail]
        value = cur if cur in avail else (avail[0] if avail else None)

        other = run if run2 in (None, "same") else run2
        avail2 = available_methods(run_schema(other))
        opts2 = [{"label": m.upper(), "value": m} for m in avail2]
        # Default the second panel to a *different* method when there is one.
        if cur2 in avail2:
            value2 = cur2
        else:
            value2 = next((m for m in avail2 if m != value), avail2[0] if avail2 else None)
        return opts, value, opts2, value2

    # Toggle visibility of mode-specific controls and the 2nd graph.
    @app.callback(
        Output("facet_box", "style"),
        Output("dual_box", "style"),
        Output("graph2", "style"),
        Input("mode", "value"),
    )
    def _toggle(mode):
        hide = {"display": "none"}
        show = {}
        g2 = {"height": "100vh", "flex": "1 1 0"}
        g2_hidden = {**g2, "display": "none"}
        return (
            show if mode == "facet" else hide,
            show if mode == "dual" else hide,
            g2 if mode == "dual" else g2_hidden,
        )

    # Merge box/lasso selections from either graph into the shared store.
    @app.callback(
        Output("sel_store", "data"),
        Input("graph", "selectedData"),
        Input("graph2", "selectedData"),
        Input("clear_sel", "n_clicks"),
    )
    def _collect_selection(sel1, sel2, _clear):
        # On the initial fire triggered_id is neither graph; returning sel2 then
        # was a latent bug, so be explicit about which input triggered.
        if ctx.triggered_id == "clear_sel":
            return None
        if ctx.triggered_id == "graph":
            sel = sel1
        elif ctx.triggered_id == "graph2":
            sel = sel2
        else:
            return None
        if not sel or not sel.get("points"):
            return None
        ids = [p["customdata"][0] for p in sel["points"] if p.get("customdata")]
        return ids or None

    # Main render.
    @app.callback(
        Output("graph", "figure"),
        Output("graph2", "figure"),
        Output("status", "children"),
        Input("run", "value"),
        Input("run2", "value"),
        Input("mode", "value"),
        Input("method", "value"),
        Input("color", "value"),
        Input("symbol", "value"),
        Input("facet_col", "value"),
        Input("method2", "value"),
        Input("color2", "value"),
        Input("size", "value"),
        Input("opacity", "value"),
        Input("maxpts", "value"),
        Input("sel_store", "data"),
        [Input(f"filter-{f}", "value") for f in facets],
    )
    def _render(run, run2, mode, method, color, symbol, facet_col, method2, color2,
                size, opacity, maxpts, sel_ids, *filter_vals):
        df = load_run(run)
        filters = {f: (v or []) for f, v in zip(facets, filter_vals)}
        selected = set(sel_ids) if sel_ids else None
        if not method:
            return px.scatter(), px.scatter(), "This run has no projection coordinates."

        rgb1 = _resolve_rgb(color, df, method)
        sub = _prepare(df, method, filters, maxpts)
        warn = ""
        if symbol and symbol not in (None, "none") and symbol not in LOW_CARD_SHAPE:
            warn = f" ⚠ shape-by '{symbol}' has many groups; webgl symbols recycle."
        if color == RGB_OPTION and rgb1 is None:
            warn += (" ⚠ this run has no rgb_* columns — write them with "
                     "src/embedding_rgb.py annotate-parquet.")

        fig1 = build_figure(
            sub, method=method, color=None if rgb1 else color,
            symbol=None if rgb1 else symbol,
            facet_col=facet_col if mode == "facet" else None,
            selected_ids=selected, size=size, opacity=opacity,
            title=None, rgb_method=rgb1,
        )

        status = f"Showing {len(sub):,} / {len(df.dropna(subset=[f'{method}_x'])):,} points ({method.upper()})."
        fig2 = px.scatter()

        if mode == "dual":
            cross = run2 not in (None, "same") and run2 != run
            df2 = load_run(run2) if cross else df
            if method2 and f"{method2}_x" in df2.columns:
                rgb2 = _resolve_rgb(color2, df2, method2)
                sub2 = _prepare(df2, method2, filters, maxpts)
                fig2 = build_figure(
                    sub2, method=method2, color=None if rgb2 else color2,
                    symbol=None, facet_col=None, selected_ids=selected,
                    size=size, opacity=opacity,
                    title=(Path(run2).parent.name if cross else method2.upper()),
                    rgb_method=rgb2,
                )
            else:
                fig2 = px.scatter(title=f"2nd run has no {method2} coordinates")
            if cross:
                # Two runs need not cover the same patches. Say so, or the
                # patches that silently vanish look like a real difference.
                shared = len(set(df["uid"]) & set(df2["uid"]))
                status += (f" | 2nd run {Path(run2).parent.name}: "
                           f"{shared:,} of {len(df):,} patches in common")
                if selected:
                    here = len(selected & set(df2["uid"]))
                    status += f"; {here:,}/{len(selected):,} selected present"

        if selected:
            status += f" {len(selected):,} selected."
        status += warn
        return fig1, fig2, status

    return app


def _resolve_rgb(color: str | None, df: pd.DataFrame, method: str) -> str | None:
    """Which rgb_* family to colour with, if the user asked and the run has one."""
    if color != RGB_OPTION:
        return None
    have = rgb_methods(df.columns)
    if not have:
        return None
    return method if method in have else have[0]


def main() -> None:
    default_dir = os.path.join(
        os.environ.get("DATA_DIR", "."),
        "output/lcz-classification/embedding_viz",
    )
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default=default_dir,
                    help="Dir containing projection_*.parquet runs")
    ap.add_argument("--parquet", default=None,
                    help="Optional single parquet; its parent becomes --data-dir")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8050)
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    data_dir = (Path(args.parquet).parent if args.parquet
                else Path(args.data_dir))
    app = make_app(data_dir)
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
