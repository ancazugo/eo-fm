"""Interactive Dash explorer for the embedding-projection parquet outputs.

Loads a ``projection_*.parquet`` written by ``src/embedding_projection.py``
(390k+ rows of UMAP/t-SNE/PCA 2-D coords + per-patch metadata) and serves a
WebGL scatter explorer so you can:

  * pick the projection method (umap / tsne / pca),
  * colour by any facet (LCZ class / city / country / continent / split) and
    independently set the **marker shape** by a second facet (two variables at
    once),
  * filter the cloud by any combination of facet values,
  * switch display mode between a single plot, side-by-side **facet panels**
    (small multiples), or **linked dual plots** where a box/lasso selection in
    one panel highlights the same patches in the other,
  * tune point size / opacity / a max-points sample for responsiveness.

Run it, then port-forward 8050::

    source /maps/acz25/envs/eo_fm-env/bin/activate
    python src/embedding_explorer.py \
        --data-dir $DATA_DIR/output/lcz-classification/embedding_viz

A run dropdown lists every ``projection_*.parquet`` under ``--data-dir`` so new
embeddings / poolings are switchable without restarting.
"""

from __future__ import annotations

import argparse
import os
from functools import lru_cache
from pathlib import Path

import pandas as pd
import plotly.express as px
from dash import Dash, Input, Output, State, ctx, dcc, html

from utils.constants import lcz_dict

# ── Facet / coordinate metadata ───────────────────────────────────────────────
FACETS = ["lcz_name", "city", "country", "continent", "split"]
METHODS = ["umap", "tsne", "pca"]
SPLIT_ORDER = ["train", "val", "test"]
HOVER_COLS = ["patch_id", "city", "country", "continent", "lcz_name", "split"]
# scattergl symbol palette is small — shape-by is only meaningful for few groups.
LOW_CARD_SHAPE = {"continent", "split"}


def _lcz_color_map() -> dict[str, str]:
    """Map the parquet's ``lcz_name`` strings ("LCZ {n}: {label}") to colours."""
    out: dict[str, str] = {}
    for code, meta in lcz_dict.items():
        out[f"LCZ {code}: {meta['name']}"] = meta["color"]
    return out


LCZ_COLORS = _lcz_color_map()


# ── Data access (cached per parquet path) ─────────────────────────────────────
@lru_cache(maxsize=4)
def load_run(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
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
    opts = []
    for p in paths:
        # Use the run folder name as the human label when it is informative.
        label = p.parent.name if p.parent != data_dir else p.stem
        opts.append({"label": label, "value": str(p)})
    return opts


def _category_orders(df: pd.DataFrame) -> dict:
    orders = {}
    if "lcz_name" in df:
        orders["lcz_name"] = list(df["lcz_name"].cat.categories)
    if "split" in df:
        orders["split"] = list(df["split"].cat.categories)
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
    sub = df.dropna(subset=[xcol, ycol])
    for col, vals in filters.items():
        if vals:
            sub = sub[sub[col].isin(vals)]
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
) -> "px.scatter":
    xcol, ycol = f"{method}_x", f"{method}_y"
    if df.empty:
        fig = px.scatter(title="No points match the current filters")
        fig.update_layout(template="plotly_white")
        return fig

    color_map = LCZ_COLORS if color == "lcz_name" else None
    kwargs = dict(
        x=xcol,
        y=ycol,
        color=color if color and color != "none" else None,
        symbol=symbol if symbol and symbol != "none" else None,
        facet_col=facet_col if facet_col and facet_col != "none" else None,
        facet_col_wrap=3,
        color_discrete_map=color_map,
        category_orders=_category_orders(df),
        custom_data=["patch_id"],  # customdata[0] == patch_id for linked select
        hover_data={c: True for c in HOVER_COLS if c in df},
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
        sel = df[df["patch_id"].isin(selected_ids)]
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

    app = Dash(__name__, title="Embedding Explorer")

    def dropdown(id_, options, value, **kw):
        return dcc.Dropdown(id=id_, options=options, value=value,
                            clearable=False, **kw)

    facet_opts = [{"label": "none", "value": "none"}] + [
        {"label": f, "value": f} for f in FACETS
    ]

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
            dcc.RadioItems(
                id="method",
                options=[{"label": m.upper(), "value": m} for m in METHODS],
                value="umap", inline=True,
            ),
            html.Label("Colour by"),
            dropdown("color", facet_opts, "lcz_name"),
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
                    html.Label("2nd plot method (dual)"),
                    dcc.RadioItems(
                        id="method2",
                        options=[{"label": m.upper(), "value": m} for m in METHODS],
                        value="tsne", inline=True,
                    ),
                    html.Label("2nd plot colour (dual)"),
                    dropdown("color2", facet_opts, "continent"),
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
                    for f in FACETS
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

    # Populate / reset filter options when the run changes.
    @app.callback(
        [Output(f"filter-{f}", "options") for f in FACETS]
        + [Output(f"filter-{f}", "value") for f in FACETS],
        Input("run", "value"),
    )
    def _refresh_filters(run):
        df = load_run(run)
        opts = []
        for f in FACETS:
            cats = (list(df[f].cat.categories)
                    if hasattr(df[f], "cat") else sorted(df[f].dropna().unique()))
            opts.append([{"label": str(c), "value": str(c)} for c in cats])
        return opts + [[] for _ in FACETS]

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
        if ctx.triggered_id == "clear_sel":
            return None
        sel = sel1 if ctx.triggered_id == "graph" else sel2
        if not sel or not sel.get("points"):
            return None
        ids = [p["customdata"][0] for p in sel["points"]
               if p.get("customdata")]
        return ids or None

    # Main render.
    @app.callback(
        Output("graph", "figure"),
        Output("graph2", "figure"),
        Output("status", "children"),
        Input("run", "value"),
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
        [Input(f"filter-{f}", "value") for f in FACETS],
    )
    def _render(run, mode, method, color, symbol, facet_col, method2, color2,
                size, opacity, maxpts, sel_ids, *filter_vals):
        df = load_run(run)
        filters = {f: (v or []) for f, v in zip(FACETS, filter_vals)}
        selected = set(sel_ids) if sel_ids else None

        sub = _prepare(df, method, filters, maxpts)
        warn = ""
        if symbol and symbol not in (None, "none") and symbol not in LOW_CARD_SHAPE:
            warn = f" ⚠ shape-by '{symbol}' has many groups; webgl symbols recycle."

        fig1 = build_figure(
            sub, method=method, color=color,
            symbol=symbol, facet_col=facet_col if mode == "facet" else None,
            selected_ids=selected, size=size, opacity=opacity,
            title=None,
        )

        if mode == "dual":
            sub2 = _prepare(df, method2, filters, maxpts)
            fig2 = build_figure(
                sub2, method=method2, color=color2, symbol=None,
                facet_col=None, selected_ids=selected, size=size,
                opacity=opacity, title=method2.upper(),
            )
        else:
            fig2 = px.scatter()

        n_total = len(df.dropna(subset=[f"{method}_x"]))
        status = (f"Showing {len(sub):,} / {n_total:,} points "
                  f"({method.upper()}).")
        if selected:
            status += f" {len(selected):,} selected."
        status += warn
        return fig1, fig2, status

    return app


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
