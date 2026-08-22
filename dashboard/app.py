"""
TurbineGuard AI Dashboard
PatchTST-powered multi-agent predictive maintenance dashboard.
"""

from pathlib import Path

import os
import dash
import dash_mantine_components as dmc
from dash import Input, Output, State, dcc, html
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.agents.graph import run_turbineguard
from src.ml.dashboard_inference import predict_engine_rul


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
DATASETS = ["FD001", "FD002", "FD003", "FD004"]

# C-MAPSS RUL labels are conventionally clipped at 125 cycles during training
# (the piecewise-linear degradation target used across most published PatchTST /
# LSTM baselines on this dataset). The radial gauge below uses that same 125-cycle
# ceiling as its full-scale mark, so a "full" gauge means "as healthy as the
# model is ever trained to predict" rather than an arbitrary UI maximum.
RUL_GAUGE_MAX = 125

_data_cache: dict[str, pd.DataFrame] = {}


# -----------------------------------------------------------------------------
# Data helpers (unchanged business logic)
# -----------------------------------------------------------------------------
def load_dataset(dataset: str) -> pd.DataFrame:
    """Load and cache a processed C-MAPSS test set."""
    dataset = dataset.upper()

    if dataset not in _data_cache:
        file_path = DATA_DIR / f"test_{dataset}_processed.csv"

        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")

        _data_cache[dataset] = pd.read_csv(file_path)

    return _data_cache[dataset]


def engine_options(dataset: str):
    data = load_dataset(dataset)
    engines = sorted(data["unit_nr"].unique().tolist())

    # Mantine's Select expects string values, so unit_nr (int) is stringified
    # here and cast back to int wherever it's used to filter the dataframe.
    return [
        {"label": f"Engine {engine}", "value": str(engine)}
        for engine in engines
    ]


def risk_status(predicted_rul: float, threshold: float) -> str:
    """Map a predicted RUL to a status key used to drive gauge / pill colors."""
    if predicted_rul < 15:
        return "critical"
    if predicted_rul < threshold:
        return "warning"
    return "healthy"


STATUS_LABEL = {
    "critical": "CRITICAL",
    "warning": "WARNING",
    "healthy": "HEALTHY",
    "neutral": "STANDBY",
}


# -----------------------------------------------------------------------------
# Small component builders (display-only helpers — no business logic here)
# -----------------------------------------------------------------------------
def build_status_pill(status: str, text: str | None = None):
    return html.Div(
        [
            html.Span(className="led-dot"),
            html.Span(text or STATUS_LABEL.get(status, status.upper())),
        ],
        className=f"status-pill status-{status}",
    )


def build_gauge(value: float, caption: str, status: str = "neutral", max_scale: float = RUL_GAUGE_MAX):
    percent = max(0.0, min(value / max_scale, 1.0)) * 100 if max_scale else 0
    return html.Div(
        [
            html.Div(
                className="gauge-ring",
                style={
                    "--gauge-value": f"{percent:.1f}",
                    "--gauge-color": f"var(--status-{status})" if status != "neutral" else "var(--status-neutral)",
                },
            ),
            html.Div(
                [
                    html.Div(f"{value:.0f}", className="gauge-value"),
                    html.Div(caption, className="gauge-caption"),
                ],
                className="gauge-inner",
            ),
        ],
        className="gauge-container",
    )


def build_gauge_row(value: float, status: str, latest_cycle, regime):
    # Two siblings, not one cramped row: the dial keeps its own line, and
    # Cycle/Regime get the full sidebar width below so their values are
    # never clipped.
    gauge_section = html.Div(
        [
            build_gauge(value, "cycles left", status),
            html.Div(
                [
                    html.Div("Predicted RUL", className="gauge-meta-label"),
                    build_status_pill(status),
                ],
                className="gauge-meta",
            ),
        ],
        className="gauge-row",
    )

    tiles_section = html.Div(
        [
            html.Div(
                [
                    html.Div("Cycle", className="metric-tile-label"),
                    html.Div(str(latest_cycle), className="metric-tile-value"),
                ],
                className="metric-tile",
            ),
            html.Div(
                [
                    html.Div("Regime", className="metric-tile-label"),
                    html.Div(str(regime), className="metric-tile-value"),
                ],
                className="metric-tile",
            ),
        ],
        className="tile-row",
    )

    return [gauge_section, tiles_section]


def build_empty_state(text: str, positive: bool = False):
    classes = "empty-state is-positive" if positive else "empty-state"
    return html.Div(text, className=classes)


# dcc.Dropdown's default light-theme CSS classnames aren't stable across Dash
# versions, which is why targeting ".Select-control" etc. from style.css was
# silently no-oping. dmc.Select is styled directly through its own `styles`
# prop instead, so it can't fall back to an unstyled default.
MANTINE_THEME = {
    "fontFamily": "var(--font-body)",
    "primaryColor": "cyan",
    "colors": {
        "cyan": [
            "#e0fbfc", "#b8f3f5", "#8ce9ed", "#5fdfe4", "#3dd8de",
            "#2dd4dd", "#22b4bc", "#18939a", "#0f6f75", "#083f42",
        ],
        "dark": [
            "#e8ecf1", "#8b95a7", "#6b7386", "#565f72", "#3a4152",
            "#2c3140", "#232837", "#1c2230", "#171c26", "#12161d",
        ],
    },
}


def dmc_select(component_id: str, data: list[dict[str, str]], value: str | None = None):
    return dmc.Select(
        id=component_id,
        data=data,
        value=value,
        searchable=False,
        clearable=False,
        allowDeselect=False,
        checkIconPosition="right",
        styles={
            "input": {
                "backgroundColor": "var(--bg-input)",
                "borderColor": "var(--border-hairline-strong)",
                "borderRadius": "8px",
                "color": "var(--text-primary)",
                "fontFamily": "var(--font-mono)",
                "fontSize": "13px",
                "fontWeight": 500,
                "minHeight": "40px",
            },
            "dropdown": {
                "backgroundColor": "var(--bg-panel-raised)",
                "borderColor": "var(--border-hairline-strong)",
                "borderRadius": "8px",
                "boxShadow": "0 12px 28px -12px rgba(0, 0, 0, 0.65)",
            },
            "option": {
                "color": "var(--text-secondary)",
                "fontFamily": "var(--font-mono)",
                "fontSize": "13px",
                "borderRadius": "6px",
            },
        },
    )


app = dash.Dash(
    __name__,
    suppress_callback_exceptions=True,
    external_stylesheets=[
        "https://fonts.googleapis.com/css2?"
        "family=Chakra+Petch:wght@500;600;700&"
        "family=Inter:wght@400;500;600;700&"
        "family=JetBrains+Mono:wght@500;600;700&display=swap"
    ],
)
app.title = "TurbineGuard AI | Predictive Maintenance"

default_dataset = "FD004"
default_engines = engine_options(default_dataset)
default_engine = default_engines[0]["value"]

dashboard_layout = html.Div(
    [
        html.Div(
            [
                html.Div(
                    [
                        html.H1(["TURBINE", html.Span("GUARD", className="brand-accent")]),
                        html.P("PatchTST RUL Prediction · Agentic Root-Cause Diagnosis · Maintenance Dispatch"),
                    ]
                ),
                html.Div(
                    [
                        html.Span(className="led-dot"),
                        html.Span("SIMULATED ENVIRONMENT"),
                    ],
                    className="header-status-pill",
                ),
            ],
            className="dashboard-header",
        ),

        html.Div(
            [
                html.Div(
                    [
                        html.H3("Model Selection"),

                        html.Label("Dataset / Model"),
                        dmc_select(
                            "dataset-selector",
                            data=[
                                {
                                    "label": f"{dataset} PatchTST model",
                                    "value": dataset,
                                }
                                for dataset in DATASETS
                            ],
                            value=default_dataset,
                        ),

                        html.H3("Engine Selection", style={"marginTop": "28px"}),

                        html.Label("Select Engine ID"),
                        dmc_select(
                            "engine-selector",
                            data=default_engines,
                            value=default_engine,
                        ),

                        html.Div(
                            id="gauge-row-container",
                            children=build_gauge_row(0, "neutral", "—", "—"),
                        ),

                        html.Label("RUL Threshold (cycles)"),
                        dcc.Slider(
                            id="rul-threshold",
                            min=10,
                            max=50,
                            step=5,
                            value=30,
                            marks={i: str(i) for i in range(10, 51, 10)},
                            tooltip={
                                "placement": "bottom",
                                "always_visible": True,
                            },
                        ),

                        html.Button(
                            "Run Model & Agent Analysis",
                            id="run-analysis-btn",
                            n_clicks=0,
                            className="run-button",
                        ),

                        html.Div(
                            id="status-message",
                            className="status-message",
                        ),
                    ],
                    className="sidebar",
                ),

                html.Div(
                    [
                        html.Div(
                            [
                                html.H3("Agent Decision"),
                                html.Div(
                                    id="agent-decision-banner",
                                    children=build_empty_state(
                                        "Choose a dataset and engine, then run analysis to obtain a "
                                        "PatchTST prediction."
                                    ),
                                ),
                            ],
                            className="agent-decision-section",
                        ),

                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.H3("Proposed Work Order"),
                                        html.Div(
                                            id="work-order-card",
                                            children=build_empty_state("No maintenance action proposed"),
                                            className="work-order-card",
                                        ),
                                    ],
                                ),

                                html.Div(
                                    [
                                        html.H3("Retrieved Manuals"),
                                        html.Div(
                                            id="manuals-card",
                                            children=build_empty_state("No manuals retrieved"),
                                            className="manuals-card",
                                        ),
                                    ],
                                ),
                            ],
                            className="card-section",
                        ),

                        html.Div(
                            [
                                html.H3("Fleet Overview"),

                                html.Div(
                                    [
                                        html.Div(
                                            [
                                                html.Div("Total Engines", className="kpi-label"),
                                                html.Div(id="total-engines-kpi", className="kpi-value kpi-value-total"),
                                            ],
                                            className="kpi-card",
                                        ),

                                        html.Div(
                                            [
                                                html.Div("High Risk", className="kpi-label"),
                                                html.Div(id="high-risk-kpi", className="kpi-value kpi-value-high-risk"),
                                            ],
                                            className="kpi-card",
                                        ),

                                        html.Div(
                                            [
                                                html.Div("Average Predicted RUL", className="kpi-label"),
                                                html.Div(id="avg-rul-kpi", className="kpi-value kpi-value-avg"),
                                            ],
                                            className="kpi-card",
                                        ),
                                    ],
                                    className="kpi-container",
                                ),

                                html.Div(
                                    [
                                        html.H4("Predicted RUL Across Fleet"),
                                        dcc.Graph(
                                            id="rul-distribution-chart",
                                            style={"height": "380px"},
                                            config={"displayModeBar": False},
                                        ),
                                    ],
                                    className="chart-container",
                                ),
                            ],
                            className="fleet-overview",
                        ),
                    ],
                    className="main-content",
                ),
            ],
            className="main-layout",
        ),

        html.Div(
            [
                html.P(
                    "TURBINEGUARD AI  ·  PatchTST + LangGraph + MCP + FAISS RAG  ·  "
                    "Work orders shown as proposed actions"
                ),
            ],
            className="dashboard-footer",
        ),
    ],
    className="dashboard-container",
)

app.layout = dmc.MantineProvider(
    forceColorScheme="dark",
    theme=MANTINE_THEME,
    children=dashboard_layout,
)


def empty_figure():
    fig = go.Figure()
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=380,
        xaxis={"visible": False},
        yaxis={"visible": False},
    )
    return fig


@app.callback(
    [
        Output("engine-selector", "data"),
        Output("engine-selector", "value"),
    ],
    Input("dataset-selector", "value"),
)
def update_engine_selector(dataset):
    options = engine_options(dataset)
    return options, options[0]["value"]


@app.callback(
    Output("gauge-row-container", "children"),
    [
        Input("dataset-selector", "value"),
        Input("engine-selector", "value"),
        Input("rul-threshold", "value"),
    ],
)
def preview_prediction(dataset, engine_id, rul_threshold):
    if not dataset or engine_id is None:
        return build_gauge_row(0, "neutral", "—", "—")

    try:
        data = load_dataset(dataset)
        engine_data = data[data["unit_nr"] == int(engine_id)]

        prediction = predict_engine_rul(dataset, engine_data)
        latest_cycle = int(engine_data["time_in_cycles"].max())
        predicted_rul = float(prediction["predicted_rul"])
        status = risk_status(predicted_rul, rul_threshold or 30)

        return build_gauge_row(
            predicted_rul,
            status,
            latest_cycle,
            prediction["operating_regime"],
        )
    except Exception:
        return build_gauge_row(0, "neutral", "—", "—")


@app.callback(
    [
        Output("agent-decision-banner", "children"),
        Output("work-order-card", "children"),
        Output("manuals-card", "children"),
        Output("status-message", "children"),
        Output("status-message", "className"),
        Output("total-engines-kpi", "children"),
        Output("high-risk-kpi", "children"),
        Output("avg-rul-kpi", "children"),
        Output("rul-distribution-chart", "figure"),
    ],
    Input("run-analysis-btn", "n_clicks"),
    [
        State("dataset-selector", "value"),
        State("engine-selector", "value"),
        State("rul-threshold", "value"),
    ],
)
def run_analysis(n_clicks, dataset, engine_id, rul_threshold):
    if n_clicks == 0:
        return (
            dash.no_update,
            dash.no_update,
            dash.no_update,
            dash.no_update,
            dash.no_update,
            dash.no_update,
            dash.no_update,
            dash.no_update,
            empty_figure(),
        )

    try:
        engine_id = int(engine_id)

        result = run_turbineguard(
            engine_id=engine_id,
            dataset=dataset,
            rul_threshold=rul_threshold,
        )

        predicted_rul = float(result["rul_prediction"])
        status = risk_status(predicted_rul, rul_threshold)

        agent_banner = html.Div(
            [
                html.Div(build_status_pill(status), className="decision-strip-badge"),
                html.Div(
                    [
                        html.Div(STATUS_LABEL.get(status, status.upper()), className="decision-strip-title"),
                        html.Div(result["final_decision"], className="decision-strip-detail"),
                    ],
                    className="decision-strip-body",
                ),
            ],
            className=f"decision-strip status-{status}",
        )

        if result.get("work_order"):
            work_order = result["work_order"]

            work_order_html = html.Div(
                [
                    html.Div("SIMULATED / PROPOSED — no CMMS record was created", className="work-order-simulated-tag"),
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Div("Status", className="work-order-label"),
                                    html.Div(work_order.status, className="work-order-value"),
                                ],
                                className="work-order-metric",
                            ),
                            html.Div(
                                [
                                    html.Div("Priority", className="work-order-label"),
                                    html.Div(
                                        work_order.priority,
                                        className=f"priority-pill priority-{work_order.priority}",
                                    ),
                                ],
                                className="work-order-metric",
                            ),
                            html.Div(
                                [
                                    html.Div("Engine", className="work-order-label"),
                                    html.Div(str(work_order.engine_id), className="work-order-value"),
                                ],
                                className="work-order-metric",
                            ),
                        ],
                        className="work-order-metrics",
                    ),
                    html.Div(
                        [
                            html.Div("Description", className="work-order-description-label"),
                            html.Div(work_order.description, className="work-order-description"),
                        ]
                    ),
                ]
            )
        else:
            work_order_html = build_empty_state(
                "No work order proposed — predicted RUL is above the threshold.", positive=True
            )

        manuals = result.get("retrieved_manuals", [])

        if manuals:
            manuals_html = html.Ul(
                [
                    html.Li(
                        [
                            html.Div(
                                [
                                    html.Span(manual["title"], className="manual-title"),
                                    html.Span(f"{manual['score']:.3f}", className="manual-score"),
                                ],
                                className="manual-title-row",
                            ),
                            html.Div(
                                html.Div(
                                    className="manual-score-fill",
                                    style={"width": f"{min(manual['score'], 1.0) * 100:.0f}%"},
                                ),
                                className="manual-score-track",
                            ),
                        ],
                        className="manual-item",
                    )
                    for manual in manuals
                ],
                className="manual-list",
            )
        else:
            manuals_html = build_empty_state("No manuals needed for this predicted risk state.")

        data = load_dataset(dataset)
        fleet_rows = []

        for fleet_engine in sorted(data["unit_nr"].unique()):
            engine_data = data[data["unit_nr"] == fleet_engine]

            prediction = predict_engine_rul(dataset, engine_data)

            fleet_rows.append(
                {
                    "unit_nr": fleet_engine,
                    "predicted_rul": prediction["predicted_rul"],
                }
            )

        fleet_predictions = pd.DataFrame(fleet_rows)
        high_risk_count = int(
            (fleet_predictions["predicted_rul"] < rul_threshold).sum()
        )
        average_rul = fleet_predictions["predicted_rul"].mean()

        figure = px.bar(
            fleet_predictions,
            x="unit_nr",
            y="predicted_rul",
            color="predicted_rul",
            color_continuous_scale=["#e5484d", "#e8a33d", "#3ecf8e"],
            labels={
                "unit_nr": "Engine ID",
                "predicted_rul": "Predicted RUL (cycles)",
            },
        )

        figure.update_traces(
            marker_line_width=0,
            hovertemplate="<b>Engine %{x}</b><br>Predicted RUL: %{y:.1f} cycles<extra></extra>",
        )

        figure.update_layout(
            showlegend=False,
            height=380,
            margin=dict(l=10, r=10, t=10, b=10),
            font=dict(family="Inter, sans-serif", color="#8b95a7", size=12),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(
                title="Engine ID",
                showgrid=False,
                zeroline=False,
                color="#8b95a7",
                linecolor="rgba(148,163,184,0.15)",
            ),
            yaxis=dict(
                title="Predicted RUL (cycles)",
                showgrid=True,
                gridcolor="rgba(148,163,184,0.08)",
                zeroline=False,
                color="#8b95a7",
            ),
            coloraxis_showscale=False,
            hoverlabel=dict(
                bgcolor="#171c26",
                bordercolor="rgba(148,163,184,0.2)",
                font=dict(family="JetBrains Mono, monospace", color="#e8ecf1", size=12),
            ),
        )

        figure.add_hline(
            y=rul_threshold,
            line_dash="dash",
            line_color="#c98a4b",
            line_width=1.5,
            annotation_text=f"Threshold: {rul_threshold} cycles",
            annotation_font_color="#c98a4b",
            annotation_font_family="JetBrains Mono, monospace",
            annotation_font_size=11,
            annotation_position="top right",
        )

        status_msg = (
            f"Analysis complete for Engine {engine_id} on {dataset} — "
            f"predicted RUL {predicted_rul:.1f} cycles."
        )

        return (
            agent_banner,
            work_order_html,
            manuals_html,
            status_msg,
            "status-message status-success",
            str(len(fleet_predictions)),
            str(high_risk_count),
            f"{average_rul:.1f}",
            figure,
        )

    except Exception as exc:
        error_message = f"Analysis failed: {exc}"

        return (
            html.Div(
                [
                    html.Div(build_status_pill("critical", "ERROR"), className="decision-strip-badge"),
                    html.Div(
                        html.Div(error_message, className="decision-strip-detail"),
                        className="decision-strip-body",
                    ),
                ],
                className="decision-strip status-critical",
            ),
            build_empty_state("No work order proposed"),
            build_empty_state("No manuals retrieved"),
            error_message,
            "status-message status-error",
            "—",
            "—",
            "—",
            empty_figure(),
        )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8050"))

    app.run(
        debug=False,
        host="0.0.0.0",
        port=port,
        threaded=True,
    )
