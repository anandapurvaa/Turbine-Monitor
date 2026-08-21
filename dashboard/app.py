"""
TurbineGuard AI Dashboard
PatchTST-powered multi-agent predictive maintenance dashboard.
"""

from pathlib import Path

import os
import dash
from dash import Input, Output, State, dcc, html
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.agents.graph import run_turbineguard
from src.ml.dashboard_inference import predict_engine_rul


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
DATASETS = ["FD001", "FD002", "FD003", "FD004"]

_data_cache: dict[str, pd.DataFrame] = {}


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

    return [
        {"label": f"Engine {engine}", "value": engine}
        for engine in engines
    ]


def risk_label(predicted_rul: float, threshold: float) -> tuple[str, str, str]:
    if predicted_rul < 15:
        return "CRITICAL", "#d62728", "⚠️"
    if predicted_rul < threshold:
        return "WARNING", "#ff7f0e", "⚠️"
    return "HEALTHY", "#2ca02c", "✅"


app = dash.Dash(__name__, suppress_callback_exceptions=True)
app.title = "TurbineGuard AI | Predictive Maintenance"

default_dataset = "FD004"
default_engines = engine_options(default_dataset)
default_engine = default_engines[0]["value"]

app.layout = html.Div(
    [
        html.Div(
            [
                html.H1("⚙️ TurbineGuard AI"),
                html.P(
                    "PatchTST RUL Prediction + Agentic Root-Cause Diagnosis "
                    "+ Proposed Maintenance Dispatch"
                ),
            ],
            className="dashboard-header",
        ),

        html.Div(
            [
                html.Div(
                    [
                        html.H3("Model Selection"),

                        html.Label("Dataset / Model:"),
                        dcc.Dropdown(
                            id="dataset-selector",
                            options=[
                                {
                                    "label": f"{dataset} PatchTST model",
                                    "value": dataset,
                                }
                                for dataset in DATASETS
                            ],
                            value=default_dataset,
                            clearable=False,
                        ),

                        html.H3("Engine Selection", style={"marginTop": "25px"}),

                        html.Label("Select Engine ID:"),
                        dcc.Dropdown(
                            id="engine-selector",
                            options=default_engines,
                            value=default_engine,
                            clearable=False,
                        ),

                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.Div(
                                            "Predicted RUL",
                                            className="metric-label",
                                        ),
                                        html.Div(
                                            id="current-rul-display",
                                            children="—",
                                            className=(
                                                "metric-value metric-value-rul"
                                            ),
                                        ),
                                        html.Div(
                                            "cycles",
                                            className="metric-unit",
                                        ),
                                    ],
                                    className="metric-card metric-card-rul",
                                ),

                                html.Div(
                                    [
                                        html.Div(
                                            "Current Cycle",
                                            className="metric-label",
                                        ),
                                        html.Div(
                                            id="current-cycle-display",
                                            children="—",
                                            className=(
                                                "metric-value metric-value-cycle"
                                            ),
                                        ),
                                    ],
                                    className="metric-card metric-card-cycle",
                                ),

                                html.Div(
                                    [
                                        html.Div(
                                            "Operating Regime",
                                            className="metric-label",
                                        ),
                                        html.Div(
                                            id="regime-display",
                                            children="—",
                                            className="metric-value",
                                        ),
                                    ],
                                    className="metric-card",
                                ),
                            ]
                        ),

                        html.Label("RUL Threshold (cycles):"),
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
                            "🔍 Run Model & Agent Analysis",
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
                                html.H3("🤖 Agent Decision"),
                                html.Div(
                                    id="agent-decision-banner",
                                    children=html.Div(
                                        "Choose a dataset and engine, then run "
                                        "analysis to obtain a PatchTST prediction.",
                                        style={
                                            "padding": "30px",
                                            "textAlign": "center",
                                            "color": "#999",
                                        },
                                    ),
                                ),
                            ],
                            className="agent-decision-section",
                        ),

                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.H3("📋 Proposed Work Order"),
                                        html.Div(
                                            id="work-order-card",
                                            children=html.Div(
                                                "No maintenance action proposed",
                                                style={
                                                    "padding": "40px",
                                                    "textAlign": "center",
                                                    "color": "#999",
                                                },
                                            ),
                                            className="work-order-card",
                                        ),
                                    ],
                                    style={
                                        "width": "48%",
                                        "display": "inline-block",
                                        "verticalAlign": "top",
                                        "marginRight": "4%",
                                    },
                                ),

                                html.Div(
                                    [
                                        html.H3("📚 Retrieved Manuals"),
                                        html.Div(
                                            id="manuals-card",
                                            children=html.Div(
                                                "No manuals retrieved",
                                                style={
                                                    "padding": "40px",
                                                    "textAlign": "center",
                                                    "color": "#999",
                                                },
                                            ),
                                            className="manuals-card",
                                        ),
                                    ],
                                    style={
                                        "width": "48%",
                                        "display": "inline-block",
                                        "verticalAlign": "top",
                                    },
                                ),
                            ],
                            className="card-section",
                        ),

                        html.Div(
                            [
                                html.H3("📊 Fleet Overview"),

                                html.Div(
                                    [
                                        html.Div(
                                            [
                                                html.Div(
                                                    "Total Engines",
                                                    className="kpi-label",
                                                ),
                                                html.Div(
                                                    id="total-engines-kpi",
                                                    className=(
                                                        "kpi-value kpi-value-total"
                                                    ),
                                                ),
                                            ],
                                            className="kpi-card kpi-total",
                                        ),

                                        html.Div(
                                            [
                                                html.Div(
                                                    "High Risk",
                                                    className="kpi-label",
                                                ),
                                                html.Div(
                                                    id="high-risk-kpi",
                                                    className=(
                                                        "kpi-value "
                                                        "kpi-value-high-risk"
                                                    ),
                                                ),
                                            ],
                                            className="kpi-card kpi-high-risk",
                                        ),

                                        html.Div(
                                            [
                                                html.Div(
                                                    "Average Predicted RUL",
                                                    className="kpi-label",
                                                ),
                                                html.Div(
                                                    id="avg-rul-kpi",
                                                    className=(
                                                        "kpi-value kpi-value-avg"
                                                    ),
                                                ),
                                            ],
                                            className="kpi-card kpi-avg",
                                        ),
                                    ],
                                    className="kpi-container",
                                ),

                                html.Div(
                                    [
                                        html.H4(
                                            "Predicted RUL Across Fleet"
                                        ),
                                        dcc.Graph(
                                            id="rul-distribution-chart",
                                            style={"height": "400px"},
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
                    "TurbineGuard AI | PatchTST + LangGraph + MCP + FAISS RAG "
                    "| Work orders shown as proposed actions"
                ),
            ],
            className="dashboard-footer",
        ),
    ],
    className="dashboard-container",
)


@app.callback(
    [
        Output("engine-selector", "options"),
        Output("engine-selector", "value"),
    ],
    Input("dataset-selector", "value"),
)
def update_engine_selector(dataset):
    options = engine_options(dataset)
    return options, options[0]["value"]


@app.callback(
    [
        Output("current-rul-display", "children"),
        Output("current-cycle-display", "children"),
        Output("regime-display", "children"),
    ],
    [
        Input("dataset-selector", "value"),
        Input("engine-selector", "value"),
    ],
)
def preview_prediction(dataset, engine_id):
    if not dataset or engine_id is None:
        return "—", "—", "—"

    try:
        data = load_dataset(dataset)
        engine_data = data[data["unit_nr"] == engine_id]

        prediction = predict_engine_rul(dataset, engine_data)
        latest_cycle = int(engine_data["time_in_cycles"].max())

        return (
            f"{prediction['predicted_rul']:.1f}",
            str(latest_cycle),
            str(prediction["operating_regime"]),
        )
    except Exception:
        return "Error", "—", "—"


@app.callback(
    [
        Output("agent-decision-banner", "children"),
        Output("work-order-card", "children"),
        Output("manuals-card", "children"),
        Output("status-message", "children"),
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
            go.Figure(),
        )

    try:
        result = run_turbineguard(
            engine_id=engine_id,
            dataset=dataset,
            rul_threshold=rul_threshold,
        )

        predicted_rul = float(result["rul_prediction"])
        status, color, icon = risk_label(predicted_rul, rul_threshold)

        agent_banner = html.Div(
            [
                html.H3(f"{icon} {status}"),
                html.Div(
                    result["final_decision"],
                    className="agent-decision-banner",
                    style={
                        "backgroundColor": color,
                        "color": "white",
                        "padding": "18px",
                        "borderRadius": "8px",
                    },
                ),
            ]
        )

        if result.get("work_order"):
            work_order = result["work_order"]
            priority_color = {
                "HIGH": "#d62728",
                "MEDIUM": "#ff7f0e",
                "LOW": "#2ca02c",
            }[work_order.priority]

            work_order_html = html.Div(
                [
                    html.Div(
                        "SIMULATED / PROPOSED — no CMMS record was created",
                        style={
                            "fontWeight": "bold",
                            "color": "#666",
                            "marginBottom": "12px",
                        },
                    ),
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Div(
                                        "Status",
                                        className="work-order-label",
                                    ),
                                    html.Div(
                                        work_order.status,
                                        className="work-order-value",
                                    ),
                                ],
                                className="work-order-metric",
                            ),
                            html.Div(
                                [
                                    html.Div(
                                        "Priority",
                                        className="work-order-label",
                                    ),
                                    html.Div(
                                        work_order.priority,
                                        className="work-order-value",
                                        style={"color": priority_color},
                                    ),
                                ],
                                className="work-order-metric",
                            ),
                            html.Div(
                                [
                                    html.Div(
                                        "Engine",
                                        className="work-order-label",
                                    ),
                                    html.Div(
                                        str(work_order.engine_id),
                                        className="work-order-value",
                                    ),
                                ],
                                className="work-order-metric",
                            ),
                        ],
                        className="work-order-metrics",
                    ),
                    html.Div(
                        [
                            html.Div(
                                "Description:",
                                style={
                                    "fontWeight": "bold",
                                    "marginBottom": "5px",
                                },
                            ),
                            html.Div(work_order.description),
                        ],
                        style={"marginTop": "15px"},
                    ),
                ]
            )
        else:
            work_order_html = html.Div(
                "No work order proposed; predicted RUL is above the threshold.",
                style={
                    "padding": "40px",
                    "textAlign": "center",
                    "color": "#2ca02c",
                },
            )

        manuals = result.get("retrieved_manuals", [])

        if manuals:
            manuals_html = html.Ul(
                [
                    html.Li(
                        [
                            html.Span(
                                manual["title"],
                                className="manual-title",
                            ),
                            html.Span(
                                f" (relevance: {manual['score']:.3f})",
                                className="manual-score",
                            ),
                        ],
                        className="manual-item",
                    )
                    for manual in manuals
                ],
                className="manual-list",
            )
        else:
            manuals_html = html.Div(
                "No manuals needed for this predicted risk state.",
                style={
                    "padding": "40px",
                    "textAlign": "center",
                    "color": "#999",
                },
            )

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
            color_continuous_scale="RdYlGn",
            labels={
                "unit_nr": "Engine ID",
                "predicted_rul": "Predicted RUL (cycles)",
            },
        )

        figure.update_layout(
            showlegend=False,
            height=400,
            xaxis_title="Engine ID",
            yaxis_title="Predicted RUL (cycles)",
            plot_bgcolor="white",
            paper_bgcolor="white",
        )

        figure.add_hline(
            y=rul_threshold,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Threshold: {rul_threshold} cycles",
            annotation_position="right",
        )

        status_msg = html.Div(
            (
                f"✅ {dataset} model analysis completed for Engine {engine_id}. "
                f"Predicted RUL: {predicted_rul:.1f} cycles."
            ),
            className="status-message status-success",
        )

        return (
            agent_banner,
            work_order_html,
            manuals_html,
            status_msg,
            str(len(fleet_predictions)),
            str(high_risk_count),
            f"{average_rul:.1f}",
            figure,
        )

    except Exception as exc:
        error_message = f"Analysis failed: {exc}"

        return (
            html.Div(error_message, style={"color": "#d62728"}),
            html.Div("No work order proposed"),
            html.Div("No manuals retrieved"),
            html.Div(
                error_message,
                className="status-message",
                style={"color": "#d62728"},
            ),
            "—",
            "—",
            "—",
            go.Figure(),
        )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8050"))

    app.run(
        debug=False,
        host="0.0.0.0",
        port=port,
        threaded=True,
    )