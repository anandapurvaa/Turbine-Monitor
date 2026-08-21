"""
TurbineGuard AI Dashboard
Professional Dash-based interface for multi-agent predictive maintenance.
"""
import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from src.agents.graph import run_turbineguard

# Initialize Dash app
app = dash.Dash(__name__, suppress_callback_exceptions=True)
app.title = "TurbineGuard AI | Predictive Maintenance"

# Load data
test_data = pd.read_csv("data/processed/test_FD001_processed.csv")
engines = sorted(test_data["unit_nr"].unique().tolist())

# App layout
app.layout = html.Div([
    # Header
    html.Div([
        html.H1("⚙️ TurbineGuard AI"),
        html.P("Agentic Root-Cause Diagnosis & Auto-Dispatch for Fleet Predictive Maintenance"),
    ], className="dashboard-header"),
    
    # Main layout
    html.Div([
        # Sidebar
        html.Div([
            html.H3("Engine Selection"),
            
            # Engine dropdown
            html.Label("Select Engine ID:"),
            dcc.Dropdown(
                id='engine-selector',
                options=[{"label": f"Engine {eng}", "value": eng} for eng in engines],
                value=engines[0],
                clearable=False,
            ),
            
            # Current RUL display
            html.Div([
                html.Div([
                    html.Div("Current RUL", className="metric-label"),
                    html.Div(id='current-rul-display', children="0.0", className="metric-value metric-value-rul"),
                    html.Div("cycles", className="metric-unit"),
                ], className="metric-card metric-card-rul"),
                
                html.Div([
                    html.Div("Current Cycle", className="metric-label"),
                    html.Div(id='current-cycle-display', children="0", className="metric-value metric-value-cycle"),
                ], className="metric-card metric-card-cycle"),
            ]),
            
            # Threshold slider
            html.Label("RUL Threshold (cycles):"),
            dcc.Slider(
                id='rul-threshold',
                min=10,
                max=50,
                step=5,
                value=30,
                marks={i: str(i) for i in range(10, 51, 10)},
                tooltip={"placement": "bottom", "always_visible": True},
            ),
            
            # Run button
            html.Button(
                "🔍 Run TurbineGuard Analysis",
                id='run-analysis-btn',
                n_clicks=0,
                className="run-button",
            ),
            
            # Status message
            html.Div(id='status-message', className="status-message"),
            
        ], className="sidebar"),
        
        # Main content
        html.Div([
            # Agent Decision Banner
            html.Div([
                html.H3("🤖 Agent Decision"),
                html.Div(id='agent-decision-banner', children=[
                    html.Div("Select an engine and click 'Run TurbineGuard Analysis' to see the agent decision.", 
                             style={"padding": "30px", "text-align": "center", "color": "#999"}),
                ]),
            ], className="agent-decision-section"),
            
            # Work Order & Manuals (2 columns)
            html.Div([
                # Work Order Card
                html.Div([
                    html.H3("📋 Work Order"),
                    html.Div(id='work-order-card', children=[
                        html.Div("No work order created", style={"padding": "40px", "text-align": "center", "color": "#999"}),
                    ], className="work-order-card"),
                ], style={"width": "48%", "display": "inline-block", "vertical-align": "top", "margin-right": "4%"}),
                
                # Retrieved Manuals Card
                html.Div([
                    html.H3("📚 Retrieved Maintenance Manuals"),
                    html.Div(id='manuals-card', children=[
                        html.Div("No manuals retrieved", style={"padding": "40px", "text-align": "center", "color": "#999"}),
                    ], className="manuals-card"),
                ], style={"width": "48%", "display": "inline-block", "vertical-align": "top"}),
            ], className="card-section"),
            
            # Fleet Overview
            html.Div([
                html.H3("📊 Fleet Overview"),
                
                # KPI cards
                html.Div([
                    html.Div([
                        html.Div("Total Engines", className="kpi-label"),
                        html.Div(id='total-engines-kpi', children=str(len(engines)), className="kpi-value kpi-value-total"),
                    ], className="kpi-card kpi-total"),
                    
                    html.Div([
                        html.Div("High Risk (RUL < threshold)", className="kpi-label"),
                        html.Div(id='high-risk-kpi', children="0", className="kpi-value kpi-value-high-risk"),
                    ], className="kpi-card kpi-high-risk"),
                    
                    html.Div([
                        html.Div("Avg RUL", className="kpi-label"),
                        html.Div(id='avg-rul-kpi', children="0.0", className="kpi-value kpi-value-avg"),
                    ], className="kpi-card kpi-avg"),
                ], className="kpi-container"),
                
                # RUL Distribution Chart
                html.Div([
                    html.H4("RUL Distribution Across Fleet"),
                    dcc.Graph(id='rul-distribution-chart', style={"height": "400px"}),
                ], className="chart-container"),
            ], className="fleet-overview"),
            
        ], className="main-content"),
        
    ], className="main-layout"),
    
    # Footer
    html.Div([
        html.P("TurbineGuard AI | Multi-agent predictive maintenance system | PatchTST + LangGraph + MCP + FAISS RAG"),
    ], className="dashboard-footer"),
    
], className="dashboard-container")


# Callbacks
@app.callback(
    [Output('current-rul-display', 'children'),
     Output('current-cycle-display', 'children')],
    Input('engine-selector', 'value')
)
def update_engine_metrics(engine_id):
    """Update current RUL and cycle displays."""
    engine_data = test_data[test_data["unit_nr"] == engine_id].iloc[-1]
    return f"{engine_data['RUL_capped']:.1f}", str(engine_data['time_in_cycles'])


@app.callback(
    [Output('agent-decision-banner', 'children'),
     Output('work-order-card', 'children'),
     Output('manuals-card', 'children'),
     Output('status-message', 'children'),
     Output('total-engines-kpi', 'children'),
     Output('high-risk-kpi', 'children'),
     Output('avg-rul-kpi', 'children'),
     Output('rul-distribution-chart', 'figure')],
    Input('run-analysis-btn', 'n_clicks'),
    [State('engine-selector', 'value'),
     State('rul-threshold', 'value')]
)
def run_analysis(n_clicks, engine_id, rul_threshold):
    """Run TurbineGuard analysis and update all displays."""
    if n_clicks == 0:
        return (
            dash.no_update, dash.no_update, dash.no_update, dash.no_update,
            str(len(engines)), "0", "0.0",
            go.Figure()
        )
    
    # Run agent analysis
    result = run_turbineguard(engine_id=engine_id, rul_threshold=rul_threshold)
    
    # 1. Agent decision banner
    decision_text = result["final_decision"]
    is_anomaly = result["anomaly_detected"]
    
    banner_class = "agent-decision-banner agent-decision-anomaly" if is_anomaly else "agent-decision-banner agent-decision-healthy"
    icon = "⚠️" if is_anomaly else "✅"
    
    agent_banner = html.Div([
        html.H3(f"{icon} Agent Decision"),
        html.Div(decision_text, className=banner_class),
    ])
    
    # 2. Work order card
    if result.get("work_order"):
        wo = result["work_order"]
        priority_color = {"HIGH": "#d62728", "MEDIUM": "#ff7f0e", "LOW": "#2ca02c"}[wo.priority]
        
        work_order_html = html.Div([
            html.Div([
                html.Div([
                    html.Div("Work Order ID", className="work-order-label"),
                    html.Div(f"#{wo.work_order_id}", className="work-order-value"),
                ], className="work-order-metric"),
                
                html.Div([
                    html.Div("Priority", className="work-order-label"),
                    html.Div(wo.priority, className="work-order-value", style={"color": priority_color}),
                ], className="work-order-metric"),
                
                html.Div([
                    html.Div("Engine", className="work-order-label"),
                    html.Div(str(wo.engine_id), className="work-order-value", style={"color": "#1f77b4"}),
                ], className="work-order-metric"),
            ], className="work-order-metrics"),
            
            html.Div([
                html.Div("Description:", style={"font-weight": "bold", "margin-bottom": "5px"}),
                html.Div(wo.description, style={"color": "#333"}),
            ], style={"margin-bottom": "10px"}),
            
            html.Div([
                html.Div("Failure Mode Hypothesis:", style={"font-weight": "bold", "margin-bottom": "5px"}),
                html.Div(wo.failure_mode_id, style={"color": "#333", "font-style": "italic"}),
            ]),
        ])
    else:
        work_order_html = html.Div("No work order created", style={"padding": "40px", "text-align": "center", "color": "#999"})
    
    # 3. Manuals card
    if result.get("retrieved_manuals"):
        manuals_html = html.Div([
            html.Ul([
                html.Li([
                    html.Span(f"{manual['title']}", className="manual-title"),
                    html.Span(f" (score: {manual['score']:.3f})", className="manual-score"),
                ], className="manual-item")
                for manual in result["retrieved_manuals"]
            ], className="manual-list"),
        ])
    else:
        manuals_html = html.Div("No manuals retrieved", style={"padding": "40px", "text-align": "center", "color": "#999"})
    
    # 4. Status message
    status_msg = html.Div("✅ Analysis completed successfully", className="status-message status-success")
    
    # 5-7. Fleet KPIs
    last_cycles = test_data.groupby("unit_nr").last().reset_index()
    high_risk_count = len(last_cycles[last_cycles["RUL_capped"] < rul_threshold])
    avg_rul = last_cycles["RUL_capped"].mean()
    
    # 8. RUL distribution chart
    fig = px.bar(
        last_cycles,
        x="unit_nr",
        y="RUL_capped",
        title="RUL Distribution Across Fleet",
        labels={"unit_nr": "Engine ID", "RUL_capped": "Remaining Useful Life (cycles)"},
        color="RUL_capped",
        color_continuous_scale="RdYlGn_r",
    )
    fig.update_layout(
        showlegend=False,
        height=400,
        xaxis_title="Engine ID",
        yaxis_title="RUL (cycles)",
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    fig.add_hline(
        y=rul_threshold,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Threshold: {rul_threshold} cycles",
        annotation_position="right",
    )
    
    return (
        agent_banner,
        work_order_html,
        manuals_html,
        status_msg,
        str(len(engines)),
        str(high_risk_count),
        f"{avg_rul:.1f}",
        fig,
    )


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8050, threaded = True)