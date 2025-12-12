"""
Water End-Use Classification Dashboard
Simplified version with timeline chart and summary table
"""

# %%
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, State, dash_table
import dash_bootstrap_components as dbc

# ============================================================================
# CONFIGURATION
# ============================================================================

CLASSIFIED_CSV_PATH = (
    Path(__file__).parent / "usage_by_meter-20251019212859-M6723 (1)_classified.csv"
)

# Color mapping for each category
CATEGORY_COLORS = {
    "Shower": "rgb(31, 119, 180)",  # Blue
    "Tap": "rgb(255, 127, 14)",  # Orange
    "Toilet": "rgb(44, 160, 44)",  # Green
    "Clothes Washer": "rgb(214, 39, 40)",  # Red
    "Dishwasher": "rgb(148, 103, 189)",  # Purple
    "Bathtub": "rgb(140, 86, 75)",  # Brown
    "Irrigation": "rgb(227, 119, 194)",  # Pink
    "Evap Cooler": "rgb(127, 127, 127)",  # Gray
    "Leak": "rgb(188, 189, 34)",  # Yellow-green
    "Other": "rgb(23, 190, 207)",  # Cyan
}

# ============================================================================
# DATA LOADING AND PREPARATION
# ============================================================================


def parse_raw_flow_data(raw_data_str, pulse=10):
    """Parse raw flow rate data from CSV string and convert for display."""
    if pd.isna(raw_data_str) or raw_data_str == "":
        return []
    try:
        flows = [float(x.strip()) for x in str(raw_data_str).split(",")]
        # Values in CSV are after /2 conversion, apply /6 for display
        flows = [f / 6.0 for f in flows]
        return flows
    except:
        return []


def parse_datetime(row):
    """Parse date and time from CSV row."""
    try:
        date_str = row["Start date"]
        time_str = row["Start time"]
        dt = pd.to_datetime(f"{date_str} {time_str}", format="%d-%b-%y %H:%M:%S")
        return dt
    except:
        return None


# Load data
print("Loading classified events...")
df = pd.read_csv(CLASSIFIED_CSV_PATH)

# Calculate end-use summary
summary_data = df.groupby("Category")["Volume (L)"].sum().reset_index()
summary_data.columns = ["Category", "Volume (L)"]
total_volume = summary_data["Volume (L)"].sum()
summary_data["Percentage (%)"] = (
    summary_data["Volume (L)"] / total_volume * 100
).round(1)

# Ensure all categories are included
all_categories = [
    "Shower",
    "Tap",
    "Clothes Washer",
    "Dishwasher",
    "Toilet",
    "Bathtub",
    "Irrigation",
    "Evap Cooler",
    "Leak",
    "Other",
]
for cat in all_categories:
    if cat not in summary_data["Category"].values:
        new_row = pd.DataFrame(
            [{"Category": cat, "Volume (L)": 0.0, "Percentage (%)": 0.0}]
        )
        summary_data = pd.concat([summary_data, new_row], ignore_index=True)

# Sort by volume descending
summary_data = summary_data.sort_values("Volume (L)", ascending=False).reset_index(
    drop=True
)

# Add Total row
total_row = pd.DataFrame(
    [{"Category": "Total", "Volume (L)": total_volume, "Percentage (%)": 100.0}]
)
summary_data = pd.concat([summary_data, total_row], ignore_index=True)

print("\n=== END-USE SUMMARY TABLE DATA ===")
print(summary_data.to_string())
print(f"\nTotal rows in summary table: {len(summary_data)}")
print("====================================\n")

# Parse datetimes and flow rates
df["datetime_start"] = df.apply(parse_datetime, axis=1)
df["flow_rates"] = df["Raw Series Data"].apply(parse_raw_flow_data)

# Remove events without valid datetime
df_events = df[df["datetime_start"].notna()].copy()
df_events = df_events.sort_values("datetime_start").reset_index(drop=True)

print(f"Loaded {len(df_events)} events")

# Calculate date range and max flow
min_date = df_events["datetime_start"].min()
max_date = df_events["datetime_start"].max()
total_days = (max_date - min_date).total_seconds() / 86400

# Calculate max flow rate for y-axis
all_flow_rates = []
for flow_list in df_events["flow_rates"]:
    all_flow_rates.extend(flow_list)
max_flow_rate = max(all_flow_rates) if all_flow_rates else 10
ymax = max_flow_rate + 1

print(f"Date range: {min_date} to {max_date}")
print(f"Max flow rate: {max_flow_rate:.2f} L/min, Y-axis max: {ymax:.2f} L/min")

# ============================================================================
# DASH APP
# ============================================================================

print("\nCreating Dash app with End-Use Summary Table...")
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = dbc.Container(
    [
        # Menu Bar
        dbc.Row(
            dbc.Col(
                dbc.ButtonGroup(
                    [
                        dbc.Button(
                            "Run", id="run-button", color="success", className="me-2"
                        ),
                        dbc.Button(
                            "Open", id="open-button", color="primary", className="me-2"
                        ),
                        dbc.Button(
                            "Save", id="save-button", color="info", className="me-2"
                        ),
                        dbc.Button("Save As", id="save-as-button", color="secondary"),
                    ],
                    className="mb-3",
                ),
            ),
        ),
        # Header
        dbc.Row(
            dbc.Col(
                html.Div(
                    [
                        html.H2(
                            "Water End-Use Classification Dashboard",
                            className="text-primary mb-1",
                        ),
                        html.P(
                            id="header-info",
                            children=f"Property: {df_events['Site'].iloc[0]} | Total Events: {len(df_events)} | "
                            f"Period: {min_date.strftime('%d-%b-%Y')} to {max_date.strftime('%d-%b-%Y')}",
                            className="text-muted mb-0",
                        ),
                    ],
                    className="mb-4",
                ),
            ),
        ),
        # Top Row: Summary Table (1/3) and Navigation Controls (2/3)
        dbc.Row(
            [
                # Left: End-Use Summary Table (1/3 width)
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody(
                            [
                                html.H5("End-Use Summary", className="mb-3"),
                                dash_table.DataTable(
                                    id="summary-table",
                                    columns=[
                                        {"name": "Category", "id": "Category"},
                                        {
                                            "name": "Volume (L)",
                                            "id": "Volume (L)",
                                            "type": "numeric",
                                            "format": {"specifier": ",.1f"},
                                        },
                                        {
                                            "name": "Percentage (%)",
                                            "id": "Percentage (%)",
                                            "type": "numeric",
                                            "format": {"specifier": ".1f"},
                                        },
                                    ],
                                    data=summary_data.to_dict("records"),
                                    style_cell={
                                        "textAlign": "left",
                                        "padding": "8px",
                                        "fontFamily": "Arial, sans-serif",
                                        "fontSize": "12px",
                                    },
                                    style_header={
                                        "backgroundColor": "rgb(230, 230, 230)",
                                        "fontWeight": "bold",
                                        "fontSize": "12px",
                                    },
                                    style_data_conditional=[
                                        {
                                            "if": {
                                                "filter_query": '{Category} = "Total"'
                                            },
                                            "fontWeight": "bold",
                                            "backgroundColor": "rgb(248, 248, 248)",
                                        }
                                    ],
                                    style_table={
                                        "height": "400px",
                                        "overflowY": "auto",
                                    },
                                ),
                            ]
                        ),
                        className="shadow-sm",
                    ),
                    width=4,
                ),
                # Right: Event Navigation Controls (2/3 width)
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody(
                            [
                                html.H5("Event Navigation", className="mb-3"),
                                dbc.Row(
                                    [
                                        dbc.Col(
                                            [
                                                html.Label(
                                                    "Select Category:", className="mb-2"
                                                ),
                                                dcc.Dropdown(
                                                    id="category-dropdown",
                                                    options=[
                                                        {
                                                            "label": "Shower",
                                                            "value": "Shower",
                                                        },
                                                        {
                                                            "label": "Tap",
                                                            "value": "Tap",
                                                        },
                                                        {
                                                            "label": "Toilet",
                                                            "value": "Toilet",
                                                        },
                                                        {
                                                            "label": "Clothes Washer",
                                                            "value": "Clothes Washer",
                                                        },
                                                        {
                                                            "label": "Dishwasher",
                                                            "value": "Dishwasher",
                                                        },
                                                        {
                                                            "label": "Bathtub",
                                                            "value": "Bathtub",
                                                        },
                                                        {
                                                            "label": "Irrigation",
                                                            "value": "Irrigation",
                                                        },
                                                        {
                                                            "label": "Evap Cooler",
                                                            "value": "Evap Cooler",
                                                        },
                                                        {
                                                            "label": "Leak",
                                                            "value": "Leak",
                                                        },
                                                        {
                                                            "label": "Other",
                                                            "value": "Other",
                                                        },
                                                    ],
                                                    value="Shower",
                                                    clearable=False,
                                                ),
                                            ],
                                            width=6,
                                        ),
                                        dbc.Col(
                                            [
                                                html.Label(
                                                    "Navigation:", className="mb-2"
                                                ),
                                                dbc.ButtonGroup(
                                                    [
                                                        dbc.Button(
                                                            "Previous",
                                                            id="prev-button",
                                                            color="primary",
                                                            outline=True,
                                                            className="me-2",
                                                        ),
                                                        dbc.Button(
                                                            "Next",
                                                            id="next-button",
                                                            color="primary",
                                                            outline=True,
                                                        ),
                                                    ],
                                                ),
                                                dbc.Checklist(
                                                    id="skip-zero-volume",
                                                    options=[
                                                        {
                                                            "label": " Skip zero-volume events",
                                                            "value": "skip",
                                                        }
                                                    ],
                                                    value=["skip"],
                                                    className="mt-2",
                                                    style={"fontSize": "12px"},
                                                ),
                                            ],
                                            width=6,
                                        ),
                                    ],
                                    className="mb-3",
                                ),
                                dbc.Row(
                                    [
                                        dbc.Col(
                                            [
                                                html.Div(
                                                    id="current-event-info",
                                                    className="alert alert-info",
                                                    children="Select a category and click Next/Previous to navigate events",
                                                )
                                            ],
                                            width=12,
                                        ),
                                    ]
                                ),
                                # Available End-Uses Section
                                html.Hr(className="my-3"),
                                dbc.Row(
                                    [
                                        dbc.Col(
                                            [
                                                html.H6(
                                                    "Available End-Uses in Property",
                                                    className="mb-2",
                                                ),
                                                html.P(
                                                    "Uncheck any end-uses that are NOT available in this property:",
                                                    style={
                                                        "fontSize": "12px",
                                                        "color": "#666",
                                                    },
                                                ),
                                                dbc.Checklist(
                                                    id="available-enduses",
                                                    options=[
                                                        {
                                                            "label": " Shower",
                                                            "value": "Shower",
                                                        },
                                                        {
                                                            "label": " Tap",
                                                            "value": "Tap",
                                                        },
                                                        {
                                                            "label": " Toilet",
                                                            "value": "Toilet",
                                                        },
                                                        {
                                                            "label": " Clothes Washer",
                                                            "value": "Clothes Washer",
                                                        },
                                                        {
                                                            "label": " Dishwasher",
                                                            "value": "Dishwasher",
                                                        },
                                                        {
                                                            "label": " Bathtub",
                                                            "value": "Bathtub",
                                                        },
                                                        {
                                                            "label": " Irrigation",
                                                            "value": "Irrigation",
                                                        },
                                                        {
                                                            "label": " Evap Cooler",
                                                            "value": "Evap Cooler",
                                                        },
                                                        {
                                                            "label": " Leak",
                                                            "value": "Leak",
                                                        },
                                                        {
                                                            "label": " Other",
                                                            "value": "Other",
                                                        },
                                                    ],
                                                    value=[
                                                        "Shower",
                                                        "Tap",
                                                        "Toilet",
                                                        "Clothes Washer",
                                                        "Dishwasher",
                                                        "Bathtub",
                                                        "Irrigation",
                                                        "Evap Cooler",
                                                        "Leak",
                                                        "Other",
                                                    ],
                                                    inline=True,
                                                    style={"fontSize": "12px"},
                                                ),
                                            ],
                                            width=12,
                                        ),
                                    ]
                                ),
                            ]
                        ),
                        className="shadow-sm",
                    ),
                    width=8,
                ),
            ],
            className="mb-4",
        ),
        # Timeline Chart
        dbc.Row(
            dbc.Col(
                dbc.Card(
                    dbc.CardBody(
                        [
                            html.H5("Water Usage Timeline", className="mb-3"),
                            dcc.Graph(
                                id="timeline-chart", config={"displayModeBar": True}
                            ),
                        ]
                    ),
                    className="shadow-sm",
                ),
                width=12,
            ),
            className="mb-4",
        ),
        # Controls
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.Label("Window Size (hours):"),
                        dcc.Input(
                            id="window-size-input",
                            type="number",
                            value=1.0,
                            min=0.5,
                            max=total_days * 24,
                            step=0.5,
                            className="form-control",
                        ),
                    ],
                    width=2,
                ),
                dbc.Col(
                    [
                        html.Label("Scroll Position:"),
                        dcc.Slider(
                            id="time-slider",
                            min=0,
                            max=total_days,
                            step=0.01,
                            value=0,
                            marks=None,
                            tooltip={"placement": "bottom", "always_visible": True},
                        ),
                    ],
                    width=10,
                ),
            ],
            className="mb-4",
        ),
        # Event Details Modal
        dbc.Modal(
            [
                dbc.ModalHeader(dbc.ModalTitle(id="modal-title")),
                dbc.ModalBody(id="modal-body"),
                dbc.ModalFooter(
                    dbc.Button("Close", id="close-modal", className="ms-auto")
                ),
            ],
            id="event-modal",
            size="lg",
            is_open=False,
        ),
        # File Selection Modal
        dbc.Modal(
            [
                dbc.ModalHeader(dbc.ModalTitle("Select CSV File to Analyze")),
                dbc.ModalBody(
                    [
                        dbc.Row(
                            [
                                dbc.Col(
                                    dcc.Input(
                                        id="file-path-input",
                                        type="text",
                                        placeholder="Select a CSV file using the Browse button",
                                        style={"width": "100%"},
                                        readOnly=True,
                                    ),
                                    width=9,
                                ),
                                dbc.Col(
                                    dbc.Button(
                                        "Browse...",
                                        id="browse-button",
                                        color="secondary",
                                        className="w-100",
                                    ),
                                    width=3,
                                ),
                            ],
                            className="mb-3",
                        ),
                        html.Div(id="file-selection-message", className="text-muted"),
                    ]
                ),
                dbc.ModalFooter(
                    [
                        dbc.Button("Cancel", id="cancel-file-button", className="me-2"),
                        dbc.Button("Analyze", id="analyze-button", color="success"),
                    ]
                ),
            ],
            id="file-modal",
            is_open=False,
        ),
        # Analysis Progress Modal
        dbc.Modal(
            [
                dbc.ModalHeader(dbc.ModalTitle("Analyzing...")),
                dbc.ModalBody(
                    [
                        html.Div(
                            "Analysis in progress. This may take a few minutes..."
                        ),
                        dbc.Progress(
                            id="progress-bar",
                            striped=True,
                            animated=True,
                            value=100,
                            className="mt-3",
                        ),
                        html.Div(id="analysis-status", className="mt-3"),
                    ]
                ),
            ],
            id="progress-modal",
            is_open=False,
            backdrop="static",
        ),
        # Store for event data and current navigation index
        dcc.Store(id="event-store", data=df_events.to_dict("records")),
        dcc.Store(id="current-event-index", data=0),
        dcc.Store(id="current-file-path", data=str(CLASSIFIED_CSV_PATH)),
        dcc.Store(id="classified-csv-path", data=str(CLASSIFIED_CSV_PATH)),
        # Interval for checking analysis progress
        dcc.Interval(id="interval", interval=1000, n_intervals=0, disabled=True),
    ],
    fluid=True,
    className="p-4",
)

# ============================================================================
# CALLBACKS
# ============================================================================


@app.callback(
    Output("timeline-chart", "figure"),
    [Input("time-slider", "value"), Input("window-size-input", "value")],
)
def update_timeline(slider_value, window_hours):
    """Update timeline chart based on slider position and window size."""

    window_days = window_hours / 24.0
    start_day = slider_value
    end_day = start_day + window_days

    start_time = min_date + timedelta(days=start_day)
    end_time = min_date + timedelta(days=end_day)

    # Filter events in window
    mask = (df_events["datetime_start"] >= start_time) & (
        df_events["datetime_start"] <= end_time
    )
    visible_events = df_events[mask]

    fig = go.Figure()

    # Add each event as a filled area
    for idx, event in visible_events.iterrows():
        flow_rates = event["flow_rates"]
        category = event["Category"]
        color = CATEGORY_COLORS.get(category, "rgb(128, 128, 128)")

        if len(flow_rates) == 0:
            continue

        # Create time points (5-second intervals)
        n_points = len(flow_rates)
        event_start = event["datetime_start"]
        time_points = [event_start + timedelta(seconds=5 * i) for i in range(n_points)]

        fig.add_trace(
            go.Scatter(
                x=time_points,
                y=flow_rates,
                fill="tozeroy",
                mode="lines",
                name=f"{category} - {event_start.strftime('%H:%M:%S')}",
                line=dict(color=color, width=1),
                fillcolor=color.replace("rgb", "rgba").replace(")", ", 0.6)"),
                hovertemplate=f"<b>{category}</b><br>Time: %{{x}}<br>Flow: %{{y:.1f}} L/min<extra></extra>",
                customdata=[[idx]] * len(time_points),
            )
        )

    fig.update_layout(
        xaxis_title="Time",
        yaxis_title="Flow Rate (L/min)",
        yaxis_range=[0, ymax],
        height=400,
        hovermode="closest",
        showlegend=False,
        margin=dict(l=50, r=20, t=20, b=50),
    )

    return fig


@app.callback(
    [
        Output("event-modal", "is_open"),
        Output("modal-title", "children"),
        Output("modal-body", "children"),
    ],
    [Input("timeline-chart", "clickData"), Input("close-modal", "n_clicks")],
    [State("event-modal", "is_open")],
)
def handle_event_click(clickData, close_clicks, is_open):
    """Handle click on timeline to show event details."""
    from dash import callback_context

    if not callback_context.triggered:
        return is_open, "", ""

    trigger_id = callback_context.triggered[0]["prop_id"].split(".")[0]

    if trigger_id == "close-modal":
        return False, "", ""

    if trigger_id == "timeline-chart" and clickData:
        point = clickData["points"][0]
        event_idx = point["customdata"][0]
        event = df_events.iloc[event_idx]

        title = f"{event['Category']}"

        body = dbc.Container(
            [
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.Strong("Start:"),
                                html.Span(
                                    f" {event['Start date']} {event['Start time']}",
                                    className="ms-2",
                                ),
                            ],
                            width=6,
                        ),
                        dbc.Col(
                            [
                                html.Strong("End:"),
                                html.Span(
                                    f" {event['End date']} {event['End time']}",
                                    className="ms-2",
                                ),
                            ],
                            width=6,
                        ),
                    ]
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.Strong("Duration:"),
                                html.Span(
                                    f" {event['Duration (h:m:s)']}", className="ms-2"
                                ),
                            ],
                            width=6,
                        ),
                        dbc.Col(
                            [
                                html.Strong("Volume:"),
                                html.Span(
                                    f" {event['Volume (L)']:.2f} L", className="ms-2"
                                ),
                            ],
                            width=6,
                        ),
                    ],
                    className="mt-2",
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.Strong("Max Flow:"),
                                html.Span(
                                    f" {event['Max flow (litre/min)']/6.0:.2f} L/min",
                                    className="ms-2",
                                ),
                            ],
                            width=6,
                        ),
                        dbc.Col(
                            [
                                html.Strong("Mode Flow:"),
                                html.Span(
                                    f" {event['Mode (L/min)']/6.0:.1f} L/min",
                                    className="ms-2",
                                ),
                            ],
                            width=6,
                        ),
                    ],
                    className="mt-2",
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.Strong("Raw Flow Data:"),
                                html.P(
                                    event["Raw Series Data"],
                                    className="mt-2",
                                    style={
                                        "fontSize": "0.9em",
                                        "maxHeight": "200px",
                                        "overflowY": "auto",
                                    },
                                ),
                            ],
                            width=12,
                        ),
                    ],
                    className="mt-3",
                ),
            ]
        )

        return True, title, body

    return is_open, "", ""


@app.callback(
    [
        Output("current-event-index", "data"),
        Output("current-event-info", "children"),
        Output("time-slider", "value"),
    ],
    [Input("prev-button", "n_clicks"), Input("next-button", "n_clicks")],
    [
        State("category-dropdown", "value"),
        State("current-event-index", "data"),
        State("skip-zero-volume", "value"),
    ],
)
def navigate_events(
    prev_clicks, next_clicks, selected_category, current_index, skip_zero
):
    """Navigate to previous/next event of selected category."""
    from dash import callback_context

    if not callback_context.triggered:
        return 0, "Select a category and click Next/Previous to navigate events", 0

    trigger_id = callback_context.triggered[0]["prop_id"].split(".")[0]

    # Filter events by selected category
    category_events = df_events[df_events["Category"] == selected_category].copy()

    # Optionally filter out zero-volume events
    if skip_zero and "skip" in skip_zero:
        category_events = category_events[category_events["Volume (L)"] > 0].copy()

    if len(category_events) == 0:
        return (
            current_index,
            f"No events found for category: {selected_category}",
            0,
        )

    # Reset index to get sequential positions
    category_events = category_events.reset_index(drop=False)
    category_events = category_events.rename(columns={"index": "original_index"})

    # Find current position in category list
    if current_index is not None:
        matching = category_events[category_events["original_index"] == current_index]
        if len(matching) > 0:
            current_pos = matching.index[0]
        else:
            current_pos = 0
    else:
        current_pos = 0

    # Navigate
    if trigger_id == "next-button":
        new_pos = (current_pos + 1) % len(category_events)
    elif trigger_id == "prev-button":
        new_pos = (current_pos - 1) % len(category_events)
    else:
        new_pos = current_pos

    event_row = category_events.iloc[new_pos]
    new_index = int(event_row["original_index"])

    # Get the actual event data from the original dataframe
    event = df_events.loc[new_index]

    # Calculate slider position to center on this event
    event_time = event["datetime_start"]
    days_from_start = (event_time - min_date).total_seconds() / 86400

    # Info message with proper data
    info_message = html.Div(
        [
            html.Strong(
                f"{selected_category} Event {new_pos + 1} of {len(category_events)}"
            ),
            html.Br(),
            html.Span(f"Start: {event['Start date']} {event['Start time']}"),
            html.Br(),
            html.Span(
                f"Volume: {event['Volume (L)']:.2f} L | Duration: {event['Duration (h:m:s)']}"
            ),
            html.Br(),
            html.Span(
                f"Max Flow: {event['Max flow (litre/min)']/6.0:.2f} L/min | Mode: {event['Mode (L/min)']/6.0:.1f} L/min"
            ),
        ]
    )

    return new_index, info_message, days_from_start


@app.callback(
    Output("file-modal", "is_open"),
    [
        Input("run-button", "n_clicks"),
        Input("cancel-file-button", "n_clicks"),
        Input("analyze-button", "n_clicks"),
    ],
    [State("file-modal", "is_open")],
)
def toggle_file_modal(run_clicks, cancel_clicks, analyze_clicks, is_open):
    """Toggle the file selection modal."""
    from dash import callback_context

    if not callback_context.triggered:
        return False

    trigger_id = callback_context.triggered[0]["prop_id"].split(".")[0]

    if trigger_id == "run-button":
        return True
    elif trigger_id in ["cancel-file-button", "analyze-button"]:
        return False

    return is_open


@app.callback(
    Output("file-path-input", "value"),
    [Input("browse-button", "n_clicks")],
    prevent_initial_call=True,
)
def browse_file(n_clicks):
    """Open file browser dialog."""
    if not n_clicks:
        return ""

    try:
        import tkinter as tk
        from tkinter import filedialog

        # Create a hidden root window
        root = tk.Tk()
        root.withdraw()
        root.wm_attributes("-topmost", 1)

        # Open file dialog
        file_path = filedialog.askopenfilename(
            title="Select CSV File",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            initialdir=str(Path(__file__).parent),
        )

        root.destroy()

        if file_path:
            return file_path
        return ""
    except Exception as e:
        print(f"Error opening file dialog: {e}")
        return ""


@app.callback(
    [
        Output("progress-modal", "is_open"),
        Output("analysis-status", "children"),
        Output("interval", "disabled"),
    ],
    [Input("analyze-button", "n_clicks")],
    [State("file-path-input", "value")],
)
def start_analysis(analyze_clicks, file_path):
    """Start the analysis process."""
    from dash import callback_context
    import threading

    if not callback_context.triggered or not file_path:
        return False, "", True

    # Check if file exists
    from pathlib import Path

    csv_path = Path(file_path)

    if not csv_path.exists():
        return False, f"Error: File not found: {file_path}", True

    if not csv_path.suffix.lower() == ".csv":
        return False, f"Error: File must be a CSV file", True

    # Start analysis in background thread
    def run_analysis():
        import analyze_real_data

        try:
            analyze_real_data.analyze_csv_file(str(csv_path))
        except Exception as e:
            print(f"Analysis error: {e}")

    analysis_thread = threading.Thread(target=run_analysis)
    analysis_thread.start()

    return True, "Starting analysis...", False


@app.callback(
    [
        Output("event-store", "data"),
        Output("header-info", "children"),
        Output("classified-csv-path", "data"),
        Output("progress-modal", "is_open", allow_duplicate=True),
    ],
    [Input("interval", "n_intervals")],
    [
        State("file-path-input", "value"),
        State("classified-csv-path", "data"),
        State("progress-modal", "is_open"),
    ],
    prevent_initial_call=True,
)
def check_analysis_progress(
    n_intervals, original_file_path, current_classified_path, progress_modal_open
):
    """Check if analysis is complete and reload data."""
    from pathlib import Path
    from dash import no_update

    if not original_file_path or not progress_modal_open:
        # Don't update anything if we're not waiting for analysis
        return no_update, no_update, no_update, False

    # Determine the expected classified CSV path
    original_path = Path(original_file_path)
    base_name = original_path.stem
    classified_csv = original_path.parent / f"{base_name}_classified.csv"

    # Check if classified file exists and has been updated recently
    if classified_csv.exists():
        import time

        file_age = time.time() - classified_csv.stat().st_mtime

        # If file was modified in the last 5 seconds, reload it
        if file_age < 5:
            try:
                # Reload the data
                df_new = pd.read_csv(classified_csv)
                df_new["datetime_start"] = df_new.apply(parse_datetime, axis=1)
                df_new["flow_rates"] = df_new["Raw Series Data"].apply(
                    parse_raw_flow_data
                )
                df_new = df_new[df_new["datetime_start"].notna()].copy()
                df_new = df_new.sort_values("datetime_start").reset_index(drop=True)

                min_date_new = df_new["datetime_start"].min()
                max_date_new = df_new["datetime_start"].max()

                header_text = (
                    f"Property: {df_new['Site'].iloc[0]} | Total Events: {len(df_new)} | "
                    f"Period: {min_date_new.strftime('%d-%b-%Y')} to {max_date_new.strftime('%d-%b-%Y')}"
                )

                print("Analysis complete! Data reloaded successfully.")
                return (
                    df_new.to_dict("records"),
                    header_text,
                    str(classified_csv),
                    False,
                )
            except Exception as e:
                print(f"Error reloading data: {e}")

    # Analysis not complete yet, don't update the store
    return no_update, no_update, no_update, progress_modal_open


@app.callback(
    Output("summary-table", "data"),
    [Input("event-store", "data"), Input("available-enduses", "value")],
)
def update_summary_table(event_data, available_enduses):
    """Update the summary table when new data is loaded or available end-uses change."""
    if not event_data or len(event_data) == 0:
        print("WARNING: update_summary_table received empty event_data")
        # Return initial summary with zeros instead of empty
        all_categories = [
            "Shower",
            "Tap",
            "Clothes Washer",
            "Dishwasher",
            "Toilet",
            "Bathtub",
            "Irrigation",
            "Evap Cooler",
            "Leak",
            "Other",
        ]
        empty_summary = pd.DataFrame(
            {
                "Category": all_categories + ["Total"],
                "Volume (L)": [0.0] * 11,
                "Percentage (%)": [0.0] * 10 + [100.0],
            }
        )
        return empty_summary.to_dict("records")

    # Convert event data to DataFrame
    df_new = pd.DataFrame(event_data)

    # Reclassify unavailable end-uses to "Other" to maintain total volume
    if available_enduses is None:
        available_enduses = []

    # Create a copy and reclassify unavailable categories to "Other"
    df_filtered = df_new.copy()
    reclassified_count = 0
    for idx, row in df_filtered.iterrows():
        if row["Category"] not in available_enduses and row["Category"] != "Other":
            df_filtered.at[idx, "Category"] = "Other"
            reclassified_count += 1

    print(f"update_summary_table: Processing {len(df_new)} events")
    print(f"Available end-uses: {available_enduses}")
    print(f"Reclassified {reclassified_count} events to 'Other' category")

    # Calculate end-use summary using reclassified data
    summary_data = df_filtered.groupby("Category")["Volume (L)"].sum().reset_index()
    summary_data.columns = ["Category", "Volume (L)"]

    # Calculate total from original data to ensure it stays constant
    total_volume = df_new["Volume (L)"].sum()

    if total_volume > 0:
        summary_data["Percentage (%)"] = (
            summary_data["Volume (L)"] / total_volume * 100
        ).round(1)
    else:
        summary_data["Percentage (%)"] = 0.0

    # Ensure all categories are included
    all_categories = [
        "Shower",
        "Tap",
        "Clothes Washer",
        "Dishwasher",
        "Toilet",
        "Bathtub",
        "Irrigation",
        "Evap Cooler",
        "Leak",
        "Other",
    ]
    for cat in all_categories:
        if cat not in summary_data["Category"].values:
            new_row = pd.DataFrame(
                [{"Category": cat, "Volume (L)": 0.0, "Percentage (%)": 0.0}]
            )
            summary_data = pd.concat([summary_data, new_row], ignore_index=True)

    # Sort by volume descending
    summary_data = summary_data.sort_values("Volume (L)", ascending=False).reset_index(
        drop=True
    )

    # Add Total row
    total_row = pd.DataFrame(
        [{"Category": "Total", "Volume (L)": total_volume, "Percentage (%)": 100.0}]
    )
    summary_data = pd.concat([summary_data, total_row], ignore_index=True)

    print("\n=== SUMMARY TABLE UPDATED ===")
    print(summary_data.to_string())
    print(f"Returning {len(summary_data)} rows to summary table")

    return summary_data.to_dict("records")


@app.callback(
    Output("current-file-path", "data"),
    [Input("save-button", "n_clicks"), Input("save-as-button", "n_clicks")],
    [State("classified-csv-path", "data"), State("event-store", "data")],
)
def save_data(save_clicks, save_as_clicks, csv_path, event_data):
    """Save the classified data back to CSV."""
    from dash import callback_context

    if not callback_context.triggered or not event_data:
        return csv_path

    trigger_id = callback_context.triggered[0]["prop_id"].split(".")[0]

    if trigger_id == "save-button":
        # Save to current file
        try:
            df_save = pd.DataFrame(event_data)
            # Remove the added columns
            cols_to_drop = ["datetime_start", "flow_rates"]
            df_save = df_save.drop(
                columns=[c for c in cols_to_drop if c in df_save.columns]
            )
            df_save.to_csv(csv_path, index=False)
            print(f"Saved to: {csv_path}")
        except Exception as e:
            print(f"Error saving: {e}")

    elif trigger_id == "save-as-button":
        # For Save As, we'd need a file dialog - for now just print message
        print("Save As functionalit