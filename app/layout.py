from dash import dcc, html, dash_table
import dash_bootstrap_components as dbc
from .data import ALL_CATEGORIES, CATEGORY_COLORS


def create_layout():
    return dbc.Container(
        [
            # --- Navbar ---
            dbc.NavbarSimple(
                children=[
                    dcc.Upload(
                        id="upload-data",
                        children=dbc.Button(
                            "📁 Upload CSV",
                            size="sm",
                            className="me-2",
                            style={
                                "backgroundColor": "#17a2b8",
                                "borderColor": "#17a2b8",
                                "color": "white",
                            },
                        ),
                        multiple=False,
                    ),
                    dbc.Button(
                        "▶ Run Analysis",
                        id="run-button",
                        size="sm",
                        className="me-3",
                        style={
                            "backgroundColor": "#28a745",
                            "borderColor": "#28a745",
                            "color": "white",
                        },
                    ),
                    # Save/Open project buttons
                    dbc.Button(
                        "💾 Save",
                        id="save-project-btn",
                        size="sm",
                        className="me-1",
                        style={
                            "backgroundColor": "#ffc107",
                            "borderColor": "#ffc107",
                            "color": "#212529",
                        },
                    ),
                    dcc.Upload(
                        id="open-project-upload",
                        children=dbc.Button(
                            "📂 Open",
                            size="sm",
                            style={
                                "backgroundColor": "#fd7e14",
                                "borderColor": "#fd7e14",
                                "color": "white",
                            },
                        ),
                        multiple=False,
                        accept=".autoflow,.json",
                    ),
                    dcc.Download(id="download-project"),
                ],
                brand="🌊 Autoflow - Water Usage Analyzer",
                brand_href="#",
                color="primary",
                dark=True,
                className="mb-3 py-2 shadow-sm",
            ),
            # --- Row 1: Data Overview ---
            dbc.Row(
                [
                    # Col 1: Summary Table
                    dbc.Col(
                        [
                            html.H6(
                                "📊 End-Use Summary",
                                className="text-center fw-bold mb-1",
                                style={"color": "#2C3E50", "fontSize": "0.85rem"},
                            ),
                            dcc.Loading(
                                id="loading-summary",
                                type="default",
                                children=html.Div(
                                    id="summary-table-container",
                                    style={"height": "160px", "overflowY": "auto"},
                                    children=[
                                        dash_table.DataTable(
                                            id="summary-table",
                                            columns=[
                                                {"name": "Category", "id": "Category"},
                                                {
                                                    "name": "Vol (L)",
                                                    "id": "Volume (L)",
                                                    "type": "numeric",
                                                    "format": {"specifier": ",.1f"},
                                                },
                                                {
                                                    "name": "%",
                                                    "id": "Percentage (%)",
                                                    "type": "numeric",
                                                    "format": {"specifier": ".1f"},
                                                },
                                            ],
                                            style_cell={
                                                "textAlign": "left",
                                                "padding": "4px",
                                                "fontFamily": "var(--bs-body-font-family)",
                                                "fontSize": "12px",
                                            },
                                            style_header={
                                                "backgroundColor": "#f8f9fa",
                                                "fontWeight": "bold",
                                                "borderBottom": "2px solid #dee2e6",
                                            },
                                            style_data_conditional=[
                                                {
                                                    "if": {
                                                        "filter_query": '{Category} = "Total"'
                                                    },
                                                    "fontWeight": "bold",
                                                    "backgroundColor": "#e9ecef",
                                                }
                                            ],
                                        )
                                    ],
                                ),
                            ),
                        ],
                        width=2,
                        className="border-end rounded-start",
                        style={"backgroundColor": "#E8F4FD", "padding": "10px"},
                    ),
                    # Col 2: Event List
                    dbc.Col(
                        [
                            html.H6(
                                "📝 Event Log",
                                className="text-center fw-bold mb-1",
                                style={"color": "#2C3E50", "fontSize": "0.85rem"},
                            ),
                            dash_table.DataTable(
                                id="event-list-table",
                                columns=[
                                    {"name": "Start Date", "id": "Start date"},
                                    {"name": "Start Time", "id": "Start time"},
                                    {"name": "Category", "id": "Category"},
                                    {"name": "Duration", "id": "Duration (h:m:s)"},
                                    {"name": "Vol (L)", "id": "Volume (L)"},
                                    {"name": "Max Flow", "id": "Max flow (litre/min)"},
                                ],
                                style_table={
                                    "height": "160px",
                                    "overflowY": "auto",
                                    "overflowX": "auto",
                                },
                                style_cell={
                                    "fontSize": "0.8em",
                                    "textAlign": "left",
                                    "padding": "4px 8px",
                                    "minWidth": "70px",
                                    "maxWidth": "150px",
                                    "whiteSpace": "nowrap",
                                    "overflow": "hidden",
                                    "textOverflow": "ellipsis",
                                },
                                style_header={
                                    "fontWeight": "bold",
                                    "backgroundColor": "#f8f9fa",
                                },
                                page_action="none",
                                fixed_rows={"headers": True},
                            ),
                        ],
                        width=8,
                        className="border-end",
                        style={"backgroundColor": "#F3E8FF", "padding": "10px"},
                    ),
                    # Col 3: Legend
                    dbc.Col(
                        [
                            html.H6(
                                "Legend",
                                className="text-center fw-bold mb-1",
                                style={"color": "#2C3E50", "fontSize": "0.85rem"},
                            ),
                            html.Div(
                                [
                                    html.Div(
                                        [
                                            html.Span(
                                                "■",
                                                style={
                                                    "color": color,
                                                    "fontSize": "1.2em",
                                                    "marginRight": "5px",
                                                },
                                            ),
                                            cat,
                                        ],
                                        className="small",
                                    )
                                    for cat, color in CATEGORY_COLORS.items()
                                ],
                                className="border p-2 bg-white",
                                style={"height": "160px", "overflowY": "auto"},
                            ),
                        ],
                        width=2,
                    ),
                ],
                className="mb-1 border-bottom pb-1",
            ),
            # --- Row 2: Global Controls ---
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.InputGroup(
                                [
                                    dbc.InputGroupText(
                                        "Zoom (hrs)", style={"fontSize": "0.8em"}
                                    ),
                                    dbc.Input(
                                        id="window-size-input",
                                        value=1.0,
                                        type="number",
                                        size="sm",
                                        style={"maxWidth": "60px"},
                                    ),
                                ],
                                size="sm",
                            ),
                        ],
                        width=2,
                    ),
                    dbc.Col(
                        [
                            html.Div(
                                [
                                    html.Span(
                                        "Duration of analysis:",
                                        className="small fw-bold me-2",
                                    ),
                                    html.Span(
                                        id="duration-display",
                                        children="0 days 0 hours 0 minutes",
                                        className="small bg-light border px-2 py-1 rounded",
                                    ),
                                ],
                                className="d-flex align-items-center h-100",
                            )
                        ],
                        width=4,
                    ),
                    dbc.Col(
                        [
                            dbc.InputGroup(
                                [
                                    dbc.InputGroupText(
                                        "From", style={"fontSize": "0.8em"}
                                    ),
                                    dbc.Input(id="date-from", type="text", size="sm"),
                                    dbc.InputGroupText(
                                        "To", style={"fontSize": "0.8em"}
                                    ),
                                    dbc.Input(id="date-to", type="text", size="sm"),
                                ],
                                size="sm",
                            )
                        ],
                        width=4,
                    ),
                ],
                className="mb-1 border-bottom pb-1 align-items-center",
            ),
            # --- Row 3: Main Workspace ---
            dbc.Row(
                [
                    # Sidebar Controls
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    dbc.CardBody(
                                        [
                                            dbc.Checklist(
                                                id="category-visibility",
                                                options=[
                                                    {"label": c, "value": c}
                                                    for c in ALL_CATEGORIES
                                                ],
                                                value=ALL_CATEGORIES,
                                                switch=False,
                                                style={"fontSize": "0.85em"},
                                            ),
                                            html.Hr(className="my-2"),
                                            html.Label(
                                                "Set all", className="small text-muted"
                                            ),
                                            dbc.Select(
                                                id="batch-source-cat",
                                                options=[
                                                    {"label": c, "value": c}
                                                    for c in ALL_CATEGORIES
                                                ],
                                                size="sm",
                                                className="mb-1",
                                            ),
                                            html.Label(
                                                "To", className="small text-muted"
                                            ),
                                            dbc.Select(
                                                id="batch-target-cat",
                                                options=[
                                                    {"label": c, "value": c}
                                                    for c in ALL_CATEGORIES
                                                ],
                                                size="sm",
                                                className="mb-2",
                                            ),
                                            dbc.Button(
                                                "Set",
                                                id="batch-set-btn",
                                                size="sm",
                                                color="danger",
                                                className="w-100 mb-2",
                                            ),
                                            # Navigation controls (moved from bottom row)
                                            html.Hr(className="my-2"),
                                            html.Label(
                                                "Navigate to",
                                                className="small text-muted",
                                            ),
                                            dbc.Select(
                                                id="nav-category",
                                                options=[
                                                    {"label": c, "value": c}
                                                    for c in ALL_CATEGORIES
                                                ],
                                                value="Shower",
                                                size="sm",
                                                className="mb-2",
                                            ),
                                            dbc.ButtonGroup(
                                                [
                                                    dbc.Button(
                                                        "◀ Prev",
                                                        id="prev-event-btn",
                                                        color="secondary",
                                                        size="sm",
                                                    ),
                                                    dbc.Button(
                                                        "Next ▶",
                                                        id="next-event-btn",
                                                        color="secondary",
                                                        size="sm",
                                                    ),
                                                ],
                                                className="w-100 mb-2",
                                            ),
                                            # Display options button
                                            dbc.Button(
                                                "Display options",
                                                id="display-options-btn",
                                                color="info",
                                                outline=True,
                                                size="sm",
                                                className="w-100 border",
                                            ),
                                        ],
                                        className="p-2",
                                    )
                                ],
                                className="border-0 rounded",
                                style={
                                    "overflow": "visible",
                                    "backgroundColor": "#E8FFF0",
                                },
                            )
                        ],
                        width=2,
                    ),
                    # Chart Area
                    dbc.Col(
                        [
                            dcc.Loading(
                                id="loading-timeline",
                                type="default",
                                parent_style={
                                    "height": "100%",
                                    "display": "flex",
                                    "flexDirection": "column",
                                },
                                children=dcc.Graph(
                                    id="timeline-chart",
                                    style={"height": "100%", "flexGrow": "1"},
                                    responsive=True,
                                ),
                            ),
                            # Slider Navigation
                            dbc.Row(
                                [
                                    dbc.Col(
                                        dbc.Button(
                                            html.I(className="bi bi-chevron-left"),
                                            id="scroll-left",
                                            color="secondary",
                                            size="sm",
                                        ),
                                        width="auto",
                                    ),
                                    dbc.Col(
                                        dcc.Slider(
                                            id="time-slider",
                                            min=0,
                                            max=100,
                                            step=0.001,
                                            value=0,
                                            marks=None,
                                        ),
                                        className="flex-grow-1",
                                    ),
                                    dbc.Col(
                                        dbc.Button(
                                            html.I(className="bi bi-chevron-right"),
                                            id="scroll-right",
                                            color="secondary",
                                            size="sm",
                                        ),
                                        width="auto",
                                    ),
                                ],
                                className="align-items-center mt-1",
                            ),
                        ],
                        width=10,
                        className="d-flex flex-column",
                        style={"height": "calc(100vh - 320px)", "minHeight": "400px"},
                    ),
                ],
                className="flex-grow-1 mb-2",
            ),
            # --- Row 4: Status Bar ---
            dbc.Row(
                [
                    dbc.Col(width=True),  # Spacer
                    dbc.Col(
                        [
                            html.Span(
                                id="filename-display", className="small text-muted"
                            ),
                        ],
                        width=4,
                        className="text-end",
                    ),
                ],
                className="py-2 border-top bg-light",
            ),
            # Stores
            dcc.Store(id="event-store", data=[]),
            dcc.Store(id="classified-csv-path", data=None),
            dcc.Store(id="current-file-path", data=None),
            dcc.Store(id="current-event-index", data=None),
            dcc.Store(id="analysis-metadata", data={}),
            dcc.Store(id="cache-version", data=0),
            dcc.Store(id="chart-initialized", data=False),  # Track if chart is built
            dcc.Store(id="nav-remaining-indices", data=[]),
            dcc.Store(id="nav-same-category", data=None),
            dcc.Store(id="nav-search-direction", data=None),
            dcc.Store(id="nav-index-for-plotting", data=[]),
            dcc.Interval(id="interval", interval=1000, n_intervals=0, disabled=True),
            # Modals
            # Save Project Modal
            dbc.Modal(
                [
                    dbc.ModalHeader(dbc.ModalTitle("💾 Save Project")),
                    dbc.ModalBody(
                        [
                            html.P(
                                "Enter a name for your project file:",
                                className="mb-2",
                            ),
                            dbc.InputGroup(
                                [
                                    dbc.Input(
                                        id="save-filename-input",
                                        type="text",
                                        placeholder="my_project",
                                        value="",
                                    ),
                                    dbc.InputGroupText(".autoflow"),
                                ],
                                className="mb-3",
                            ),
                            html.Small(
                                "💡 Tip: Enable 'Ask where to save each file' in your browser settings for full folder selection.",
                                className="text-muted",
                            ),
                        ]
                    ),
                    dbc.ModalFooter(
                        [
                            dbc.Button(
                                "Cancel",
                                id="save-modal-cancel",
                                color="secondary",
                                className="me-2",
                            ),
                            dbc.Button(
                                "Save",
                                id="save-modal-confirm",
                                color="primary",
                            ),
                        ]
                    ),
                ],
                id="save-modal",
                is_open=False,
                centered=True,
            ),
            dbc.Modal(
                [
                    dbc.ModalHeader(dbc.ModalTitle("Event Details", id="modal-title")),
                    dbc.ModalBody(id="modal-body"),
                    dbc.ModalFooter(
                        [
                            dbc.Button(
                                "✓ Confirm Reclassification",
                                id="confirm-reclassify-btn",
                                color="success",
                                className="me-2",
                                style={"display": "none"},
                            ),
                            dbc.Button(
                                "Close",
                                id="close-modal",
                                className="ms-auto",
                                n_clicks=0,
                            ),
                        ]
                    ),
                ],
                id="event-modal",
                size="xl",
                is_open=False,
            ),
            # Store for current event being viewed in modal
            dcc.Store(id="modal-event-index", data=None),
            dcc.Store(id="modal-original-category", data=None),
            # Progress Modal
            dbc.Modal(
                [
                    dbc.ModalHeader(dbc.ModalTitle("Analyzing Data")),
                    dbc.ModalBody(
                        [
                            html.Div(
                                "Processing flow events...",
                                className="text-center mb-3",
                            ),
                            dbc.Progress(
                                id="progress-bar",
                                striped=True,
                                animated=True,
                                value=100,
                                color="success",
                            ),
                            html.Div(
                                id="analysis-status",
                                className="text-center mt-3 small text-muted",
                            ),
                        ]
                    ),
                ],
                id="progress-modal",
                is_open=False,
                backdrop="static",
                centered=True,
            ),
        ],
        fluid=True,
        className="vh-100 d-flex flex-column p-2",
        style={"backgroundColor": "#f8f9fa"},
    )
