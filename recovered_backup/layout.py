from dash import dcc, html, dash_table, dash_table
import dash_bootstrap_components as dbc
from .data import ALL_CATEGORIES, CATEGORY_COLORS


def create_layout():
    return dbc.Container(
        [
            # --- Navbar (Keep for file upload, but simplify) ---
            dbc.NavbarSimple(
                children=[
                    dcc.Upload(
                        id="upload-data",
                        children=dbc.Button(
                            "Upload CSV", color="light", size="sm", className="me-2"
                        ),
                        multiple=False,
                    ),
                    dbc.Button(
                        "Run Analysis", id="run-button", color="primary", size="sm"
                    ),
                ],
                brand="Autoflow - Version 5.5",
                brand_href="#",
                color="light",
                dark=False,
                className="mb-2 py-1 border-bottom",
            ),
            # --- Row 1: Data Overview ---
            dbc.Row(
                [
                    # Col 1: Summary Table
                    dbc.Col(
                        [
                            html.H6(
                                "End-Use Summary",
                                className="text-center small fw-bold mb-1",
                            ),
                            dcc.Loading(
                                id="loading-summary",
                                type="default",
                                children=html.Div(
                                    id="summary-table-container",
                                    style={"height": "200px", "overflowY": "auto"},
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
                        width=3,
                        className="border-end",
                    ),
                    # Col 2: Event List (NEW)
                    dbc.Col(
                        [
                            html.H6(
                                "Event Log", className="text-center small fw-bold mb-1"
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
                                style_table={"height": "200px", "overflowY": "auto"},
                                style_cell={
                                    "fontSize": "0.8em",
                                    "textAlign": "left",
                                    "padding": "2px",
                                },
                                style_header={
                                    "fontWeight": "bold",
                                    "backgroundColor": "#f8f9fa",
                                },
                                page_action="none",
                                fixed_rows={"headers": True},
                            ),
                        ],
                        width=7,
                        className="border-end",
                    ),
                    # Col 3: Legend & Quick View
                    dbc.Col(
                        [
                            dbc.Row(
                                [
                                    dbc.Col(
                                        dbc.Button(
                                            "Quick view",
                                            size="sm",
                                            color="light",
                                            className="border w-100 mb-1",
                                        ),
                                        width=8,
                                    ),
                                    dbc.Col(
                                        dbc.Button(
                                            "Off",
                                            size="sm",
                                            color="light",
                                            className="border w-100 mb-1",
                                        ),
                                        width=4,
                                    ),
                                ]
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
                                className="border p-1 bg-white",
                                style={"height": "170px", "overflowY": "auto"},
                            ),
                        ],
                        width=2,
                    ),
                ],
                className="mb-2 border-bottom pb-2",
            ),
            # --- Row 2: Global Controls ---
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Button(
                                "Normal view",
                                color="success",
                                size="sm",
                                className="w-100 mb-1 fw-bold",
                            ),
                            dbc.Button(
                                "Trial mode",
                                color="success",
                                size="sm",
                                className="w-100 fw-bold",
                            ),
                        ],
                        width=2,
                    ),
                    dbc.Col(
                        [
                            dbc.InputGroup(
                                [
                                    dbc.InputGroupText(
                                        "Resolution (pulse/litre)",
                                        style={"fontSize": "0.8em"},
                                    ),
                                    dbc.Input(
                                        value="72",
                                        type="number",
                                        size="sm",
                                        style={"maxWidth": "60px"},
                                    ),
                                ],
                                size="sm",
                                className="mb-1",
                            ),
                            # Hidden or visible window size input for zoom control
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
                                className="mb-1",
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
                className="mb-2 border-bottom pb-2 align-items-center",
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
                                                color="secondary",
                                                className="w-100",
                                            ),
                                        ],
                                        className="p-2",
                                    )
                                ],
                                className="h-100 border-0 bg-light",
                            )
                        ],
                        width=2,
                        className="h-100",
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
                                className="align-items-center mt-1 flex-shrink-0",
                            ),
                        ],
                        width=10,
                        className="d-flex flex-column h-100",
                    ),
                ],
                className="mb-2 flex-grow-1 overflow-hidden",
            ),
            # --- Row 4: Footer ---
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.InputGroup(
                                [
                                    dbc.Select(
                                        id="nav-category-filter",
                                        options=[{"label": "All", "value": "All"}]
                                        + [
                                            {"label": c, "value": c}
                                            for c in ALL_CATEGORIES
                                        ],
                                        value="All",
                                        size="sm",
                                        style={"maxWidth": "120px"},
                                    ),
                                    dbc.Button(
                                        "Previous",
                                        id="prev-event-btn",
                                        size="sm",
                                        color="light",
                                        className="border",
                                    ),
                                    dbc.Button(
                                        "Next",
                                        id="next-event-btn",
                                        size="sm",
                                        color="light",
                                        className="border",
                                    ),
                                ],
                                size="sm",
                            )
                        ],
                        width=4,
                    ),
                    dbc.Col(
                        [
                            dbc.Button(
                                "Display options",
                                color="danger",
                                outline=True,
                                size="sm",
                                className="me-2",
                            ),
                            dbc.Checklist(
                                options=[
                                    {"label": "Residential", "value": "res"},
                                    {"label": "Non-Residential", "value": "non-res"},
                                ],
                                value=["res"],
                                inline=True,
                                style={"fontSize": "0.85em", "display": "inline-block"},
                            ),
                        ],
                        width=4,
                        className="text-center",
                    ),
                    dbc.Col(
                        [
                            html.Span(
                                id="filename-display", className="small text-muted me-2"
                            ),
                            dbc.InputGroup(
                                [
                                    dbc.Input(
                                        placeholder="$/kL",
                                        size="sm",
                                        style={"maxWidth": "60px"},
                                    ),
                                    dbc.Button(
                                        "Cost",
                                        size="sm",
                                        color="light",
                                        className="border",
                                    ),
                                ],
                                size="sm",
                                className="d-inline-flex w-auto",
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
            dcc.Store(id="current-event-index", data=0),
            dcc.Store(id="selected-event-index", data=None),
            dcc.Store(id="current-file-path", data=""),
            dcc.Store(id="classified-csv-path", data=""),
            dcc.Store(id="cache-version", data=0),
            dcc.Store(id="analysis-metadata", data={}),
            dcc.Store(id="reclassify-action", data={}),
            # Navigation stores (like MATLAB's remaining_table_for_searching, etc.)
            dcc.Store(
                id="nav-remaining-indices", data=[]
            ),  # remaining_table_for_searching
            dcc.Store(id="nav-same-category", data=None),  # same_category
            dcc.Store(id="nav-search-direction", data=None),  # 'next' or 'previous'
            dcc.Store(id="nav-index-for-plotting", data=[]),  # index_table_for_plotting
            dcc.Interval(id="interval", interval=1000, n_intervals=0, disabled=True),
            # Modals
            dbc.Modal(
                [
                    dbc.ModalHeader(dbc.ModalTitle("Event Details", id="modal-title")),
                    dbc.ModalBody(id="modal-body"),
                    dbc.ModalFooter(
                        dbc.Button(
                            "Close", id="close-modal", className="ms-auto", n_clicks=0
                        )
                    ),
                ],
                id="event-modal",
                size="xl",
                is_open=False,
            ),
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
