from dash import (
    Input,
    Output,
    State,
    html,
    dcc,
    callback_context,
    no_update,
    ClientsideFunction,
    Patch,
)
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from datetime import timedelta
import pandas as pd
import base64
import io
import os
import copy
from pathlib import Path

from .data import (
    CATEGORY_COLORS,
    ALL_CATEGORIES,
    calculate_summary_stats,
    load_classified_data,
    run_analysis_thread,
    parse_datetime,
    get_cached_data,
    update_cached_data,
    invalidate_cache,
)

# Track the last CSV path that was used to build the chart
_last_built_csv_path = {"value": None}


def register_callbacks(app):

    @app.callback(
        [Output("current-file-path", "data"), Output("filename-display", "children")],
        Input("upload-data", "contents"),
        State("upload-data", "filename"),
        prevent_initial_call=True,
    )
    def handle_file_upload(contents, filename):
        if contents is None:
            return no_update, no_update

        content_type, content_string = contents.split(",")
        decoded = base64.b64decode(content_string)

        # Save to a temporary file
        data_dir = Path("data_uploads")
        data_dir.mkdir(exist_ok=True)
        file_path = data_dir / filename

        try:
            with open(file_path, "wb") as f:
                f.write(decoded)
            return (
                str(file_path),
                f"Loaded file: {filename}. Click 'Run Analysis' to process.",
            )
        except Exception as e:
            return no_update, f"Error saving file: {e}"

    @app.callback(
        [
            Output("progress-modal", "is_open"),
            Output("analysis-status", "children"),
            Output("interval", "disabled"),
        ],
        [Input("run-button", "n_clicks")],
        [State("current-file-path", "data")],
    )
    def start_analysis(analyze_clicks, file_path):
        if not analyze_clicks or not file_path:
            return False, "", True

        # Check if file exists
        csv_path = Path(file_path)
        if not csv_path.exists():
            return False, f"Error: File not found: {file_path}", True

        # Start analysis
        run_analysis_thread(csv_path)
        return True, "Starting analysis...", False

    @app.callback(
        [
            Output("event-store", "data"),
            Output("classified-csv-path", "data"),
            Output("progress-modal", "is_open", allow_duplicate=True),
            Output("time-slider", "max"),
            Output("window-size-input", "max"),
            Output("duration-display", "children"),
            Output("analysis-metadata", "data"),
            Output("date-from", "value", allow_duplicate=True),
            Output("date-to", "value", allow_duplicate=True),
        ],
        [Input("interval", "n_intervals")],
        [
            State("current-file-path", "data"),
            State("classified-csv-path", "data"),
            State("progress-modal", "is_open"),
        ],
        prevent_initial_call=True,
    )
    def check_analysis_progress(
        n_intervals, original_file_path, current_classified_path, progress_modal_open
    ):
        if not original_file_path or not progress_modal_open:
            return (
                no_update,
                no_update,
                no_update,
                no_update,
                no_update,
                no_update,
                no_update,
                no_update,
                no_update,
            )

        original_path = Path(original_file_path)
        base_name = original_path.stem
        classified_csv = original_path.parent / f"{base_name}_classified.csv"

        if classified_csv.exists():
            import time

            file_age = time.time() - classified_csv.stat().st_mtime

            if file_age < 5:
                df_events = load_classified_data(classified_csv)

                if not df_events.empty:
                    min_date = df_events["datetime_start"].min()
                    max_date = df_events["datetime_start"].max()
                    total_days = (max_date - min_date).total_seconds() / 86400

                    # Calculate timestamp in milliseconds for JS
                    # Force UTC to prevent browser timezone conversion issues
                    min_date_ts = (
                        min_date.tz_localize(None).tz_localize("UTC").timestamp() * 1000
                    )

                    # Calculate initial window end time (1 hour from start)
                    initial_window_hours = 1.0
                    initial_end_date = min_date + pd.Timedelta(
                        hours=initial_window_hours
                    )

                    # Format dates as DD/MM/YYYY HH:MM:SS
                    from_date_str = min_date.strftime("%d/%m/%Y %H:%M:%S")
                    to_date_str = initial_end_date.strftime("%d/%m/%Y %H:%M:%S")

                    return (
                        df_events.to_dict("records"),
                        str(classified_csv),
                        False,
                        total_days,
                        total_days * 24,
                        f"{int(total_days)} days {int((total_days % 1) * 24)} hours {int(((total_days * 24) % 1) * 60)} minutes",
                        {
                            "min_date": str(min_date),
                            "max_date": str(max_date),
                            "min_date_ts": min_date_ts,
                        },
                        from_date_str,
                        to_date_str,
                    )

        return (
            no_update,
            no_update,
            progress_modal_open,
            no_update,
            no_update,
            no_update,
            no_update,
            no_update,
            no_update,
        )

    @app.callback(
        Output("summary-table", "data"),
        [Input("classified-csv-path", "data")],
    )
    def update_summary_table_callback(classified_path):
        if not classified_path:
            return calculate_summary_stats(pd.DataFrame())

        df_events = get_cached_data(classified_path)
        if df_events is None:
            return calculate_summary_stats(pd.DataFrame())

        return calculate_summary_stats(df_events, None)

    @app.callback(
        Output("event-list-table", "data"),
        [
            Input("event-store", "data"),
            Input("timeline-chart", "figure"),
        ],
    )
    def update_event_list(event_data, figure):
        """Update event list to show only events in the current timeline window."""
        if not event_data:
            return []

        df = pd.DataFrame(event_data)

        # Get the actual displayed range from the figure's xaxis
        if figure and "layout" in figure and "xaxis" in figure["layout"]:
            xaxis = figure["layout"]["xaxis"]
            if "range" in xaxis and len(xaxis["range"]) == 2:
                try:
                    window_start = pd.to_datetime(xaxis["range"][0])
                    window_end = pd.to_datetime(xaxis["range"][1])

                    # Filter events that fall within the displayed window
                    df["datetime_start"] = pd.to_datetime(df["datetime_start"])
                    mask = (df["datetime_start"] >= window_start) & (
                        df["datetime_start"] <= window_end
                    )
                    df = df[mask]
                except Exception:
                    pass  # If parsing fails, show all events

        # Only return columns needed for display
        cols_to_keep = [
            "Start date",
            "Start time",
            "Category",
            "Duration (h:m:s)",
            "Volume (L)",
            "Max flow (litre/min)",
        ]
        cols = [c for c in cols_to_keep if c in df.columns]

        return df[cols].to_dict("records")

    @app.callback(
        [
            Output("event-store", "data", allow_duplicate=True),
            Output("cache-version", "data", allow_duplicate=True),
            Output("summary-table", "data", allow_duplicate=True),
        ],
        [Input("batch-set-btn", "n_clicks")],
        [
            State("batch-source-cat", "value"),
            State("batch-target-cat", "value"),
            State("event-store", "data"),
            State("classified-csv-path", "data"),
            State("cache-version", "data"),
        ],
        prevent_initial_call=True,
    )
    def batch_reclassify_events(
        n_clicks, source_cat, target_cat, event_data, classified_path, cache_version
    ):
        """Batch reclassify all events from source category to target category."""
        if not n_clicks or not source_cat or not target_cat or not event_data:
            return no_update, no_update, no_update

        if source_cat == target_cat:
            return no_update, no_update, no_update

        df_events = pd.DataFrame(event_data)

        # Update all events matching source_cat to target_cat
        mask = df_events["Category"] == source_cat
        count_changed = mask.sum()

        if count_changed > 0:
            print(
                f"DEBUG: Batch reclassifying {count_changed} events from '{source_cat}' to '{target_cat}'",
                flush=True,
            )

            df_events.loc[mask, "Category"] = target_cat

            # Save to CSV file
            if classified_path:
                try:
                    df_events.to_csv(classified_path, index=False)
                    print(f"DEBUG: Saved changes to {classified_path}", flush=True)
                except Exception as e:
                    print(f"ERROR saving CSV: {e}", flush=True)

            # Update the cache
            if classified_path:
                invalidate_cache(classified_path)
                update_cached_data(classified_path, df_events)

            # Recalculate summary
            summary_data = calculate_summary_stats(df_events)

            return (
                df_events.to_dict("records"),
                (cache_version or 0) + 1,
                summary_data,
            )

        return no_update, no_update, no_update

    # ========================================
    # Timeline Chart - Build full chart (only on initial load or visibility change)
    # ========================================
    @app.callback(
        [
            Output("timeline-chart", "figure"),
            Output("chart-initialized", "data"),
        ],
        [
            Input("classified-csv-path", "data"),  # Triggers on new analysis
            Input("category-visibility", "value"),
        ],
        [
            State("event-store", "data"),
            State("analysis-metadata", "data"),
            State("window-size-input", "value"),
            State("time-slider", "value"),  # Get current slider position
            State("chart-initialized", "data"),
        ],
    )
    def build_timeline_chart(
        csv_path,
        visible_categories,
        event_data,
        metadata,
        window_hours,
        slider_value,
        is_initialized,
    ):
        """Build the full timeline chart - only on initial load or visibility change."""
        from dash import callback_context

        ctx = callback_context
        trigger = ctx.triggered[0]["prop_id"] if ctx.triggered else "none"
        print(
            f"DEBUG build_timeline_chart TRIGGERED by: {trigger}, csv_path={csv_path}, last_built={_last_built_csv_path['value']}",
            flush=True,
        )

        # Skip if this csv_path was already built (prevents duplicate builds)
        if (
            trigger == "classified-csv-path.data"
            and csv_path == _last_built_csv_path["value"]
        ):
            print("DEBUG: Skipping rebuild - same csv_path already built", flush=True)
            return no_update, no_update

        if not event_data or not metadata:
            fig = go.Figure()
            fig.update_layout(
                xaxis_title="Time",
                yaxis_title="Flow Rate (L/min)",
                height=400,
                margin=dict(t=20, b=40, l=50, r=20),
            )
            return fig, False

        df = pd.DataFrame(event_data)
        df["datetime_start"] = pd.to_datetime(df["datetime_start"])

        min_date = pd.to_datetime(metadata.get("min_date"))
        window_hours = window_hours or 1.0
        slider_value = slider_value or 0

        # Use current slider position to maintain view
        window_start = min_date + timedelta(days=slider_value)
        window_end = window_start + timedelta(hours=window_hours)

        # Filter by visible categories
        if visible_categories:
            df = df[df["Category"].isin(visible_categories)]

        # Create figure with all traces
        fig = go.Figure()

        for idx, row in df.iterrows():
            category = row.get("Category", "Other")
            if category not in (visible_categories or ALL_CATEGORIES):
                continue

            color = CATEGORY_COLORS.get(category, "gray")

            flow_rates = row.get("flow_rates", [])
            if isinstance(flow_rates, str):
                try:
                    import ast

                    flow_rates = ast.literal_eval(flow_rates)
                except:
                    flow_rates = []

            if flow_rates:
                start_time = row["datetime_start"]
                times = [
                    start_time + timedelta(seconds=i) for i in range(len(flow_rates))
                ]
                custom_data = [[idx]] * len(flow_rates)

                fig.add_trace(
                    go.Scatter(
                        x=times,
                        y=flow_rates,
                        mode="lines",
                        fill="tozeroy",
                        name=category,
                        line=dict(color=color, width=1),
                        fillcolor=color,
                        showlegend=False,
                        customdata=custom_data,
                        hovertemplate=f"{category}<br>Time: %{{x}}<br>Flow: %{{y:.2f}} L/min<br><i>Click for details</i><extra></extra>",
                    )
                )

        fig.update_layout(
            xaxis=dict(
                range=[
                    window_start.strftime("%Y-%m-%dT%H:%M:%S"),
                    window_end.strftime("%Y-%m-%dT%H:%M:%S"),
                ],
                title="Time",
            ),
            yaxis=dict(title="Flow Rate (L/min)", rangemode="tozero"),
            height=400,
            margin=dict(l=50, r=20, t=30, b=40),
            showlegend=False,
            hovermode="closest",
        )

        # Mark this csv_path as built to prevent duplicate builds
        _last_built_csv_path["value"] = csv_path
        print(
            f"DEBUG: Chart built successfully, marked csv_path={csv_path}", flush=True
        )

        return fig, True  # Mark chart as initialized

    # ========================================
    # Timeline Chart - Update window range only (FAST)
    # ========================================
    # ========================================
    # Timeline Chart - Update window range (CLIENT-SIDE)
    # ========================================
    app.clientside_callback(
        ClientsideFunction(namespace="clientside", function_name="updateTimeline"),
        [
            Output("timeline-chart", "figure", allow_duplicate=True),
            Output("date-from", "value", allow_duplicate=True),
            Output("date-to", "value", allow_duplicate=True),
        ],
        [
            Input("time-slider", "value"),
            Input("window-size-input", "value"),
        ],
        [
            State("analysis-metadata", "data"),
            State("timeline-chart", "figure"),
        ],
        prevent_initial_call=True,
    )

    # Clientside callback for slider scroll buttons
    app.clientside_callback(
        ClientsideFunction(namespace="clientside", function_name="scrollSlider"),
        Output("time-slider", "value", allow_duplicate=True),
        [Input("scroll-left", "n_clicks"), Input("scroll-right", "n_clicks")],
        [
            State("time-slider", "value"),
            State("time-slider", "max"),
            State("window-size-input", "value"),
        ],
        prevent_initial_call=True,
    )

    # ========================================
    # Event Navigation (CLIENT-SIDE - INSTANT)
    # ========================================
    app.clientside_callback(
        ClientsideFunction(namespace="clientside", function_name="navigateEvent"),
        Output("time-slider", "value", allow_duplicate=True),
        [
            Input("prev-event-btn", "n_clicks"),
            Input("next-event-btn", "n_clicks"),
        ],
        [
            State("nav-category", "value"),
            State("event-store", "data"),
            State("analysis-metadata", "data"),
            State("time-slider", "value"),
            State("time-slider", "max"),
            State("window-size-input", "value"),
        ],
        prevent_initial_call=True,
    )

    # ========================================
    # Save Project - Open Modal
    # ========================================
    @app.callback(
        [
            Output("save-modal", "is_open"),
            Output("save-filename-input", "value"),
        ],
        [
            Input("save-project-btn", "n_clicks"),
            Input("save-modal-cancel", "n_clicks"),
            Input("save-modal-confirm", "n_clicks"),
        ],
        [
            State("save-modal", "is_open"),
            State("classified-csv-path", "data"),
        ],
        prevent_initial_call=True,
    )
    def toggle_save_modal(
        open_clicks, cancel_clicks, confirm_clicks, is_open, classified_path
    ):
        """Open/close the save modal and set default filename."""
        ctx = callback_context
        if not ctx.triggered:
            return no_update, no_update

        trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]

        if trigger_id == "save-project-btn":
            # Opening modal - set default filename from classified path
            if classified_path:
                default_name = Path(classified_path).stem.replace("_classified", "")
            else:
                default_name = "my_project"
            return True, default_name

        # Cancel or Confirm closes the modal
        return False, no_update

    # ========================================
    # Save Project - Download File
    # ========================================
    @app.callback(
        Output("download-project", "data"),
        Input("save-modal-confirm", "n_clicks"),
        [
            State("save-filename-input", "value"),
            State("event-store", "data"),
            State("classified-csv-path", "data"),
            State("analysis-metadata", "data"),
            State("time-slider", "value"),
            State("window-size-input", "value"),
        ],
        prevent_initial_call=True,
    )
    def save_project(
        n_clicks,
        filename_input,
        event_data,
        classified_path,
        metadata,
        slider_value,
        window_size,
    ):
        """Save the current project state to a .autoflow file."""
        import json
        from datetime import datetime

        if not n_clicks or not event_data:
            return no_update

        # Build project data structure
        project_data = {
            "version": "1.0",
            "saved_at": datetime.now().isoformat(),
            "original_file": classified_path,
            "metadata": metadata,
            "view_state": {
                "slider_value": slider_value,
                "window_size": window_size,
            },
            "events": event_data,
        }

        # Use user-provided filename or default
        if filename_input and filename_input.strip():
            filename = f"{filename_input.strip()}.autoflow"
        elif classified_path:
            original_name = Path(classified_path).stem.replace("_classified", "")
            filename = f"{original_name}.autoflow"
        else:
            filename = "project.autoflow"

        return dict(
            content=json.dumps(project_data, indent=2),
            filename=filename,
            type="application/json",
        )

    # ========================================
    # Open Project Callback
    # ========================================
    @app.callback(
        [
            Output("event-store", "data", allow_duplicate=True),
            Output("classified-csv-path", "data", allow_duplicate=True),
            Output("analysis-metadata", "data", allow_duplicate=True),
            Output("time-slider", "max", allow_duplicate=True),
            Output("time-slider", "value", allow_duplicate=True),
            Output("window-size-input", "max", allow_duplicate=True),
            Output("window-size-input", "value", allow_duplicate=True),
            Output("duration-display", "children", allow_duplicate=True),
            Output("date-from", "value", allow_duplicate=True),
            Output("date-to", "value", allow_duplicate=True),
            Output("filename-display", "children", allow_duplicate=True),
        ],
        Input("open-project-upload", "contents"),
        State("open-project-upload", "filename"),
        prevent_initial_call=True,
    )
    def open_project(contents, filename):
        """Load a project from a .autoflow file."""
        import json

        if contents is None:
            return [no_update] * 11

        try:
            # Decode the uploaded file
            content_type, content_string = contents.split(",")
            decoded = base64.b64decode(content_string).decode("utf-8")
            project_data = json.loads(decoded)

            # Extract data from project file
            event_data = project_data.get("events", [])
            metadata = project_data.get("metadata", {})
            view_state = project_data.get("view_state", {})
            original_file = project_data.get("original_file", "")

            if not event_data:
                return [no_update] * 11

            # Convert event data to DataFrame to get date range
            df_events = pd.DataFrame(event_data)
            df_events["datetime_start"] = pd.to_datetime(df_events["datetime_start"])

            min_date = df_events["datetime_start"].min()
            max_date = df_events["datetime_start"].max()
            total_days = (max_date - min_date).total_seconds() / 86400

            # Update the cache with the loaded data
            if original_file:
                update_cached_data(original_file, df_events)

            # Calculate min_date timestamp for JS
            min_date_ts = min_date.timestamp() * 1000

            # Restore or calculate metadata
            if not metadata:
                metadata = {
                    "min_date": str(min_date),
                    "max_date": str(max_date),
                    "min_date_ts": min_date_ts,
                }

            # Get view state values
            slider_value = view_state.get("slider_value", 0)
            window_size = view_state.get("window_size", 1.0)

            # Calculate From/To dates based on slider position
            start_time = min_date + timedelta(days=slider_value)
            end_time = start_time + timedelta(hours=window_size)
            from_date_str = start_time.strftime("%d/%m/%Y %H:%M:%S")
            to_date_str = end_time.strftime("%d/%m/%Y %H:%M:%S")

            # Duration display
            duration_str = f"{int(total_days)} days {int((total_days % 1) * 24)} hours {int(((total_days * 24) % 1) * 60)} minutes"

            # Display message
            display_msg = f"Opened project: {filename}"

            return (
                event_data,
                original_file,
                metadata,
                total_days,
                slider_value,
                total_days * 24,
                window_size,
                duration_str,
                from_date_str,
                to_date_str,
                display_msg,
            )

        except Exception as e:
            print(f"Error opening project: {e}", flush=True)
            return [no_update] * 11

    # ========================================
    # Event Details Modal Callback
    # ========================================
    @app.callback(
        [
            Output("event-modal", "is_open"),
            Output("modal-title", "children"),
            Output("modal-body", "children"),
            Output("modal-event-index", "data"),
            Output("modal-original-category", "data"),
            Output("confirm-reclassify-btn", "style"),
        ],
        [
            Input("timeline-chart", "clickData"),
            Input("close-modal", "n_clicks"),
        ],
        [
            State("event-modal", "is_open"),
            State("event-store", "data"),
        ],
    )
    def handle_event_click(click_data, close_clicks, is_open, event_data):
        """Handle click on timeline to show event details with chart."""
        from dash import callback_context, no_update

        if not callback_context.triggered:
            return is_open, no_update, no_update, no_update, no_update, no_update

        trigger_id = callback_context.triggered[0]["prop_id"].split(".")[0]

        # Handle close button
        if trigger_id == "close-modal":
            return False, no_update, no_update, None, None, {"display": "none"}

        # Handle chart click
        if trigger_id == "timeline-chart" and click_data and event_data:
            try:
                point = click_data["points"][0]
                event_idx = point.get("customdata", [None])[0]

                if event_idx is None:
                    return (
                        is_open,
                        no_update,
                        no_update,
                        no_update,
                        no_update,
                        no_update,
                    )

                # Find the event in event_data
                df_events = pd.DataFrame(event_data)

                if event_idx not in df_events.index:
                    if isinstance(event_idx, int) and event_idx < len(df_events):
                        event = df_events.iloc[event_idx]
                        actual_idx = event_idx
                    else:
                        return (
                            is_open,
                            no_update,
                            no_update,
                            no_update,
                            no_update,
                            no_update,
                        )
                else:
                    event = df_events.loc[event_idx]
                    actual_idx = event_idx

                category = event.get("Category", "Unknown")
                color = CATEGORY_COLORS.get(category, "red")

                # Build modal title
                title = f"📊 {category} Event Details"

                # Create flow rate chart
                flow_rates = event.get("flow_rates", [])
                if isinstance(flow_rates, str):
                    try:
                        import ast

                        flow_rates = ast.literal_eval(flow_rates)
                    except:
                        flow_rates = []

                # Create the event shape chart
                event_fig = go.Figure()
                if flow_rates:
                    times = list(range(len(flow_rates)))
                    event_fig.add_trace(
                        go.Scatter(
                            x=times,
                            y=flow_rates,
                            mode="lines",
                            fill="tozeroy",
                            line=dict(color=color, width=2),
                            fillcolor=color,
                            hovertemplate="Time: %{x}s<br>Flow: %{y:.2f} L/min<extra></extra>",
                        )
                    )
                event_fig.update_layout(
                    xaxis=dict(title="Time (seconds)", showgrid=True, gridcolor="#eee"),
                    yaxis=dict(
                        title="Flow rate (L/min)",
                        showgrid=True,
                        gridcolor="#eee",
                        rangemode="tozero",
                    ),
                    height=250,
                    margin=dict(l=50, r=20, t=10, b=40),
                    plot_bgcolor="white",
                    showlegend=False,
                )

                # Build modal body with two columns: left=details, right=chart
                body = dbc.Container(
                    [
                        dbc.Row(
                            [
                                # Left column - Event details
                                dbc.Col(
                                    [
                                        # Event info table
                                        html.Table(
                                            [
                                                html.Tbody(
                                                    [
                                                        html.Tr(
                                                            [
                                                                html.Td(
                                                                    "Date",
                                                                    style={
                                                                        "fontWeight": "bold",
                                                                        "padding": "5px 15px 5px 5px",
                                                                    },
                                                                ),
                                                                html.Td(
                                                                    event.get(
                                                                        "Start date", ""
                                                                    ),
                                                                    style={
                                                                        "padding": "5px"
                                                                    },
                                                                ),
                                                            ]
                                                        ),
                                                        html.Tr(
                                                            [
                                                                html.Td(
                                                                    "Start time",
                                                                    style={
                                                                        "fontWeight": "bold",
                                                                        "padding": "5px 15px 5px 5px",
                                                                    },
                                                                ),
                                                                html.Td(
                                                                    event.get(
                                                                        "Start time", ""
                                                                    ),
                                                                    style={
                                                                        "padding": "5px"
                                                                    },
                                                                ),
                                                            ]
                                                        ),
                                                        html.Tr(
                                                            [
                                                                html.Td(
                                                                    "End time",
                                                                    style={
                                                                        "fontWeight": "bold",
                                                                        "padding": "5px 15px 5px 5px",
                                                                    },
                                                                ),
                                                                html.Td(
                                                                    event.get(
                                                                        "End time", ""
                                                                    ),
                                                                    style={
                                                                        "padding": "5px"
                                                                    },
                                                                ),
                                                            ]
                                                        ),
                                                        html.Tr(
                                                            [
                                                                html.Td(
                                                                    "Category",
                                                                    style={
                                                                        "fontWeight": "bold",
                                                                        "padding": "5px 15px 5px 5px",
                                                                    },
                                                                ),
                                                                html.Td(
                                                                    html.Span(
                                                                        category,
                                                                        style={
                                                                            "color": color,
                                                                            "fontWeight": "bold",
                                                                        },
                                                                    ),
                                                                    style={
                                                                        "padding": "5px"
                                                                    },
                                                                ),
                                                            ]
                                                        ),
                                                        html.Tr(
                                                            [
                                                                html.Td(
                                                                    "Duration",
                                                                    style={
                                                                        "fontWeight": "bold",
                                                                        "padding": "5px 15px 5px 5px",
                                                                    },
                                                                ),
                                                                html.Td(
                                                                    event.get(
                                                                        "Duration (h:m:s)",
                                                                        "N/A",
                                                                    ),
                                                                    style={
                                                                        "padding": "5px"
                                                                    },
                                                                ),
                                                            ]
                                                        ),
                                                        html.Tr(
                                                            [
                                                                html.Td(
                                                                    "Total Volume (L)",
                                                                    style={
                                                                        "fontWeight": "bold",
                                                                        "padding": "5px 15px 5px 5px",
                                                                    },
                                                                ),
                                                                html.Td(
                                                                    f"{float(event.get('Volume (L)', 0)):.4f}",
                                                                    style={
                                                                        "padding": "5px"
                                                                    },
                                                                ),
                                                            ]
                                                        ),
                                                        html.Tr(
                                                            [
                                                                html.Td(
                                                                    "Max Flow (L/min)",
                                                                    style={
                                                                        "fontWeight": "bold",
                                                                        "padding": "5px 15px 5px 5px",
                                                                    },
                                                                ),
                                                                html.Td(
                                                                    f"{float(event.get('Max flow (litre/min)', 0)):.4f}",
                                                                    style={
                                                                        "padding": "5px"
                                                                    },
                                                                ),
                                                            ]
                                                        ),
                                                        html.Tr(
                                                            [
                                                                html.Td(
                                                                    "Mode Flow (L/min)",
                                                                    style={
                                                                        "fontWeight": "bold",
                                                                        "padding": "5px 15px 5px 5px",
                                                                    },
                                                                ),
                                                                html.Td(
                                                                    f"{float(event.get('Mode (L/min)', 0)):.4f}",
                                                                    style={
                                                                        "padding": "5px"
                                                                    },
                                                                ),
                                                            ]
                                                        ),
                                                    ]
                                                )
                                            ],
                                            style={
                                                "borderCollapse": "collapse",
                                                "width": "100%",
                                                "fontSize": "0.9em",
                                            },
                                        ),
                                        html.Hr(className="my-3"),
                                        # Reclassify section
                                        html.Div(
                                            [
                                                html.Label(
                                                    "Select one of the following options",
                                                    className="small text-muted mb-1",
                                                ),
                                                dbc.Select(
                                                    id="modal-reclassify-select",
                                                    options=[
                                                        {"label": c, "value": c}
                                                        for c in ALL_CATEGORIES
                                                    ],
                                                    value=category,
                                                    size="sm",
                                                    className="mb-2",
                                                ),
                                            ]
                                        ),
                                    ],
                                    width=5,
                                    style={
                                        "borderRight": "1px solid #dee2e6",
                                        "paddingRight": "15px",
                                    },
                                ),
                                # Right column - Chart
                                dbc.Col(
                                    [
                                        dcc.Graph(
                                            figure=event_fig,
                                            config={"displayModeBar": False},
                                            style={"height": "100%"},
                                        ),
                                    ],
                                    width=7,
                                    style={"paddingLeft": "15px"},
                                ),
                            ]
                        ),
                    ],
                    fluid=True,
                )

                return (
                    True,
                    title,
                    body,
                    actual_idx,
                    category,
                    {"display": "inline-block"},
                )

            except Exception as e:
                print(f"Error handling click: {e}", flush=True)
                import traceback

                traceback.print_exc()
                return is_open, no_update, no_update, no_update, no_update, no_update

        return is_open, no_update, no_update, no_update, no_update, no_update

    # ========================================
    # Reclassification Confirmation Callback (OPTIMIZED - index-based)
    # ========================================
    @app.callback(
        [
            Output("event-store", "data", allow_duplicate=True),
            Output("summary-table", "data", allow_duplicate=True),
            Output("event-list-table", "data", allow_duplicate=True),
            Output("event-modal", "is_open", allow_duplicate=True),
            Output("timeline-chart", "figure", allow_duplicate=True),
        ],
        [Input("confirm-reclassify-btn", "n_clicks")],
        [
            State("modal-reclassify-select", "value"),
            State("modal-event-index", "data"),
            State("modal-original-category", "data"),
            State("event-store", "data"),
            State("classified-csv-path", "data"),
            State("summary-table", "data"),
            State("event-list-table", "data"),
            State("timeline-chart", "figure"),
        ],
        prevent_initial_call=True,
    )
    def confirm_reclassification(
        n_clicks,
        new_category,
        event_idx,
        original_category,
        event_data,
        csv_path,
        current_summary,
        current_event_list,
        current_figure,
    ):
        """Optimized reclassification - only update what changed."""
        if not n_clicks or event_idx is None or not event_data:
            return no_update, no_update, no_update, no_update, no_update

        # Skip if category didn't change
        if new_category == original_category:
            return no_update, no_update, no_update, False, no_update

        try:
            # Get the volume of the event being reclassified
            event_volume = 0
            if isinstance(event_idx, int) and event_idx < len(event_data):
                event_volume = float(event_data[event_idx].get("Volume (L)", 0))

            # === 1. Update event_store (just change category at index) ===
            updated_event_data = copy.deepcopy(event_data)
            if isinstance(event_idx, int) and event_idx < len(updated_event_data):
                updated_event_data[event_idx]["Category"] = new_category

            # === 2. Update summary table (add/subtract volume) ===
            updated_summary = copy.deepcopy(current_summary) if current_summary else []

            # Find category indices in summary
            old_cat_idx = None
            new_cat_idx = None
            total_idx = None

            for i, row in enumerate(updated_summary):
                if row.get("Category") == original_category:
                    old_cat_idx = i
                if row.get("Category") == new_category:
                    new_cat_idx = i
                if row.get("Category") == "Total":
                    total_idx = i

            # Subtract from old category, add to new category
            if old_cat_idx is not None and event_volume > 0:
                updated_summary[old_cat_idx]["Volume (L)"] = round(
                    updated_summary[old_cat_idx].get("Volume (L)", 0) - event_volume, 2
                )
            if new_cat_idx is not None:
                updated_summary[new_cat_idx]["Volume (L)"] = round(
                    updated_summary[new_cat_idx].get("Volume (L)", 0) + event_volume, 2
                )

            # Recalculate percentages
            total_volume = sum(
                row.get("Volume (L)", 0)
                for row in updated_summary
                if row.get("Category") != "Total"
            )
            for row in updated_summary:
                if row.get("Category") != "Total":
                    if total_volume > 0:
                        row["Percentage (%)"] = round(
                            (row.get("Volume (L)", 0) / total_volume) * 100, 1
                        )
                    else:
                        row["Percentage (%)"] = 0

            # === 3. Update event list table (just change category at index) ===
            updated_event_list = (
                copy.deepcopy(current_event_list) if current_event_list else []
            )
            if isinstance(event_idx, int) and event_idx < len(updated_event_list):
                updated_event_list[event_idx]["Category"] = new_category

            # === 4. Update chart (just change color of the specific trace) ===
            new_color = CATEGORY_COLORS.get(new_category, "gray")
            patched_figure = Patch()

            # Find the trace index that corresponds to this event
            # Since each event creates one trace, event_idx should correspond to trace index
            # But we need to account for filtered categories
            if current_figure and "data" in current_figure:
                # Find trace by matching customdata
                for trace_idx, trace in enumerate(current_figure.get("data", [])):
                    customdata = trace.get("customdata", [])
                    if customdata and len(customdata) > 0:
                        trace_event_idx = (
                            customdata[0][0]
                            if isinstance(customdata[0], list)
                            else customdata[0]
                        )
                        if trace_event_idx == event_idx:
                            # Update this trace's color and name
                            patched_figure["data"][trace_idx]["line"][
                                "color"
                            ] = new_color
                            patched_figure["data"][trace_idx]["fillcolor"] = new_color
                            patched_figure["data"][trace_idx]["name"] = new_category
                            patched_figure["data"][trace_idx][
                                "hovertemplate"
                            ] = f"{new_category}<br>Time: %{{x}}<br>Flow: %{{y:.2f}} L/min<br><i>Click for details</i><extra></extra>"
                            break

            # === 5. Update CSV file in background ===
            if csv_path and Path(csv_path).exists():
                try:
                    csv_df = pd.read_csv(csv_path)
                    if event_idx < len(csv_df):
                        csv_df.iloc[event_idx, csv_df.columns.get_loc("Category")] = (
                            new_category
                        )
                        csv_df.to_csv(csv_path, index=False)
                except Exception as e:
                    print(f"Error updating CSV: {e}", flush=True)

            print(
                f"Reclassified event {event_idx} from {original_category} to {new_category}",
                flush=True,
            )

            return (
                updated_event_data,
                updated_summary,
                updated_event_list,
                False,
                patched_figure,
            )

        except Exception as e:
            print(f"Error in reclassification: {e}", flush=True)
            import traceback

            traceback.print_exc()
            return no_update, no_update, no_update, no_update, no_update
