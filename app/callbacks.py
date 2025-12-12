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
                    min_date_ts = min_date.timestamp() * 1000

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
            return calculate_summary_stats(pd.DataFrame()).to_dict("records")

        df_events = get_cached_data(classified_path)
        if df_events is None:
            return calculate_summary_stats(pd.DataFrame()).to_dict("records")

        summary_df = calculate_summary_stats(df_events, None)
        return summary_df.to_dict("records")

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
            summary_df = calculate_summary_stats(df_events)

            return (
                df_events.to_dict("records"),
                (cache_version or 0) + 1,
                summary_df.to_dict("records"),
            )

        return no_update, no_update, no_update

    @app.callback(
        Output("timeline-chart", "figure"),
        [
            Input("event-store", "data"),
            Input("time-slider", "value"),
            Input("window-size-input", "value"),
            Input("category-visibility", "value"),
        ],
        [State("analysis-metadata", "data")],
    )
    def update_timeline_chart(
        event_data, slider_value, window_hours, visible_categories, metadata
    ):
        if not event_data or not metadata:
            # Return empty figure
            fig = go.Figure()
            fig.update_layout(
                title="Upload a CSV file and run analysis to see the timeline",
                xaxis_title="Time",
                yaxis_title="Flow Rate (L/min)",
                height=300,
            )
            return fig

        df = pd.DataFrame(event_data)
        df["datetime_start"] = pd.to_datetime(df["datetime_start"])

        min_date = pd.to_datetime(metadata.get("min_date"))
        max_date = pd.to_datetime(metadata.get("max_date"))

        # Calculate window
        window_hours = window_hours or 1.0
        slider_value = slider_value or 0

        window_start = min_date + timedelta(days=slider_value)
        window_end = window_start + timedelta(hours=window_hours)

        # Filter by visible categories
        if visible_categories:
            df = df[df["Category"].isin(visible_categories)]

        # Create figure
        fig = go.Figure()

        # Add traces for each category
        for category in ALL_CATEGORIES:
            if category not in (visible_categories or []):
                continue

            cat_df = df[df["Category"] == category]
            if cat_df.empty:
                continue

            color = CATEGORY_COLORS.get(category, "gray")

            for _, row in cat_df.iterrows():
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
                        start_time + timedelta(seconds=i)
                        for i in range(len(flow_rates))
                    ]

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
                            hovertemplate=f"{category}<br>Time: %{{x}}<br>Flow: %{{y:.2f}} L/min<extra></extra>",
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
            height=300,
            margin=dict(l=50, r=20, t=30, b=40),
            showlegend=False,
            hovermode="closest",
        )

        return fig

    # Clientside callback for slider scroll buttons
    app.clientside_callback(
        ClientsideFunction(namespace="clientside", function_name="scrollSlider"),
        Output("time-slider", "value"),
        [Input("scroll-left", "n_clicks"), Input("scroll-right", "n_clicks")],
        [
            State("time-slider", "value"),
            State("time-slider", "max"),
            State("window-size-input", "value"),
        ],
    )

    # ========================================
    # Navigation Callbacks
    # ========================================
    @app.callback(
        [
            Output("timeline-chart", "figure", allow_duplicate=True),
            Output("current-event-index", "data"),
            Output("nav-remaining-indices", "data"),
            Output("nav-same-category", "data"),
            Output("nav-search-direction", "data"),
            Output("nav-index-for-plotting", "data"),
            Output("date-from", "value"),
            Output("date-to", "value"),
        ],
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
            State("current-event-index", "data"),
            State("nav-remaining-indices", "data"),
            State("nav-same-category", "data"),
            State("nav-search-direction", "data"),
            State("nav-index-for-plotting", "data"),
            State("timeline-chart", "figure"),
        ],
        prevent_initial_call=True,
    )
    def navigate_event(
        prev_clicks,
        next_clicks,
        category,
        event_data,
        metadata,
        slider_value,
        slider_max,
        window_days,
        current_index,
        remaining_indices,
        same_category,
        search_direction,
        index_for_plotting,
        current_figure,
    ):
        ctx = callback_context
        if not ctx.triggered or not event_data or not metadata or not category:
            return (
                no_update,
                no_update,
                no_update,
                no_update,
                no_update,
                no_update,
                no_update,
                no_update,
            )

        trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]

        df = pd.DataFrame(event_data)
        df["datetime_start"] = pd.to_datetime(df["datetime_start"])

        min_date = pd.to_datetime(metadata.get("min_date"))
        window_days = (window_days or 1.0) / 24  # Convert hours to days

        # Get current window position from figure
        current_window_start = min_date + timedelta(days=slider_value or 0)

        # Filter by category
        cat_df = df[df["Category"] == category].copy()
        if cat_df.empty:
            return (
                no_update,
                no_update,
                no_update,
                no_update,
                no_update,
                no_update,
                no_update,
                no_update,
            )

        cat_df = cat_df.sort_values("datetime_start").reset_index(drop=True)

        # Determine search direction
        is_next = trigger_id == "next-event-btn"
        new_direction = "next" if is_next else "previous"

        # Check if we need to rebuild the index list
        rebuild_list = (
            same_category != category
            or search_direction != new_direction
            or not remaining_indices
        )

        if rebuild_list:
            # Build list of all events of this category
            if is_next:
                # Find events after current window start
                future_events = cat_df[cat_df["datetime_start"] >= current_window_start]
                remaining_indices = future_events.index.tolist()
            else:
                # Find events before current window start
                past_events = cat_df[cat_df["datetime_start"] < current_window_start]
                remaining_indices = past_events.index.tolist()[::-1]  # Reverse for prev

        if not remaining_indices:
            # No more events in this direction, wrap around
            if is_next:
                remaining_indices = cat_df.index.tolist()
            else:
                remaining_indices = cat_df.index.tolist()[::-1]

        if not remaining_indices:
            return (
                no_update,
                no_update,
                [],
                category,
                new_direction,
                [],
                no_update,
                no_update,
            )

        # Get next event index
        index_of_searched_event = remaining_indices[0]
        remaining_indices = remaining_indices[1:]

        # Get event details
        event_row = cat_df.loc[index_of_searched_event]
        event_start = event_row["datetime_start"]

        # Calculate slider position to center the event
        offset_days = window_days / 2
        new_slider_days = (event_start - min_date).total_seconds() / 86400 - offset_days

        # Clamp to valid range
        new_slider_value = max(0, min(slider_max or 100, new_slider_days))

        # Calculate new time range
        start_time = min_date + timedelta(days=new_slider_value)
        end_time = start_time + timedelta(days=window_days)

        # Create a Patch object to update only the xaxis range
        patched_figure = Patch()
        patched_figure["layout"]["xaxis"]["range"] = [
            start_time.strftime("%Y-%m-%dT%H:%M:%S"),
            end_time.strftime("%Y-%m-%dT%H:%M:%S"),
        ]

        # Format dates for From/To textboxes (DD/MM/YYYY HH:MM:SS)
        from_date_str = start_time.strftime("%d/%m/%Y %H:%M:%S")
        to_date_str = end_time.strftime("%d/%m/%Y %H:%M:%S")

        return (
            patched_figure,
            index_of_searched_event,
            remaining_indices,
            category,
            new_direction,
            index_for_plotting,
            from_date_str,
            to_date_str,
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

    # Modal close callback
    @app.callback(
        Output("event-modal", "is_open"),
        [Input("close-modal", "n_clicks")],
        [State("event-modal", "is_open")],
    )
    def toggle_modal(n_clicks, is_open):
        if n_clicks:
            return False
        return is_open
