import dash
import dash_bootstrap_components as dbc
from .layout import create_layout
from .callbacks import register_callbacks

# Create Dash app
app = dash.Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.BOOTSTRAP,
        "https://cdn.jsdelivr.net/npm/bootstrap-icons@1.10.0/font/bootstrap-icons.css",
    ],
    suppress_callback_exceptions=True,
)

app.title = "Autoflow - Water Usage Analyzer"

# Set layout
app.layout = create_layout()

# Register callbacks
register_callbacks(app)

# Run server
if __name__ == "__main__":
    app.run_server(debug=True, port=8052)
