import dash
import dash_bootstrap_components as dbc
from .layout import create_layout
from .callbacks import register_callbacks

# Create Dash app with FLATLY theme (modern, clean look)
app = dash.Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.FLATLY,  # Modern flat design theme
        "https://cdn.jsdelivr.net/npm/bootstrap-icons@1.10.0/font/bootstrap-icons.css",
        "https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap",
    ],
    suppress_callback_exceptions=True,
    assets_folder="assets",  # Ensure assets folder is used
)

app.title = "Autoflow - Water Usage Analyzer"

# Set layout
app.layout = create_layout()

# Register callbacks
register_callbacks(app)

# Run server
if __name__ == "__main__":
    app.run_server(debug=True, port=8052)
