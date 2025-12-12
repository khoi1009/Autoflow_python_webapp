#%%
from dash import Dash
import dash_bootstrap_components as dbc
from app.layout import create_layout
from app.callbacks import register_callbacks

def main():
    print("Starting Autoflow Web Dashboard...")
    
    app = Dash(__name__, external_stylesheets=[dbc.themes.MINTY, dbc.icons.BOOTSTRAP], suppress_callback_exceptions=True)
    app.title = "Water End-Use Dashboard"
    
    app.layout = create_layout()
    register_callbacks(app)
    
    app.run_server(debug=True, port=8051)

if __name__ == "__main__":
    main()

# %%
