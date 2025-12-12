# Autoflow - Water Usage Analyzer

A Python-based web application for analyzing and classifying residential water usage data.

## Features

- 📊 **End-Use Classification**: Automatically classifies water events into categories (Shower, Tap, Toilet, Clothes Washer, Dishwasher, etc.)
- 📈 **Interactive Timeline**: Visualize water usage over time with zoomable charts
- 🔍 **Event Navigation**: Navigate between events by category (Previous/Next)
- 📝 **Event Log**: View and filter events in the current time window
- 🔄 **Batch Reclassification**: Quickly change categories for multiple events
- 💾 **Project Save/Load**: Save your analysis and resume later (.autoflow files)

## Installation

1. Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/Autoflow_Python.git
cd Autoflow_Python
```

2. Create a virtual environment (recommended):

```bash
python -m venv venv
venv\Scripts\activate  # Windows
# or
source venv/bin/activate  # Linux/Mac
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

1. Run the application:

```bash
python -m app.main
```

2. Open your browser and navigate to: http://127.0.0.1:8052/

3. Upload a CSV file with water usage data and click "Run Analysis"

## Project Structure

```
Autoflow_Python/
├── app/                    # Dash web application
│   ├── main.py            # App entry point
│   ├── layout.py          # UI layout
│   ├── callbacks.py       # Event handlers
│   ├── data.py            # Data loading/caching
│   └── assets/            # Static files (JS, CSS)
├── src/                    # Analysis modules
│   ├── main.py            # Analysis pipeline
│   └── core/              # Core analysis logic
│       ├── event_extraction.py
│       └── classifier_simple.py
├── data_uploads/          # Uploaded data files
├── checkpoints/           # ML model checkpoints
└── requirements.txt       # Python dependencies
```

## License

MIT License
