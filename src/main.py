"""
Main analysis module for Autoflow water usage analyzer.
"""

import pandas as pd
from pathlib import Path
from .core.event_extraction import extract_events
from .core.classifier_simple import classify_events


def run_analysis(csv_path):
    """
    Run the complete analysis pipeline on a CSV file.

    Args:
        csv_path: Path to the input CSV file with water usage data

    Returns:
        Path to the classified CSV file
    """
    csv_path = Path(csv_path)

    print(f"Starting analysis for: {csv_path}")

    # Load data
    df = pd.read_csv(csv_path)

    # Extract events
    events = extract_events(df)

    # Classify events
    classified_events = classify_events(events)

    # Create output path
    output_path = csv_path.parent / f"{csv_path.stem}_classified.csv"

    # Save classified events
    classified_events.to_csv(output_path, index=False)

    print(f"Analysis complete. Output saved to: {output_path}")

    return output_path
