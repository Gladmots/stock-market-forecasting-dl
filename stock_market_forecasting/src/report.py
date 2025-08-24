# src/report.py

# run with python -m src.report to get outputs/report.pdf
# report.py
# Run: python -m stock_market_forecasting.report
from pathlib import Path
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table
from reportlab.lib.styles import getSampleStyleSheet
import json

ROOT = Path(__file__).resolve().parent
OUTPUTS_DIR = ROOT / "outputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
REPORT_PATH = OUTPUTS_DIR / "report.pdf"

def main():
    styles = getSampleStyleSheet()
    elems = []

    elems.append(Paragraph("Stock Forecasting Report", styles['Title']))
    elems.append(Spacer(1, 12))

    # Metrics table
    metrics_path = OUTPUTS_DIR / "metrics.json"
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text())
        rows = [[k, round(float(v), 4)] for k, v in metrics.items()]
        elems.append(Paragraph("Validation Metrics", styles['Heading2']))
        elems.append(Table([["Metric", "Value"]] + rows))
        elems.append(Spacer(1, 12))

    # Plots (if present)
    for plot_name in ("val_plot.png", "future_plot.png"):
        plot_path = OUTPUTS_DIR / plot_name
        if plot_path.exists():
            title = plot_name.replace("_", " ").replace(".png", "").title()
            elems.append(Paragraph(title, styles['Heading2']))
            elems.append(Image(str(plot_path), width=480, height=260))
            elems.append(Spacer(1, 12))

    # CSV notes
    elems.append(Paragraph("Data Exports", styles['Heading2']))
    elems.append(Paragraph("The following CSVs are saved in outputs/:", styles['Normal']))
    elems.append(Paragraph("• val_predictions.csv", styles['Normal']))
    elems.append(Paragraph("• future_predictions.csv", styles['Normal']))

    SimpleDocTemplate(str(REPORT_PATH)).build(elems)
    print(f"[report] Generated {REPORT_PATH}")

if __name__ == "__main__":
    main()
