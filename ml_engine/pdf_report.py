"""
PDF Report Generator

Exports a comprehensive intrusion detection report including
all available visualizations and metrics tables.
"""

import os
from datetime import datetime

from django.conf import settings
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Image,
    Table,
    TableStyle,
    PageBreak,
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER

from ml_engine.attack_knowledge_base import get_attack_info


# -
# Shared helpers
# -
RESULTS_DIR = os.path.join(settings.MEDIA_ROOT, "results")


def _styles():
    base = getSampleStyleSheet()
    base.add(ParagraphStyle(
        name="CenteredNormal", parent=base["Normal"], alignment=TA_CENTER
    ))
    base.add(ParagraphStyle(
        name="SubHeading", parent=base["Heading3"],
        spaceBefore=12, spaceAfter=6, textColor=colors.HexColor("#1e3a5f")
    ))
    base.add(ParagraphStyle(
        name="SmallCenter", parent=base["Normal"],
        alignment=TA_CENTER, fontSize=8
    ))
    return base


def _table_style():
    return TableStyle([
        ("BackgroundColor", (0, 0), (-1, 0), colors.HexColor("#2563eb")),
        ("TEXTCOLOR",       (0, 0), (-1, 0), colors.white),
        ("FONTNAME",        (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",        (0, 0), (-1, 0), 9),
        ("ALIGN",           (0, 0), (-1, -1), "CENTER"),
        ("VALIGN",          (0, 0), (-1, -1), "MIDDLE"),
        ("FONTNAME",        (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE",        (0, 1), (-1, -1), 8),
        ("GRID",            (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ("ROWBACKGROUNDS",  (0, 1), (-1, -1), [colors.white, colors.HexColor("#f1f5f9")]),
        ("TOPPADDING",      (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING",   (0, 0), (-1, -1), 5),
        ("LEFTPADDING",     (0, 0), (-1, -1), 6),
        ("RIGHTPADDING",    (0, 0), (-1, -1), 6),
    ])


def _add_image_if_exists(el, path, caption, styles, width=440, height=280):
    if os.path.exists(path):
        el.append(Spacer(1, 8))
        el.append(Image(path, width=width, height=height))
        el.append(Paragraph(caption, styles["SmallCenter"]))
        el.append(Spacer(1, 12))


# -
# Public API
# -
def generate_pdf_report(output_path, dataset_results, overall_metrics):
    """
    Build the full PDF.

    Parameters
    
    output_path     : str
    dataset_results : list[dict]  – one entry per dataset (from train_eval)
    overall_metrics : dict        – avg / max / min accuracy
    """
    doc = SimpleDocTemplate(
        output_path, pagesize=A4,
        rightMargin=0.6*inch, leftMargin=0.6*inch,
        topMargin=0.6*inch,   bottomMargin=0.6*inch
    )
    styles = _styles()
    el = []

    # ===
    # TITLE
    # ===
    el.append(Spacer(1, 40))
    el.append(Paragraph(
        "<b>Intrusion Detection System</b>",
        ParagraphStyle("BigTitle", parent=styles["Title"],
                       fontSize=26, spaceBefore=0, spaceAfter=4,
                       textColor=colors.HexColor("#1e3a5f"))
    ))
    el.append(Paragraph("Analysis Report", styles["CenteredNormal"]))
    el.append(Spacer(1, 8))
    el.append(Paragraph(
        f"Generated: {datetime.now().strftime('%B %d, %Y - %H:%M')}",
        styles["SmallCenter"]
    ))
    el.append(Paragraph("CNN + BiLSTM + Attention Architecture", styles["SmallCenter"]))
    el.append(Spacer(1, 30))

    # ===
    # OVERALL SUMMARY
    # ===
    el.append(Paragraph("<b>Overall Performance Summary</b>", styles["Heading2"]))

    summary_data = [
        ["Metric", "Value"],
        ["Average Accuracy", f"{overall_metrics.get('average_accuracy', 0) * 100:.2f} %"],
        ["Best Accuracy",    f"{overall_metrics.get('max_accuracy',      0) * 100:.2f} %"],
        ["Lowest Accuracy",  f"{overall_metrics.get('min_accuracy',      0) * 100:.2f} %"],
        ["Datasets Analysed", str(len(dataset_results))],
    ]
    t = Table(summary_data, colWidths=[300, 150])
    t.setStyle(_table_style())
    el.append(t)
    el.append(Spacer(1, 20))
    el.append(PageBreak())

    # ===
    # PER-DATASET SECTIONS
    # ===
    for result in dataset_results:
        ds_name      = result["dataset"]
        ds_dir       = os.path.join(RESULTS_DIR, ds_name)
        accuracy     = result.get("accuracy")
        is_labelled  = result.get("is_labelled", True)
        rows         = result.get("rows", 0)
        columns      = result.get("columns", 0)
        size_mb      = result.get("size_mb", 0)
        cls_report   = result.get("classification_report")

        # - heading
        el.append(Paragraph(f"<b>Dataset: {ds_name}</b>", styles["Heading2"]))

        # - quick-facts
        facts = [
            ["Property",       "Value"],
            ["Total Samples",  str(rows)],
            ["Features",       str(columns)],
            ["Size",           f"{size_mb} MB"],
            ["Labelled",       "Yes" if is_labelled else "No"],
        ]
        if accuracy is not None:
            facts.insert(1, ["Accuracy", f"{accuracy * 100:.2f} %"])

        t = Table(facts, colWidths=[200, 100])
        t.setStyle(_table_style())
        el.append(t)
        el.append(Spacer(1, 14))

        # - classification report (labelled datasets only)
        if is_labelled and cls_report:
            el.append(Paragraph("<b>Classification Report</b>", styles["SubHeading"]))

            cr_rows = [["Class", "Attack Type", "Precision", "Recall", "F1-Score", "Support"]]

            for key in sorted(cls_report.keys()):
                if key in ("accuracy", "macro avg", "weighted avg"):
                    continue
                info     = get_attack_info(int(key))
                row_data = cls_report[key]
                cr_rows.append([
                    key,
                    info["attack_type"],
                    f"{row_data.get('precision', 0):.3f}",
                    f"{row_data.get('recall',    0):.3f}",
                    f"{row_data.get('f1-score',  0):.3f}",
                    str(int(row_data.get('support', 0)))
                ])

            for avg_key in ("macro avg", "weighted avg"):
                if avg_key in cls_report:
                    row_data = cls_report[avg_key]
                    cr_rows.append([
                        "",
                        avg_key.title(),
                        f"{row_data.get('precision', 0):.3f}",
                        f"{row_data.get('recall',    0):.3f}",
                        f"{row_data.get('f1-score',  0):.3f}",
                        str(int(row_data.get('support', 0)))
                    ])

            t     = Table(cr_rows, colWidths=[40, 100, 70, 70, 70, 70])
            style = _table_style()
            avg_start = len(cr_rows) - 2
            style.add("FONTNAME",        (0, avg_start), (-1, -1), "Helvetica-Bold")
            style.add("BackgroundColor",  (0, avg_start), (-1, -1), colors.HexColor("#e0e7ff"))
            t.setStyle(style)
            el.append(t)
            el.append(Spacer(1, 16))

        # - visualisations (each added only when the file exists)
        _add_image_if_exists(
            el, os.path.join(ds_dir, "confusion_matrix.png"),
            "Confusion Matrix", styles, width=380, height=300
        )
        _add_image_if_exists(
            el, os.path.join(ds_dir, "accuracy_curve.png"),
            "Training vs Validation Accuracy", styles, width=420, height=240
        )
        _add_image_if_exists(
            el, os.path.join(ds_dir, "loss_curve.png"),
            "Training vs Validation Loss", styles, width=420, height=240
        )
        _add_image_if_exists(
            el, os.path.join(ds_dir, "prediction_analysis.png"),
            "Prediction Distribution & Per-Class Accuracy", styles, width=440, height=240
        )

        el.append(PageBreak())

    
    # FOOTER
    
    el.append(Spacer(1, 30))
    el.append(Paragraph(
        "<i>This report was auto-generated by the IDS Platform. "
        "All visualisations reflect the latest evaluation run per dataset.</i>",
        styles["SmallCenter"]
    ))

    doc.build(el)