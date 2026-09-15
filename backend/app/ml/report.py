"""
MoodMax Executive PDF Report Generator.
Stateless, in-memory PDF generation using ReportLab and Matplotlib.
Zero disk writes: streams raw bytes directly via io.BytesIO.
"""

import io
import datetime
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # Thread-safe headless rendering
import matplotlib.pyplot as plt
import numpy as np

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image,
    PageBreak,
    HRFlowable,
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import os
from pathlib import Path

# Register Fonts
try:
    _font_dir = Path(__file__).resolve().parent.parent / "assets" / "fonts"
    pdfmetrics.registerFont(TTFont("NotoSans", str(_font_dir / "NotoSans-Regular.ttf")))
    pdfmetrics.registerFont(TTFont("NotoSans-Bold", str(_font_dir / "NotoSans-Bold.ttf")))
    pdfmetrics.registerFont(TTFont("NotoSansDevanagari", str(_font_dir / "NotoSansDevanagari-Regular.ttf")))
    pdfmetrics.registerFont(TTFont("NotoSansDevanagari-Bold", str(_font_dir / "NotoSansDevanagari-Bold.ttf")))
    FONT_NORMAL = "NotoSans"
    FONT_BOLD = "NotoSans-Bold"
    FONT_DEV_NORMAL = "NotoSansDevanagari"
    FONT_DEV_BOLD = "NotoSansDevanagari-Bold"
except Exception as e:
    FONT_NORMAL = "Helvetica"
    FONT_BOLD = "Helvetica-Bold"
    FONT_DEV_NORMAL = "Helvetica"
    FONT_DEV_BOLD = "Helvetica-Bold"


# Brand Color Palette (matching frontend/src/index.css)
COLOR_PRIMARY = colors.HexColor("#267F4B")       # Forest Green
COLOR_PRIMARY_DARK = colors.HexColor("#1B5E36")  # Dark Green
COLOR_PRIMARY_LIGHT = colors.HexColor("#EAF5EE") # Soft Mint Tint
COLOR_DARK = colors.HexColor("#172033")          # Deep Navy
COLOR_MUTED = colors.HexColor("#64748B")         # Cool Slate
COLOR_BORDER = colors.HexColor("#E5E7F2")        # Card Border
COLOR_BG_CARD = colors.HexColor("#F8FAFC")       # Slate light
COLOR_RED = colors.HexColor("#DC2626")           # Negative Red
COLOR_GREEN = colors.HexColor("#16A34A")         # Positive Green
COLOR_AMBER = colors.HexColor("#D97706")         # Neutral/Warning Amber

EMOTION_HEX_MAP = {
    "joy": "#E5A100",
    "neutral": "#64748B",
    "surprise": "#DB2777",
    "anger": "#DC2626",
    "sadness": "#2563EB",
    "disgust": "#059669",
    "fear": "#7C3AED",
}


class NumberedCanvas(canvas.Canvas):
    """Two-pass canvas for precise 'Page X of Y' numbering and running header/footer."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count):
        self.saveState()

        # Top Accent Bar (Brand Forest Green)
        self.setFillColor(COLOR_PRIMARY)
        self.rect(0, 786, 612, 6, fill=True, stroke=False)

        # Running Footer
        self.setFont(FONT_NORMAL, 8)
        self.setFillColor(COLOR_MUTED)
        self.drawString(36, 25, "MoodMax™ Applied AI Intelligence — Confidential & Proprietary")
        self.drawRightString(576, 25, f"Page {self._pageNumber} of {page_count}")

        # Footer divider rule
        self.setStrokeColor(COLOR_BORDER)
        self.setLineWidth(0.5)
        self.line(36, 36, 576, 36)

        self.restoreState()


def render_sentiment_donut(sentiment_breakdown: dict) -> io.BytesIO:
    """Render an in-memory donut chart for sentiment distribution."""
    fig, ax = plt.subplots(figsize=(3.4, 2.3), dpi=200)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    labels = ["Positive", "Neutral", "Negative"]
    values = [
        sentiment_breakdown.get("Positive", 0),
        sentiment_breakdown.get("Neutral", 0),
        sentiment_breakdown.get("Negative", 0),
    ]
    palette = ["#16A34A", "#94A3B8", "#DC2626"]

    # If all zero, default
    if sum(values) == 0:
        values = [1, 1, 1]

    wedges, texts, autotexts = ax.pie(
        values,
        labels=labels,
        colors=palette,
        autopct="%1.0f%%",
        startangle=140,
        pctdistance=0.75,
        textprops={"fontsize": 8, "color": "#172033", "weight": "bold"},
        wedgeprops={"width": 0.42, "edgecolor": "white", "linewidth": 2},
    )
    for at in autotexts:
        at.set_color("white")
        at.set_fontsize(7.5)
        at.set_weight("bold")

    ax.axis("equal")
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor(), transparent=False)
    plt.close(fig)
    buf.seek(0)
    return buf


def render_emotion_radar(emotion_breakdown: dict) -> io.BytesIO:
    """Render an in-memory radar / spider chart for 7-class emotion distribution."""
    classes = ["joy", "surprise", "neutral", "sadness", "fear", "anger", "disgust"]
    display_names = ["Joy", "Surprise", "Neutral", "Sadness", "Fear", "Anger", "Disgust"]

    total = max(sum(emotion_breakdown.values()), 1)
    values = [(emotion_breakdown.get(c, 0) / total) * 100 for c in classes]

    # Close the polygon
    values += values[:1]
    angles = np.linspace(0, 2 * np.pi, len(classes), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(3.4, 2.3), subplot_kw=dict(polar=True), dpi=200)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#F8FAFC")

    ax.plot(angles, values, color="#267F4B", linewidth=2)
    ax.fill(angles, values, color="#267F4B", alpha=0.25)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(display_names, fontsize=7.5, color="#172033", weight="bold")
    ax.tick_params(pad=3)

    ax.set_yticks([10, 25, 40])
    ax.set_yticklabels(["10%", "25%", "40%"], fontsize=6, color="#64748B")
    ax.grid(color="#E2E8F0", linestyle="--", linewidth=0.6)
    ax.spines["polar"].set_color("#CBD5E1")

    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor(), transparent=False)
    plt.close(fig)
    buf.seek(0)
    return buf


def generate_actionable_takeaways(emotion_breakdown: dict, sentiment_breakdown: dict, total: int) -> list[str]:
    """Deterministic rule-based business takeaways based on aggregate metrics."""
    takeaways = []
    tot = max(total, 1)

    anger_pct = (emotion_breakdown.get("anger", 0) / tot) * 100
    disgust_pct = (emotion_breakdown.get("disgust", 0) / tot) * 100
    fear_pct = (emotion_breakdown.get("fear", 0) / tot) * 100
    joy_pct = (emotion_breakdown.get("joy", 0) / tot) * 100
    sadness_pct = (emotion_breakdown.get("sadness", 0) / tot) * 100
    pos_pct = (sentiment_breakdown.get("Positive", 0) / tot) * 100
    neg_pct = (sentiment_breakdown.get("Negative", 0) / tot) * 100

    # Rule 1: High anger / escalation
    if anger_pct >= 12.0 or neg_pct >= 30.0:
        takeaways.append(
            f"<b>Urgent Support Escalation ({anger_pct:.1f}% Anger):</b> High customer friction identified. "
            "Recommend auditing recent support ticket response times and prioritizing angry customer outreach."
        )
    # Rule 2: High disgust / product rejection
    if disgust_pct >= 6.0:
        takeaways.append(
            f"<b>Quality & Brand Audit ({disgust_pct:.1f}% Disgust):</b> Elevated revulsion/distaste detected. "
            "Examine product defect rates, recent UI redesign changes, or delivery hygiene."
        )
    # Rule 3: High fear / trust hesitation
    if fear_pct >= 4.0:
        takeaways.append(
            f"<b>Trust & Transparency Clarification ({fear_pct:.1f}% Fear):</b> Anxiety signals detected. "
            "Review payment gateway notices, security FAQs, and return/refund policies to reassure customers."
        )
    # Rule 4: High sadness / let-down
    if sadness_pct >= 10.0:
        takeaways.append(
            f"<b>Customer Retention Alert ({sadness_pct:.1f}% Sadness):</b> High disappointment signals churn risk. "
            "Deploy proactive re-engagement incentives and personalized apology workflows."
        )
    # Rule 5: High joy / brand advocates
    if joy_pct >= 35.0 or pos_pct >= 50.0:
        takeaways.append(
            f"<b>Advocacy & Referral Opportunity ({joy_pct:.1f}% Joy):</b> Strong organic delight detected. "
            "Capitalize on positive sentiment with targeted review requests, referral rewards, and testimonial campaigns."
        )

    # Fallback if feedback is predominantly neutral
    if len(takeaways) < 2:
        takeaways.append(
            "<b>Informational Engagement:</b> Significant neutral feedback indicates customers are seeking clarity or making factual inquiries. "
            "Enhance self-serve knowledge base articles and documentation."
        )

    return takeaways[:3]


def generate_executive_pdf_report(
    filename: Optional[str],
    summary: dict,
    results: list[dict],
) -> io.BytesIO:
    """
    Generate an executive 2-page PDF report entirely in-memory.
    Returns io.BytesIO containing the complete PDF binary.
    """
    output_buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        output_buffer,
        pagesize=letter,
        leftMargin=36,
        rightMargin=36,
        topMargin=36,
        bottomMargin=42,
    )

    styles = getSampleStyleSheet()

    # Custom Typography Styles
    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Normal"],
        fontName=FONT_BOLD,
        fontSize=20,
        leading=24,
        textColor=COLOR_DARK,
    )
    subtitle_style = ParagraphStyle(
        "DocSubtitle",
        parent=styles["Normal"],
        fontName=FONT_NORMAL,
        fontSize=9,
        leading=12,
        textColor=COLOR_MUTED,
    )
    section_heading = ParagraphStyle(
        "SectionHeading",
        parent=styles["Normal"],
        fontName=FONT_BOLD,
        fontSize=12,
        leading=16,
        textColor=COLOR_PRIMARY_DARK,
        spaceBefore=8,
        spaceAfter=6,
    )
    card_title = ParagraphStyle(
        "CardTitle",
        parent=styles["Normal"],
        fontName=FONT_NORMAL,
        fontSize=8,
        leading=10,
        textColor=COLOR_MUTED,
    )
    card_value = ParagraphStyle(
        "CardValue",
        parent=styles["Normal"],
        fontName=FONT_BOLD,
        fontSize=16,
        leading=18,
        textColor=COLOR_DARK,
    )
    card_sub = ParagraphStyle(
        "CardSub",
        parent=styles["Normal"],
        fontName=FONT_NORMAL,
        fontSize=7.5,
        leading=9,
        textColor=COLOR_PRIMARY,
    )
    body_style = ParagraphStyle(
        "Body",
        parent=styles["Normal"],
        fontName=FONT_NORMAL,
        fontSize=8.5,
        leading=11,
        textColor=COLOR_DARK,
    )
    grievance_style = ParagraphStyle(
        "GrievanceText",
        parent=styles["Normal"],
        fontName=FONT_DEV_NORMAL,
        fontSize=8,
        leading=10.5,
        textColor=COLOR_DARK,
    )

    story = []

    # ─────────────────────────────────────────────────────────────
    # PAGE 1: Executive Summary & Overview
    # ─────────────────────────────────────────────────────────────

    # Header Row
    timestamp_str = datetime.datetime.now(datetime.timezone.utc).strftime("%B %d, %Y • %H:%M UTC")
    clean_name = filename or "Batch Upload"
    header_data = [
        [
            Paragraph("<b>MoodMax™</b> <font color='#267F4B'>Executive Audit Report</font>", title_style),
            Paragraph(f"<b>Generated:</b> {timestamp_str}<br/><b>Source:</b> {clean_name}", subtitle_style),
        ]
    ]
    header_table = Table(header_data, colWidths=[360, 180])
    header_table.setStyle(
        TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("ALIGN", (1, 0), (1, 0), "RIGHT"),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
        ])
    )
    story.append(header_table)
    story.append(Spacer(1, 10))
    story.append(HRFlowable(width="100%", thickness=1, color=COLOR_BORDER, spaceBefore=2, spaceAfter=10))

    # Executive KPI Cards Table
    total_analyses = summary.get("total_analyses", len(results))
    sentiment_breakdown = summary.get("sentiment_breakdown", {})
    emotion_breakdown = summary.get("emotion_breakdown", {})
    top_languages = summary.get("top_languages", {})

    pos_count = sentiment_breakdown.get("Positive", 0)
    neg_count = sentiment_breakdown.get("Negative", 0)
    pos_pct = (pos_count / max(total_analyses, 1)) * 100
    neg_pct = (neg_count / max(total_analyses, 1)) * 100

    dominant_overall = max(emotion_breakdown, key=emotion_breakdown.get) if emotion_breakdown else "joy"
    dom_count = emotion_breakdown.get(dominant_overall, 0)
    dom_pct = (dom_count / max(total_analyses, 1)) * 100

    lang_summary = ", ".join([f"{k.upper()} ({v})" for k, v in list(top_languages.items())[:3]]) or "EN"

    kpi_cards = [
        [
            Paragraph("TOTAL ANALYZED", card_title),
            Paragraph("DOMINANT EMOTION", card_title),
            Paragraph("SENTIMENT PROFILE", card_title),
            Paragraph("TOP LANGUAGES", card_title),
        ],
        [
            Paragraph(f"{total_analyses:,}", card_value),
            Paragraph(f"{dominant_overall.capitalize()}", card_value),
            Paragraph(f"{pos_pct:.0f}% Pos / {neg_pct:.0f}% Neg", card_value),
            Paragraph(f"{lang_summary}", ParagraphStyle("LangVal", parent=card_value, fontSize=11, leading=13)),
        ],
        [
            Paragraph("100% In-Memory Processed", card_sub),
            Paragraph(f"{dom_pct:.1f}% share of batch", card_sub),
            Paragraph("Calibrated Probability Vectors", card_sub),
            Paragraph("fastText 176-Language Engine", card_sub),
        ],
    ]

    kpi_table = Table(kpi_cards, colWidths=[130, 135, 140, 135])
    kpi_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), COLOR_BG_CARD),
            ("BOX", (0, 0), (-1, -1), 1, COLOR_BORDER),
            ("INNERGRID", (0, 0), (-1, -1), 0.5, COLOR_BORDER),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ])
    )
    story.append(kpi_table)
    story.append(Spacer(1, 14))

    # Visual Analytics Section: Charts Side-by-Side
    story.append(Paragraph("Visual Sentiment & Emotion Analytics", section_heading))

    donut_buf = render_sentiment_donut(sentiment_breakdown)
    radar_buf = render_emotion_radar(emotion_breakdown)

    chart_table = Table(
        [
            [
                Paragraph("<b>Ternary Sentiment Distribution</b>", ParagraphStyle("CT1", parent=body_style, alignment=1, textColor=COLOR_MUTED)),
                Paragraph("<b>7-Class Ekman Emotion Distribution</b>", ParagraphStyle("CT2", parent=body_style, alignment=1, textColor=COLOR_MUTED)),
            ],
            [
                Image(donut_buf, width=255, height=170),
                Image(radar_buf, width=255, height=170),
            ],
        ],
        colWidths=[270, 270],
    )
    chart_table.setStyle(
        TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("BOX", (0, 0), (0, 1), 1, COLOR_BORDER),
            ("BOX", (1, 0), (1, 1), 1, COLOR_BORDER),
            ("BACKGROUND", (0, 0), (-1, -1), colors.white),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ])
    )
    story.append(chart_table)
    story.append(Spacer(1, 14))

    # Executive Overview Paragraph
    overview_text = (
        f"This executive intelligence report synthesizes <b>{total_analyses:,} unstructured customer inputs</b> "
        f"using MoodMax's fine-tuned multilingual transformer engine (135.3M parameters). "
        f"The batch demonstrates a <b>{pos_pct:.1f}% positive</b> and <b>{neg_pct:.1f}% negative</b> sentiment split, "
        f"anchored primarily around <b>{dominant_overall.capitalize()}</b> ({dom_pct:.1f}%). "
        "Post-hoc temperature scaling (T=1.1724) and per-class decision boundaries ensure probability fidelity across all categories."
    )
    story.append(Paragraph(overview_text, body_style))

    # ─────────────────────────────────────────────────────────────
    # PAGE 2: Grievances, Praises & Strategic Action Items
    # ─────────────────────────────────────────────────────────────
    story.append(PageBreak())

    story.append(Paragraph("<b>MoodMax™</b> <font color='#267F4B'>Customer Voice & Action Plan</font>", title_style))
    story.append(Paragraph("Granular Grievances, Brand Praises, and Heuristic Action Items", subtitle_style))
    story.append(Spacer(1, 8))
    story.append(HRFlowable(width="100%", thickness=1, color=COLOR_BORDER, spaceBefore=2, spaceAfter=8))


    # Strategic Actionable Takeaways (Rule-Based Heuristics)
    story.append(Paragraph("🎯 Strategic Actionable Takeaways", section_heading))

    takeaways = generate_actionable_takeaways(emotion_breakdown, sentiment_breakdown, total_analyses)
    takeaway_cards = []
    for t in takeaways:
        takeaway_cards.append([Paragraph(f"• {t}", body_style)])

    t_table = Table(takeaway_cards, colWidths=[540])
    t_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), COLOR_BG_CARD),
            ("BOX", (0, 0), (-1, -1), 1, COLOR_BORDER),
            ("INNERGRID", (0, 0), (-1, -1), 0.5, COLOR_BORDER),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ])
    )
    story.append(t_table)

    # Build the document
    doc.build(story, canvasmaker=NumberedCanvas)
    output_buffer.seek(0)
    return output_buffer
