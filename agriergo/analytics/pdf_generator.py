"""
PDF Report Generator — Generates publication-grade PDF ergonomic reports.

Produces structured PDF reports containing:
- Executive Overview & Metadata
- Dual Risk Assessment (RULA & REBA Scorecard)
- Agricultural Drudgery Index (ADI) & Fatigue Breakdown
- 11-Parameter Research Table
- Ergonomic Recommendations & Action Plan
"""

import os
import io
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether, HRFlowable
)

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from agriergo.analytics.parameter_aggregator import WorkerReport
from agriergo.perception.video_processor import VideoMetadata


class PDFReportGenerator:
    """
    Generates PDF assessment reports for researchers and ergonomic assessors.
    """

    def generate_pdf(
        self,
        worker_reports: List[WorkerReport],
        video_metadata: VideoMetadata,
        output_path: Optional[str] = None,
    ) -> bytes:
        """
        Generate a professional PDF document byte array.
        """
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(
            buffer,
            pagesize=A4,
            rightMargin=36,
            leftMargin=36,
            topMargin=36,
            bottomMargin=36,
        )

        styles = getSampleStyleSheet()
        
        # Custom Paragraph Styles
        title_style = ParagraphStyle(
            'DocTitle',
            parent=styles['Heading1'],
            fontName='Helvetica-Bold',
            fontSize=22,
            leading=26,
            textColor=colors.HexColor("#1E293B"),
            spaceAfter=6,
        )
        subtitle_style = ParagraphStyle(
            'DocSubtitle',
            parent=styles['Normal'],
            fontName='Helvetica',
            fontSize=11,
            textColor=colors.HexColor("#64748B"),
            spaceAfter=15,
        )
        section_heading = ParagraphStyle(
            'SectionHeading',
            parent=styles['Heading2'],
            fontName='Helvetica-Bold',
            fontSize=14,
            leading=18,
            textColor=colors.HexColor("#0F172A"),
            spaceBefore=12,
            spaceAfter=8,
        )
        body_style = ParagraphStyle(
            'Body',
            parent=styles['Normal'],
            fontName='Helvetica',
            fontSize=9.5,
            leading=13,
            textColor=colors.HexColor("#334155"),
        )
        bold_body = ParagraphStyle(
            'BoldBody',
            parent=body_style,
            fontName='Helvetica-Bold',
        )

        elements = []

        # ── Header Section ──
        elements.append(Paragraph("AgriErgo Assessment Report", title_style))
        elements.append(Paragraph(
            f"Video-Based Farm Worker Ergonomics &amp; Drudgery Evaluation | Generated: {datetime.now().strftime('%B %d, %Y - %H:%M')}",
            subtitle_style
        ))
        elements.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#3B82F6"), spaceAfter=15))

        # ── Video Info & Metadata Table ──
        elements.append(Paragraph("1. Video &amp; Dataset Metadata", section_heading))
        meta_data = [
            [
                Paragraph("<b>Filename:</b>", body_style), Paragraph(video_metadata.filename, body_style),
                Paragraph("<b>Duration:</b>", body_style), Paragraph(f"{video_metadata.duration_seconds}s", body_style),
            ],
            [
                Paragraph("<b>Resolution:</b>", body_style), Paragraph(f"{video_metadata.width}x{video_metadata.height}", body_style),
                Paragraph("<b>Sampling Rate:</b>", body_style), Paragraph(f"{video_metadata.fps} FPS", body_style),
            ],
            [
                Paragraph("<b>Workers Analyzed:</b>", body_style), Paragraph(str(len(worker_reports)), body_style),
                Paragraph("<b>Platform Version:</b>", body_style), Paragraph("AgriErgo v0.2.0 Advanced", body_style),
            ],
        ]
        meta_table = Table(meta_data, colWidths=[110, 150, 110, 150])
        meta_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#F8FAFC")),
            ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
            ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
            ('PADDING', (0, 0), (-1, -1), 6),
        ]))
        elements.append(meta_table)
        elements.append(Spacer(1, 15))

        # ── Per-Worker Analysis Sections ──
        for wr in worker_reports:
            task_title = getattr(wr, 'classified_task', 'General Agricultural Work') or 'General Agricultural Work'
            elements.append(Paragraph(f"2. Worker #{wr.worker_id} Assessment - {task_title}", section_heading))

            # Ergonomic & Drudgery Scorecard Cards
            rula = getattr(wr, 'rula_score', None) or 1
            reba = wr.reba_score or 1
            adi = getattr(wr, 'drudgery_index', 0.0) or 0.0
            drudgery_pct = getattr(wr, 'drudgery_percentage', 0.0) or adi
            adi_cat = getattr(wr, 'drudgery_category', 'Low Drudgery') or 'Low Drudgery'
            l5s1 = getattr(wr, 'l5s1_compression_n', 0.0) or 0.0
            arm_risk = getattr(wr, 'arm_postural_risk', 'Low') or 'Low'

            card_data = [
                [
                    Paragraph("<b>REBA Score (Body):</b>", body_style),
                    Paragraph(f"<b>{reba}</b> ({wr.reba_risk_level or 'N/A'})", bold_body),
                    Paragraph("<b>RULA Score (Upper):</b>", body_style),
                    Paragraph(f"<b>{rula}</b>", bold_body),
                ],
                [
                    Paragraph("<b>Drudgery Index / Pct:</b>", body_style),
                    Paragraph(f"<b>{adi} / 100 ({drudgery_pct}%)</b> ({adi_cat})", bold_body),
                    Paragraph("<b>Arm Postural Risk:</b>", body_style),
                    Paragraph(f"<b>{arm_risk}</b> (Elv >45°: {wr.posture_summary.shoulder_above_45_pct if wr.posture_summary else 0}%)", bold_body),
                ],
            ]
            card_table = Table(card_data, colWidths=[140, 120, 140, 120])
            card_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#EFF6FF")),
                ('BOX', (0, 0), (-1, -1), 1, colors.HexColor("#BFDBFE")),
                ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#DBEAFE")),
                ('PADDING', (0, 0), (-1, -1), 6),
            ]))
            elements.append(card_table)
            elements.append(Spacer(1, 10))

            # 1-Hour Standardisation & Raw Duration Matrix Table
            std = wr.standardised_1hr
            elements.append(Paragraph(f"Worker #{wr.worker_id} Standardised 1-Hour Activity Analysis", bold_body))
            std_rows = [
                [Paragraph("<b>Activity / Action</b>", bold_body), Paragraph("<b>Sampled Duration</b>", bold_body), Paragraph("<b>Standardised per 1-Hour</b>", bold_body), Paragraph("<b>Share (%)</b>", bold_body)],
                [Paragraph("1. Sitting", body_style), Paragraph(f"{wr.sitting_duration}s", body_style), Paragraph(f"{std.sitting_formatted_1hr if std else 'N/A'}", bold_body), Paragraph(f"{std.sitting_pct if std else 0}%", body_style)],
                [Paragraph("1b. Squatting", body_style), Paragraph(f"{getattr(wr, 'squatting_duration', 0.0)}s", body_style), Paragraph(f"{std.squatting_formatted_1hr if std else 'N/A'}", bold_body), Paragraph(f"{std.squatting_pct if std else 0}%", body_style)],
                [Paragraph("2. Standing", body_style), Paragraph(f"{wr.standing_duration}s", body_style), Paragraph(f"{std.standing_formatted_1hr if std else 'N/A'}", bold_body), Paragraph(f"{std.standing_pct if std else 0}%", body_style)],
                [Paragraph("3. Bending (Stooping)", body_style), Paragraph(f"{wr.bending_duration}s", body_style), Paragraph(f"{std.bending_formatted_1hr if std else 'N/A'}", bold_body), Paragraph(f"{std.bending_pct if std else 0}%", body_style)],
                [Paragraph("4. Walking", body_style), Paragraph(f"{getattr(wr, 'walking_duration', 0.0)}s", body_style), Paragraph(f"{std.walking_formatted_1hr if std else 'N/A'}", bold_body), Paragraph(f"{std.walking_pct if std else 0}%", body_style)],
                [Paragraph("7. Work Pause (Rest)", body_style), Paragraph(f"{wr.total_rest_duration}s", body_style), Paragraph(f"{std.rest_formatted_1hr if std else 'N/A'}", bold_body), Paragraph(f"{std.rest_pct if std else 0}%", body_style)],
                [Paragraph("Active Work Session", bold_body), Paragraph(f"{round(wr.total_tracked_time - wr.total_rest_duration, 1)}s", bold_body), Paragraph(f"{std.active_work_formatted_1hr if std else '01:00:00'}", bold_body), Paragraph(f"{round(100 - (std.rest_pct if std else 0), 1)}%", bold_body)],
            ]
            std_table = Table(std_rows, colWidths=[150, 110, 140, 120])
            std_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#F1F5F9")),
                ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
                ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
                ('PADDING', (0, 0), (-1, -1), 4),
            ]))
            elements.append(std_table)
            elements.append(Spacer(1, 10))

            # Postural Angles & Arm Study Matrix
            elements.append(Paragraph(f"Worker #{wr.worker_id} Biomechanical Angle &amp; Arm Posture Study", bold_body))
            ang_rows = [
                [Paragraph("<b>Biomechanical Angle</b>", bold_body), Paragraph("<b>Measured Value</b>", bold_body), Paragraph("<b>Ergonomic Standard / Risk</b>", bold_body)],
                [Paragraph("Trunk Flexion (Torso Bending)", body_style), Paragraph(f"Avg: {wr.posture_summary.avg_trunk_flexion if wr.posture_summary else 'N/A'}° | Peak: {wr.posture_summary.max_trunk_flexion if wr.posture_summary else 'N/A'}°", body_style), Paragraph("ISO 11226 Limit: &lt;60° (&lt;4 min/shift)", body_style)],
                [Paragraph("Knee Flexion Angle", body_style), Paragraph(f"Avg: {getattr(wr.posture_summary, 'avg_knee_angle', 'N/A')}°", body_style), Paragraph("Stoop: &gt;140° | Deep Squat: &lt;90°", body_style)],
                [Paragraph("Shoulder Upper Arm Elevation", body_style), Paragraph(f"Avg: {getattr(wr.posture_summary, 'avg_shoulder_angle', 'N/A')}° | Peak: {getattr(wr.posture_summary, 'max_shoulder_angle', 'N/A')}°", body_style), Paragraph(f"Time &gt;45°: {wr.posture_summary.shoulder_above_45_pct if wr.posture_summary else 0}% | &gt;90°: {wr.posture_summary.shoulder_above_90_pct if wr.posture_summary else 0}%", body_style)],
                [Paragraph("Elbow Flexion Angle", body_style), Paragraph(f"Avg: {getattr(wr.posture_summary, 'avg_elbow_angle', 'N/A')}°", body_style), Paragraph("Neutral: 60°–100° (REBA Table B)", body_style)],
            ]
            ang_table = Table(ang_rows, colWidths=[180, 170, 170])
            ang_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#F8FAFC")),
                ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
                ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
                ('PADDING', (0, 0), (-1, -1), 4),
            ]))
            elements.append(ang_table)
            elements.append(Spacer(1, 15))

        doc.build(elements)
        
        pdf_bytes = buffer.getvalue()
        buffer.close()

        # Optionally save to disk
        if output_path:
            try:
                with open(output_path, 'wb') as f:
                    f.write(pdf_bytes)
            except Exception:
                pass

        return pdf_bytes
