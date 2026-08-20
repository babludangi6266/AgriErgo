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
            output_path if output_path else buffer,
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
        elements.append(Paragraph("🌾 AgriErgo Assessment Report", title_style))
        elements.append(Paragraph(
            f"Video-Based Farm Worker Ergonomics & Drudgery Evaluation | Generated: {datetime.now().strftime('%B %d, %Y - %H:%M')}",
            subtitle_style
        ))
        elements.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#3B82F6"), spaceAfter=15))

        # ── Video Info & Metadata Table ──
        elements.append(Paragraph("1. Video & Dataset Metadata", section_heading))
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
            elements.append(Paragraph(f"2. Worker #{wr.worker_id} Assessment — <i>{task_title}</i>", section_heading))

            # Ergonomic & Drudgery Scorecard Cards
            rula = getattr(wr, 'rula_score', None) or 1
            reba = wr.reba_score or 1
            adi = getattr(wr, 'drudgery_index', 0.0) or 0.0
            adi_cat = getattr(wr, 'drudgery_category', 'Low Drudgery') or 'Low Drudgery'
            l5s1 = getattr(wr, 'l5s1_compression_n', 0.0) or 0.0
            rwl = getattr(wr, 'niosh_rwl_kg', 0.0) or 0.0

            card_data = [
                [
                    Paragraph("<b>REBA Score (Body):</b>", body_style),
                    Paragraph(f"<b>{reba}</b> ({wr.reba_risk_level or 'N/A'})", bold_body),
                    Paragraph("<b>RULA Score (Upper):</b>", body_style),
                    Paragraph(f"<b>{rula}</b>", bold_body),
                ],
                [
                    Paragraph("<b>Drudgery Index (ADI):</b>", body_style),
                    Paragraph(f"<b>{adi} / 100</b> ({adi_cat})", bold_body),
                    Paragraph("<b>L5/S1 Compression:</b>", body_style),
                    Paragraph(f"<b>{l5s1} N</b> ({'EXCEEDED 3.4kN' if l5s1 > 3400 else 'Safe Limit'})", bold_body),
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
            elements.append(Spacer(1, 12))

            # 11 Parameters Breakdown Table
            elements.append(Paragraph(f"Worker #{wr.worker_id} Parameter Matrix", bold_body))
            param_rows = [
                [Paragraph("<b>Parameter</b>", bold_body), Paragraph("<b>Measurement / Value</b>", bold_body)],
                [Paragraph("1. Sitting Duration", body_style), Paragraph(f"{wr.sitting_duration}s ({round(wr.sitting_duration/max(1, wr.total_tracked_time)*100,1)}%)", body_style)],
                [Paragraph("1b. Squatting Duration", body_style), Paragraph(f"{getattr(wr, 'squatting_duration', 0.0)}s ({round(getattr(wr, 'squatting_duration', 0.0)/max(1, wr.total_tracked_time)*100,1)}%)", body_style)],
                [Paragraph("2. Standing Duration", body_style), Paragraph(f"{wr.standing_duration}s ({round(wr.standing_duration/max(1, wr.total_tracked_time)*100,1)}%)", body_style)],
                [Paragraph("3. Bending Duration", body_style), Paragraph(f"{wr.bending_duration}s ({round(wr.bending_duration/max(1, wr.total_tracked_time)*100,1)}%)", body_style)],
                [Paragraph("4. Walking Duration", body_style), Paragraph(f"{getattr(wr, 'walking_duration', 0.0)}s ({round(getattr(wr, 'walking_duration', 0.0)/max(1, wr.total_tracked_time)*100,1)}%)", body_style)],
                [Paragraph("5. Carried Load Events", body_style), Paragraph(f"{wr.total_load_events} load events", body_style)],
                [Paragraph("6. Repetitive Motion", body_style), Paragraph(f"{wr.repetitive_movement.cycles_per_minute} cycles/min" if wr.repetitive_movement and wr.repetitive_movement.is_repetitive else "None detected", body_style)],
                [Paragraph("7. Field Trips Count", body_style), Paragraph(f"{wr.trip_count_result.trip_count if wr.trip_count_result else 0} trips", body_style)],
                [Paragraph("8. Tools/Equipment Detected", body_style), Paragraph(", ".join([t.tool_name for t in wr.tools_used]) if wr.tools_used else "None", body_style)],
                [Paragraph("9. Trunk Flexion Angle", body_style), Paragraph(f"Avg: {wr.posture_summary.avg_trunk_flexion if wr.posture_summary else 'N/A'}° | Max: {wr.posture_summary.max_trunk_flexion if wr.posture_summary else 'N/A'}°", body_style)],
                [Paragraph("10. Continuous Work Bout", body_style), Paragraph(f"Longest: {wr.longest_work_bout}s | Avg: {wr.avg_work_bout}s", body_style)],
                [Paragraph("11. Rest Duration & Count", body_style), Paragraph(f"Total Rest: {wr.total_rest_duration}s ({wr.rest_count} rest periods)", body_style)],
            ]
            p_table = Table(param_rows, colWidths=[200, 320])
            p_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#E2E8F0")),
                ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
                ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#F1F5F9")),
                ('PADDING', (0, 0), (-1, -1), 5),
            ]))
            elements.append(p_table)
            elements.append(Spacer(1, 15))

        doc.build(elements)
        
        pdf_bytes = buffer.getvalue()
        buffer.close()
        return pdf_bytes
