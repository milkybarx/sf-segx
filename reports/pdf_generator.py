"""
Automated Space Weather Alert Bulletin Generator (NOAA SWPC / NASA GSFC Style)
==============================================================================
Generates publication-quality Space Weather Advisory Bulletins in PDF format:
1. Executive Risk Summary & Active Filament Morphology Metrics
2. Eruption Probability & Predicted Flare Severity
3. Hydrodynamic CME Drag Transit Time & Geomagnetic Storm Warning (Kp Index)
4. Parker Spiral Magnetic Connectivity & Satellite Fleet Directives
"""

import os
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from datetime import datetime
from typing import Dict, Any, Optional


def generate_space_weather_pdf(
    report_data: Optional[Dict[str, Any]] = None,
    output_pdf_path: str = "outputs/reports/Space_Weather_Alert_Bulletin.pdf"
) -> str:
    """
    Builds a professional 2-page Space Weather Warning Bulletin in PDF format.
    """
    os.makedirs(os.path.dirname(output_pdf_path), exist_ok=True)

    if report_data is None:
        report_data = {
            'timestamp': datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
            'filament_loc': 'N19.4° W68.1° (Stonyhurst)',
            'filament_length_km': 142500,
            'filament_area_km2': 2.1e9,
            'shear_angle_deg': 68.2,
            'contrast_dip': 0.412,
            'eruption_24h': 45.8,
            'eruption_48h': 70.6,
            'risk_tier': 'CRITICAL ELEVATED RISK',
            'flare_class': 'M5.0 / X1.0',
            'cme_v0_kms': 1250.0,
            'cme_arrival_kms': 620.4,
            'cme_transit_hours': 34.2,
            'cme_arrival_utc': '2026-08-23 02:45 UTC',
            'kp_index': 7.0,
            'storm_scale': 'G3 (Strong)',
            'earth_footpoint': 'W61.5°',
            'delta_lon': '6.6° (Direct Hit)',
            'critical_sats': ['DSCOVR (L1)', 'JWST (L2)', 'GPS III (MEO)', 'NOAA GOES-16 (GEO)', 'Starlink (LEO)', 'ISS (LEO)'],
            'eva_directive': 'SUSPEND ALL EXTRAVEHICULAR ACTIVITIES (EVAs) & COMMAND CREW SHELTER'
        }

    with PdfPages(output_pdf_path) as pdf:
        # ── PAGE 1: EXECUTIVE WARNING & SPACE WEATHER OVERVIEW ──
        fig, ax = plt.subplots(figsize=(8.5, 11), dpi=300)
        ax.axis('off')
        fig.patch.set_facecolor('#0B0F19')

        # Header Banner
        ax.fill_between([0, 1], [0.92, 0.92], [1.0, 1.0], color='#1E293B', transform=ax.transAxes)
        ax.fill_between([0, 1], [0.915, 0.915], [0.92, 0.92], color='#DC2626', transform=ax.transAxes)

        ax.text(0.5, 0.965, "SOLAR FILAMENT ERUPTION & SATELLITE RADIATION ALERT", color='#F8FAFC', fontsize=14, fontweight='bold', ha='center', transform=ax.transAxes)
        ax.text(0.5, 0.935, f"OFFICIAL SPACE ENVIRONMENT TELEMETRY BULLETIN | ISSUED: {report_data.get('timestamp', 'UTC')}", color='#94A3B8', fontsize=8.5, ha='center', transform=ax.transAxes)

        # Section 1: Executive Status Box
        y_cursor = 0.87
        ax.text(0.06, y_cursor, "1. EXECUTIVE EVENT SUMMARY", color='#38BDF8', fontsize=11, fontweight='bold', transform=ax.transAxes)
        
        box_text = (
            f"SOURCE REGION     : {report_data.get('filament_loc')}\n"
            f"ERUPTION RISK     : 24h: {report_data.get('eruption_24h')}% | 48h: {report_data.get('eruption_48h')}% [{report_data.get('risk_tier')}]\n"
            f"PROJECTED FLARE   : {report_data.get('flare_class')} Soft X-Ray Class\n"
            f"EARTH CONNECTIVITY: Magnetic Separation Δφ = {report_data.get('delta_lon')} (High-Flux Parker Spiral Path)\n"
            f"ASTRONAUT SAFETY  : {report_data.get('eva_directive')}"
        )
        ax.text(0.08, y_cursor - 0.09, box_text, color='#F1F5F9', fontsize=8.5, family='monospace', bbox=dict(boxstyle='round,pad=0.5', fc='#1E293B', ec='#DC2626', lw=1.2), transform=ax.transAxes)

        # Section 2: Filament Quantitative Morphology Breakdown
        y_cursor = 0.68
        ax.text(0.06, y_cursor, "2. DETECTED FILAMENT MORPHOMETRIC MEASUREMENTS", color='#38BDF8', fontsize=11, fontweight='bold', transform=ax.transAxes)
        
        morph_lines = [
            f"• True Physical Spine Length : {report_data.get('filament_length_km', 0):,} km ({report_data.get('filament_length_km', 0)/696340:.3f} R_Sun)",
            f"• True Plasma Surface Area   : {report_data.get('filament_area_km2', 0):.2e} km²",
            f"• Magnetic Neutral Line Shear: {report_data.get('shear_angle_deg', 0):.1f}° (High Free Energy Polarity Inversion Line)",
            f"• Mean Chromospheric Contrast: {report_data.get('contrast_dip', 0):.3f} ΔI/I_0"
        ]
        for idx, l in enumerate(morph_lines):
            ax.text(0.08, y_cursor - 0.035 * (idx + 1), l, color='#E2E8F0', fontsize=9, transform=ax.transAxes)

        # Section 3: CME Kinematics & Geomagnetic Storm Warning
        y_cursor = 0.46
        ax.text(0.06, y_cursor, "3. CORONAL MASS EJECTION (CME) DRAG-BASED IMPACT FORECAST", color='#38BDF8', fontsize=11, fontweight='bold', transform=ax.transAxes)
        
        cme_box = (
            f"CME LAUNCH SPEED  : {report_data.get('cme_v0_kms')} km/s (Estimated Initial Plasma Ejection)\n"
            f"1-AU ARRIVAL SPEED: {report_data.get('cme_arrival_kms')} km/s (Aerodynamically Decelerated)\n"
            f"TRANSIT TIME      : {report_data.get('cme_transit_hours')} Hours ({report_data.get('cme_transit_hours', 0)/24.0:.1f} Days)\n"
            f"ESTIMATED IMPACT  : {report_data.get('cme_arrival_utc')}\n"
            f"GEOMAGNETIC SCALE : NOAA {report_data.get('storm_scale')} | Kp Index: {report_data.get('kp_index')}/9\n"
            f"POWER GRID IMPACT : Intermittent voltage regulation alarms; induced pipeline currents"
        )
        ax.text(0.08, y_cursor - 0.10, cme_box, color='#FEF08A', fontsize=8.5, family='monospace', bbox=dict(boxstyle='round,pad=0.5', fc='#1E293B', ec='#F59E0B', lw=1.2), transform=ax.transAxes)

        # Section 4: Mission Advisory Directives
        y_cursor = 0.22
        ax.text(0.06, y_cursor, "4. OPERATIONAL AEROSPACE MISSION DIRECTIVES", color='#38BDF8', fontsize=11, fontweight='bold', transform=ax.transAxes)
        
        directives = [
            "• Human Spaceflight (ISS / Tiangong) : Restrict crew to radiation-hardened service modules.",
            "• Lagrange Sentinels (DSCOVR / JWST) : Throttle optical sensors; enable radiation scrubbing on SSR.",
            "• GNSS Navigation (GPS III / Galileo): Upload clock phase corrections; monitor L-band scintillation.",
            "• LEO Mega-Constellations (Starlink) : Orient satellites edge-on (knife-edge) to mitigate drag decay."
        ]
        for idx, d in enumerate(directives):
            ax.text(0.08, y_cursor - 0.035 * (idx + 1), d, color='#CBD5E1', fontsize=8.2, transform=ax.transAxes)

        # Footer
        ax.text(0.5, 0.02, "Generated by Solar Filament AI & Space Weather Intelligence Platform (2-Stage Coarse-to-Fine Pipeline)", color='#64748B', fontsize=7.5, ha='center', transform=ax.transAxes)

        pdf.savefig(fig, facecolor=fig.get_facecolor(), edgecolor='none')
        plt.close(fig)

    return output_pdf_path


if __name__ == '__main__':
    p = generate_space_weather_pdf()
    print(f"[+] Successfully generated Space Weather PDF: {p}")
