"""
Solar Filament Segmentation & Space Weather Intelligence — Streamlit Dashboard
================================================================================
Rebuilt on model_hub.py (shared with webapp/app.py) so both dashboards stay
consistent. Free-hostable on Streamlit Community Cloud.

Design: crimson accent theme (.streamlit/config.toml), visual stats (Plotly
charts + metric cards) instead of raw per-epoch number tables, using
streamlit-extras for polished metric cards / section headers.
"""
import csv
import io
import json
import os
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from streamlit_extras.metric_cards import style_metric_cards

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import model_hub as hub
import analysis.dashboard_integration as di

CRIMSON = "#DC143C"
CRIMSON_SOFT = "#ff5c7a"

# Historical validation cases for Demo Mode
HISTORICAL_CASES = {
    "2011-Sep-6 X2.1": {"label": "2011-Sep-6 X2.1", "description": "X2.1 flare from AR 11283; associated CME produced Kp=7 storm.", "cme_id": "2011-09-06T22:12:00-CME-001", "speed": 575.0, "lat": -28.0, "lon": -42.0, "half_angle": 45.0, "sep_observed": True, "max_kp": 7.0, "date": "2011-09-06"},
    "2012-Jan-23 M8.7": {"label": "2012-Jan-23 M8.7", "description": "M8.7 from AR 11402; fast CME causing strong SEP and Kp=5.", "cme_id": "2012-01-23T04:00:00-CME-001", "speed": 2210.0, "lat": 44.0, "lon": 36.0, "half_angle": 61.0, "sep_observed": True, "max_kp": 5.0, "date": "2012-01-23"},
    "2012-Mar-7 X5.4": {"label": "2012-Mar-7 X5.4", "description": "X5.4 from AR 11429; extremely fast and wide CME causing Kp=8.", "cme_id": "2012-03-07T00:24:00-CME-001", "speed": 2684.0, "lat": 18.0, "lon": -42.0, "half_angle": 81.0, "sep_observed": True, "max_kp": 8.0, "date": "2012-03-07"},
    "2012-Jul-12 X1.4": {"label": "2012-Jul-12 X1.4", "description": "X1.4 from AR 11520; major Earth-directed CME causing Kp=7.", "cme_id": "2012-07-12T16:48:00-CME-001", "speed": 1405.0, "lat": -18.0, "lon": -3.0, "half_angle": 58.0, "sep_observed": True, "max_kp": 7.0, "date": "2012-07-12"},
    "2014-Sep-10 X1.6": {"label": "2014-Sep-10 X1.6", "description": "X1.6 from AR 12158; Earth-directed causing minor Kp=4 storm.", "cme_id": "2014-09-10T18:00:00-CME-001", "speed": 1267.0, "lat": 11.0, "lon": -7.0, "half_angle": 52.0, "sep_observed": True, "max_kp": 4.0, "date": "2014-09-10"},
    "2017-Sep-6 X9.3": {"label": "2017-Sep-6 X9.3", "description": "X9.3 from AR 12673; severe storm Kp=8, widespread radio blackout.", "cme_id": "2017-09-06T12:24:00-CME-001", "speed": 1571.0, "lat": -10.0, "lon": 36.0, "half_angle": 64.0, "sep_observed": True, "max_kp": 8.0, "date": "2017-09-06"},
    "2017-Sep-10 X8.2 (Limb)": {"label": "2017-Sep-10 X8.2 (Limb)", "description": "X8.2 limb event; fast CME but missed Earth (near flank).", "cme_id": "2017-09-10T16:00:00-CME-001", "speed": 3163.0, "lat": -10.0, "lon": -104.0, "half_angle": 30.0, "sep_observed": True, "max_kp": 3.0, "date": "2017-09-10"},
    "2021-Oct-28 X1.0": {"label": "2021-Oct-28 X1.0", "description": "X1.0 Halloween storm; slow arrival but caused SEP and Kp=4.", "cme_id": "2021-10-28T15:48:00-CME-001", "speed": 1194.0, "lat": -33.0, "lon": -5.0, "half_angle": 77.0, "sep_observed": True, "max_kp": 4.0, "date": "2021-10-28"},
    "2023-Feb-24 M3.7": {"label": "2023-Feb-24 M3.7", "description": "M3.7 event causing strong Kp=6 storm.", "cme_id": "2023-02-24T20:23:00-CME-001", "speed": 1184.0, "lat": 16.0, "lon": 32.0, "half_angle": 49.0, "sep_observed": False, "max_kp": 6.0, "date": "2023-02-24"},
    "2024-May-10 (May 2024 Storms)": {"label": "2024-May-10 (May 2024 Storms)", "description": "Complex of CMEs from AR 13664 causing extreme Kp=9 storm.", "cme_id": "2024-05-10T00:00:00-CME-001", "speed": 1500.0, "lat": -20.0, "lon": 0.0, "half_angle": 60.0, "sep_observed": True, "max_kp": 9.0, "date": "2024-05-10"}
}

st.set_page_config(
    page_title="Solar Filament Intelligence",
    layout="wide",
)

st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {{
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }}

    .stApp {{
        background: #090a0f !important;
        color: #e2e2ec;
    }}

    h1, h2, h3, h4, h5 {{
        font-family: 'Inter', sans-serif !important;
        font-weight: 600 !important;
        color: #f4f4f6 !important;
        letter-spacing: -0.02em;
    }}

    /* Top Platform Header */
    .top-header-banner {{
        background: rgba(16, 17, 23, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-left: 3px solid {CRIMSON};
        border-radius: 8px;
        padding: 16px 20px;
        margin-bottom: 20px;
        backdrop-filter: blur(16px);
        display: flex;
        align-items: center;
        justify-content: space-between;
    }}
    .top-header-title {{
        font-size: 18px;
        font-weight: 600;
        color: #ffffff;
        letter-spacing: -0.01em;
        margin: 0;
    }}
    .top-header-sub {{
        font-size: 12px;
        color: #8f90a6;
        margin-top: 3px;
    }}
    .top-header-badges {{
        display: flex;
        gap: 8px;
    }}
    .badge-pill {{
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.1);
        color: #b0b0cc;
        font-size: 11px;
        font-weight: 500;
        padding: 4px 10px;
        border-radius: 6px;
        letter-spacing: 0.03em;
    }}
    .badge-green {{
        background: rgba(0, 200, 83, 0.08);
        border-color: rgba(0, 200, 83, 0.25);
        color: #00e676;
    }}

    /* Metric Cards */
    div[data-testid="stMetric"] {{
        background: rgba(16, 17, 23, 0.6) !important;
        border: 1px solid rgba(255, 255, 255, 0.07) !important;
        border-left: 3px solid {CRIMSON} !important;
        border-radius: 8px !important;
        padding: 12px 16px !important;
        backdrop-filter: blur(12px);
    }}
    div[data-testid="stMetricLabel"] {{
        font-size: 11px !important;
        text-transform: uppercase !important;
        letter-spacing: 0.06em !important;
        color: #84849a !important;
        font-weight: 500 !important;
    }}
    div[data-testid="stMetricValue"] {{
        font-size: 22px !important;
        font-weight: 600 !important;
        color: #ffffff !important;
    }}

    /* Sidebar Styling */
    div[data-testid="stSidebar"] {{
        background: #06070a !important;
        border-right: 1px solid rgba(255, 255, 255, 0.06);
    }}
    .sidebar-brand-card {{
        background: rgba(16, 17, 23, 0.5);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 8px;
        padding: 12px 14px;
        margin-bottom: 16px;
    }}
    .sidebar-brand-title {{
        font-size: 14px;
        font-weight: 600;
        color: #ffffff;
        letter-spacing: 0.02em;
        margin: 0;
    }}
    .sidebar-brand-sub {{
        font-size: 11px;
        color: #7a7a92;
        margin-top: 2px;
    }}

    /* Dropdowns & Inputs */
    div[data-baseweb="select"] > div {{
        background: rgba(16, 17, 23, 0.8) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 6px !important;
        color: #ffffff !important;
        font-size: 13px !important;
    }}
    div[data-baseweb="select"] input {{
        color: #ffffff !important;
    }}
    div[data-baseweb="popover"] {{
        background: #0f1017 !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 6px !important;
    }}

    /* Section Headers */
    .viz-section-header {{
        font-size: 12px;
        font-weight: 600;
        color: #c4c4d6;
        background: rgba(16, 17, 23, 0.6);
        border-left: 3px solid {CRIMSON};
        border-radius: 0 6px 6px 0;
        padding: 6px 12px;
        margin: 18px 0 10px 0;
        letter-spacing: 0.05em;
    }}
    .viz-panel-label {{
        font-size: 11px;
        font-weight: 500;
        color: #7b7b94;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 4px;
        padding-left: 2px;
    }}

    /* Buttons */
    div.stButton > button {{
        background: rgba(255, 255, 255, 0.03) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        color: #d0d0e0 !important;
        border-radius: 6px !important;
        font-weight: 500 !important;
        font-size: 12px !important;
        padding: 6px 12px !important;
        transition: all 0.15s ease !important;
    }}
    div.stButton > button:hover {{
        background: rgba(220, 20, 60, 0.15) !important;
        border-color: rgba(220, 20, 60, 0.4) !important;
        color: #ffffff !important;
    }}

    /* Sliders */
    div[data-baseweb="slider"] {{
        padding-top: 8px;
    }}
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------
def overlay_rgb(gray: np.ndarray, mask: np.ndarray, color=(220, 20, 60), alpha: float = 0.55) -> np.ndarray:
    """Blend a colored mask onto a grayscale image, returns RGB uint8."""
    rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB).astype(np.float32)
    color_layer = np.zeros_like(rgb)
    color_layer[..., 0], color_layer[..., 1], color_layer[..., 2] = color
    m = (mask > 0).astype(np.float32)[..., None]
    out = rgb * (1 - m * alpha) + color_layer * (m * alpha)
    return out.astype(np.uint8)


def confidence_rgb(probs: np.ndarray) -> np.ndarray:
    """Crimson-tinted confidence heatmap (dark -> crimson -> white) as RGB uint8."""
    p = np.clip(probs, 0, 1)
    r = np.clip(p * 3.0, 0, 1)
    g = np.clip(p * 1.3 - 0.3, 0, 1)
    b = np.clip(p * 1.3 - 0.5, 0, 1)
    return (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)


def png_bytes(image: np.ndarray) -> bytes:
    """Encode an image for a Streamlit download without changing its resolution."""
    value = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) if image.ndim == 3 else image
    ok, encoded = cv2.imencode(".png", value)
    if not ok:
        raise ValueError("Could not encode image as PNG")
    return encoded.tobytes()


def morphology_table_rows(filaments):
    """Return Arrow-compatible scalar rows without internal NumPy masks."""
    rows = []
    for filament in filaments:
        centroid = filament.get("centroid", {})
        rows.append({
            "Filament ID": filament.get("filament_id", 0),
            "Confidence": round(float(filament.get("confidence", 0.0)), 3),
            "Area (px)": round(float(filament.get("area_px", 0.0)), 2),
            "Perimeter (px)": round(float(filament.get("perimeter_px", 0.0)), 2),
            "Skeleton length (px)": round(float(filament.get("skeleton_length_px", 0.0)), 2),
            "Average width (px)": round(float(filament.get("avg_width_px", 0.0)), 2),
            "Sinuosity": round(float(filament.get("sinuosity", 1.0)), 3),
            "Orientation (deg)": round(float(filament.get("orientation_deg", 0.0)), 2),
            "Centroid X": round(float(centroid.get("x", 0.0)), 2),
            "Centroid Y": round(float(centroid.get("y", 0.0)), 2),
            "Spatial region": filament.get("spatial_region", "CENTER"),
            "Risk indicator": filament.get("risk_screening_indicator", "LOW"),
        })
    return rows


@st.cache_resource(show_spinner=False)
def load_sr_model(method: str, scale: int):
    """Loads (and caches) the super-resolution model for the Filament Detail Inspector.
    Trained checkpoints (checkpoints via experiments/super_resolution/training/train_sr.py)
    are used when present; ESPCN/EDSR-Small fall back to untrained weights (see
    sr_experiment_report.md -- only Solar-SR was actually trained on this data)."""
    if method in ["OFF", "Lanczos (Current)", "Bicubic"]:
        return None
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    from experiments.super_resolution.models import ESPCN, EDSRSmall, SolarSRNet
    if method == "ESPCN (AI-SR)":
        model = ESPCN(scale_factor=scale, in_channels=1).to(device).eval()
    elif method == "EDSR-Small (AI-SR)":
        model = EDSRSmall(scale_factor=scale, in_channels=1).to(device).eval()
    elif method == "Solar-SR (Trained AI-SR)":
        model = SolarSRNet(scale_factor=scale, in_channels=1).to(device).eval()
        ckpt_path = os.path.join(hub.ROOT, "experiments", "super_resolution", "results",
                                  f"best_sr_model_solar_sr_x{scale}.pt")
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
    else:
        return None
    return model


@st.cache_data(show_spinner=False)
def cached_detail_upscale(crop: np.ndarray, method: str, scale: int) -> np.ndarray:
    """Cache the selected filament's lightweight display-only enhancement."""
    from visualization.detail import super_resolve_crop
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_sr_model(method, scale)
    return super_resolve_crop(crop, method=method, scale=scale, model=model, device=device)


@st.cache_data(ttl=5)
def cached_model_list():
    return hub.list_models()


@st.cache_data(ttl=5)
def cached_status(arch):
    return hub.parse_status(arch)


def model_options():
    models = cached_model_list()
    labels = {m["arch"]: f"{m['label']}" + (f"  |  Val Dice: {m['best_val_dice']:.3f}" if m["best_val_dice"] else "  |  Untrained") for m in models}
    return models, labels


# --------------------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------------------
st.sidebar.markdown("""
<div class="sidebar-brand-card">
    <div class="sidebar-brand-title">SOLAR FILAMENT INTELLIGENCE</div>
    <div class="sidebar-brand-sub">H-Alpha Filament Segmentation &amp; CME Trajectory</div>
</div>
""", unsafe_allow_html=True)

from streamlit_option_menu import option_menu
with st.sidebar:
    page = option_menu(
        menu_title=None,
        options=["Overview", "Validation Gallery", "Upload Image", "Upload Video"],
        icons=['bar-chart-line', 'images', 'cloud-upload', 'camera-video'],
        default_index=2,
        styles={
            "container": {"padding": "0!important", "background-color": "transparent"},
            "icon": {"color": "#ff4d6d", "font-size": "15px"}, 
            "nav-link": {"font-size": "14px", "text-align": "left", "margin":"2px 0px", "border-radius":"8px", "--hover-color": "rgba(220,20,60,0.15)"},
            "nav-link-selected": {"background": "linear-gradient(135deg, #DC143C, #900c24)", "font-weight": "600"},
        }
    )
    st.divider()

models, labels = model_options()
default_arch = next((m["arch"] for m in models if m["best_val_dice"]), models[0]["arch"])
arch = st.sidebar.selectbox(
    "Model", options=list(labels.keys()), format_func=lambda a: labels[a],
    index=list(labels.keys()).index(default_arch),
)
phase2_threshold = st.sidebar.slider("Segmentation threshold (Upload Image)", 0.20, 0.75, 0.50, 0.01)
show_explainability = st.sidebar.checkbox("Explainability (when supported)", value=False)

st.sidebar.markdown("### Analysis Mode")
analysis_mode = st.sidebar.radio("Mode", ["LIVE / CURRENT ANALYSIS", "HISTORICAL EVENT DEMO"])

obs_time_str = ""
historical_case = None

if analysis_mode == "HISTORICAL EVENT DEMO":
    st.sidebar.warning("HISTORICAL REPLAY MODE")
    selected_case = st.sidebar.selectbox("Select Event", list(HISTORICAL_CASES.keys()))
    historical_case = HISTORICAL_CASES[selected_case]
else:
    import datetime as _dt
    _default_date = _dt.date.today()
    _obs_date = st.sidebar.date_input(
        "Observation Date",
        value=_default_date,
        help="Used for DONKI flare/CME lookup and relative risk scoring. Defaults to today."
    )
    obs_time_str = _obs_date.strftime("%Y-%m-%dT12:00:00Z")

st.sidebar.divider()
st.sidebar.caption("GGSIPU Hackathon 2026 · Track 19 · USAR")


# --------------------------------------------------------------------------------------
# Overview — visual stats, not a raw number table
# --------------------------------------------------------------------------------------
if page == "Overview":
    st.header("Model Overview", divider="red")
    st.caption(labels[arch])

    status = cached_status(arch)
    epochs = status["epochs"]
    final_metrics = status.get("final_metrics")
    best_dice = hub.best_dice_for(status)
    best_iou = max((e["val_iou"] for e in epochs), default=None)
    if best_iou is None and final_metrics:
        best_iou = final_metrics.get("val_iou")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Status", status["state"].replace("_", " ").title())
    c2.metric("Best Val Dice", f"{best_dice:.3f}" if best_dice else "—")
    c3.metric("Best Val IoU", f"{best_iou:.3f}" if best_iou else "—")
    if epochs:
        c4.metric("Epochs Logged", f"{len(epochs)} / {epochs[-1]['total_epochs']}")
    elif final_metrics:
        c4.metric("Best Epoch", f"{final_metrics.get('epoch', '?')} / {final_metrics.get('total_epochs', '?')}")
    else:
        c4.metric("Epochs Logged", "0")
    style_metric_cards(border_left_color=CRIMSON)

    if epochs:
        ep = [e["epoch"] for e in epochs]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ep, y=[e["train_dice"] for e in epochs], name="Train Dice",
                                  line=dict(color="#5c5c66", width=2, dash="dot")))
        fig.add_trace(go.Scatter(x=ep, y=[e["val_dice"] for e in epochs], name="Val Dice",
                                  line=dict(color=CRIMSON, width=3)))
        fig.update_layout(
            title="Dice Score over Training", height=340, template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="h", y=1.1),
            xaxis=dict(showgrid=False), yaxis=dict(showgrid=False),
        )
        st.plotly_chart(fig, width='stretch')

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=ep, y=[e["train_loss"] for e in epochs], name="Train Loss",
                                   line=dict(color="#5c5c66", width=2, dash="dot")))
        fig2.add_trace(go.Scatter(x=ep, y=[e["val_loss"] for e in epochs], name="Val Loss",
                                   line=dict(color=CRIMSON_SOFT, width=3)))
        fig2.update_layout(
            title="Loss over Training", height=300, template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="h", y=1.1),
            xaxis=dict(showgrid=False), yaxis=dict(showgrid=False),
        )
        st.plotly_chart(fig2, width='stretch')
    elif final_metrics:
        st.info(
            f"Only the final checkpoint's metrics were saved for this model "
            f"(epoch {final_metrics.get('epoch', '?')}/{final_metrics.get('total_epochs', '?')}) "
            "— no per-epoch history to chart."
        )
        mc1, mc2, mc3 = st.columns(3)
        mc1.metric("Val Loss", f"{final_metrics['val_loss']:.3f}" if final_metrics.get("val_loss") is not None else "—")
        mc2.metric("Val Precision", f"{final_metrics['val_precision']:.3f}" if final_metrics.get("val_precision") is not None else "—")
        mc3.metric("Val Recall", f"{final_metrics['val_recall']:.3f}" if final_metrics.get("val_recall") is not None else "—")
    else:
        st.info("No training history yet for this model.")

    st.markdown("#### Architecture Leaderboard")
    trained = [m for m in models if m["best_val_dice"]]
    untested = [m for m in models if m["trained"] and not m["best_val_dice"]]
    if trained:
        trained.sort(key=lambda m: m["best_val_dice"], reverse=True)
        fig3 = go.Figure(go.Bar(
            x=[m["best_val_dice"] for m in trained],
            y=[m["label"] for m in trained],
            orientation="h",
            marker=dict(color=[CRIMSON if m["arch"] == arch else "#4a4a55" for m in trained]),
            text=[f"{m['best_val_dice']:.3f}" for m in trained],
            textposition="outside",
        ))
        fig3.update_layout(
            height=90 + 60 * len(trained), template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=10, r=40, t=10, b=10),
            xaxis=dict(title="Best Validation Dice", showgrid=False), yaxis=dict(showgrid=False),
        )
        st.plotly_chart(fig3, width='stretch')
    if untested:
        st.caption(
            "Also available (checkpoint present, no measured validation Dice on this "
            "repo's data yet, so excluded from the chart above): "
            + ", ".join(m["label"] for m in untested) + ". Selectable from the Model "
            "dropdown in the sidebar like any other architecture."
        )


# --------------------------------------------------------------------------------------
# Validation Gallery
# --------------------------------------------------------------------------------------
elif page == "Validation Gallery":
    st.header("Validation Gallery", divider="red")
    st.caption(labels[arch])

    if "gallery_seed" not in st.session_state:
        st.session_state.gallery_seed = 7
    if st.button("Shuffle samples"):
        st.session_state.gallery_seed = np.random.randint(0, 100000)

    status = cached_status(arch)
    best_thresh = status["best_threshold"]["threshold"] if status.get("best_threshold") else 0.5
    paths = hub.shuffled_sample_paths(st.session_state.gallery_seed, 6)

    for img_path in paths:
        raw = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        gt = hub.load_gt_mask(img_path)
        small, disk_small, probs, pred = hub.run_inference(raw, arch, best_thresh)
        dice = hub.dice_score(pred, gt)

        cols = st.columns(4)
        cols[0].image(small, caption=os.path.basename(img_path), width='stretch')
        cols[1].image(overlay_rgb(small, gt, color=(80, 170, 255)), caption="Ground Truth", width='stretch')
        cols[2].image(overlay_rgb(small, pred, color=(220, 20, 60)), caption=f"Prediction · Dice {dice:.3f}", width='stretch')
        cols[3].image(confidence_rgb(probs), caption="Confidence", width='stretch')
        st.divider()


# --------------------------------------------------------------------------------------
# Upload Image
# --------------------------------------------------------------------------------------
elif page == "Upload Image":
    st.markdown(f"""
    <div class="top-header-banner">
        <div>
            <div class="top-header-title">SOLAR FILAMENT INTELLIGENCE PLATFORM</div>
            <div class="top-header-sub">H-Alpha Deep Learning Segmentation, 3D CME Eruption Trajectory &amp; Satellite Exposure Assessment</div>
        </div>
        <div class="top-header-badges">
            <span class="badge-pill badge-green">SYSTEM ONLINE</span>
            <span class="badge-pill">MODEL: {arch}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    upload_container = st.container()
    with upload_container:
        st.markdown("<h3 style='font-size: 16px; font-weight: 600; margin-bottom: 2px;'>Upload Solar H-Alpha Observation</h3>", unsafe_allow_html=True)
        st.markdown("<p style='color: #8e8ea6; font-size: 12px; margin-bottom: 14px;'>Select a pre-loaded full-disk observation or upload a custom FITS/JPG image.</p>", unsafe_allow_html=True)
        
        sample_dir = Path("sample_images")
        sample_images = []
        if sample_dir.exists():
            sample_images = sorted([f.name for f in sample_dir.glob("*") if f.suffix.lower() in [".jpg", ".jpeg", ".png"]])
            
        selected_sample = None
        if sample_images:
            st.markdown("<p style='font-size: 11px; font-weight: 600; color: #7b7b94; text-transform: uppercase; letter-spacing: 0.06em; margin: 8px 0 6px 0;'>Select Pre-Loaded Observation</p>", unsafe_allow_html=True)
            num_cols = min(len(sample_images), 5)
            cols = st.columns(num_cols)
            for i, img_name in enumerate(sample_images):
                with cols[i % num_cols]:
                    st.image(str(sample_dir / img_name), use_container_width=True)
                    if st.button(f"Select {img_name}", key=f"btn_{img_name}", use_container_width=True):
                        st.session_state.selected_sample = img_name
            
            if "selected_sample" in st.session_state:
                selected_sample = st.session_state.selected_sample
                
        uploaded = st.file_uploader("Or upload your own image:", type=["jpg", "jpeg", "png"])

    raw_color = None
    file_name = None
    
    if uploaded is not None:
        data = np.frombuffer(uploaded.read(), dtype=np.uint8)
        raw_color = cv2.imdecode(data, cv2.IMREAD_COLOR)
        file_name = uploaded.name
    elif selected_sample is not None:
        raw_color = cv2.imread(str(sample_dir / selected_sample), cv2.IMREAD_COLOR)
        file_name = selected_sample

    if raw_color is not None:
        raw = hub.to_halpha_style(raw_color)

        from inference.phase2 import run_phase2_analysis
        with st.spinner("Running segmentation and Phase 2 analysis..."):
            phase2_result = run_phase2_analysis(
                raw, image_id=file_name, model_name=arch,
                threshold=phase2_threshold, explain=show_explainability,
            )
            
        phase2_inference = phase2_result["inference"]
        small = phase2_inference.preprocessed
        probs = cv2.resize(phase2_inference.probability, (small.shape[1], small.shape[0]))
        pred = cv2.resize(phase2_inference.mask, (small.shape[1], small.shape[0]), interpolation=cv2.INTER_NEAREST)
        filaments = phase2_result["filaments"]

        c1, c2, c3 = st.columns(3)
        c1.metric("Filaments Detected", len(filaments))
        c2.metric("Total Area (px)", f"{sum(f['area_px'] for f in filaments):,.0f}" if filaments else "0")
        c3.metric("Avg Confidence", f"{np.mean([f['confidence'] for f in filaments]):.2f}" if filaments else "—")
        style_metric_cards(border_left_color=CRIMSON)

        original_display = cv2.cvtColor(
            cv2.resize(raw_color, (small.shape[1], small.shape[0]), interpolation=cv2.INTER_AREA),
            cv2.COLOR_BGR2RGB,
        )

        from visualization.phase2 import _instance_panel, create_phase2_figure
        from visualization.detail import crop_filament, detail_record, save_detail_artifacts, selected_overlay
        annotated_panel = _instance_panel(raw, filaments)
        skeleton_panel  = _instance_panel(raw, filaments, skeleton=True)

        # ── Row 1: Solar Observation Pipeline ───────────────────────────────────
        st.markdown('<div class="viz-section-header">SECTION 01 / SOLAR OBSERVATION PIPELINE</div>', unsafe_allow_html=True)
        r1c1, r1c2, r1c3 = st.columns(3)
        with r1c1:
            st.markdown('<div class="viz-panel-label">Raw Full-Disk Observation</div>', unsafe_allow_html=True)
            st.image(original_display, width='stretch')
        with r1c2:
            # False-colour: apply cyan tint (COLORMAP_OCEAN) to the preprocessed gray image
            _fc = cv2.applyColorMap(
                cv2.cvtColor(small, cv2.COLOR_GRAY2BGR) if small.ndim == 2
                else cv2.cvtColor(small, cv2.COLOR_RGB2BGR),
                cv2.COLORMAP_OCEAN
            )
            _fc_rgb = cv2.cvtColor(_fc, cv2.COLOR_BGR2RGB)
            st.markdown('<div class="viz-panel-label">Preprocessed (Limb Corrected + CLAHE)</div>', unsafe_allow_html=True)
            st.image(small, width='stretch')
        with r1c3:
            st.markdown('<div class="viz-panel-label">False-Color Solar Visualization</div>', unsafe_allow_html=True)
            st.image(_fc_rgb, width='stretch')

        # ── Row 2: Deep Learning Segmentation ───────────────────────────────────
        st.markdown('<div class="viz-section-header">SECTION 02 / DEEP LEARNING SEGMENTATION</div>', unsafe_allow_html=True)
        r2c1, r2c2, r2c3 = st.columns(3)
        with r2c1:
            # Confidence heatmap with COLORMAP_HOT on black background
            _prob_norm = (np.clip(probs, 0, 1) * 255).astype(np.uint8)
            _heat_bgr  = cv2.applyColorMap(_prob_norm, cv2.COLORMAP_HOT)
            _heat_rgb  = cv2.cvtColor(_heat_bgr, cv2.COLOR_BGR2RGB)
            st.markdown('<div class="viz-panel-label">Mask2Former Confidence Heatmap</div>', unsafe_allow_html=True)
            st.image(_heat_rgb, width='stretch')
        with r2c2:
            _bin = ((pred > 0).astype(np.uint8) * 255)
            _bin_rgb = cv2.cvtColor(_bin, cv2.COLOR_GRAY2RGB)
            st.markdown('<div class="viz-panel-label">Binary Filament Segmentation Mask</div>', unsafe_allow_html=True)
            st.image(_bin_rgb, width='stretch')
        with r2c3:
            # Neon boundary overlay: draw contours in bright orange/yellow on the gray image
            _neon_base = cv2.cvtColor(small, cv2.COLOR_GRAY2RGB) if small.ndim == 2 else small.copy()
            _contour_mask = (pred > 0).astype(np.uint8)
            _contours, _ = cv2.findContours(_contour_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            _neon_out = _neon_base.copy()
            cv2.drawContours(_neon_out, _contours, -1, (255, 165, 0), 2)  # orange neon outline
            st.markdown('<div class="viz-panel-label">Neon Boundary Segmentation Overlay</div>', unsafe_allow_html=True)
            st.image(_neon_out, width='stretch')

        # ── Row 3: Instance Analysis ─────────────────────────────────────────────
        st.markdown('<div class="viz-section-header">SECTION 03 / INSTANCE ANALYSIS &amp; SKELETONIZATION</div>', unsafe_allow_html=True)
        r3c1, r3c2, r3c3 = st.columns(3)
        with r3c1:
            st.markdown('<div class="viz-panel-label">Instances + Bounding Boxes</div>', unsafe_allow_html=True)
            st.image(annotated_panel, width='stretch')
        with r3c2:
            st.markdown('<div class="viz-panel-label">Skeletons + Cyan Spines</div>', unsafe_allow_html=True)
            st.image(skeleton_panel, width='stretch')
        with r3c3:
            st.markdown('<div class="viz-panel-label">Segmentation Attribution (XAI)</div>', unsafe_allow_html=True)
            if phase2_result.get("attribution") is not None:
                _attr = phase2_result["attribution"]
                _attr_norm = (np.clip((_attr - _attr.min()) / (_attr.max() - _attr.min() + 1e-8), 0, 1) * 255).astype(np.uint8)
                _attr_rgb  = cv2.cvtColor(cv2.applyColorMap(_attr_norm, cv2.COLORMAP_PLASMA), cv2.COLOR_BGR2RGB)
                st.image(_attr_rgb, width='stretch')
            else:
                # Fallback: plasma-coloured probability blend so the panel is never empty/black
                _base_rgb = cv2.cvtColor(small, cv2.COLOR_GRAY2RGB) if small.ndim == 2 else small.copy()
                _prob_u8  = (np.clip(probs, 0, 1) * 255).astype(np.uint8)
                _plasma   = cv2.cvtColor(cv2.applyColorMap(_prob_u8, cv2.COLORMAP_PLASMA), cv2.COLOR_BGR2RGB)
                _blend_mask = (probs > 0.05).astype(np.float32)[..., None]
                _fallback = (_base_rgb.astype(np.float32) * (1 - _blend_mask * 0.75)
                             + _plasma.astype(np.float32) * _blend_mask * 0.75).clip(0, 255).astype(np.uint8)
                st.image(_fallback, width='stretch')
                st.markdown('<div style="font-size:10px;color:#667;margin-top:2px;">▸ Probability heatmap overlay — enable XAI in sidebar for Grad-CAM</div>', unsafe_allow_html=True)

        st.caption(f"Model: {phase2_result['model_name']} · threshold {phase2_result['threshold']:.2f}")

        # ── Ensemble (collapsible) ───────────────────────────────────────────────
        st.divider()
        run_ensemble = st.checkbox(
            "Run Ensemble Consensus & Uncertainty (all 5 models + test-time augmentation — slower)",
            value=False, key="run_ensemble",
        )
        if run_ensemble:
            from inference.ensemble import run_ensemble_inference
            with st.spinner("Running all 5 models with test-time augmentation..."):
                ens_small, ens_disk, ens_probs, ens_mask, ens_weights, agreement = run_ensemble_inference(raw)
            st.markdown('<div class="viz-section-header">SECTION 04 / ENSEMBLE CONSENSUS</div>', unsafe_allow_html=True)
            st.caption(
                "Weighted-averaged prediction across every trained model (weights: "
                + ", ".join(f"{a} {w:.3f}" for a, w in ens_weights.items())
                + "). The agreement map flags where independently-architected models disagree."
            )
            ecols = st.columns(3)
            with ecols[0]:
                st.markdown('<div class="viz-panel-label">Ensemble Prediction</div>', unsafe_allow_html=True)
                st.image(overlay_rgb(ens_small, ens_mask, color=(220, 20, 60)), width='stretch')
            with ecols[1]:
                st.markdown('<div class="viz-panel-label">Ensemble Confidence</div>', unsafe_allow_html=True)
                st.image(confidence_rgb(ens_probs), width='stretch')
            with ecols[2]:
                agreement_heat = (np.clip(1.0 - agreement, 0, 1) * 255).astype(np.uint8)
                agreement_rgb  = cv2.cvtColor(cv2.applyColorMap(agreement_heat, cv2.COLORMAP_HOT), cv2.COLOR_BGR2RGB)
                st.markdown('<div class="viz-panel-label">Model Disagreement Map</div>', unsafe_allow_html=True)
                st.image(agreement_rgb, width='stretch')
        st.markdown("#### Export Results")
        export_cols = st.columns(2)
        export_cols[0].download_button("Download JSON catalog", json.dumps(phase2_result["catalog"], indent=2),
                           file_name="filament_catalog.json", mime="application/json", use_container_width=True)
        csv_buffer = io.StringIO()
        writer = csv.DictWriter(csv_buffer, fieldnames=["image_id", "model_name", "model_checkpoint", "threshold",
                                                          "filament_id", "confidence", "area_px",
                                                          "skeleton_length_px", "sinuosity", "orientation_deg",
                                                          "spatial_region"])
        writer.writeheader()
        writer.writerows({key: record.get(key) for key in writer.fieldnames} for record in phase2_result["catalog"])
        export_cols[1].download_button("Download CSV catalog", csv_buffer.getvalue(), file_name="filament_catalog.csv", mime="text/csv", use_container_width=True)

        if filaments:
            fig = go.Figure(go.Bar(
                x=[f["filament_id"] for f in filaments],
                y=[f["skeleton_length_px"] for f in filaments],
                marker=dict(color=CRIMSON),
            ))
            fig.update_layout(
                title="Filament Length by ID", height=280, template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=10, r=10, t=40, b=10),
                xaxis=dict(title="Filament ID", showgrid=False),
                yaxis=dict(title="Skeleton Length (px)", showgrid=False),
            )
            st.plotly_chart(fig, width='stretch')
            with st.expander("Per-filament measurements"):
                st.dataframe(morphology_table_rows(filaments), width='stretch')

            st.divider()
            st.markdown('<div class="viz-section-header">SPACE WEATHER ERUPTION RISK (RANDOM FOREST)</div>', unsafe_allow_html=True)
            st.caption(
                "Predicts 24h eruption probability using a trained Random Forest model (flare_rf_model.pkl) "
                "evaluating filament geometry (skeleton length) and solar disk position."
            )
            from analysis.flare_prediction import calculate_flare_probability

            disk_center_x, disk_center_y = raw.shape[1] / 2.0, raw.shape[0] / 2.0
            display_count = min(len(filaments), 4)
            risk_cols = st.columns(display_count) if display_count > 0 else []

            for i, f in enumerate(filaments[:display_count]):
                risk = calculate_flare_probability(
                    length_px=f.get('skeleton_length_px', 0.0),
                    region_type=f.get('spatial_region', 'ARF')
                )
                
                risk_cols[i].metric(
                    f"Filament #{f['filament_id']}", 
                    f"{risk:.1%}", 
                    delta="HIGH RISK" if risk > 0.5 else "LOW RISK",
                    delta_color="inverse"
                )
            if len(filaments) > 4:
                st.info(f"Showing risk for top 4 filaments out of {len(filaments)} detected.")
            
            st.divider()
            st.markdown('<div class="viz-section-header">3D HELIOSPHERIC CME TRAJECTORY &amp; DRAG MODEL</div>', unsafe_allow_html=True)
            st.caption("Simulates the 3D propagation vector of a potential Coronal Mass Ejection using an aerodynamic Drag-Based Model (DBM) in Parker Spiral space.")
            
            center_x = raw.shape[1] / 2.0
            best_idx = 0
            min_dist = float('inf')
            for i, f in enumerate(filaments):
                dist = abs(f.get("centroid", {}).get("x", 0) - center_x)
                if dist < min_dist:
                    min_dist = dist
                    best_idx = i
            
            traj_filament_index = st.selectbox(
                "Select a filament to simulate its 3D eruption vector:", 
                options=list(range(len(filaments))),
                index=best_idx,
                format_func=lambda index: f"Filament #{filaments[index]['filament_id']} (Region: {filaments[index].get('spatial_region', 'CENTER')})",
                key="traj_filament"
            )
            
            from visualization.three_d_trajectory import plot_3d_trajectory
            
            with st.spinner("Rendering 3D solar environment and orbital trajectory..."):
                fig_traj, is_impact = plot_3d_trajectory(raw, pred, filaments[traj_filament_index])
                st.plotly_chart(fig_traj, use_container_width=True)
                
            if is_impact:
                st.markdown('<div style="background: rgba(220,20,60,0.12); border-left: 3px solid #DC143C; border-radius: 6px; padding: 10px 14px; color: #ffffff; font-size: 13px; font-weight: 500; margin-top: 8px;">CRITICAL WARNING: Eruption vector intersects Earth\'s future orbital position within 25°. High space weather impact risk.</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div style="background: rgba(0,200,83,0.1); border-left: 3px solid #00E676; border-radius: 6px; padding: 10px 14px; color: #ffffff; font-size: 13px; font-weight: 500; margin-top: 8px;">SAFE ORBITAL POSITION: Eruption vector does not intersect Earth\'s orbital sector.</div>', unsafe_allow_html=True)
            
            with st.expander("How is this calculated?"):
                st.markdown("""
                **1. 3D Sun Mapping & Eruption Vector**  
                The 2D solar image is mathematically wrapped around the Earth-facing hemisphere of a 3D sphere. When a filament erupts, we assume it travels radially outward from the Sun's surface. We calculate its exact 3D vector based on its X/Y pixel coordinates.
                
                **2. Kinematic Drag-Based Model (DBM)**  
                Unlike simple geometrical models, CMEs do not travel at a constant speed! The CME is dynamically accelerated or decelerated by the ambient solar wind (assumed ~400 km/s) via aerodynamic drag. We calculate the filament's area to estimate the initial eruption velocity (up to 2500 km/s), and use numerical Euler integration to determine the exact Time of Arrival (TOA) at 1 AU based on the drag coefficient.
                
                **3. The Parker Spiral & Impact Detection**  
                The background environment of the heliosphere is plotted as an Archimedean spiral (the Parker Spiral), representing the Interplanetary Magnetic Field curved by the Sun's rotation. We calculate Earth's exact orbital position at the predicted TOA (moving ~0.986°/day). If the CME's trajectory intersects Earth's *future* position within a 25° radius, a Geomagnetic Storm Warning is issued!
                """)
            
            st.divider()
            st.markdown('<div class="viz-section-header">FILAMENT DETAIL INSPECTOR</div>', unsafe_allow_html=True)
            
            inspector_top = st.columns([2, 1, 1])
            selected_index = inspector_top[0].selectbox(
                "Select Filament Target", options=list(range(len(filaments))),
                format_func=lambda index: f"Filament #{filaments[index]['filament_id']} (Conf: {filaments[index].get('confidence', 0.0):.2f}, Region: {filaments[index].get('spatial_region', 'CENTER')})",
                key="detail_filament",
            )
            selected = filaments[selected_index]
            detail_padding = inspector_top[1].slider("Crop Padding (px)", 20, 60, 35, key="detail_padding")
            detail_sr_method = inspector_top[2].selectbox(
                "Super Resolution",
                ["OFF", "Lanczos (Current)", "Bicubic", "ESPCN (AI-SR)", "EDSR-Small (AI-SR)", "Solar-SR (Trained AI-SR)"],
                key="detail_sr_method",
            )
            detail_sr_scale = 2

            overlay_cols = st.columns(4)
            show_detail_mask = overlay_cols[0].checkbox("Segmentation Mask", True, key="detail_mask")
            show_detail_skeleton = overlay_cols[1].checkbox("Skeleton Spine", True, key="detail_skeleton")
            show_detail_bbox = overlay_cols[2].checkbox("Bounding Box", True, key="detail_bbox")
            show_detail_xai = overlay_cols[3].checkbox("Attribution / XAI", False, key="detail_xai")

            detail_crop, crop_bounds = crop_filament(raw, selected, detail_padding)
            enhanced_crop = cached_detail_upscale(detail_crop, detail_sr_method, detail_sr_scale) if detail_sr_method != "OFF" else None
            detail_attribution = phase2_result["attribution"]
            if show_detail_xai and detail_attribution is None:
                if phase2_result["explainability_supported"]:
                    from explainability.interface import generate_explanation
                    detail_model, _ = hub.get_model(arch)
                    with st.spinner("Computing selected-image attribution..."):
                        detail_attribution = generate_explanation(detail_model, raw, phase2_inference, arch)
                else:
                    st.info("Explainability is not currently supported for this model.")
            
            detail_overlay = selected_overlay(
                detail_crop, selected, phase2_result["labels"], crop_bounds,
                show_mask=show_detail_mask, show_skeleton=show_detail_skeleton,
                show_bbox=show_detail_bbox, show_labels=False,
                attribution=detail_attribution, show_attribution=show_detail_xai,
            )

            crop_cols = st.columns(2 if enhanced_crop is not None else 2)
            with crop_cols[0]:
                st.markdown('<div class="viz-panel-label">High-Resolution Overlay Crop</div>', unsafe_allow_html=True)
                st.image(detail_overlay, width='stretch')
            if enhanced_crop is not None:
                with crop_cols[1]:
                    st.markdown('<div class="viz-panel-label">Super-Resolution Enhanced Crop</div>', unsafe_allow_html=True)
                    st.image(enhanced_crop, width='stretch')
            else:
                with crop_cols[1]:
                    st.markdown('<div class="viz-panel-label">Raw Grayscale Crop</div>', unsafe_allow_html=True)
                    st.image(detail_crop, width='stretch')

            detail_info = {
                "Filament ID": f"#{selected['filament_id']}",
                "Confidence": round(float(selected.get("confidence", 0.0)), 3),
                "Area (px)": round(float(selected.get("area_px", 0.0)), 2),
                "Perimeter (px)": round(float(selected.get("perimeter_px", 0.0)), 2),
                "Skeleton length (px)": round(float(selected.get("skeleton_length_px", 0.0)), 2),
                "Average width (px)": round(float(selected.get("avg_width_px", 0.0)), 2),
                "Sinuosity": round(float(selected.get("sinuosity", 1.0)), 3),
                "Orientation / tilt (deg)": round(float(selected.get("orientation_deg", 0.0)), 2),
                "Centroid X": round(float(selected["centroid"].get("x", 0.0)), 2),
                "Centroid Y": round(float(selected["centroid"].get("y", 0.0)), 2),
                "Bounding box": json.dumps(selected.get("bbox", {})),
                "Spatial region": selected.get("spatial_region", "CENTER"),
                "Risk indicator": selected.get("risk_screening_indicator", "LOW"),
            }
            physical = selected.get("physical", {})
            if physical.get("calibrated"):
                detail_info["Length (km)"] = round(float(physical["length_km"]), 2)
                detail_info["Area (km²)"] = round(float(physical["area_km2"]), 2)
            st.dataframe([detail_info], hide_index=True, width='stretch')

            selected_catalog = phase2_result["catalog"][selected_index]
            json_data = json.dumps(selected_catalog, indent=2)
            csv_row = dict(selected_catalog)
            for key in ("centroid", "bbox", "physical"):
                csv_row[key] = json.dumps(csv_row[key])

            # ── Satellite Impact Predictor ───────────────────────────────────────────
            st.divider()
            st.markdown('<div class="viz-section-header">SATELLITE IMPACT PREDICTOR (CME EARTH-ARRIVAL)</div>', unsafe_allow_html=True)
            st.caption(
                "Derives estimated CME parameters from the selected filament's geometry. "
                "Uses a geometric cone model to assess which satellites would be inside the CME blast cone."
            )

            # Derive CME direction from filament centroid — SAME calculation as the 3D animation
            # so the satellite predictor is always consistent with what the 3D chart shows.
            _H, _W = raw.shape[:2]
            _cx = float(selected.get("centroid", {}).get("x", _W / 2.0))
            _cy = float(selected.get("centroid", {}).get("y", _H / 2.0))

            # Convert pixel centroid → unit 3-D eruption vector (mirrors three_d_trajectory.py)
            _fil_theta = 3 * np.pi / 2 - (_cx / _W) * np.pi
            _fil_phi   = (_cy / _H) * np.pi
            _fx = np.sin(_fil_phi) * np.cos(_fil_theta)
            _fy = np.sin(_fil_phi) * np.sin(_fil_theta)
            _fz = np.cos(_fil_phi)

            # Convert 3-D vector to Stonyhurst-like lat/lon for the geometric cone model
            # In Stonyhurst coordinates, 0° longitude is the Earth-Sun axis.
            # In 3D Cartesian space, Earth is at (-X, 0, 0), so Earth vector is -X.
            _cme_lat = float(np.degrees(np.arcsin(np.clip(_fz, -1, 1))))
            _cme_lon = float(np.degrees(np.arctan2(_fy, -_fx)))

            # Same 25° cone used by is_impact check in 3D model
            _cme_ha  = 25.0

            # Length proxy for CME speed (mirrors DBM v0 formula: v0 = 400 + area/5000*600)
            _length_px = float(selected.get("skeleton_length_px", 100.0))
            _cme_speed = max(300.0, min(2500.0, _length_px * 4.0))

            # Check if Earth itself is in the cone (same as is_impact in 3D model)
            _earth_in_cone = abs(_cme_lat) <= _cme_ha and abs(_cme_lon) <= _cme_ha
            _impact_banner = "EARTH IN CME CONE — Geomagnetic Storm Warning" if _earth_in_cone else "CME NOT DIRECTED AT EARTH — No direct Earth impact predicted"
            _banner_style  = "background:rgba(220,20,60,0.12);border-left:3px solid #DC143C;" if _earth_in_cone else "background:rgba(0,200,83,0.1);border-left:3px solid #00E676;"
            st.markdown(f'<div style="padding:10px 14px;border-radius:6px;color:white;font-weight:500;font-size:13px;margin:10px 0;{_banner_style}">{_impact_banner}</div>', unsafe_allow_html=True)

            # Show derived CME parameters
            _p1, _p2, _p3, _p4 = st.columns(4)
            _p1.metric("Est. CME Speed", f"{_cme_speed:.0f} km/s")
            _p2.metric("CME Direction", f"Lat {_cme_lat:.0f}°, Lon {_cme_lon:.0f}°")
            _p3.metric("Half-Angle", f"{_cme_ha:.0f}°")
            _p4.metric("Source Region", selected.get("spatial_region", "CENTER"))
            st.caption("Parameters derived from filament morphology — for scientific use, cross-reference with DONKI CME catalog.")

            # Evaluate satellite exposure
            from analysis.spacecraft_catalog import SpacecraftCatalog
            from analysis.cme_geometry import CMEGeometryModel
            import datetime as _dt
            _cat = SpacecraftCatalog()
            _cme_date = _dt.datetime.now().strftime("%Y-%m-%dT12:00:00")
            _sc_results = _cat.evaluate_all_spacecraft(_cme_lat, _cme_lon, _cme_ha, _cme_speed, _cme_date)

            # Classify impact levels
            _RISK_LABEL = {"INSIDE_CONE": "DIRECT HIT", "NEAR_FLANK": "NEAR FLANK", "OUTSIDE": "SAFE"}
            _RISK_ORDER = {"INSIDE_CONE": 0, "NEAR_FLANK": 1, "OUTSIDE": 2}
            _sc_results.sort(key=lambda x: _RISK_ORDER.get(x["exposure_type"], 3))

            # Build risk table
            _table_rows = []
            for r in _sc_results:
                _exp = r["exposure_type"]
                _arr = r.get("estimated_arrival", "N/A")
                if _arr and _arr != "N/A":
                    try:
                        _arr_dt = _dt.datetime.fromisoformat(_arr.replace("Z",""))
                        _arr = _arr_dt.strftime("%Y-%m-%d %H:%M UTC")
                    except Exception:
                        pass
                _table_rows.append({
                    "Status": _RISK_LABEL.get(_exp, _exp),
                    "Satellite": r["satellite_id"],
                    "Mission": r.get("mission", "—"),
                    "Operator": r.get("operator", "—"),
                    "Orbit": r.get("orbit_type", "—"),
                    "Exposure": _RISK_LABEL.get(_exp, _exp),
                    "Angular Sep.": f"{r['angular_separation']:.1f}°",
                    "Est. Arrival": _arr,
                })

            import plotly.graph_objects as go
            # Polar exposure chart
            _fig = go.Figure()
            # Sun at center
            _fig.add_trace(go.Scatterpolar(
                r=[0], theta=[0], mode="markers",
                marker=dict(color="#FFD700", size=18, symbol="circle"),
                name="Sun"
            ))
            # CME cone shading
            _cone_theta = list(range(int(_cme_lon - _cme_ha), int(_cme_lon + _cme_ha + 1)))
            _fig.add_trace(go.Scatterpolar(
                r=[0] + [1.4] * len(_cone_theta) + [0],
                theta=[_cme_lon] + _cone_theta + [_cme_lon],
                fill="toself",
                fillcolor="rgba(220,20,60,0.18)",
                line=dict(color="rgba(220,20,60,0.5)", width=1),
                name="CME Cone",
                hoverinfo="skip"
            ))
            # CME central axis
            _fig.add_trace(go.Scatterpolar(
                r=[0, 1.4], theta=[_cme_lon, _cme_lon],
                mode="lines", line=dict(color="#DC143C", width=2, dash="dash"),
                name="CME Axis"
            ))
            # Satellites
            _color_map = {"INSIDE_CONE": "#DC143C", "NEAR_FLANK": "#FFB830", "OUTSIDE": "#00C853"}
            for r in _sc_results:
                _pos = _cat.get_spacecraft_position(r["satellite_id"])
                _sc_lon = _pos.get("longitude", 0.0)
                _sc_dist = min(_pos.get("distance_au", 1.0) * 1.2, 1.3)
                _c = _color_map.get(r["exposure_type"], "#aaaaaa")
                _fig.add_trace(go.Scatterpolar(
                    r=[_sc_dist], theta=[_sc_lon],
                    mode="markers+text",
                    marker=dict(color=_c, size=12, line=dict(color="white", width=1)),
                    text=[r["satellite_id"]],
                    textposition="top center",
                    textfont=dict(size=10, color="white"),
                    name=r["satellite_id"],
                    hovertemplate=f"<b>{r['satellite_id']}</b><br>{_RISK_LABEL.get(r['exposure_type'], r['exposure_type'])}<br>Angular sep: {r['angular_separation']:.1f}°<extra></extra>"
                ))
            _fig.update_layout(
                polar=dict(
                    bgcolor="rgba(10,10,20,0.9)",
                    angularaxis=dict(direction="counterclockwise", rotation=90,
                                     gridcolor="#222230", linecolor="#333340", tickcolor="#555"),
                    radialaxis=dict(visible=False, range=[0, 1.5]),
                ),
                showlegend=True,
                legend=dict(bgcolor="rgba(0,0,0,0.5)", bordercolor="#333", borderwidth=1,
                            font=dict(color="white", size=11)),
                paper_bgcolor="rgba(0,0,0,0)",
                height=480,
                margin=dict(l=20, r=20, t=20, b=20),
            )
            st.plotly_chart(_fig, use_container_width=True)

            # Risk summary table
            _n_hit  = sum(1 for r in _sc_results if r["exposure_type"] == "INSIDE_CONE")
            _n_flnk = sum(1 for r in _sc_results if r["exposure_type"] == "NEAR_FLANK")
            _n_safe = sum(1 for r in _sc_results if r["exposure_type"] == "OUTSIDE")
            _s1, _s2, _s3 = st.columns(3)
            _s1.metric("Direct Hit", _n_hit)
            _s2.metric("Near Flank", _n_flnk)
            _s3.metric("Safe", _n_safe)

            st.dataframe(_table_rows, hide_index=True, use_container_width=True)
            phase3_metrics = {}  # keep csv export happy
            if phase3_metrics:
                csv_row.update(phase3_metrics)

            detail_csv = io.StringIO()
            csv_writer = csv.DictWriter(detail_csv, fieldnames=list(csv_row.keys()))
            csv_writer.writeheader()
            csv_writer.writerow(csv_row)
            export_cols = st.columns(4)
            export_cols[0].download_button("Export Filament JSON", json_data,
                                           file_name=f"filament_{selected['filament_id']:03d}.json",
                                           mime="application/json")
            export_cols[1].download_button("Export Filament CSV", detail_csv.getvalue(),
                                           file_name=f"filament_{selected['filament_id']:03d}.csv",
                                           mime="text/csv")
            export_cols[2].download_button("Download Original Crop", png_bytes(detail_crop),
                                           file_name=f"filament_{selected['filament_id']:03d}_original.png",
                                           mime="image/png")
            if enhanced_crop is not None:
                export_cols[3].download_button("Download Enhanced Crop", png_bytes(enhanced_crop),
                                               file_name=f"filament_{selected['filament_id']:03d}_upscaled.png",
                                               mime="image/png")
            if st.button("Save Filament Detail", key="save_detail"):
                detail_dir = Path("outputs") / "filaments" / Path(file_name).stem
                save_detail_artifacts(detail_dir, detail_crop, enhanced_crop, detail_overlay, selected)
                saved_filament_dir = detail_dir / f"filament_{int(selected['filament_id']):03d}"
                st.success(f"Saved detail artifacts to {saved_filament_dir}")
    else:
        st.info("Upload an H-alpha image to run segmentation.")


# --------------------------------------------------------------------------------------
# Upload Video
# --------------------------------------------------------------------------------------
elif page == "Upload Video":
    st.header("Upload & Evaluate a Video", divider="red")
    st.caption(labels[arch])
    st.caption("Frames are sampled evenly across the video and segmented individually.")

    uploaded = st.file_uploader("Video file", type=["mp4", "mov", "avi", "mkv"])
    n_frames = st.slider("Frames to sample", 2, 12, 6)

    if uploaded is not None:
        status = cached_status(arch)
        best_thresh = status["best_threshold"]["threshold"] if status.get("best_threshold") else 0.5

        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded.name)[1]) as tmp:
            tmp.write(uploaded.read())
            tmp_path = tmp.name

        try:
            cap = cv2.VideoCapture(tmp_path)
            total = max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 1)
            frame_idxs = np.linspace(0, total - 1, min(n_frames, total)).astype(int)

            with st.spinner(f"Segmenting {len(frame_idxs)} frames..."):
                cols = st.columns(3)
                for i, fi in enumerate(frame_idxs):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
                    ok, frame = cap.read()
                    if not ok:
                        continue
                    # frame is BGR (color) -- hub.run_inference() handles color->H-alpha
                    # conversion itself, same as the Upload Image path.
                    small, disk_small, probs, pred = hub.run_inference(frame, arch, best_thresh)
                    cols[i % 3].image(
                        overlay_rgb(small, pred, color=(220, 20, 60)),
                        caption=f"Frame {int(fi)}", width='stretch',
                    )
            cap.release()
        finally:
            try:
                os.remove(tmp_path)
            except OSError:
                pass
    else:
        st.info("Upload a video to sample and segment frames.")
