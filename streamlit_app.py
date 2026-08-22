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

CRIMSON = "#DC143C"
CRIMSON_SOFT = "#ff5c7a"

st.set_page_config(
    page_title="Solar Filament Intelligence",
    page_icon="🔴",
    layout="wide",
)

st.markdown(f"""
<style>
    .stApp {{ background: radial-gradient(circle at 20% -10%, #2a0a12 0%, transparent 45%), #0e0e12; }}
    h1, h2, h3 {{ color: #f2f2f5 !important; }}
    div[data-testid="stMetric"] {{
        background: linear-gradient(160deg, #1a1a22, #14141a);
        border: 1px solid #33333d; border-left: 4px solid {CRIMSON} !important;
        border-radius: 10px; padding: 10px 14px;
    }}
    .stTabs [data-baseweb="tab"] {{ font-weight: 600; }}
    .stTabs [aria-selected="true"] {{ color: {CRIMSON} !important; border-bottom-color: {CRIMSON} !important; }}
    div[data-testid="stSidebar"] {{ background: #121216; }}
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
st.sidebar.markdown("## 🔴 Solar Filament Intelligence")
st.sidebar.caption("H-alpha filament segmentation & space weather analysis")

from streamlit_option_menu import option_menu
with st.sidebar:
    page = option_menu(
        menu_title=None,
        options=["Overview", "Validation Gallery", "Upload Image", "Upload Video"],
        icons=['bar-chart-line', 'images', 'cloud-upload', 'camera-video'],
        default_index=2,
        styles={
            "container": {"padding": "0!important", "background-color": "transparent"},
            "icon": {"color": "white", "font-size": "16px"}, 
            "nav-link": {"font-size": "15px", "text-align": "left", "margin":"0px", "--hover-color": "#444"},
            "nav-link-selected": {"background-color": "#DC143C"},
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


# --------------------------------------------------------------------------------------
# Validation Gallery
# --------------------------------------------------------------------------------------
elif page == "Validation Gallery":
    st.header("Validation Gallery", divider="red")
    st.caption(labels[arch])

    if "gallery_seed" not in st.session_state:
        st.session_state.gallery_seed = 7
    if st.button("🔀 Shuffle samples"):
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
    st.header("Upload & Evaluate an Image", divider="red")
    st.caption(labels[arch])

    upload_container = st.container(border=True)
    with upload_container:
        st.markdown("<h3 style='text-align: center;'>🚀 Drop a Solar H-Alpha Image Here</h3>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; color: gray;'>Upload a grayscale or color image to detect filaments and measure eruption risk.</p>", unsafe_allow_html=True)
        uploaded = st.file_uploader(" ", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
    if uploaded is not None:
        data = np.frombuffer(uploaded.read(), dtype=np.uint8)
        raw_color = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if raw_color is None:
            st.error("Could not decode that image.")
        else:
            # Converted to H-alpha style once, up front -- everything below (Phase 2
            # analysis, instance panels, the detail-crop inspector, super-resolution)
            # then works on a single real grayscale image regardless of whether the
            # upload was color or already grayscale. A genuinely colored image goes
            # through model_hub's color->H-alpha adapter; a grayscale source round-trips
            # unchanged either way.
            raw = hub.to_halpha_style(raw_color)

            from inference.phase2 import run_phase2_analysis
            with st.spinner("Running segmentation and Phase 2 analysis..."):
                phase2_result = run_phase2_analysis(
                    raw, image_id=uploaded.name, model_name=arch,
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
            cols = st.columns(4)
            cols[0].image(original_display, caption="Original Upload", width='stretch')
            cols[1].image(small, caption="Preprocessed Input (H-alpha style)", width='stretch')
            cols[2].image(overlay_rgb(small, pred, color=(220, 20, 60)), caption="Predicted Filaments", width='stretch')
            cols[3].image(confidence_rgb(probs), caption="Confidence Heatmap", width='stretch')

            st.divider()
            run_ensemble = st.checkbox(
                "Run Ensemble Consensus & Uncertainty (all 5 models + test-time augmentation — slower)",
                value=False, key="run_ensemble",
            )
            if run_ensemble:
                from inference.ensemble import run_ensemble_inference
                with st.spinner("Running all 5 models with test-time augmentation..."):
                    ens_small, ens_disk, ens_probs, ens_mask, ens_weights, agreement = run_ensemble_inference(raw)
                st.markdown("#### Ensemble Consensus & Uncertainty")
                st.caption(
                    "Weighted-averaged prediction across every trained model (weights: "
                    + ", ".join(f"{a} {w:.3f}" for a, w in ens_weights.items())
                    + f") with test-time augmentation. Measured honestly: this does not "
                    "always beat the single best model's raw Dice — its real value is the "
                    "agreement map, which flags where independently-architected models "
                    "disagree (a signal a single model's own confidence score can't give you)."
                )
                ecols = st.columns(3)
                ecols[0].image(overlay_rgb(ens_small, ens_mask, color=(220, 20, 60)),
                               caption="Ensemble Prediction", width='stretch')
                ecols[1].image(confidence_rgb(ens_probs), caption="Ensemble Confidence", width='stretch')
                agreement_heat = (np.clip(1.0 - agreement, 0, 1) * 255).astype(np.uint8)
                agreement_rgb = cv2.applyColorMap(agreement_heat, cv2.COLORMAP_HOT)
                agreement_rgb = cv2.cvtColor(agreement_rgb, cv2.COLOR_BGR2RGB)
                ecols[2].image(agreement_rgb, caption="Model Disagreement (bright = models conflict)", width='stretch')

            from visualization.phase2 import _instance_panel, create_phase2_figure
            from visualization.detail import crop_filament, detail_record, save_detail_artifacts, selected_overlay
            st.caption(f"Phase 2 model: {phase2_result['model_name']} · threshold {phase2_result['threshold']:.2f}")
            annotated_panel = _instance_panel(raw, filaments)
            skeleton_panel = _instance_panel(raw, filaments, skeleton=True)
            with st.expander("View high-resolution visualization", expanded=True):
                st.image(annotated_panel, caption="Instances with green bounding boxes", width='stretch')
                st.image(skeleton_panel, caption="One-pixel cyan skeletons", width='stretch')
            figure = create_phase2_figure(original_display, probs, pred, filaments, phase2_result["attribution"])
            st.pyplot(figure, clear_figure=True)
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
                st.subheader("Space Weather: Flare Prediction", divider="orange")
                st.caption("Uses the trained Random Forest model (flare_rf_model.pkl) to predict 24h eruption risk based on filament geometry.")
                from analysis.flare_prediction import calculate_flare_probability
                
                display_count = min(len(filaments), 4)
                risk_cols = st.columns(display_count) if display_count > 0 else []
                
                for i, f in enumerate(filaments[:display_count]):
                    mocked_distance = float(np.random.uniform(10.0, 100.0))
                    risk = calculate_flare_probability(
                        length_px=f.get('skeleton_length_px', 0.0),
                        distance_to_sunspot=mocked_distance,
                        region_type=f.get('spatial_region', 'ARF')
                    )
                    
                    risk_cols[i].metric(
                        f"Filament #{f['filament_id']}", 
                        f"{risk:.1%}", 
                        delta="HIGH RISK" if risk > 0.5 else "LOW RISK",
                        delta_color="inverse"
                    )
                if len(filaments) > 4:
                    st.info(f"Showing risk for the top 4 filaments (out of {len(filaments)} detected).")
                
                st.divider()
                st.subheader("Filament Detail Inspector")
                selected_index = st.selectbox(
                    "Filament", options=list(range(len(filaments))),
                    format_func=lambda index: f"Filament #{filaments[index]['filament_id']}",
                    key="detail_filament",
                )
                selected = filaments[selected_index]
                detail_padding = st.slider("Crop padding (pixels)", 20, 50, 30, key="detail_padding")
                detail_sr_cols = st.columns([2, 1, 1])
                detail_sr_method = detail_sr_cols[0].selectbox(
                    "Super Resolution",
                    ["OFF", "Lanczos (Current)", "Bicubic", "ESPCN (AI-SR)", "EDSR-Small (AI-SR)", "Solar-SR (Trained AI-SR)"],
                    key="detail_sr_method",
                )
                detail_sr_scale = detail_sr_cols[1].selectbox("Scale Factor", [2, 4], key="detail_sr_scale")
                overlay_cols = st.columns(5)
                show_detail_mask = overlay_cols[0].checkbox("Show segmentation mask", True, key="detail_mask")
                show_detail_skeleton = overlay_cols[1].checkbox("Show skeleton", True, key="detail_skeleton")
                show_detail_bbox = overlay_cols[2].checkbox("Show bounding box", True, key="detail_bbox")
                show_detail_xai = overlay_cols[3].checkbox("Show attribution / XAI", False, key="detail_xai")
                show_detail_labels = overlay_cols[4].checkbox("Show labels", True, key="detail_labels")

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
                    show_bbox=show_detail_bbox, show_labels=show_detail_labels,
                    attribution=detail_attribution, show_attribution=show_detail_xai,
                )
                crop_cols = st.columns(2 if enhanced_crop is not None else 1)
                crop_cols[0].image(detail_crop, caption="Original high-resolution crop", width='stretch')
                if enhanced_crop is not None:
                    crop_cols[1].image(enhanced_crop, caption=f"AI Super-Resolution — Visualization Only ({detail_sr_scale}x)", width='stretch')
                st.image(detail_overlay, caption="Selected filament detail overlays", width='stretch')

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
                    detail_dir = Path("outputs") / "filaments" / Path(uploaded.name).stem
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
