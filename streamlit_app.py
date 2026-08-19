"""
Solar Filament Segmentation & Space Weather Intelligence — Streamlit Dashboard
================================================================================
Rebuilt on model_hub.py (shared with webapp/app.py) so both dashboards stay
consistent. Free-hostable on Streamlit Community Cloud.

Design: crimson accent theme (.streamlit/config.toml), visual stats (Plotly
charts + metric cards) instead of raw per-epoch number tables, using
streamlit-extras for polished metric cards / section headers.
"""
import os
import sys
import tempfile
import json
import io
import csv
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


def morphology_table_rows(filaments):
    """Return Arrow-compatible scalar rows without internal NumPy masks."""
    rows = []
    for filament in filaments:
        bbox = filament.get("bbox", {})
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
            "BBox": f"({bbox.get('x_min', 0)}, {bbox.get('y_min', 0)}) - "
                    f"({bbox.get('x_max', 0)}, {bbox.get('y_max', 0)})",
            "Spatial region": filament.get("spatial_region", "CENTER"),
            "Risk screening": filament.get("risk_screening_indicator", "LOW"),
        })
    return rows


def png_bytes(image: np.ndarray) -> bytes:
    """Encode an image for a Streamlit download without changing its resolution."""
    value = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) if image.ndim == 3 else image
    ok, encoded = cv2.imencode(".png", value)
    if not ok:
        raise ValueError("Could not encode image as PNG")
    return encoded.tobytes()


@st.cache_data(show_spinner=False)
def cached_detail_upscale(crop: np.ndarray, scale: int) -> np.ndarray:
    """Cache the selected filament's lightweight display-only enhancement."""
    from visualization.detail import high_quality_upscale
    return high_quality_upscale(crop, scale)


@st.cache_data(ttl=5)
def cached_model_list():
    return hub.list_models()


@st.cache_data(ttl=5)
def cached_status(arch):
    return hub.parse_status(arch)


def model_options():
    models = cached_model_list()
    labels = {m["arch"]: f"{m['label']}" + (f" · Dice {m['best_val_dice']:.3f}" if m["best_val_dice"] else " · not trained") for m in models}
    return models, labels


# --------------------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------------------
st.sidebar.markdown("## 🔴 Solar Filament Intelligence")
st.sidebar.caption("H-alpha filament segmentation & space weather analysis")

models, labels = model_options()
default_arch = next((m["arch"] for m in models if m["arch"] == "mask2former"), models[0]["arch"])
arch = st.sidebar.selectbox(
    "Model", options=list(labels.keys()), format_func=lambda a: labels[a],
    index=list(labels.keys()).index(default_arch),
)
phase2_threshold = st.sidebar.slider("Mask2Former threshold", 0.20, 0.75, 0.50, 0.01)
show_explainability = st.sidebar.checkbox("Mask2Former explainability", value=False)

page = st.sidebar.radio("View", ["Overview", "Validation Gallery", "Upload Image", "Upload Video"])

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
    best_dice = hub.best_dice_for(status)
    best_iou = max((e["val_iou"] for e in epochs), default=None)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Status", status["state"].replace("_", " ").title())
    c2.metric("Best Val Dice", f"{best_dice:.3f}" if best_dice else "—")
    c3.metric("Best Val IoU", f"{best_iou:.3f}" if best_iou else "—")
    c4.metric("Epochs Logged", f"{len(epochs)}" + (f" / {epochs[-1]['total_epochs']}" if epochs else ""))
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
        )
        st.plotly_chart(fig2, width='stretch')
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
            margin=dict(l=10, r=40, t=10, b=10), xaxis_title="Best Validation Dice",
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

    uploaded = st.file_uploader("H-alpha solar image", type=["jpg", "jpeg", "png"])
    if uploaded is not None:
        data = np.frombuffer(uploaded.read(), dtype=np.uint8)
        raw = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
        if raw is None:
            st.error("Could not decode that image.")
        else:
            status = cached_status(arch)
            best_thresh = status["best_threshold"]["threshold"] if status.get("best_threshold") else 0.5
            with st.spinner("Running segmentation..."):
                if arch == "mask2former":
                    from inference.phase2 import run_phase2_analysis
                    phase2_result = run_phase2_analysis(raw, image_id=uploaded.name, threshold=phase2_threshold,
                                                        explain=show_explainability)
                    phase2_inference = phase2_result["inference"]
                    small = phase2_inference.preprocessed
                    probs = cv2.resize(phase2_inference.probability, (small.shape[1], small.shape[0]))
                    pred = cv2.resize(phase2_inference.mask, (small.shape[1], small.shape[0]), interpolation=cv2.INTER_NEAREST)
                else:
                    phase2_result = None
                    small, disk_small, probs, pred = hub.run_inference(raw, arch, best_thresh)

            from analysis.filament_morphology import analyze_filaments
            filaments = phase2_result["filaments"] if phase2_result else analyze_filaments(pred, probs, min_area=40)

            c1, c2, c3 = st.columns(3)
            c1.metric("Filaments Detected", len(filaments))
            c2.metric("Total Area (px)", f"{sum(f['area_px'] for f in filaments):,.0f}" if filaments else "0")
            c3.metric("Avg Confidence", f"{np.mean([f['confidence'] for f in filaments]):.2f}" if filaments else "—")
            style_metric_cards(border_left_color=CRIMSON)

            cols = st.columns(3)
            cols[0].image(small, caption="Preprocessed Input", width='stretch')
            cols[1].image(overlay_rgb(small, pred, color=(220, 20, 60)), caption="Predicted Filaments", width='stretch')
            cols[2].image(confidence_rgb(probs), caption="Confidence Heatmap", width='stretch')

            if phase2_result:
                from visualization.phase2 import _instance_panel, create_phase2_figure
                from visualization.detail import crop_filament, detail_record, save_detail_artifacts, selected_overlay
                annotated_panel = _instance_panel(raw, filaments)
                skeleton_panel = _instance_panel(raw, filaments, skeleton=True)
                with st.expander("View high-resolution visualization", expanded=True):
                    st.image(annotated_panel, caption="Instances with green bounding boxes", width='stretch')
                    st.image(skeleton_panel, caption="One-pixel cyan skeletons", width='stretch')
                figure = create_phase2_figure(small, probs, pred, filaments, phase2_result["attribution"])
                st.pyplot(figure, clear_figure=True)
                st.download_button("Download JSON catalog", json.dumps(phase2_result["catalog"], indent=2),
                                   file_name="filament_catalog.json", mime="application/json")
                csv_buffer = io.StringIO()
                writer = csv.DictWriter(csv_buffer, fieldnames=["image_id", "model", "filament_id", "confidence", "area_px", "skeleton_length_px", "sinuosity", "orientation_deg", "spatial_region"])
                writer.writeheader()
                writer.writerows({key: record.get(key) for key in writer.fieldnames} for record in phase2_result["catalog"])
                st.download_button("Download CSV catalog", csv_buffer.getvalue(), file_name="filament_catalog.csv", mime="text/csv")

                st.divider()
                st.subheader("Filament Detail Inspector")
                selected_index = st.selectbox(
                    "Filament", options=list(range(len(filaments))),
                    format_func=lambda index: f"Filament #{filaments[index]['filament_id']}",
                    key="detail_filament",
                )
                selected = filaments[selected_index]
                detail_padding = st.slider("Crop padding (pixels)", 20, 50, 30, key="detail_padding")
                detail_sr = st.checkbox("Enable Super Resolution", key="detail_super_resolution")
                overlay_cols = st.columns(5)
                show_detail_mask = overlay_cols[0].checkbox("Show segmentation mask", True, key="detail_mask")
                show_detail_skeleton = overlay_cols[1].checkbox("Show skeleton", True, key="detail_skeleton")
                show_detail_bbox = overlay_cols[2].checkbox("Show bounding box", True, key="detail_bbox")
                show_detail_xai = overlay_cols[3].checkbox("Show attribution / XAI", False, key="detail_xai")
                show_detail_labels = overlay_cols[4].checkbox("Show labels", True, key="detail_labels")

                detail_crop, crop_bounds = crop_filament(raw, selected, detail_padding)
                enhanced_crop = cached_detail_upscale(detail_crop, 2) if detail_sr else None
                detail_attribution = phase2_result["attribution"]
                if show_detail_xai and detail_attribution is None:
                    from explainability.segmentation_attribution import segmentation_attribution
                    detail_model, _ = hub.get_model("mask2former")
                    if detail_model is not None:
                        detail_input_mask = cv2.resize(
                            phase2_inference.mask,
                            (phase2_inference.preprocessed.shape[1], phase2_inference.preprocessed.shape[0]),
                            interpolation=cv2.INTER_NEAREST,
                        )
                        with st.spinner("Computing selected-image attribution..."):
                            detail_attribution = segmentation_attribution(
                                detail_model, phase2_inference.preprocessed, detail_input_mask,
                            )
                detail_overlay = selected_overlay(
                    detail_crop, selected, phase2_result["labels"], crop_bounds,
                    show_mask=show_detail_mask, show_skeleton=show_detail_skeleton,
                    show_bbox=show_detail_bbox, show_labels=show_detail_labels,
                    attribution=detail_attribution, show_attribution=show_detail_xai,
                )
                crop_cols = st.columns(2 if enhanced_crop is not None else 1)
                crop_cols[0].image(detail_crop, caption="Original high-resolution crop", width='stretch')
                if enhanced_crop is not None:
                    crop_cols[1].image(enhanced_crop, caption="High-quality upscaled (display enhancement; no new information)", width='stretch')
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
                    paths = save_detail_artifacts(detail_dir, detail_crop, enhanced_crop, detail_overlay, selected)
                    saved_filament_dir = detail_dir / f"filament_{int(selected['filament_id']):03d}"
                    st.success(f"Saved detail artifacts to {saved_filament_dir}")

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
                    xaxis_title="Filament ID", yaxis_title="Skeleton Length (px)",
                )
                st.plotly_chart(fig, width='stretch')
                with st.expander("Per-filament measurements"):
                    st.dataframe(morphology_table_rows(filaments), width='stretch')
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
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    small, disk_small, probs, pred = hub.run_inference(gray, arch, best_thresh)
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
