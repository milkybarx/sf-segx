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
default_arch = next((m["arch"] for m in models if m["best_val_dice"]), models[0]["arch"])
arch = st.sidebar.selectbox(
    "Model", options=list(labels.keys()), format_func=lambda a: labels[a],
    index=list(labels.keys()).index(default_arch),
)

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
                small, disk_small, probs, pred = hub.run_inference(raw, arch, best_thresh)

            from analysis.filament_morphology import analyze_filaments
            filaments = analyze_filaments(pred, probs, min_area=40)

            c1, c2, c3 = st.columns(3)
            c1.metric("Filaments Detected", len(filaments))
            c2.metric("Total Area (px)", f"{sum(f['area_px'] for f in filaments):,.0f}" if filaments else "0")
            c3.metric("Avg Confidence", f"{np.mean([f['confidence'] for f in filaments]):.2f}" if filaments else "—")
            style_metric_cards(border_left_color=CRIMSON)

            cols = st.columns(3)
            cols[0].image(small, caption="Preprocessed Input", width='stretch')
            cols[1].image(overlay_rgb(small, pred, color=(220, 20, 60)), caption="Predicted Filaments", width='stretch')
            cols[2].image(confidence_rgb(probs), caption="Confidence Heatmap", width='stretch')

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
                    st.dataframe(filaments, width='stretch')
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
