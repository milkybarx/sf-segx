"""
Shared model registry, checkpoint loading, training-log parsing, and inference logic
for every dashboard in this repo (webapp/app.py [FastAPI], streamlit_app.py, dashboard/app.py).
Framework-agnostic: no FastAPI/Streamlit/Gradio imports here, just torch/cv2/numpy.

Two families of models are exposed under one interface:
  - MODEL_REGISTRY (from train_smp.py): ImageNet-pretrained smp architectures
    (U-Net/DeepLabV3+/FPN/PSPNet/Attention-U-Net), trained + logged by this repo.
  - EXTERNAL_MODELS: models trained outside train_smp.py, each with its own "kind" (build
    function + preprocessing pipeline) and a static history (JSON/CSV/scalar) instead of a
    live train_log*.txt:
      - "mask2former": Mask2Former with a torchvision ResNet-34 pixel-decoder backbone
        (models/mask2former.py), trained at 768px with preprocessing.solar_preprocessor.
        SolarPreprocessor ([0,1]-normalized grayscale).
      - "segformer": HuggingFace SegformerForSemanticSegmentation (nvidia/mit-b2 config,
        2-class head), trained at 640px on plain-resized (no astronomical preprocessing)
        ImageNet-normalized RGB -- confirmed empirically via threshold/channel sweeps against
        real ground truth: channel 1 of the 2-class output is the filament logit (channel 0
        scores near-zero Dice), and feeding CLAHE/limb-corrected GONGPreprocessor output
        instead of a plain resize measurably hurts Dice, i.e. wrong preprocessing silently
        produces a working-looking but badly wrong model.

Color input: every model above was trained exclusively on single-channel H-alpha imagery.
run_inference() calls to_halpha_style() on every raw_img first -- a truly colored input is
converted to H-alpha style by models/color_adapter.py's ColorToHAlphaNet (checkpoints/
color_to_halpha_adapter.pth) before reaching any model-specific preprocessing, so every
architecture above gets color-image support without being retrained itself. A grayscale (or
grayscale-stored-as-3-channel, like this repo's own dataset) input passes straight through.
"""
import csv
import glob
import json
import os
import re

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2

ROOT = os.path.dirname(os.path.abspath(__file__))
DISPLAY_SIZE = 512
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])

from train_smp import GONGPreprocessor, IMG_DIR, MASK_DIR, CHECKPOINT_DIR, MODEL_REGISTRY  # noqa: E402

# data/ (the 3GB Kaggle dataset) is gitignored -- deployed environments (e.g. Streamlit
# Community Cloud) never have it. The Validation Gallery needs *some* real images+masks to
# show, so it prefers the full dataset when present (local dev) and falls back to a small
# curated set bundled directly in the repo (assets/gallery_samples/, ~32MB, 40 pairs chosen
# for visible filament content) when it isn't -- rather than silently showing nothing.
_BUNDLED_IMG_DIR = os.path.join(ROOT, "assets", "gallery_samples", "images")
_BUNDLED_MASK_DIR = os.path.join(ROOT, "assets", "gallery_samples", "masks")
if os.path.isdir(IMG_DIR) and glob.glob(f"{IMG_DIR}/*.jpeg"):
    GALLERY_IMG_DIR, GALLERY_MASK_DIR = IMG_DIR, MASK_DIR
else:
    GALLERY_IMG_DIR, GALLERY_MASK_DIR = _BUNDLED_IMG_DIR, _BUNDLED_MASK_DIR

EXTERNAL_MODELS = {
    "mask2former_phase2_hybrid": {
        "label": "Mask2Former (ResNet-34 backbone, dice+focal+boundary loss, 512px)",
        "kind": "mask2former",
        "checkpoint": os.path.join(ROOT, "checkpoints", "mask2former_phase2_hybrid_512_best.pth"),
        "resolution": 512,
        "best_threshold": 0.5,
        # Self-reported at training time (its own val split); independently re-measured
        # via this repo's real inference pipeline on an 80-image sample of the gallery
        # set and confirmed as a genuine improvement over mask2former_phase3_768
        # (0.7173 vs 0.6919 real measured mean Dice) before being added here.
        "final_metrics": {
            "epoch": 49, "total_epochs": 50,
            "val_loss": 0.19984761522761707, "val_dice": 0.7248574258952305,
            "val_iou": 0.5722777781815365, "val_precision": 0.7237881098327965,
            "val_recall": 0.7350614394607216,
        },
    },
    "mask2former_phase3": {
        "label": "Mask2Former (ResNet-34 backbone, 768px)",
        "kind": "mask2former",
        "checkpoint": os.path.join(ROOT, "checkpoints", "mask2former_phase3_768_best.pth"),
        # Native training/inference resolution -- resized up from the 512 used elsewhere
        # in this dashboard, then back down to DISPLAY_SIZE for consistent UI display.
        "resolution": 768,
        "best_threshold": 0.5,
        # No per-epoch history was kept for this checkpoint, only its final (best) epoch --
        # shown as a metrics summary instead of a Dice-over-epochs chart.
        "final_metrics": {
            "epoch": 50, "total_epochs": 50,
            "val_loss": 0.21066286736007395, "val_dice": 0.7206500790499407,
            "val_iou": 0.5708135900183998, "val_precision": 0.7056684835717596,
            "val_recall": 0.7571822695177177,
        },
    },
    "segformer_b2": {
        "label": "SegFormer (MiT-B2, 640px)",
        "kind": "segformer",
        "checkpoint": os.path.join(ROOT, "checkpoints", "segformer_b2_best.pt"),
        "config_path": os.path.join(ROOT, "configs", "segformer_mitb2_config.json"),
        # 2-class classifier head (background, filament) -- channel 1 is the filament
        # logit (confirmed empirically: channel 0 scores near-zero Dice against real
        # ground truth, channel 1 matches the checkpoint's reported val_dice).
        "num_labels": 2,
        "resolution": 640,
        # Best threshold + Dice measured empirically on this repo's data/eval pipeline
        # (raw resize, no astronomical preprocessing, ImageNet-normalized RGB, sigmoid
        # on the channel-1/filament logit -- see the "segformer" branch of run_inference).
        "best_threshold": 0.5,
        "final_metrics": {"epoch": 36, "total_epochs": 36, "val_dice": 0.6969595923322902},
    },
}

preprocessor = GONGPreprocessor()
device = torch.device("cpu")

infer_tf = A.Compose([
    A.Resize(DISPLAY_SIZE, DISPLAY_SIZE),
    A.Normalize(mean=(0.5,), std=(0.5,)),
    ToTensorV2(),
])

_models = {}          # arch -> (torch.nn.Module, checkpoint_mtime)
_sample_paths_cache = {}
_ext_preprocessor = None
_color_adapter = None
COLOR_ADAPTER_CKPT = os.path.join(ROOT, "checkpoints", "color_to_halpha_adapter.pth")
COLOR_ADAPTER_RES = 512


def log_path_for(arch: str) -> str:
    name = "train_log_unet_resnet34.txt" if arch == "unet_resnet34" else f"train_log_{arch}.txt"
    return os.path.join(ROOT, "outputs", "logs", name)


def checkpoint_path_for(arch: str) -> str:
    if arch in EXTERNAL_MODELS:
        return EXTERNAL_MODELS[arch]["checkpoint"]
    return os.path.join(CHECKPOINT_DIR, MODEL_REGISTRY[arch]["checkpoint"])


def all_archs():
    return list(MODEL_REGISTRY.keys()) + list(EXTERNAL_MODELS.keys())


def label_for(arch: str) -> str:
    if arch in EXTERNAL_MODELS:
        return EXTERNAL_MODELS[arch]["label"]
    return MODEL_REGISTRY[arch]["label"]


def shuffled_sample_paths(seed: int, n: int = 6):
    key = (seed, n)
    if key not in _sample_paths_cache:
        paths = sorted(
            glob.glob(f"{GALLERY_IMG_DIR}/*.jpg") + glob.glob(f"{GALLERY_IMG_DIR}/*.jpeg")
            + glob.glob(f"{GALLERY_IMG_DIR}/*.png")
        )
        rng = np.random.RandomState(seed)
        idx = rng.choice(len(paths), size=min(n, len(paths)), replace=False)
        _sample_paths_cache[key] = [paths[i] for i in idx]
    return _sample_paths_cache[key]


def _build_mask2former_model(checkpoint: dict):
    """Builds the from-scratch models/{unet,mask2former}.py architecture the checkpoint
    was actually trained with (read from the checkpoint's own saved config), not the
    segmentation-models-pytorch architectures in train_smp.MODEL_REGISTRY."""
    saved_config = checkpoint.get("config", {})
    model_name = saved_config.get("model", {}).get("name", "mask2former").lower()
    if model_name == "mask2former":
        from models.mask2former import build_mask2former
        return build_mask2former(saved_config.get("model", {}))
    from models.unet import build_unet
    return build_unet(saved_config.get("model", {}))


def _build_segformer_model(spec: dict):
    # Loads the MiT-B2 config from a local JSON snapshot (configs/segformer_mitb2_config.json)
    # instead of SegformerConfig.from_pretrained("nvidia/mit-b2", ...), which hits HuggingFace
    # Hub over the network every time -- pointless here since the checkpoint already has every
    # trained weight; the "pretrained" config is only used for its architecture shape (incl.
    # the 2-class classifier head), not any actual pretrained values.
    from transformers import SegformerConfig, SegformerForSemanticSegmentation
    config = SegformerConfig.from_json_file(spec["config_path"])
    return SegformerForSemanticSegmentation(config)


def get_model(arch: str):
    if arch not in MODEL_REGISTRY and arch not in EXTERNAL_MODELS:
        return None, None
    ckpt = checkpoint_path_for(arch)
    if not os.path.exists(ckpt):
        return None, None
    mtime = os.path.getmtime(ckpt)
    cached = _models.get(arch)
    if cached is None or cached[1] != mtime:
        try:
            if arch in EXTERNAL_MODELS:
                kind = EXTERNAL_MODELS[arch]["kind"]
                checkpoint = torch.load(ckpt, map_location=device, weights_only=False)
                if kind == "segformer":
                    m = _build_segformer_model(EXTERNAL_MODELS[arch])
                else:
                    m = _build_mask2former_model(checkpoint)
                m.load_state_dict(checkpoint["model_state_dict"])
            else:
                spec = MODEL_REGISTRY[arch]
                m = spec.get("build_inference", spec["build"])()
                state = torch.load(ckpt, map_location=device)
                # Some checkpoints are stored fp16 on disk to stay under GitHub's 100MB
                # single-file limit (e.g. deeplabv3plus_resnet50_best.pth, 107MB -> 54MB) --
                # upcast to fp32 so inference runs at normal precision regardless of how
                # the file was saved. A no-op for checkpoints already in fp32.
                state = {k: (v.float() if torch.is_tensor(v) and v.is_floating_point() else v)
                         for k, v in state.items()}
                m.load_state_dict(state)
        except Exception:
            return None, None
        m.eval()
        _models[arch] = (m, mtime)
    return _models[arch]


def release_model(arch: str):
    """Drop a loaded model from the in-memory cache and force garbage collection.

    Used after ensemble inference (which loads every trained architecture at once)
    to give the process's memory back -- important on memory-constrained hosts like
    Streamlit Community Cloud's free tier, where holding 5+ model checkpoints
    (including two ~270-375MB Mask2Former variants) in memory simultaneously and
    indefinitely risks an out-of-memory kill that shows up to the user as a plain
    app crash with no Python traceback."""
    if arch in _models:
        del _models[arch]
    import gc
    gc.collect()


def get_ext_preprocessor():
    global _ext_preprocessor
    if _ext_preprocessor is None:
        from preprocessing.solar_preprocessor import SolarPreprocessor
        _ext_preprocessor = SolarPreprocessor(target_size=DISPLAY_SIZE)
    return _ext_preprocessor


def _is_true_grayscale(img: np.ndarray, tol: int = 2) -> bool:
    """MAGFiLO's own JPEGs are single-channel content saved as 3-channel files (R==G==B
    exactly) -- cheap to detect and skip the color adapter entirely for those."""
    if img.ndim == 2:
        return True
    b, g, r = img[..., 0].astype(np.int16), img[..., 1].astype(np.int16), img[..., 2].astype(np.int16)
    return bool(np.abs(r - g).max() <= tol and np.abs(g - b).max() <= tol)


def get_color_adapter():
    """Lazily loads models/color_adapter.py's ColorToHAlphaNet. Returns None if no
    checkpoint has been trained yet (scripts/train_color_adapter.py) -- callers fall back
    to plain grayscale conversion in that case, never crash."""
    global _color_adapter
    if _color_adapter is None and os.path.exists(COLOR_ADAPTER_CKPT):
        from models.color_adapter import build_color_adapter
        ckpt = torch.load(COLOR_ADAPTER_CKPT, map_location=device, weights_only=False)
        m = build_color_adapter(base=ckpt.get("config", {}).get("base", 24))
        state = {k: v.float() for k, v in ckpt["model_state_dict"].items()}
        m.load_state_dict(state)
        m.eval()
        _color_adapter = m
    return _color_adapter


def to_halpha_style(raw_img: np.ndarray) -> np.ndarray:
    """Every model in this repo was trained exclusively on genuinely single-channel H-alpha
    imagery (see models/color_adapter.py docstring). A truly colored input is run through
    the trained color->H-alpha adapter so every existing model can use it unmodified; a
    grayscale/near-grayscale input (incl. this repo's own dataset, stored as 3-channel
    files with R==G==B) passes straight through as before.

    Runs the adapter at a fixed COLOR_ADAPTER_RES (512) regardless of the calling model's
    own resolution. Tried making this per-model (768 for Mask2Former, matching its own
    working resolution) on the theory that a fixed 512px pass was discarding detail
    Mask2Former needs -- measured it against real ground truth and it made Mask2Former's
    colored-input Dice *worse* (0.523 vs 0.549 hue-tinted), not better: the adapter was
    trained on 320px crops, and running it at 768px pushes it further from that training
    distribution than 512px does, which hurt more than any resolution-matching helped.
    Reverted; Mask2Former remains the one model with a real (documented) accuracy gap on
    colored input, per the Results section."""
    if _is_true_grayscale(raw_img):
        return raw_img if raw_img.ndim == 2 else cv2.cvtColor(raw_img, cv2.COLOR_BGR2GRAY)

    adapter = get_color_adapter()
    if adapter is None:
        # No trained adapter available -- fall back to a plain (imperfect) conversion
        # rather than crashing.
        return cv2.cvtColor(raw_img, cv2.COLOR_BGR2GRAY)

    h, w = raw_img.shape[:2]
    res = COLOR_ADAPTER_RES
    small = cv2.resize(raw_img, (res, res), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    inp = torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0).float()
    with torch.no_grad():
        out = adapter(inp).squeeze().numpy()
    out_u8 = np.clip(out * 255.0, 0, 255).astype(np.uint8)
    return cv2.resize(out_u8, (w, h), interpolation=cv2.INTER_LINEAR)


# --------------------------------------------------------------------------------------
# Log parsing (per architecture)
# --------------------------------------------------------------------------------------
EPOCH_RE = re.compile(
    r"Epoch \[(\d+)/(\d+)\] \| Train Loss: ([\d.]+) - Dice: ([\d.]+) \| "
    r"Val Loss: ([\d.]+) - Dice: ([\d.]+) - IoU: ([\d.]+)"
)
PROGRESS_RE = re.compile(
    r"Epoch (\d+)/(\d+) \[(train|val)\]:\s*(\d+)%\|.*?\|\s*(\d+)/(\d+)\s*\[([^\]]+)\]"
)
THRESH_RE = re.compile(r"Threshold: ([\d.]+) .*Validation Dice: ([\d.]+)")
BEST_THRESH_RE = re.compile(r"Optimal [Tt]hreshold: ([\d.]+).*Dice: ([\d.]+)")


def parse_external_status(arch: str):
    """Static status for a model trained outside this session (own results.json/csv, no live log)."""
    spec = EXTERNAL_MODELS[arch]
    ckpt = spec["checkpoint"]
    epochs, best_threshold = [], None

    if "results_json" in spec and os.path.exists(spec["results_json"]):
        with open(spec["results_json"], "r", encoding="utf-8") as f:
            results = json.load(f)
        total_epochs = results.get("total_epochs", len(results.get("history", [])))
        for e in results.get("history", []):
            epochs.append({
                "epoch": e["epoch"], "total_epochs": total_epochs,
                "train_loss": e["train"]["loss"], "train_dice": e["train"]["dice"],
                "val_loss": e["val"]["loss"], "val_dice": e["val"]["dice"], "val_iou": e["val"]["iou"],
            })
        best_threshold = {"threshold": 0.5, "dice": results.get("best_val_dice", 0.0)}

    elif "history_csv" in spec and os.path.exists(spec["history_csv"]):
        with open(spec["history_csv"], "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        total_epochs = len(rows)
        for row in rows:
            epochs.append({
                "epoch": int(row["epoch"]), "total_epochs": total_epochs,
                "train_loss": float(row["train_loss"]), "train_dice": float(row["train_dice"]),
                "val_loss": float(row["val_loss"]), "val_dice": float(row["val_dice"]),
                "val_iou": float(row["val_iou"]),
            })
        # The checkpoint is the best-Dice epoch, not necessarily the last logged row.
        best_epoch_dice = max((e["val_dice"] for e in epochs), default=0.0)
        best_threshold = {"threshold": spec.get("best_threshold", 0.5), "dice": best_epoch_dice}

    elif "final_metrics" in spec:
        # No per-epoch history was kept for this checkpoint (only its final/best epoch) --
        # no Dice-over-epochs chart to build, so just surface the scalar metrics directly.
        fm = spec["final_metrics"]
        best_threshold = {"threshold": spec.get("best_threshold", 0.5), "dice": fm.get("val_dice", 0.0)}

    return {
        "arch": arch, "label": spec["label"],
        "state": "complete" if os.path.exists(ckpt) else "not_started",
        "epochs": epochs, "current_progress": None, "thresholds": [],
        "best_threshold": best_threshold,
        "final_metrics": spec.get("final_metrics"),
        "checkpoint_exists": os.path.exists(ckpt),
        "checkpoint_mtime": os.path.getmtime(ckpt) if os.path.exists(ckpt) else None,
    }


def parse_status(arch: str):
    if arch in EXTERNAL_MODELS:
        return parse_external_status(arch)

    path = log_path_for(arch)
    if not os.path.exists(path):
        ckpt = checkpoint_path_for(arch)
        return {
            "arch": arch, "label": label_for(arch), "state": "not_started",
            "epochs": [], "current_progress": None, "thresholds": [], "best_threshold": None,
            "checkpoint_exists": os.path.exists(ckpt),
            "checkpoint_mtime": os.path.getmtime(ckpt) if os.path.exists(ckpt) else None,
        }

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        raw = f.read()
    lines = [l.strip() for l in re.split(r"[\r\n]", raw) if l.strip()]

    epochs = [m.groups() for m in (EPOCH_RE.match(l) for l in lines) if m]
    last_progress = None
    for l in reversed(lines):
        m = PROGRESS_RE.search(l)
        if m:
            last_progress = m.groups()
            break

    thresholds = [(float(m.group(1)), float(m.group(2))) for m in (THRESH_RE.search(l) for l in lines) if m]
    best_thresh_match = None
    for l in reversed(lines):
        m = BEST_THRESH_RE.search(l)
        if m:
            best_thresh_match = m.groups()
            break

    state = "idle"
    if epochs and not thresholds:
        state = "training"
    elif thresholds and not best_thresh_match:
        state = "optimizing_threshold"
    elif best_thresh_match:
        state = "complete"
    elif epochs or last_progress:
        state = "training"

    ckpt = checkpoint_path_for(arch)
    return {
        "arch": arch,
        "label": label_for(arch),
        "state": state,
        "epochs": [
            {
                "epoch": int(e[0]), "total_epochs": int(e[1]),
                "train_loss": float(e[2]), "train_dice": float(e[3]),
                "val_loss": float(e[4]), "val_dice": float(e[5]), "val_iou": float(e[6]),
            }
            for e in epochs
        ],
        "current_progress": (
            {
                "epoch": int(last_progress[0]), "total_epochs": int(last_progress[1]),
                "phase": last_progress[2], "pct": int(last_progress[3]),
                "batch": int(last_progress[4]), "total_batches": int(last_progress[5]),
                "eta": last_progress[6],
            }
            if last_progress else None
        ),
        "thresholds": thresholds,
        "best_threshold": (
            {"threshold": float(best_thresh_match[0]), "dice": float(best_thresh_match[1])}
            if best_thresh_match else None
        ),
        "checkpoint_exists": os.path.exists(ckpt),
        "checkpoint_mtime": os.path.getmtime(ckpt) if os.path.exists(ckpt) else None,
    }


def best_dice_for(status: dict):
    best_dice = max((e["val_dice"] for e in status["epochs"]), default=None)
    if status["best_threshold"] and (best_dice is None or status["best_threshold"]["dice"] > best_dice):
        best_dice = status["best_threshold"]["dice"]
    return best_dice


def list_models():
    """One row per architecture: label, trained?, state, best dice/threshold."""
    out = []
    for arch in all_archs():
        st = parse_status(arch)
        out.append({
            "arch": arch,
            "label": label_for(arch),
            "trained": st["checkpoint_exists"],
            "state": st["state"],
            "best_val_dice": best_dice_for(st),
            "best_threshold": st["best_threshold"]["threshold"] if st["best_threshold"] else None,
        })
    return out


# --------------------------------------------------------------------------------------
# Inference (all at DISPLAY_SIZE=512, not full 2048 res -> fast)
# --------------------------------------------------------------------------------------
# EXTERNAL_MODELS were trained with preprocessing.solar_preprocessor.SolarPreprocessor
# (own disk detection + [0,1] normalization), NOT our GONGPreprocessor + [-1,1] Normalize
# pipeline -- feeding the wrong preprocessing/normalization into that checkpoint would
# silently produce garbage predictions, so inference branches on which pipeline trained it.
def run_inference(raw_img: np.ndarray, arch: str, best_thresh: float):
    """Returns (display_image_uint8, disk_mask_bool, prob_map_float, pred_mask_uint8), all at DISPLAY_SIZE.

    raw_img may be single-channel grayscale OR a 3-channel color image (BGR, as from
    cv2.imread/cv2.imdecode/cv2.VideoCapture) -- a genuinely colored input is converted to
    H-alpha style via to_halpha_style() before any model-specific preprocessing runs, so
    every architecture below gets color support without being retrained itself."""
    raw_img = to_halpha_style(raw_img)
    model, _ = get_model(arch)
    if arch in EXTERNAL_MODELS and EXTERNAL_MODELS[arch]["kind"] == "segformer":
        spec = EXTERNAL_MODELS[arch]
        res = spec.get("resolution", DISPLAY_SIZE)
        # No astronomical preprocessing (see EXTERNAL_MODELS docstring) -- just resize.
        # disk_mask is still computed (geometry only, cheap) so predictions can be
        # clipped to the solar disk like every other model here.
        _, disk_mask = preprocessor.preprocess(raw_img)
        small = cv2.resize(raw_img, (DISPLAY_SIZE, DISPLAY_SIZE), interpolation=cv2.INTER_AREA)
        disk_small = cv2.resize(disk_mask.astype(np.uint8), (DISPLAY_SIZE, DISPLAY_SIZE),
                                 interpolation=cv2.INTER_NEAREST).astype(bool)
        if model is None:
            return small, disk_small, np.zeros((DISPLAY_SIZE, DISPLAY_SIZE), dtype=np.float32), \
                np.zeros((DISPLAY_SIZE, DISPLAY_SIZE), dtype=np.uint8)
        with torch.no_grad():
            # Resized directly from raw_img at the model's own trained resolution (not
            # from the already-downsampled DISPLAY_SIZE "small") to match training-time input.
            model_in = cv2.resize(raw_img, (res, res), interpolation=cv2.INTER_AREA)
            img3 = cv2.cvtColor(model_in, cv2.COLOR_GRAY2RGB).astype(np.float32) / 255.0
            img3n = (img3 - IMAGENET_MEAN) / IMAGENET_STD
            inp = torch.from_numpy(img3n.transpose(2, 0, 1)).unsqueeze(0).float()
            logits = model(pixel_values=inp).logits
            # 2-class head: channel 0 = background, channel 1 = filament (see EXTERNAL_MODELS).
            fg_logits = logits[:, 1:2] if spec.get("num_labels", 1) == 2 else logits
            probs_native = torch.sigmoid(fg_logits)
            probs_up = torch.nn.functional.interpolate(
                probs_native, size=(DISPLAY_SIZE, DISPLAY_SIZE), mode="bilinear", align_corners=False
            )
            probs = probs_up.squeeze().numpy()

    elif arch in EXTERNAL_MODELS:
        spec = EXTERNAL_MODELS[arch]
        res = spec.get("resolution", DISPLAY_SIZE)
        pre = get_ext_preprocessor()
        result = pre.preprocess(raw_img, return_intermediates=True)
        small = cv2.resize(result["preprocessed"], (DISPLAY_SIZE, DISPLAY_SIZE), interpolation=cv2.INTER_AREA)
        disk_small = cv2.resize(result["disk_mask"], (DISPLAY_SIZE, DISPLAY_SIZE),
                                 interpolation=cv2.INTER_NEAREST) > 127
        if model is None:
            return small, disk_small, np.zeros((DISPLAY_SIZE, DISPLAY_SIZE), dtype=np.float32), \
                np.zeros((DISPLAY_SIZE, DISPLAY_SIZE), dtype=np.uint8)
        with torch.no_grad():
            # Resized directly from the full-resolution preprocessed image at the model's
            # own trained resolution (not from the already-downsampled "small") for best accuracy.
            model_in = cv2.resize(result["preprocessed"], (res, res), interpolation=cv2.INTER_AREA)
            inp = torch.from_numpy(model_in.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0)
            probs_native = torch.sigmoid(model(inp))
            if res != DISPLAY_SIZE:
                probs_native = torch.nn.functional.interpolate(
                    probs_native, size=(DISPLAY_SIZE, DISPLAY_SIZE), mode="bilinear", align_corners=False
                )
            probs = probs_native.squeeze().numpy()
    else:
        enhanced_img, disk_mask = preprocessor.preprocess(raw_img)
        small = cv2.resize(enhanced_img, (DISPLAY_SIZE, DISPLAY_SIZE), interpolation=cv2.INTER_AREA)
        disk_small = cv2.resize(disk_mask.astype(np.uint8), (DISPLAY_SIZE, DISPLAY_SIZE),
                                 interpolation=cv2.INTER_NEAREST).astype(bool)
        if model is None:
            return small, disk_small, np.zeros((DISPLAY_SIZE, DISPLAY_SIZE), dtype=np.float32), \
                np.zeros((DISPLAY_SIZE, DISPLAY_SIZE), dtype=np.uint8)
        with torch.no_grad():
            inp = infer_tf(image=enhanced_img)["image"].unsqueeze(0)
            probs = torch.sigmoid(model(inp)).squeeze().numpy()

    pred_mask = (probs > best_thresh).astype(np.uint8)
    pred_mask[~disk_small] = 0
    return small, disk_small, probs, pred_mask


def load_gt_mask(img_path: str, display_size: int = DISPLAY_SIZE):
    fn = os.path.splitext(os.path.basename(img_path))[0]
    gt_full = cv2.imread(os.path.join(GALLERY_MASK_DIR, fn + ".png"), cv2.IMREAD_GRAYSCALE)
    if gt_full is None:
        return np.zeros((display_size, display_size), dtype=np.uint8)
    gt_full = (gt_full > 127).astype(np.uint8)
    return cv2.resize(gt_full, (display_size, display_size), interpolation=cv2.INTER_NEAREST)


def dice_score(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    return float((2.0 * (pred_mask & gt_mask).sum()) / (pred_mask.sum() + gt_mask.sum() + 1e-6))
