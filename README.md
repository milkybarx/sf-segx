# Solar Filament Segmentation & Space Weather Intelligence System

AI-powered detection and segmentation of solar filaments in H-alpha full-disk solar
imagery, combining deep learning, classical computer vision, and hybrid fusion, with
morphology analysis, explainability, and an interactive dashboard.

Built for **GGSIPU Hackathon 2026 — Track 19: AI-Based Automated Solar Filament
Segmentation & Space Weather Intelligence System**, University School of Automation &
Robotics (USAR), GGSIPU, using the [Kaggle "filament-segmentation-2026"](https://www.kaggle.com/competitions/filament-segmentation-2026)
competition's MAGFiLO 1.0 dataset.

**Beyond segmentation — actual space weather intelligence**: `notebooks/Copy_of_FINAL_SOLAR.ipynb`'s
post-processing step doesn't stop at drawing masks. Every detected filament gets a
**space-weather risk rating** (LOW/MODERATE/HIGH/SEVERE) from its physical length —
longer filaments store more free magnetic energy along their polarity-inversion-line
channel and are empirically more eruption-prone and geoeffective, with >150,000 km used
as the "giant filament" cutoff. Bounding boxes are color-coded by risk (green → yellow →
orange → red) directly on the full-disk overlay, and every detection is exported with its
risk tag to `outputs/filament_catalog.json`/`.csv` — turning a segmentation mask into a
scannable risk catalog, which is the actual point of a *Space Weather Intelligence System*.

## Results

One backbone/encoder per architecture family — no redundant variants of the same model:

| Model | Backbone | Params | Epochs | Best Val Dice | Best Val IoU | Notes |
|---|---|---|---|---|---|---|
| **Mask2Former** | ResNet-34 (ImageNet), 768px | 22.6M | 50 | **0.7207** | 0.5708 | Best model overall. See *leakage caveat* below |
| SegFormer | MiT-B2 (ImageNet), 640px | 27.3M | 36 | 0.6970 | — | See *SegFormer preprocessing* note below |
| U-Net | ResNet-34 (ImageNet) | 24.4M | 15 | 0.6611 | 0.4945 | Clean file-level split |
| DeepLabV3+ | ResNet-50 (ImageNet) | 26.7M | 35 | 0.6521 | 0.4844 | Clean file-level split |
| Attention U-Net | MONAI, from scratch | 7.9M | 20/25 | 0.6507 | 0.4829 | Run crashed epoch 22 on a transient file-read error (fixed); finalized from the epoch-20 checkpoint rather than re-run, since Dice had plateaued/was oscillating in 0.63-0.65 for ~10 epochs |

All trained models are in `checkpoints/` and selectable live in the dashboard
(`streamlit_app.py`), including on your own uploaded images/videos. A fine-tuned
SAM/MedSAM is still scoped for a future pass — not included yet (see below).

### A correction worth being upfront about

The original codebase's README described the shipped `checkpoints/best_model.pth` as a
"U-Net" achieving 0.699 Dice. **It was actually a Mask2Former** (`models/mask2former.py`,
2.76M params, from-scratch FPN pixel decoder) — confirmed by loading it and inspecting
`checkpoint['config']['model']['name']`, not by re-reading prose.

Separately, that same run's validation split (`preprocessing/dataset.py`) split by raw COCO
`image_id`, but **707 physical images have 1,154 `image_id` entries** (up to 3 separate
annotation sessions per image, sharing the same `file_name`/pixels). Splitting by `image_id`
let **121 of 296 duplicate-session images leak across train/val** — the model could see a
given photo in training and be "validated" on the same photo with a different session's
mask. This has been fixed (`create_data_splits` now splits by unique `file_name` first,
then assigns all of a file's sessions to one side) — verified zero file overlap after the
fix. This repo's `train_smp.py`-trained models always used a clean file-level 85/15 split
with single-session (not unioned) ground-truth masks — see the next section.

**That original 0.6990-Dice checkpoint has since been superseded and removed from the repo.**
The current `checkpoints/mask2former_phase3_768_best.pth` is a *different, later* training
run — a Mask2Former with a real `torchvision.models.resnet34` pixel-decoder backbone
(pretrained on ImageNet) instead of the from-scratch FPN, trained at 768px instead of 512px,
reaching 0.7207 Dice at epoch 50. It arrived as a bare `state_dict` with no accompanying
code, so its architecture (`models/mask2former.py`'s `ResNetPixelDecoder`) was reconstructed
entirely from the checkpoint's own key names and tensor shapes and validated by a strict
`load_state_dict()` (zero missing, zero unexpected keys) plus real Dice scores on held-out
ground truth matching the checkpoint's own claimed number. Its own saved config records
`train_ratio: 0.8, val_ratio: 0.2, seed: 42`, but which split *methodology* (leakage-prone
by-`image_id`, or the fixed by-`file_name`) produced that split was not independently
re-verified — treat 0.7207 with the same "possibly optimistic" caution as the original run
until re-confirmed.

### A second, similar data-quality fix

`scripts/prepare_masks.py` (used by `train_smp.py`) originally **unioned** every
annotation session's polygons for a duplicate-session file into one mask. Measuring this
on a sample file: union area was **2.4x** the intersection area — sessions disagree
substantially, so unioning inflates/noises the target rather than producing consensus.
Fixed to use one canonical session per file instead.

### SegFormer preprocessing — a silent-failure trap

`checkpoints/segformer_b2_best.pt` (HuggingFace `SegformerForSemanticSegmentation`, MiT-B2,
640px, 2-class head) was trained on **plain-resized RGB images with ImageNet normalization —
no CLAHE, no limb-darkening correction, no disk detection**, unlike every other model in
this repo. Feeding it this repo's `GONGPreprocessor` output instead (the "obviously more
sophisticated" choice) doesn't error — it just measurably drops Dice on a held-out sample,
i.e. a wrong-but-plausible-looking model. It also arrived as a bare `state_dict` with a
2-channel classifier output and no indication of which channel is foreground or whether to
read it via softmax or independent sigmoid; a threshold/channel sweep against real ground
truth showed **channel 0 scores near-zero Dice (background) and channel 1 matches the
checkpoint's reported val_dice (filament)** — sigmoid on channel 1, not softmax. Both facts
were confirmed by testing against real ground truth before wiring it into `model_hub.py`,
not assumed. The lesson generalizes: *a model loading successfully proves the architecture
matches, not that the preprocessing or output convention does* — every external checkpoint
in this repo (`checkpoints/*.pth`/`*.pt` not produced by `train_smp.py`) needed its exact
training-time preprocessing and output convention reverse-engineered and verified against
ground truth before being trusted, not assumed from what "should" work.

### MedSAM — attempted, not included

A `03_MedSAM_Training.ipynb` was found alongside the SegFormer work, but its saved outputs
show it crashed on cell 7 (`ModuleNotFoundError: No module named 'sklearn'`) before the
dataset/model were ever built, and no checkpoint was produced. Rather than ship a notebook
that doesn't run, it wasn't carried into this repo — a fine-tuned SAM/MedSAM remains future
work.

## Project Structure

```
.
├── README.md
├── requirements.txt
├── streamlit_app.py             # the dashboard (only one -- see Dashboard section)
├── .streamlit/config.toml       # crimson theme
├── model_hub.py                 # shared model registry + checkpoint loading + inference,
│                                 #   the single source of truth streamlit_app.py runs on
├── configs/
│   ├── default_config.yaml      # hyperparameters for training/train.py
│   └── segformer_mitb2_config.json  # local SegFormer MiT-B2 architecture snapshot (offline-safe)
├── data/                        # MAGFiLO dataset (download separately, see Setup)
│   └── MAGFiLO_1.0_Kaggle_2026/
│       ├── train/{train_images, train_masks, train_preprocessed}/
│       └── test/test_images/
├── scripts/
│   ├── prepare_masks.py         # rasterizes COCO polygons -> per-image PNG masks
│   ├── prepare_cache.py         # precomputes+caches GONGPreprocessor output
│   └── finalize_attention_unet.py  # one-off: threshold-sweep+finalize a checkpoint
│                                    #   without a live training run (see Results table)
├── preprocessing/
│   ├── solar_preprocessor.py    # disk detection, limb correction, CLAHE
│   ├── dataset.py               # COCO parsing, PyTorch Dataset, .npy caching
│   └── build_cache.py
├── classical/                   # Frangi / Hessian ridge-detection pipeline
│   ├── frangi.py, hessian.py, morphology.py, advanced_extractor.py
├── models/
│   ├── unet.py                  # from-scratch U-Net
│   └── mask2former.py           # from-scratch, lightweight Mask2Former
├── training/                    # original from-scratch trainer (U-Net / Mask2Former)
│   ├── train.py, losses.py, metrics.py
├── train_smp.py                  # multi-architecture trainer (this session's addition):
│                                  #   ImageNet-pretrained U-Net/DeepLabV3+/Attention-U-Net
│                                  #   via segmentation-models-pytorch/MONAI, `--arch <name>`
├── hybrid/
│   └── fusion.py                # weighted DL + Frangi fusion, alpha sweep
├── analysis/
│   └── filament_morphology.py   # per-filament area, length, orientation, width...
├── explainability/
│   └── confidence.py            # confidence / uncertainty maps
├── visualization/
│   └── viz.py
├── inference/
│   └── predict.py               # SolarFilamentPredictor: unet/frangi/hybrid inference
├── checkpoints/                  (tracked in git via Git LFS -- see "Checkpoints ship in the repo")
│   ├── mask2former_phase3_768_best.pth  # 0.7207 Dice, best model (fp16 on disk, ~45MB)
│   ├── segformer_b2_best.pt     # 0.6970 Dice (fp16 on disk, ~55MB, see preprocessing caveat above)
│   ├── unet_resnet34_best.pth   # 0.6611 Dice (fp16 on disk, ~46MB)
│   ├── deeplabv3plus_resnet50_best.pth  # 0.6521 Dice (fp16 on disk, ~51MB)
│   └── attention_unet_best.pth  # 0.6507 Dice (epoch 20/25, see caveat above)
├── experiments/
│   ├── evaluate.py, generate_report.py, plot_results.py
├── outputs/
│   ├── logs/                    # train_smp.py per-architecture training logs
│   ├── dataset_report/          # dataset statistics + sample annotation figures
│   └── training_summary_*.txt, training_curves_unet_scratch.png
├── notebooks/
│   ├── Copy_of_FINAL_SOLAR.ipynb  # end-to-end Colab-style walkthrough: preprocessing ->
│   │                              #   rotation-augmentation sanity check -> training ->
│   │                              #   evaluation -> post-processing (instance separation,
│   │                              #   skeleton/sinuosity/tilt measurements, Grad-CAM,
│   │                              #   JSON/CSV catalog export, space-weather risk rating)
│   ├── reference_segformer_01_data_prep.ipynb  # data prep for the earlier MiT-B0 SegFormer run
│   └── reference_segformer_02_training.ipynb   # training for the earlier MiT-B0 SegFormer run
│                                    #   (superseded by segformer_b2_best.pt, no equivalent
│                                    #   notebook exists for the MiT-B2/Mask2Former-phase3
│                                    #   checkpoints -- they arrived pre-trained, see caveats above)
├── docs/
│   ├── Solar_Filament_Complete_Guide_Zero_To_Hundred.pdf
│   └── Solar_Filament_Technical_Audit.pdf
└── dataset_report.md
```

## Setup

```bash
pip install -r requirements.txt

# GPU (CUDA 12.4) build of torch/torchvision instead of the CPU wheel, for local training:
pip install --index-url https://download.pytorch.org/whl/cu124 torch==2.6.0 torchvision==0.21.0
```

### Get the data

Requires a Kaggle account that has joined the
[filament-segmentation-2026](https://www.kaggle.com/competitions/filament-segmentation-2026)
competition (accept its rules on the website — the API can't do this step) and an API
token in `~/.kaggle/`.

```bash
kaggle competitions download -c filament-segmentation-2026 -p data
python -c "import zipfile; zipfile.ZipFile('data/filament-segmentation-2026.zip').extractall('data')"

python scripts/prepare_masks.py   # rasterize ground-truth masks (~15s)
python scripts/prepare_cache.py   # precompute preprocessing cache (~3min, speeds up train_smp.py a lot)
```

### Checkpoints ship in the repo

All 5 trained checkpoints are tracked in `checkpoints/` via **Git LFS** — `git lfs install`
once, then a normal `git clone`/`pull` transparently fetches the real weights instead of
pointer files (GitHub's `git clone` UI and `git lfs env` both confirm this without any extra
flags). Every checkpoint is stored with float16 weights and stripped of optimizer/scheduler
state (inference-only) to stay small: `unet_resnet34_best.pth` and
`deeplabv3plus_resnet50_best.pth` were re-saved this way specifically to clear GitHub's
100MB single-file limit (107MB → 51MB and 93MB → 46MB), and the same fp16-on-disk approach
was used for the two newest checkpoints (`mask2former_phase3_768_best.pth` ~45MB,
`segformer_b2_best.pt` ~55MB) to conserve GitHub LFS's free-tier bandwidth quota.
`model_hub.get_model()` transparently upcasts fp16 weights back to float32 on load, so
inference precision is unaffected either way. Retraining the `train_smp.py`-based models is
still just `python train_smp.py --arch <name>` (a few minutes to ~1.5 hours each on an
RTX-3050-class GPU) if you want to regenerate from scratch; the Mask2Former-phase3 and
SegFormer-B2 checkpoints arrived pre-trained (see caveats above) and have no equivalent
one-command retrain path in this repo. **Without checkpoints present** (e.g. if you delete
`checkpoints/`), the dashboard still runs — it shows training history/status for every
architecture and labels image/video evaluation as unavailable rather than crashing.

## Training

**Multi-architecture trainer** (ImageNet-pretrained encoders via `segmentation-models-pytorch`,
plus MONAI's Attention U-Net):

```bash
python train_smp.py --arch unet_resnet34 --epochs 15
python train_smp.py --arch deeplabv3plus_resnet50 --epochs 35 --batch_size 8
python train_smp.py --arch attention_unet --epochs 25
```

**Original from-scratch trainer** (U-Net or Mask2Former, config-driven):

```bash
python training/train.py configs/default_config.yaml
```

The earlier MiT-B0 SegFormer run was trained via `notebooks/reference_segformer_02_training.ipynb`
(HuggingFace `transformers`, not `train_smp.py`). The current `segformer_b2_best.pt` and
`mask2former_phase3_768_best.pth` checkpoints arrived pre-trained with no accompanying
training code — see their preprocessing/architecture caveats above.

Both trainers write checkpoints to `checkpoints/` and print per-epoch Dice/IoU/loss.

## Dashboard

One dashboard, `streamlit_app.py` — crimson theme, visual stats (Plotly Dice/loss curves +
a model leaderboard chart, no raw per-epoch tables), model selector across every
architecture in the Results table, a validation gallery with shuffle, and upload-your-own
image/video evaluation. Built on `model_hub.py` so there's one model registry, one
inference path, no logic duplicated across UIs.

```bash
streamlit run streamlit_app.py
# -> http://localhost:8501
```

### Deploy to Streamlit Community Cloud

1. Push this repo to GitHub (`.gitignore` excludes only `data/` — the 3GB Kaggle dataset,
   not needed at inference time — everything else, checkpoints included, is tracked, with
   `checkpoints/*.pth`/`*.pt` served via Git LFS — see `.gitattributes`).
2. On [share.streamlit.io](https://share.streamlit.io), point a new app at the repo with
   main file path `streamlit_app.py`. No secrets/config needed — `requirements.txt` and
   `.streamlit/config.toml` are picked up automatically; Streamlit Community Cloud resolves
   Git LFS pointers on its own during the build.
3. That's it — all 5 trained models are in the repo, so the deployed app is fully live
   (Overview/Gallery/Upload Image/Upload Video all work) with no extra setup.

## Methods

**Preprocessing** (`preprocessing/solar_preprocessor.py`, `classical/`): grayscale
conversion, solar-disk detection (contour + `minEnclosingCircle`, shrunk 7% to drop the
limb edge artifact ring), limb-darkening correction, CLAHE contrast enhancement,
multi-scale Black Top-Hat + Frangi vesselness ridge filtering.

**Deep learning**: interchangeable architectures — a lightweight Mask2Former (`models/`,
with either a from-scratch FPN or a `torchvision` ResNet-34 pixel-decoder backbone),
from-scratch U-Net, ImageNet-pretrained U-Net/DeepLabV3+/Attention-U-Net (`train_smp.py`,
via `segmentation-models-pytorch`/MONAI), and a HuggingFace SegFormer (MiT-B2). Trained with
a compound Dice+Focal/BCE loss under severe class imbalance (filaments cover <2% of the
disk), AMP mixed precision, and (in `training/train.py`) cosine LR annealing + early
stopping.

**Hybrid fusion** (`hybrid/fusion.py`): `P_final = alpha * P_DeepLearning + (1-alpha) * P_Frangi`,
with an alpha sweep to find the best blend on the validation set; also supports strict
intersection/union fusion.

**Morphology & explainability** (`analysis/`, `explainability/`): per-filament area,
perimeter, skeleton length, average width, orientation, bounding box, centroid, and a
confidence/uncertainty map from the model's probability output. `hybrid/fusion.py`'s
astronomical calibration (0.6 arcsec/px ≈ 435 km/px) converts pixel measurements to
physical units. `notebooks/Copy_of_FINAL_SOLAR.ipynb` additionally computes sinuosity,
tilt/orientation, a Grad-CAM attention heatmap, and a space-weather risk rating per
filament (see top of this README).

**Inference** (`inference/predict.py`): `SolarFilamentPredictor` runs U-Net/Mask2Former,
Frangi, or hybrid prediction on a single image and returns every intermediate
(preprocessed image, Frangi/Hessian response, probability maps, final mask, overlay).

## Known limitations

- The current Mask2Former checkpoint's 0.7207 Dice has an unverified split methodology (see
  *A correction worth being upfront about* above) — the split code in this repo is fixed for
  future from-scratch runs, but this particular checkpoint arrived pre-trained, so whether it
  used the leakage-prone or fixed split can't be confirmed from the checkpoint alone; treat
  0.7207 as a possible upper-bound estimate, not a guaranteed clean benchmark.
- DeepLabV3+ initially targeted an EfficientNet-B4 encoder; switched to ResNet-50 after
  measuring EfficientNet-B4 at ~7-12x slower per batch on this GPU/driver/cuDNN combo (and
  hitting a CUDA OOM at batch size 8 before AMP was added). Only the ResNet-50 variant is
  kept in `train_smp.py` — one backbone per architecture, no redundant registry entries.
- Windows: `DataLoader(num_workers=...)` must stay at `0` in `train_smp.py` — the
  preprocessor holds a `cv2.CLAHE` object that Windows' `spawn` multiprocessing start
  method can't pickle. `SolarFilamentDataset.__getitem__` also retries transient file
  reads (a real failure hit mid-run on a cloud-synced `Downloads` folder — see the
  Attention U-Net row in Results).
- None of the models here produce the Kaggle competition's actual submission format
  (per-filament instance RLE masks scored by Panoptic Quality) — they're binary/
  semantic segmenters. Turning predictions into per-filament instances (for a leaderboard
  submission) is future work; the full-disk instance panel in `notebooks/Copy_of_FINAL_SOLAR.ipynb`
  demonstrates one connected-components-based approach to doing so.

## Tech Stack

PyTorch, torchvision, segmentation-models-pytorch, MONAI, HuggingFace Transformers,
OpenCV, scikit-image, Albumentations, Streamlit + Plotly, Kaggle API.

## License

Data from NSO/GONG (National Solar Observatory / Global Oscillation Network Group),
MAGFiLO 1.0 (MLEcoFi 2024, Earth-Space AI Research Lab).
