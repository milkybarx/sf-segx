# Solar Filament Segmentation & Space Weather Intelligence System

## Phase 2: Mask2Former Scientific Analysis

Phase 2 uses the existing custom grayscale Mask2Former implementation and
`checkpoints/mask2former_best.pth`. Its training-time preprocessing is preserved:
solar-disk detection, limb-darkening correction, percentile normalization, denoising,
CLAHE, disk masking, resize to 512 x 512, and single-channel float input in [0, 1].
The model emits one dense `[B, 1, H, W]` logit map. Phase 2 applies sigmoid to that
map; confidence is the mean sigmoid probability over each downstream connected
component. The checkpoint does not emit native panoptic instances, so components are
explicitly reported as semantic-mask instance separation.

Run the CLI from the repository root:

```text
python scripts/analyze_filament.py --image path/to/image.png --model mask2former --threshold 0.5 --output outputs/
```

This writes `outputs/phase2_analysis.png`, per-filament crops,
`outputs/catalog/filament_catalog.json`, and `outputs/catalog/filament_catalog.csv`.
The existing Streamlit dashboard defaults to Mask2Former for uploads. Set the threshold
in the sidebar, optionally enable segmentation attribution, and use the catalog download
buttons in the upload view. SegFormer experiments remain comparison models and are not
used by this Phase 2 pipeline.

All morphology values are pixel-based unless valid metadata supplies `km_per_px`, or a
solar radius in pixels and its known physical radius. Without that metadata, physical
values are `null` and `physical.calibrated` is `false`. The risk field is a
**Morphological Eruption-Risk Screening Indicator** based on length, area, and confidence;
it is not a validated eruption or CME probability and does not claim that a long filament
will erupt.

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
| **Mask2Former** (from scratch) | custom FPN | 2.76M | 50 | **0.6990** | 0.5399 | See *leakage caveat* below |
| U-Net | ResNet-34 (ImageNet) | 24.4M | 15 | 0.6611 | 0.4945 | Clean file-level split |
| DeepLabV3+ | ResNet-50 (ImageNet) | 26.7M | 35 | 0.6521 | 0.4844 | Clean file-level split |
| SegFormer | MiT-B0 (ImageNet) | 3.7M | 50 | 0.6320 | — | See *SegFormer preprocessing* note below |
| Attention U-Net | MONAI, from scratch | 7.9M | 20/25 | 0.6507 | 0.4829 | Run crashed epoch 22 on a transient file-read error (fixed); finalized from the epoch-20 checkpoint rather than re-run, since Dice had plateaued/was oscillating in 0.63-0.65 for ~10 epochs |

All trained models are in `checkpoints/` and selectable live in the dashboard
(`streamlit_app.py`), including on your own uploaded images/videos. A fine-tuned
SAM/MedSAM is still scoped for a future pass — not included yet (see below).

### A correction worth being upfront about

The original codebase's README described the shipped `checkpoints/best_model.pth` as a
"U-Net" achieving 0.699 Dice. **It is actually a Mask2Former** (`models/mask2former.py`,
2.76M params) — confirmed by loading it and inspecting `checkpoint['config']['model']['name']`,
not by re-reading prose. Renamed to `checkpoints/mask2former_best.pth` here.

Separately, that same run's validation split (`preprocessing/dataset.py`) split by raw COCO
`image_id`, but **707 physical images have 1,154 `image_id` entries** (up to 3 separate
annotation sessions per image, sharing the same `file_name`/pixels). Splitting by `image_id`
let **121 of 296 duplicate-session images leak across train/val** — the model could see a
given photo in training and be "validated" on the same photo with a different session's
mask. This has been fixed (`create_data_splits` now splits by unique `file_name` first,
then assigns all of a file's sessions to one side) — verified zero file overlap after the
fix. The shipped 0.6990 Dice is from *before* this fix and is likely a bit optimistic; a
clean re-run would be needed for an apples-to-apples number. This repo's other models
(trained via `train_smp.py`) always used a clean file-level 85/15 split with single-session
(not unioned) ground-truth masks — see the next section.

### A second, similar data-quality fix

`scripts/prepare_masks.py` (used by `train_smp.py`) originally **unioned** every
annotation session's polygons for a duplicate-session file into one mask. Measuring this
on a sample file: union area was **2.4x** the intersection area — sessions disagree
substantially, so unioning inflates/noises the target rather than producing consensus.
Fixed to use one canonical session per file instead.

### SegFormer preprocessing — a silent-failure trap

`checkpoints/segformer_b0_best.pt` (HuggingFace `SegformerForSemanticSegmentation`, MiT-B0)
was trained on **plain-resized RGB images with ImageNet normalization — no CLAHE, no
limb-darkening correction, no disk detection**, unlike every other model in this repo.
Feeding it this repo's `GONGPreprocessor` output instead (the "obviously more sophisticated"
choice) doesn't error — it just silently drops Dice from ~0.60 to ~0.22 on a held-out
sample, i.e. a wrong-but-plausible-looking model. Confirmed by testing both pipelines
against real ground truth before wiring it into `model_hub.py`. The lesson generalizes:
*a model loading successfully proves the architecture matches, not that the preprocessing
does* — every external checkpoint in this repo (`checkpoints/*.pth`/`*.pt` not produced by
`train_smp.py`) needed its exact training-time preprocessing reverse-engineered and
verified against ground truth before being trusted, not assumed from what "should" work.

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
│   └── default_config.yaml      # hyperparameters for training/train.py
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
├── checkpoints/                  (tracked in git -- see "Checkpoints ship in the repo")
│   ├── mask2former_best.pth     # 0.6990 Dice (pre-leakage-fix, see caveat above)
│   ├── unet_resnet34_best.pth   # 0.6611 Dice (fp16 on disk, ~46MB)
│   ├── deeplabv3plus_resnet50_best.pth  # 0.6521 Dice (fp16 on disk, ~51MB)
│   ├── segformer_b0_best.pt     # 0.6320 Dice (see SegFormer preprocessing caveat above)
│   └── attention_unet_best.pth  # 0.6507 Dice (epoch 20/25, see caveat above)
├── experiments/
│   ├── mask2former_training_results.json  # full 50-epoch history for that checkpoint
│   ├── segformer_training_history.csv     # full 50-epoch history for that checkpoint
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
│   ├── reference_segformer_01_data_prep.ipynb  # how segformer_b0_best.pt's data was prepped
│   └── reference_segformer_02_training.ipynb   # how segformer_b0_best.pt was trained
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

All 5 trained checkpoints are tracked directly in `checkpoints/` — no Git LFS, no external
download step. Two of them (`unet_resnet34_best.pth`, `deeplabv3plus_resnet50_best.pth`)
were re-saved with float16 weights specifically to clear GitHub's 100MB single-file limit
(107MB → 51MB and 93MB → 46MB); `model_hub.get_model()` transparently upcasts them back to
float32 on load, so inference precision is unaffected either way. Retraining any of them is
still just `python train_smp.py --arch <name>` (a few minutes to ~1.5 hours each on an
RTX-3050-class GPU) if you want to regenerate from scratch. **Without checkpoints present**
(e.g. if you delete `checkpoints/`), the dashboard still runs — it shows training
history/status for every architecture and labels image/video evaluation as unavailable
rather than crashing.

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

SegFormer was trained via `notebooks/reference_segformer_02_training.ipynb` (HuggingFace
`transformers`, not `train_smp.py` — see its own preprocessing caveat above).

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
   not needed at inference time — everything else, checkpoints included, is tracked).
2. On [share.streamlit.io](https://share.streamlit.io), point a new app at the repo with
   main file path `streamlit_app.py`. No secrets/config needed — `requirements.txt` and
   `.streamlit/config.toml` are picked up automatically.
3. That's it — all 5 trained models are in the repo, so the deployed app is fully live
   (Overview/Gallery/Upload Image/Upload Video all work) with no extra setup.

## Methods

**Preprocessing** (`preprocessing/solar_preprocessor.py`, `classical/`): grayscale
conversion, solar-disk detection (contour + `minEnclosingCircle`, shrunk 7% to drop the
limb edge artifact ring), limb-darkening correction, CLAHE contrast enhancement,
multi-scale Black Top-Hat + Frangi vesselness ridge filtering.

**Deep learning**: interchangeable architectures — from-scratch U-Net and a lightweight
Mask2Former (`models/`), ImageNet-pretrained U-Net/DeepLabV3+/Attention-U-Net
(`train_smp.py`, via `segmentation-models-pytorch`/MONAI), and a HuggingFace SegFormer
(MiT-B0). Trained with a compound Dice+Focal/BCE loss under severe class imbalance
(filaments cover <2% of the disk), AMP mixed precision, and (in `training/train.py`)
cosine LR annealing + early stopping.

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

- The shipped Mask2Former checkpoint's 0.6990 Dice was measured under a validation split
  with train/val leakage (see above) — treat it as an upper-bound estimate, not a clean
  benchmark. The split code is now fixed for future runs; re-training was out of scope
  for this pass given time constraints (the original run took ~96 minutes).
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
