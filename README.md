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
| SegFormer | MiT-B2 (ImageNet), 640px | 27.3M | 36 | 0.6970 | — | See *SegFormer preprocessing* note; a 768px retry landed slightly lower, see *second honest result* below |
| U-Net | ResNet-34 (ImageNet) | 24.4M | 80 | 0.6629 | 0.4960 | Clean file-level split. Retrained 15→80 epochs + cosine LR; plateaued at epoch 37, see note below |
| DeepLabV3+ | ResNet-50 (ImageNet) | 26.7M | 35 | 0.6521 | 0.4844 | Clean file-level split |
| Attention U-Net | MONAI, from scratch | 7.9M | 20/25 | 0.6507 | 0.4829 | Run crashed epoch 22 on a transient file-read error (fixed); finalized from the epoch-20 checkpoint rather than re-run, since Dice had plateaued/was oscillating in 0.63-0.65 for ~10 epochs |

All trained models are in `checkpoints/` and selectable live in the dashboard
(`streamlit_app.py`), including on your own uploaded images/videos. A fine-tuned
SAM/MedSAM is still scoped for a future pass — not included yet (see below).

### An honest result: pushing U-Net further

U-Net was originally trained for only 15 epochs (0.6611 Dice) — clearly undertrained next to
Mask2Former's 50. Re-trained for 80 epochs with a cosine LR schedule added to `train_smp.py`
(previously flat for the whole run) to see how far it could go. Reported plainly: **it
plateaued at 0.6629 by epoch 37 and never improved again** — by epoch 80, train Dice had
climbed to 0.74 while validation Dice sat at 0.656, a widening train/val gap that's a
textbook overfitting signal, not a sign more epochs would help. More training time alone
wasn't the bottleneck this architecture/resolution combination had; closing the remaining gap
to Mask2Former would need a different lever (higher resolution, stronger regularization or
augmentation, or a bigger backbone), not just a longer run. Kept as the new checkpoint anyway
since it is a genuine, if small, improvement (0.6611 → 0.6636 at its optimal threshold).

### A second honest result: retraining SegFormer-B2 at higher resolution

Found the actual training notebook the shipped `segformer_b2_best.pt` came from
(`notebooks/SegFormer_B2_Solar_Filament_Training.ipynb`) and adapted it into
`scripts/train_segformer_b2.py` to retry with two changes: 768px input instead of the
shipped 640px (on the theory that higher resolution helped Mask2Former, so it might help
here too), and `focal_dice` loss instead of the shipped `dice_bce` (both loss options were
already in the source notebook, just never compared). 40 epochs, early stopping patience 8,
~5 minutes/epoch. **Result: 0.6960 best validation Dice — very slightly worse than the
shipped 640px/dice_bce checkpoint's 0.6970**, not better. Growth clearly plateaued by epoch
27 (0.695) and never meaningfully improved after, even as the LR scheduler halved twice.
The shipped checkpoint was kept; the new one was not adopted. `experiments/
segformer_b2_history.csv` and `segformer_b2_config.json` are kept as a record of the
attempt. Between this and the U-Net result above, resolution/loss-function tweaks and
longer training runs have now twice failed to meaningfully beat what's already shipped —
closing the remaining gap to a >0.73 target across this project would need a genuinely
different approach (architecture, more/better data, or an ensemble), not further tuning of
the current recipes.

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

### Color images — the color→H-alpha adapter

Every model above was trained exclusively on MAGFiLO H-alpha imagery, which is genuinely
single-channel (verified: every training JPEG has R==G==B exactly pixel-for-pixel, just
saved as a 3-channel file). None of them have ever seen a real color image, and a naive
grayscale conversion (`cv2.COLOR_BGR2GRAY`'s fixed 0.299R+0.587G+0.114B weighting) isn't
guaranteed to preserve filament contrast for an arbitrarily color-graded or false-color
input — a hue rotation can shift which channel actually carries the true luminance
structure, silently degrading predictions without erroring.

`models/color_adapter.py`'s `ColorToHAlphaNet` (a small U-Net, 3-channel RGB → 1-channel,
264K params) fixes this as a shared preprocessing step ahead of every model, so none of them
need retraining. There's no real color solar imagery anywhere in this repo or dataset to
train it on directly, so `scripts/train_color_adapter.py` trains it **self-supervised**:
take a real grayscale H-alpha training image, synthetically re-color it with a random
hue/saturation/gamma transform (HSV recompose, `V` = the true grayscale so luminance
structure is preserved, `H`/`S` randomized per example, 15% of examples left untouched as
pure grayscale), and train the network to recover the original grayscale from the synthetic
color version (L1 + a Sobel-gradient term for edge sharpness). A wide spread of random
tints — not one fixed "look" — forces the network to learn a general *undo an unknown color
cast* mapping instead of memorizing a specific color scheme, which is what lets it
generalize to actually-unseen colored inputs at inference.

`model_hub.run_inference()` calls `to_halpha_style()` on every input first: a genuinely
colored image (channels differ) goes through the adapter; a grayscale or
grayscale-stored-as-3-channel image (like this repo's own dataset, or the Validation
Gallery) is detected and passed straight through unchanged, so nothing regresses for the
existing use case.

Trained for 40 epochs (cosine LR decay, val L1 = 0.0255 on held-out grayscale-vs-synthetic-
color reconstruction) and verified end-to-end against real ground truth with two different
synthetic color families — a global hue/saturation tint (HSV recompose) and an independent
per-channel gain/bias transform (not part of the training augmentation, a harder
generalization test) — run through the full `run_inference()` pipeline:

| Model | True grayscale | Hue-tinted | Independent-channel |
|---|---|---|---|
| U-Net (ResNet-34) | 0.592 | 0.585 (−1%) | 0.592 (±0%) |
| SegFormer (MiT-B2) | 0.575 | 0.572 (−1%) | 0.554 (−4%) |
| Mask2Former (ResNet-34, 768px) | 0.679 | 0.549 (−19%) | 0.600 (−12%) |

U-Net and SegFormer hold up close to their true-grayscale accuracy on both color families.
Mask2Former shows a real, honest gap — it's the highest-resolution (768px) and highest-
baseline-accuracy model here, so it depends the most on fine texture the adapter's
grayscale reconstruction softens slightly; treat color-image predictions from it as
noticeably less reliable than from the other two until the adapter (or a Mask2Former-
specific fix) improves further.

### Phase 2 analysis — instance separation, catalog export, super-resolution

A second contributor (ET3RYX) built a scientific post-processing layer on top of the
segmentation models, generalized here to work with every architecture in the Results table
(not just one): `postprocessing/` (connected-component instance separation, skeleton
analysis, spatial-region tagging, physical-unit calibration), `catalog/` (a validated
JSON/CSV filament schema, exported per-image and per-filament from the Upload Image page),
`explainability/interface.py` (a pluggable input-gradient attribution hook, currently
supported for Mask2Former only — see *Known limitations*), and `inference/adapters.py` (a
model-agnostic `StandardizedPrediction` wrapper around `model_hub.run_inference()`, so the
whole pipeline works for any registered model without per-architecture special-casing).
`inference/phase2.run_phase2_analysis()` ties these together and is what the Upload Image
page's "Filament Detail Inspector" runs.

**Super-resolution** (`experiments/super_resolution/`): the Filament Detail Inspector's
crop view is visualization-only (it never feeds back into scientific measurements — those
are computed from the model's native-resolution output). Four lightweight architectures are
available (ESPCN, FSRCNN, EDSR-Small, and a custom `Solar-SRNet`), trained with a physics-
informed degradation pipeline (atmospheric-seeing blur, sensor noise, JPEG compression) on
this repo's own bundled gallery images, using a Charbonnier + SSIM loss. Only Solar-SRNet
(~460K params) is actually trained here — ESPCN/EDSR-Small are available as untrained
architectures for comparison, exactly as documented in `experiments/super_resolution/
sr_experiment_report.md`. Retrained in this session (checkpoints ship in `experiments/
super_resolution/results/`): **35.60 dB PSNR / 0.90 SSIM at 2x**, **32.73 dB PSNR / 0.84
SSIM at 4x** on held-out patches, both beating plain Lanczos-4 interpolation. Retrain with:

```bash
python experiments/super_resolution/training/train_sr.py --scale 2 --model solar_sr --epochs 20
python experiments/super_resolution/training/train_sr.py --scale 4 --model solar_sr --epochs 20
```

Four real bugs were found and fixed while integrating this:
- `tests/test_phase2.py` imported a function (`high_quality_upscale`) that had been renamed
  to `super_resolve_crop` in the same commit that added AI super-resolution.
- `postprocessing/instances.py` assumed every filament dict already carried a `component_id`
  key that `analysis/filament_morphology.analyze_filaments()` never actually set (fixed by
  adding it there).
- `explainability/interface.py`'s attribution dispatch only recognized `"mask2former"`/
  `"mask2former_scratch"`, arch names from an earlier version of this repo that no longer
  exist in `model_hub.EXTERNAL_MODELS` (the current key is `"mask2former_phase3"`) — the
  attribution panel was silently blank for every model, always, not from any per-image issue.
- Even after that fix, attribution was still blank on any image where the model detected
  *no* filaments: `segmentation_attribution()`'s objective is `(logits * target_mask).mean()`,
  and an all-zero `target_mask` makes that a constant zero regardless of the input, so its
  gradient — and the whole attribution map — is exactly zero everywhere. Fixed to fall back
  to the raw sigmoid probability map (never uniformly zero) whenever the predicted mask is
  empty.

All were silent failures with no exception raised — the kind that only surface by actually
looking at the output, which is why each was caught by running the pipeline end-to-end
against real images rather than trusting that "it imports without errors" meant it worked.

### Ensemble consensus & uncertainty

`inference/ensemble.py` weighted-averages every trained model's probability map (weighted by
each model's own best validation Dice) with test-time augmentation (the same model's
prediction on horizontal/vertical flips of the input, flipped back and averaged in). Measured
honestly on an 8-image held-out sample rather than assumed: the full 5-model ensemble
(0.628 mean Dice) did **not** beat Mask2Former alone (0.647) — Mask2Former is enough stronger
than the other four that averaging them in mostly dilutes it. TTA alone did measurably help
(ensemble without TTA: 0.623, with TTA: 0.628).

So this feature's real value isn't "beats the best single model's Dice" — it's the **per-pixel
model-agreement map** it also returns: the fraction of models whose own thresholded
prediction matches the ensemble's final call at each pixel. Where several architecturally
distinct models (different backbones, resolutions, training recipes) agree, that detection is
corroborated across independent failure modes; where they conflict, that's a genuine
uncertainty signal a single model's own confidence score can't provide (a model can be
confidently wrong). Available as an opt-in checkbox on the Upload Image page (it costs ~5x a
single model's inference, or more with TTA, so it isn't run by default).

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
│   ├── finalize_attention_unet.py  # one-off: threshold-sweep+finalize a checkpoint
│   │                                #   without a live training run (see Results table)
│   ├── train_color_adapter.py   # trains models/color_adapter.py, self-supervised
│   │                              #   (synthetic re-colorization, see Results section)
│   └── train_segformer_b2.py    # adapted from the notebook segformer_b2_best.pt was
│                                  #   actually trained with (see Results section)
├── preprocessing/
│   ├── solar_preprocessor.py    # disk detection, limb correction, CLAHE
│   ├── dataset.py               # COCO parsing, PyTorch Dataset, .npy caching
│   └── build_cache.py
├── classical/                   # Frangi / Hessian ridge-detection pipeline
│   ├── frangi.py, hessian.py, morphology.py, advanced_extractor.py
├── models/
│   ├── unet.py                  # from-scratch U-Net
│   ├── mask2former.py           # from-scratch/ResNet-34-backbone Mask2Former
│   └── color_adapter.py         # ColorToHAlphaNet -- color image -> H-alpha style,
│                                  #   shared by every model above (see Results section)
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
│   ├── predict.py                # SolarFilamentPredictor: unet/frangi/hybrid inference
│   ├── adapters.py, phase2.py, mask2former.py  # Phase 2 pipeline (see Results section)
│   └── ensemble.py               # multi-model + TTA consensus/uncertainty (see Results section)
├── checkpoints/                  (tracked in git via Git LFS -- see "Checkpoints ship in the repo")
│   ├── mask2former_phase3_768_best.pth  # 0.7207 Dice, best model (fp16 on disk, ~45MB)
│   ├── segformer_b2_best.pt     # 0.6970 Dice (fp16 on disk, ~55MB, see preprocessing caveat above)
│   ├── unet_resnet34_best.pth   # 0.6629 Dice, 80 epochs (fp16 on disk, ~46MB)
│   ├── deeplabv3plus_resnet50_best.pth  # 0.6521 Dice (fp16 on disk, ~51MB)
│   ├── attention_unet_best.pth  # 0.6507 Dice (epoch 20/25, see caveat above)
│   └── color_to_halpha_adapter.pth  # color image -> H-alpha style, shared by all 5 above
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

All 6 trained checkpoints (5 segmentation models + the color→H-alpha adapter) are tracked
in `checkpoints/` via **Git LFS** — `git lfs install`
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
(HuggingFace `transformers`, not `train_smp.py`). `mask2former_phase3_768_best.pth` arrived
pre-trained with no accompanying training code in this repo — see its architecture caveat
above. `segformer_b2_best.pt`'s real training notebook (`notebooks/SegFormer_B2_Solar_Filament_
Training.ipynb`) was found and adapted into `scripts/train_segformer_b2.py`:

```bash
python scripts/train_segformer_b2.py --epochs 40 --image_size 640 --loss dice_bce
```

(The shipped checkpoint used `--image_size 640 --loss dice_bce`; a retry at 768px/`focal_dice`
landed slightly worse — see *A second honest result* above.)

Both trainers write checkpoints to `checkpoints/` and print per-epoch Dice/IoU/loss.

The color→H-alpha adapter is trained separately (no ground-truth masks involved, just the
raw training images):

```bash
python scripts/train_color_adapter.py --epochs 40 --batch_size 24
```

## Dashboard

One dashboard, `streamlit_app.py` — crimson theme, visual stats (Plotly Dice/loss curves +
a model leaderboard chart, no raw per-epoch tables), model selector across every
architecture in the Results table, a validation gallery with shuffle, and upload-your-own
image/video evaluation — color images included, via the color→H-alpha adapter (see Results
section). Built on `model_hub.py` so there's one model registry, one inference path, no
logic duplicated across UIs.

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
- `explainability/interface.py`'s input-gradient attribution only works for Mask2Former --
  its tensor construction (single-channel, positional `model(tensor)` call) doesn't match
  SegFormer's `pixel_values=` keyword/3-channel-RGB calling convention or the exact
  normalization the `train_smp.py` models were trained with, so it isn't wired up for those.
- The 5-model ensemble in `inference/ensemble.py` does not reliably beat Mask2Former alone on
  raw Dice (see *Ensemble consensus & uncertainty* above) -- use it for the agreement/
  uncertainty map, not as a way to get a higher-Dice prediction than the best single model.

## Tech Stack

PyTorch, torchvision, segmentation-models-pytorch, MONAI, HuggingFace Transformers,
OpenCV, scikit-image, Albumentations, Streamlit + Plotly, Kaggle API.

## License

Data from NSO/GONG (National Solar Observatory / Global Oscillation Network Group),
MAGFiLO 1.0 (MLEcoFi 2024, Earth-Space AI Research Lab).
