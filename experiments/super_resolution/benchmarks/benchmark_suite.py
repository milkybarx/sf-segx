"""
Super-Resolution Benchmark & Evaluation Suite
=============================================
Runs quantitative benchmarks across conventional interpolation methods
and AI super-resolution architectures on solar filament crops.
Measures PSNR, SSIM, Sharpness, Edge Preservation, Latency, and Memory.
"""

import os
import sys
import time
import glob
import json
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from classical.morphology import connected_components_analysis
from analysis.filament_morphology import analyze_filaments
from visualization.detail import crop_filament
from experiments.super_resolution.models import (
    ESPCN, FSRCNN, EDSRSmall, SolarSRNet, SSIMLoss
)
from experiments.super_resolution.training.dataset import apply_solar_degradation


def measure_laplacian_variance(img: np.ndarray) -> float:
    """Measure image sharpness using the variance of the Laplacian."""
    gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return float(lap.var())


def measure_tenengrad_gradient(img: np.ndarray) -> float:
    """Measure edge contrast and preservation using the Tenengrad gradient."""
    gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    mag_sq = gx ** 2 + gy ** 2
    return float(np.mean(mag_sq))


def calculate_psnr_np(target: np.ndarray, pred: np.ndarray) -> float:
    """Compute PSNR between two numpy uint8 or float images."""
    t = target.astype(np.float32)
    p = pred.astype(np.float32)
    if t.max() > 1.0:
        max_val = 255.0
    else:
        max_val = 1.0
    mse = np.mean((t - p) ** 2)
    if mse == 0:
        return 100.0
    return float(10.0 * np.log10((max_val ** 2) / mse))


def calculate_ssim_np(target: np.ndarray, pred: np.ndarray) -> float:
    """Compute SSIM between two numpy images."""
    t = torch.from_numpy(target.astype(np.float32) / (255.0 if target.max() > 1.0 else 1.0)).unsqueeze(0).unsqueeze(0)
    p = torch.from_numpy(pred.astype(np.float32) / (255.0 if pred.max() > 1.0 else 1.0)).unsqueeze(0).unsqueeze(0)
    ssim_mod = SSIMLoss(channels=1)
    loss = ssim_mod(p, t).item()
    return float(max(0.0, 1.0 - loss))


def extract_representative_filaments(image_dir: str, mask_dir: str, num_filaments: int = 8):
    """
    Extract a diverse set of representative filament crops covering small,
    medium, and large area structures from multiple solar images.
    """
    img_files = sorted(glob.glob(os.path.join(image_dir, "*.jpeg")) + glob.glob(os.path.join(image_dir, "*.png")))
    filaments_collected = []

    for img_path in img_files:
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        mask_path = os.path.join(mask_dir, f"{base_name}.png")
        if not os.path.exists(mask_path):
            continue

        raw = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if raw is None or mask is None:
            continue

        binary_mask = (mask > 0).astype(np.uint8)
        fils = analyze_filaments(binary_mask, min_area=100)

        for f in fils:
            bbox = f.get("bbox", {})
            w, h = bbox.get("width", 0), bbox.get("height", 0)
            area = f.get("area_px", 0)
            if 30 <= w <= 400 and 30 <= h <= 400 and area >= 150:
                crop, bounds = crop_filament(raw, f, padding=25)
                if crop.shape[0] >= 32 and crop.shape[1] >= 32:
                    filaments_collected.append({
                        "image_id": base_name,
                        "filament_id": f["filament_id"],
                        "area_px": area,
                        "length_px": f.get("skeleton_length_px", 0),
                        "width_px": f.get("avg_width_px", 0),
                        "crop": crop,
                        "bounds": bounds,
                        "filament": f
                    })
        if len(filaments_collected) >= num_filaments * 2:
            break

    # Sort by area and select small, medium, large representatives
    filaments_collected.sort(key=lambda x: x["area_px"])
    selected = []
    if len(filaments_collected) >= num_filaments:
        indices = np.linspace(0, len(filaments_collected) - 1, num_filaments, dtype=int)
        selected = [filaments_collected[i] for i in indices]
    else:
        selected = filaments_collected

    return selected


def run_comprehensive_benchmark():
    """
    Run full benchmarking suite comparing all methods on solar filament crops.
    """
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
    image_dir = os.path.join(project_root, "assets/gallery_samples/images")
    mask_dir = os.path.join(project_root, "assets/gallery_samples/masks")
    results_dir = os.path.join(project_root, "experiments/super_resolution/results")
    vis_dir = os.path.join(project_root, "experiments/super_resolution/baseline_comparison/visual_comparisons")
    bench_dir = os.path.join(project_root, "experiments/super_resolution/benchmarks")
    base_dir = os.path.join(project_root, "experiments/super_resolution/baseline_comparison")
    os.makedirs(vis_dir, exist_ok=True)
    os.makedirs(bench_dir, exist_ok=True)
    os.makedirs(base_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Benchmarking on device: {device}")

    # Extract test crops
    test_filaments = extract_representative_filaments(image_dir, mask_dir, num_filaments=8)
    print(f"Extracted {len(test_filaments)} representative filament crops for benchmarking.")

    # Load neural models
    scales = [2, 4]
    neural_models = {}

    for scale in scales:
        neural_models[scale] = {}

        # 1. ESPCN
        espcn = ESPCN(scale_factor=scale, in_channels=1).to(device).eval()
        neural_models[scale]["ESPCN (AI-SR)"] = espcn

        # 2. FSRCNN
        fsrcnn = FSRCNN(scale_factor=scale, in_channels=1).to(device).eval()
        neural_models[scale]["FSRCNN (AI-SR)"] = fsrcnn

        # 3. EDSR-Small
        edsr = EDSRSmall(scale_factor=scale, in_channels=1).to(device).eval()
        neural_models[scale]["EDSR-Small (AI-SR)"] = edsr

        # 4. SolarSRNet (custom)
        solar_sr = SolarSRNet(scale_factor=scale, in_channels=1).to(device).eval()
        
        # Load trained checkpoint if exists
        ckpt_path = os.path.join(results_dir, f"best_sr_model_x{scale}.pt")
        if not os.path.exists(ckpt_path):
            ckpt_path = os.path.join(results_dir, f"best_sr_model_solar_sr_x{scale}.pt")

        if os.path.exists(ckpt_path):
            print(f"Loading trained weights for SolarSRNet x{scale} from {ckpt_path}")
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            solar_sr.load_state_dict(ckpt["model_state_dict"])
        else:
            print(f"Using initialized weights for SolarSRNet x{scale} (train model first for trained metrics)")

        neural_models[scale]["Solar-SR (Custom AI-SR)"] = solar_sr

    benchmark_rows = []
    baseline_rows = []

    for item_idx, item in enumerate(test_filaments):
        hr_crop = item["crop"]
        h_orig, w_orig = hr_crop.shape

        # Measure baseline metrics on original crop
        orig_sharpness = measure_laplacian_variance(hr_crop)
        orig_edge = measure_tenengrad_gradient(hr_crop)

        baseline_rows.append({
            "Filament_ID": f"{item['image_id']}_f{item['filament_id']}",
            "Method": "Original Crop (Reference)",
            "Scale": 1,
            "Resolution": f"{w_orig}x{h_orig}",
            "Laplacian_Sharpness": round(orig_sharpness, 1),
            "Tenengrad_Edge_Preservation": round(orig_edge, 1),
            "Latency_ms": 0.0,
            "Peak_VRAM_MB": 0.0
        })

        for scale in scales:
            # Create synthetic degraded input for objective PSNR/SSIM evaluation against HR ground-truth
            lr_crop_float = apply_solar_degradation(hr_crop, scale=scale)
            lr_crop_uint8 = (lr_crop_float * 255.0).astype(np.uint8)

            # Target HR resized to exact scale multiple for metric alignment
            target_hr = hr_crop[:lr_crop_uint8.shape[0] * scale, :lr_crop_uint8.shape[1] * scale]

            methods = {
                "Nearest Neighbor": lambda x, s: cv2.resize(x, (x.shape[1] * s, x.shape[0] * s), interpolation=cv2.INTER_NEAREST),
                "Bilinear": lambda x, s: cv2.resize(x, (x.shape[1] * s, x.shape[0] * s), interpolation=cv2.INTER_LINEAR),
                "Bicubic": lambda x, s: cv2.resize(x, (x.shape[1] * s, x.shape[0] * s), interpolation=cv2.INTER_CUBIC),
                "Lanczos-4 (Current)": lambda x, s: cv2.resize(x, (x.shape[1] * s, x.shape[0] * s), interpolation=cv2.INTER_LANCZOS4),
            }

            visual_outputs = {
                "Original HR": target_hr,
                "Degraded LR": cv2.resize(lr_crop_uint8, (target_hr.shape[1], target_hr.shape[0]), interpolation=cv2.INTER_NEAREST)
            }

            # 1. Classical Interpolation Benchmarking
            for method_name, func in methods.items():
                t0 = time.perf_counter()
                for _ in range(5):  # warm & average
                    out = func(lr_crop_uint8, scale)
                t_ms = (time.perf_counter() - t0) * 1000.0 / 5.0

                psnr = calculate_psnr_np(target_hr, out)
                ssim = calculate_ssim_np(target_hr, out)
                sharpness = measure_laplacian_variance(out)
                edge_val = measure_tenengrad_gradient(out)

                visual_outputs[method_name] = out

                row = {
                    "Filament_ID": f"{item['image_id']}_f{item['filament_id']}",
                    "Method": method_name,
                    "Scale": scale,
                    "Input_Resolution": f"{lr_crop_uint8.shape[1]}x{lr_crop_uint8.shape[0]}",
                    "Output_Resolution": f"{out.shape[1]}x{out.shape[0]}",
                    "PSNR_dB": round(psnr, 2),
                    "SSIM": round(ssim, 4),
                    "Laplacian_Sharpness": round(sharpness, 1),
                    "Tenengrad_Edge_Preservation": round(edge_val, 1),
                    "Latency_ms": round(t_ms, 3),
                    "Peak_VRAM_MB": 0.0,
                    "Category": "Classical Interpolation"
                }
                benchmark_rows.append(row)
                if method_name in ["Bicubic", "Lanczos-4 (Current)"]:
                    baseline_rows.append(row)

            # 2. Neural Super-Resolution Benchmarking
            lr_tensor = torch.from_numpy(lr_crop_float).unsqueeze(0).unsqueeze(0).to(device)

            for ai_name, model in neural_models[scale].items():
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
                    torch.cuda.synchronize()

                t0 = time.perf_counter()
                with torch.inference_mode():
                    for _ in range(5):
                        sr_tensor = model(lr_tensor)
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                t_ms = (time.perf_counter() - t0) * 1000.0 / 5.0

                vram_mb = 0.0
                if torch.cuda.is_available():
                    vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

                sr_out_np = (sr_tensor.squeeze().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)

                # Ensure dimensions match target_hr for metric computation
                if sr_out_np.shape != target_hr.shape:
                    sr_out_np = cv2.resize(sr_out_np, (target_hr.shape[1], target_hr.shape[0]))

                psnr = calculate_psnr_np(target_hr, sr_out_np)
                ssim = calculate_ssim_np(target_hr, sr_out_np)
                sharpness = measure_laplacian_variance(sr_out_np)
                edge_val = measure_tenengrad_gradient(sr_out_np)

                visual_outputs[ai_name] = sr_out_np

                params_k = sum(p.numel() for p in model.parameters()) / 1000.0

                row = {
                    "Filament_ID": f"{item['image_id']}_f{item['filament_id']}",
                    "Method": ai_name,
                    "Scale": scale,
                    "Input_Resolution": f"{lr_crop_uint8.shape[1]}x{lr_crop_uint8.shape[0]}",
                    "Output_Resolution": f"{sr_out_np.shape[1]}x{sr_out_np.shape[0]}",
                    "PSNR_dB": round(psnr, 2),
                    "SSIM": round(ssim, 4),
                    "Laplacian_Sharpness": round(sharpness, 1),
                    "Tenengrad_Edge_Preservation": round(edge_val, 1),
                    "Latency_ms": round(t_ms, 3),
                    "Peak_VRAM_MB": round(vram_mb, 2),
                    "Params_K": round(params_k, 1),
                    "Category": "AI Super-Resolution"
                }
                benchmark_rows.append(row)

            # Generate side-by-side visual montage for this filament crop
            if item_idx < 4:  # Save detailed visual comparisons for first 4 representative samples
                _save_visual_comparison_montage(
                    visual_outputs,
                    save_path=os.path.join(vis_dir, f"comparison_{item['image_id']}_f{item['filament_id']}_x{scale}.png"),
                    title=f"Solar Filament SR Comparison (Scale {scale}x) - ID: {item['image_id']} #{item['filament_id']}"
                )

    # Save summary tables
    bench_df = pd.DataFrame(benchmark_rows)
    bench_df.to_csv(os.path.join(bench_dir, "benchmark_results.csv"), index=False)

    base_df = pd.DataFrame(baseline_rows)
    base_df.to_csv(os.path.join(base_dir, "baseline_metrics.csv"), index=False)

    # Compute aggregate summary
    summary_df = bench_df.groupby(["Method", "Scale", "Category"]).agg({
        "PSNR_dB": ["mean", "std"],
        "SSIM": ["mean", "std"],
        "Laplacian_Sharpness": ["mean"],
        "Tenengrad_Edge_Preservation": ["mean"],
        "Latency_ms": ["mean"]
    }).reset_index()
    summary_df.to_csv(os.path.join(bench_dir, "benchmark_summary.csv"), index=False)

    print(f"\n==================================================")
    print(f"BENCHMARK COMPLETED SUCCESSFULLY!")
    print(f"Results saved to:")
    print(f" - {os.path.join(bench_dir, 'benchmark_results.csv')}")
    print(f" - {os.path.join(base_dir, 'baseline_metrics.csv')}")
    print(f" - {os.path.join(bench_dir, 'benchmark_summary.csv')}")
    print(f" - Visual comparisons saved in: {vis_dir}")
    print(f"==================================================")

    return bench_df


def _save_visual_comparison_montage(images_dict: dict, save_path: str, title: str):
    """Render a structured multi-panel comparison image."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(images_dict)
    cols = min(4, n)
    rows = int(np.ceil(n / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.2, rows * 4.2))
    axes = np.array(axes).flatten()

    for idx, (name, img) in enumerate(images_dict.items()):
        ax = axes[idx]
        if img.ndim == 2:
            ax.imshow(img, cmap="gray", vmin=0, vmax=255)
        else:
            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        
        sharpness = measure_laplacian_variance(img)
        ax.set_title(f"{name}\nSharpness (Lap): {sharpness:.1f}", fontsize=10, pad=4)
        ax.axis("off")

    for idx in range(n, len(axes)):
        axes[idx].axis("off")

    plt.suptitle(title, fontsize=13, fontweight="bold", y=0.98)
    plt.tight_layout()
    plt.savefig(save_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    run_comprehensive_benchmark()
