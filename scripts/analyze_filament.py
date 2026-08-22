"""CLI for Mask2Former Phase 2 filament analysis."""
import argparse
import sys
from pathlib import Path
import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from inference.phase2 import run_phase2_analysis


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze a solar image with Mask2Former Phase 2.")
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--model", default="mask2former", help="Any model registered in model_hub")
    parser.add_argument("--threshold", default=0.5, type=float)
    parser.add_argument("--min-area", default=50, type=int)
    parser.add_argument("--output", default="outputs", type=Path)
    args = parser.parse_args()
    from model_hub import all_archs
    if args.model not in all_archs():
        raise SystemExit(f"Unknown model '{args.model}'. Available models: {', '.join(all_archs())}")
    image = cv2.imread(str(args.image), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise SystemExit(f"Could not read image: {args.image}")
    result = run_phase2_analysis(image, image_id=args.image.stem, model_name=args.model, threshold=args.threshold,
                                 min_area=args.min_area, output_dir=args.output)
    print(f"{args.model} detected {len(result['filaments'])} filament(s)")
    print(f"Catalog: {args.output / 'catalog' / 'filament_catalog.json'}")


if __name__ == "__main__":
    main()
