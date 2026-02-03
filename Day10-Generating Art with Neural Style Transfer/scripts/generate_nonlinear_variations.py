import os
import sys
from pathlib import Path
import argparse

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.gan_inference import (
    get_project_paths,
    load_generator,
    resolve_model_path,
    generate_nonlinear_variations,
    save_figure,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate nonlinear GAN variations.")
    parser.add_argument("--model-step", type=int, default=100, help="Model step (e.g., 100, 200, ...)")
    parser.add_argument("--models-dir", type=str, default=None, help="Path to model repo directory")
    parser.add_argument("--rows", type=int, default=6, help="Number of rows")
    parser.add_argument("--cols", type=int, default=10, help="Number of columns")
    parser.add_argument("--output", type=str, default=None, help="Output image path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = get_project_paths()
    models_dir = Path(args.models_dir) if args.models_dir else paths.models_dir
    model_path = resolve_model_path(models_dir, args.model_step)
    output_path = Path(args.output) if args.output else paths.outputs_dir / "nonlinear_variations.png"

    generator = load_generator(model_path)
    fig = generate_nonlinear_variations(generator, rows=args.rows, cols=args.cols)
    save_figure(fig, output_path)

    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
