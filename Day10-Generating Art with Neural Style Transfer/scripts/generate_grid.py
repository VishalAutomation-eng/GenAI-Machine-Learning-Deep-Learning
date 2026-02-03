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
    generate_images_grid,
    save_figure,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a 1xN grid of GAN samples.")
    parser.add_argument("--model-step", type=int, default=100, help="Model step (e.g., 100, 200, ...)")
    parser.add_argument("--models-dir", type=str, default=None, help="Path to model repo directory")
    parser.add_argument("--num-images", type=int, default=10, help="Number of images")
    parser.add_argument("--output", type=str, default=None, help="Output image path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = get_project_paths()
    models_dir = Path(args.models_dir) if args.models_dir else paths.models_dir
    model_path = resolve_model_path(models_dir, args.model_step)
    output_path = Path(args.output) if args.output else paths.outputs_dir / "grid_1x10.png"

    generator = load_generator(model_path)
    fig = generate_images_grid(generator, num_images=args.num_images)
    save_figure(fig, output_path)

    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
