import os
import sys
from pathlib import Path
import argparse

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.gan_inference import (
    get_project_paths,
    generate_progression_gif,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a GIF across model checkpoints.")
    parser.add_argument("--models-dir", type=str, default=None, help="Path to model repo directory")
    parser.add_argument("--start-step", type=int, default=100, help="First model step")
    parser.add_argument("--end-step", type=int, default=700, help="Last model step")
    parser.add_argument("--step-size", type=int, default=100, help="Step size between models")
    parser.add_argument("--fps", type=int, default=1, help="Frames per second")
    parser.add_argument("--output", type=str, default=None, help="Output GIF path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = get_project_paths()
    models_dir = Path(args.models_dir) if args.models_dir else paths.models_dir
    output_path = Path(args.output) if args.output else paths.outputs_dir / "gan_progression.gif"

    steps = range(args.start_step, args.end_step + 1, args.step_size)
    generate_progression_gif(models_dir, output_path, steps=steps, fps=args.fps)

    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
