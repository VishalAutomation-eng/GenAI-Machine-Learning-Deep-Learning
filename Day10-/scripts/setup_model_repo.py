import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.gan_inference import get_project_paths


def main() -> None:
    paths = get_project_paths()
    target_dir = paths.models_dir
    target_dir.parent.mkdir(parents=True, exist_ok=True)

    if target_dir.exists():
        print(f"Model repo already exists at: {target_dir}")
        return

    repo_url = "https://github.com/AshishJangra27/Face-Generator-with-GAN"
    print(f"Cloning {repo_url} -> {target_dir}")
    subprocess.run(["git", "clone", repo_url, str(target_dir)], check=True)


if __name__ == "__main__":
    main()
