from src.load_pipeline import load_diffusion_pipeline
from src.generate_images import generate_and_display

if __name__ == "__main__":
    pipe = load_diffusion_pipeline()
    generate_and_display(pipe)
