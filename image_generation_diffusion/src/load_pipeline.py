import torch
from diffusers import StableDiffusionPipeline

def load_diffusion_pipeline():
    model_id = "stabilityai/stable-diffusion-xl-base-1.0"

    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16
    )

    pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")
    return pipe
