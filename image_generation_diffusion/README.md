## Image Generation with Diffusion Models

### Model Used
For this assignment, I used the **stabilityai/stable-diffusion-2-1** model from the Hugging Face Hub.  
This model is well-known for producing high-quality, realistic images from textual prompts and is widely adopted in both research and industry.

### Prompts and Parameters
The following prompts were used:
1. *"A futuristic city at sunset, cyberpunk style"*
2. *"A realistic portrait of an astronaut riding a horse on Mars"*

Key parameters:
- **num_inference_steps = 40**  
  Controls the number of denoising steps. Higher values improve image quality.
- **guidance_scale = 7.5**  
  Balances creativity and prompt adherence.

### Observations
- The generated images were visually detailed and aligned well with the prompts.
- The cyberpunk city showed strong lighting and color contrast.
- The astronaut image demonstrated the model’s ability to combine realistic and imaginative concepts.

### Challenges and Findings
- Image generation is computationally expensive and performs best on a GPU.
- Prompt wording significantly impacts image quality.
- Adjusting guidance scale helped control creativity vs realism.

### Conclusion
This experiment demonstrates the power of diffusion models for high-quality image generation. Stable Diffusion provides an accessible and flexible way to explore generative AI using Hugging Face tools.
