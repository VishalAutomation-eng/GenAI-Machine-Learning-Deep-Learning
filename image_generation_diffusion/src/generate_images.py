import matplotlib.pyplot as plt

def generate_and_display(pipe):
    prompts = [
        "A futuristic city at sunset, cyberpunk style",
        "A realistic portrait of an astronaut riding a horse on Mars"
    ]

    images = []

    for prompt in prompts:
        image = pipe(
            prompt=prompt,
            num_inference_steps=40,
            guidance_scale=7.5
        ).images[0]

        images.append((prompt, image))

    # Display images
    plt.figure(figsize=(12, 6))
    for i, (prompt, img) in enumerate(images):
        plt.subplot(1, 2, i + 1)
        plt.imshow(img)
        plt.axis("off")
        plt.title(prompt)

    plt.show()
