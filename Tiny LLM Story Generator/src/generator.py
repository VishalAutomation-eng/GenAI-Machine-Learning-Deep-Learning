import torch

def generate_response(model, tokenizer, device, prompt, max_length=150):
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        padding=True
    ).to(device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_length=max_length,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.7
        )

    return tokenizer.decode(output_ids[0], skip_special_tokens=True)
