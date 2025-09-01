from typing import AsyncGenerator
from transformers import pipeline

# Create a small text-generation pipeline (demo-friendly model)
_generator = pipeline('text-generation', model='distilgpt2')

def generate_text(prompt: str, max_new_tokens: int = 128) -> str:
    out = _generator(prompt, max_new_tokens=max_new_tokens, do_sample=True, temperature=0.7)
    return out[0]['generated_text']

async def small_generate(prompt: str, max_new_tokens: int = 64) -> AsyncGenerator[str, None]:
    # Simulate streaming tokens chunk-by-chunk from a tiny local generator
    text = generate_text(prompt, max_new_tokens=max_new_tokens)
    for tok in text.split():
        yield tok + ' '
