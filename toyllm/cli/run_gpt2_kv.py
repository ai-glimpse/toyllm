import time

import typer

from toyllm.core import GenerationConfig
from toyllm.gpt2 import GPTModelSize
from toyllm.gpt2_kv import GPTKVModel, GPTTextGenerator


def main(
    prompt: str = "Alan Turing theorized that computers would one day become",
    model_size: GPTModelSize = GPTModelSize.SMALL,
    max_new_tokens: int = 40,
    top_k: int | None = None,
    temperature: float | None = None,
) -> None:
    """Generate text using a GPT-2 model."""
    gpt_model = GPTKVModel(model_size).load()
    text_generator = GPTTextGenerator(gpt_model=gpt_model)

    start_time = time.time()
    generate_text = text_generator.generate(
        prompt=prompt,
        config=GenerationConfig(
            max_new_tokens=max_new_tokens,
            top_k=top_k,
            temperature=temperature,
        ),
    )
    print(generate_text)
    end_time = time.time()
    print(f"Time elapsed: {end_time - start_time:.2f}s")


if __name__ == "__main__":
    typer.run(main)
