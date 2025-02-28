import time

import typer

from toyllm.gpt2 import GPTModel
from toyllm.gpt2 import TextGenerator as GptTextGenerator
from toyllm.gpt2 import gpt2_tokenizer
from toyllm.sps import GPTSpsModel, SpsTextGenerator


def main(
    prompt_text: str = "Alan Turing theorized that computers would one day become",
    generate_tokens: int = 256,
    k: int = 4,  # K in sps paper
):
    # Test the speculative sampling
    sps_text_generator = SpsTextGenerator(
        tokenizer=gpt2_tokenizer,
        target_model=GPTSpsModel(model_name="1558M"),
        draft_model=GPTSpsModel(model_name="124M"),
        lookahead=k,
    )

    start_time = time.time()
    generate_text = sps_text_generator.generate(
        prompt_text=prompt_text,
        min_gen_tokens=generate_tokens,
        temperature=0,
    )
    end_time = time.time()
    print(
        f"[Speculative Sampling]: Time elapsed: {end_time - start_time:.2f}s\n"
        f"Prompt: {prompt_text}\n"
        f"Generated: {generate_text[:200]}"
    )

    # Test the GPT2 model
    gpt = GPTModel("1558M").load("../../models/gpt_1558m.pt")
    gpt_text_generator = GptTextGenerator(gpt_model=gpt)

    start_time = time.time()
    generate_text = gpt_text_generator.generate(
        prompt_text=prompt_text,
        max_gen_tokens=generate_tokens,
    )
    end_time = time.time()
    print(
        f"[Naive GPT2 Auto-Regressive]: Time elapsed: {end_time - start_time:.2f}s\n"
        f"Prompt: {prompt_text}\n"
        f"Generated: {generate_text[:200]}"
    )


if __name__ == "__main__":
    typer.run(main)
