import time

import typer

from toyllm.gpt2 import GPTModel, GPTModelSize, GPTTextGenerator


def main(
    prompt: str = "Alan Turing theorized that computers would one day become",
    model_size: GPTModelSize = GPTModelSize.SMALL,
    max_gen_tokens: int = 40,
):
    gpt_model = GPTModel(model_size).load()
    text_generator = GPTTextGenerator(gpt_model=gpt_model)

    start_time = time.time()
    generate_text = text_generator.generate(
        prompt=prompt,
        max_gen_tokens=max_gen_tokens,
    )
    print(generate_text)
    end_time = time.time()
    print("Time elapsed: {:.2f}s".format(end_time - start_time))


if __name__ == "__main__":
    typer.run(main)
