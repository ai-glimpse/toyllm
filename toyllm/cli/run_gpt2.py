import pathlib
import time
from typing import Literal

from toyllm.gpt2.generate import TextGenerator
from toyllm.gpt2.gpt import GPTModel


def run(
    prompt_text: str,
    model_name: Literal["124M", "355M", "774M", "1558M"] = "124M",
    max_gen_tokens: int = 40,
):
    gpt_model = GPTModel(model_name).load(f"{pathlib.Path(__file__).parent}/../../models/gpt_{model_name.lower()}.pt")
    text_generator = TextGenerator(gpt_model=gpt_model)

    start_time = time.time()
    generate_text = text_generator.generate(
        prompt_text=prompt_text,
        max_gen_tokens=max_gen_tokens,
    )
    print(generate_text)
    end_time = time.time()
    print("Time elapsed: {:.2f}s".format(end_time - start_time))


if __name__ == "__main__":
    run("Alan Turing theorized that computers would one day become")
