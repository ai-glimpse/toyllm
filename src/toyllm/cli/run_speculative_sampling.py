import typer
from rich.console import Console

from toyllm.core import GenerationConfig
from toyllm.gpt2 import GPTModel, GPTModelSize, GPTTextGenerator, gpt2_tokenizer
from toyllm.sps import GPTSpsModel, SpsTextGenerator
from toyllm.util import Timer


def main(
    prompt_text: str = "Alan Turing theorized that computers would one day become",
    max_new_tokens: int = 256,
    temperature: float = 0.0,
    top_k: int | None = None,
    k: int = 4,  # K in sps paper
) -> None:
    """Generate text using a GPT-2 model and a Speculative Sampling model."""
    generate_config = GenerationConfig(
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
    )

    console = Console()
    console.print(f"Prompt: {prompt_text}", style="bold blue")
    # Test the speculative sampling
    sps_text_generator = SpsTextGenerator(
        tokenizer=gpt2_tokenizer,
        target_model=GPTSpsModel(model_size=GPTModelSize.XLARGE),
        draft_model=GPTSpsModel(model_size=GPTModelSize.SMALL),
        lookahead=k,
    )

    console.print(f"{'-' * 20} Speculative Sampling {'-' * 20}", style="bold blue")
    with Timer(name="Speculative Sampling"):
        generate_text = sps_text_generator.generate(
            prompt=prompt_text,
            config=generate_config,
        )
    console.print(f"Generated: {generate_text[:200]}", style="bold green")
    console.print(f"{'-' * 20} Speculative Sampling {'-' * 20}", style="bold blue")

    # Test the GPT2 model
    gpt = GPTModel(GPTModelSize.XLARGE).load()
    gpt_text_generator = GPTTextGenerator(gpt_model=gpt)

    console.print(f"{'-' * 20} Naive GPT2 Auto-Regressive {'-' * 20}", style="bold blue")
    with Timer(name="Naive GPT2 Auto-Regressive"):
        generate_text = gpt_text_generator.generate(
            prompt=prompt_text,
            config=generate_config,
        )
    console.print(f"Generated: {generate_text[:200]}", style="bold green")
    console.print(f"{'-' * 20} Naive GPT2 Auto-Regressive {'-' * 20}", style="bold blue")


if __name__ == "__main__":
    typer.run(main)
