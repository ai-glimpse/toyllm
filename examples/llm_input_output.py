import marimo

__generated_with = "0.8.22"
app = marimo.App(width="medium")


@app.cell
def __():
    import marimo as mo

    mo.md("# Welcome to [ToyLLM](https://github.com/ai-glimpse/toyllm)! 🌊")
    return (mo,)


@app.cell
def __():
    from toyllm.model.gpt import GPTModel
    from toyllm.model.generate import TextGenerator


    gpt = GPTModel("124M").load("../models/gpt_124m.pt")
    text_generator = TextGenerator(gpt_model=gpt)
    return GPTModel, TextGenerator, gpt, text_generator


@app.cell
def __(text_generator):
    prompt_text = "Alan Turing theorized that computers would one day become"
    generate_text = text_generator.generate(
        prompt_text=prompt_text,
        max_gen_tokens=40,
        top_k=10,
        temperature=1.5,
    )

    generate_text
    return generate_text, prompt_text


@app.cell
def __():
    return


if __name__ == "__main__":
    app.run()
