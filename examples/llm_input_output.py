import marimo

__generated_with = "0.8.22"
app = marimo.App(width="medium")


@app.cell
def __():
    import marimo as mo

    mo.md("""# LLM Glimpse: A Top-Down Approach""").center()
    return (mo,)


@app.cell
def __():
    import torch

    from toyllm import GPTModel, TextGenerator, gpt2_tokenizer

    gpt = GPTModel("124M").load("../models/gpt_124m.pt")
    text_generator = TextGenerator(gpt_model=gpt)
    return (
        GPTModel,
        TextGenerator,
        gpt,
        gpt2_tokenizer,
        text_generator,
        torch,
    )


@app.cell
def __(mo):
    mo.md(r"""## 1. Overview: Prompt & Response""")
    return


@app.cell
def __(mo):
    prompt_text = "Alan Turing theorized that computers would one day become"

    mo.md(f"**Prompt**: {prompt_text}").style({"color": "green", "font-weight": "bold"})
    return (prompt_text,)


@app.cell
def __(mo, prompt_text, text_generator):
    generate_text = text_generator.generate(
        prompt_text=prompt_text,
        max_gen_tokens=40,
        top_k=10,
    )

    mo.md(f"**LLM(GPT2) Response**: {generate_text}").style({"color": "green", "font-weight": "bold"})
    return (generate_text,)


@app.cell
def __(mo):
    mo.md(r"""## 2. The steps: Prompt -> First Token in Response -> ... -> Full Response""")
    return


@app.cell
def __(mo, prompt_text):
    mo.md(
        f"""Given the prompt: *{prompt_text}*, \n
    LLM generate the next token: **' the'** \n
    After that we change the prompt to: *{prompt_text}***{" the"}**\n
    LLM generate the next token: **' most'** \n
    After that we change the prompt to: *{prompt_text}***{" the most"}**\n
    ... ... \n
    Finally, we got the full response as shown before!
    """
    )
    return


@app.cell
def __(mo):
    mo.md(r"""## 3. The details: Prompt -> First Token in Response""")
    return


@app.cell
def __(mo):
    mo.md(r"""### 3.1 Prompt text -> Prompt tokens""")
    return


@app.cell
def __(gpt2_tokenizer, mo, prompt_text):
    prompt_tokens = gpt2_tokenizer.encode(prompt_text)

    mo.md(f"Prompt tokens(length={len(prompt_tokens)}): **{prompt_tokens}**")
    return (prompt_tokens,)


@app.cell
def __(gpt2_tokenizer, mo, prompt_tokens):
    subwords = [gpt2_tokenizer.decode([token]) for token in prompt_tokens]

    mo.md(
        f"""From tokens we can know the subwords of the prompt (generated by BPE): **{subwords}**\n
    Note that **theorized** is split into **theor** and **ized**. So we get 10 subwords with 9 words in the prompt.
    \n(Also note the white spaces before some subwords)"""
    )
    return (subwords,)


@app.cell
def __(mo, prompt_tokens, text_generator, torch):
    prompt_tokens_tensor = torch.tensor(prompt_tokens).unsqueeze(0)[:, -text_generator.context_length :]

    mo.md(
        f"""Transform the prompt token into torch tensor and truncate it by context length(**{text_generator.context_length}**). \n
    Then we got prompt_tokens_tensor **{prompt_tokens_tensor}** with shape **{prompt_tokens_tensor.shape}**"""
    )
    return (prompt_tokens_tensor,)


@app.cell
def __(mo):
    mo.md(r"""### 3.2 Prompt tokens -> Next token logits""")
    return


@app.cell
def __(mo, prompt_tokens_tensor, text_generator, torch):
    with torch.inference_mode():
        logits = text_generator.gpt_model(prompt_tokens_tensor)

    mo.md(
        f"Feed `prompt_tokens_tensor` into the GPT2 Neural Network, we get the **logits**, a tensor with shape **{logits.shape}**"
    )
    return (logits,)


@app.cell
def __(logits, mo):
    logits_last = logits[:, -1, :]

    mo.md(
        f"""Select the last logits only (with shape **{logits_last.shape}**), which are the next token's logits (can be treated as unnormalized probability) on the vocabulary set (contains 50,527 items)"""
    )
    return (logits_last,)


@app.cell
def __(mo):
    mo.md(r"""### 3.3 Next token logits -> Next Token(First Token in Response)""")
    return


@app.cell
def __(logits_last, mo, text_generator, torch):
    logits_last_with_top_k = text_generator._logits_top_k_filter(logits_last, top_k=10)
    next_token_id = torch.argmax(logits_last_with_top_k, dim=-1, keepdim=True)

    mo.md(
        f"""Apply topK sampling and temperature scaling to the last logit, we get the next token id:
    **{next_token_id}**"""
    )
    return logits_last_with_top_k, next_token_id


@app.cell
def __(gpt2_tokenizer, mo, next_token_id):
    next_token = gpt2_tokenizer.decode(next_token_id.squeeze(0).tolist())

    mo.md(f"""Decode with the gpt2 tokenizer, we get the next token: **'{next_token}'**""")
    return (next_token,)


@app.cell
def __(mo):
    mo.md(r"""### 3.4 Next Token(First Token in Response) -> Full Response""")
    return


@app.cell
def __(mo):
    mo.md(
        "Add the first token in response, we got a new prompt, feed it into the llm, we got a new next token, ..., finally we will got the full response as shown in beginning!"
    )
    return


@app.cell
def __(mo):
    mo.md(
        """Powered by [ToyLLM](https://github.com/ai-glimpse/toyllm) | Author: [MathewShen](https://github.com/shenxiangzhuang)."""
    ).center()
    return


@app.cell
def __():
    return


if __name__ == "__main__":
    app.run()
