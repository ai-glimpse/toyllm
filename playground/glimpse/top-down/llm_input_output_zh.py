import marimo

__generated_with = "0.8.22"
app = marimo.App(width="medium")


@app.cell
def __():
    import marimo as mo

    mo.md("""# LLM Glimpse：自上而下的方法""").center()
    return (mo,)


@app.cell
def __():
    import torch

    from toyllm.gpt2 import GPTModel, GPTTextGenerator, gpt2_tokenizer

    gpt = GPTModel("124M").load("../../../models/gpt_124m.pt")
    text_generator = GPTTextGenerator(gpt_model=gpt)
    return (
        GPTModel,
        GPTTextGenerator,
        gpt,
        gpt2_tokenizer,
        text_generator,
        torch,
    )


@app.cell
def __(mo):
    mo.md(r"""## 1. 概述：Prompt与Response""")


@app.cell
def __(mo):
    prompt_text = "Alan Turing theorized that computers would one day become"

    mo.md(f"**Prompt**: {prompt_text}").style({"color": "green", "font-weight": "bold"})
    return (prompt_text,)


@app.cell
def __(mo, prompt_text, text_generator):
    generate_text = text_generator.generate(
        prompt=prompt_text,
        max_gen_tokens=40,
        top_k=10,
    )

    mo.md(f"**LLM(GPT2) Response**: {generate_text}").style({"color": "green", "font-weight": "bold"})
    return (generate_text,)


@app.cell
def __(mo):
    mo.md(r"""## 2. 步骤：Prompt -> 第一个Token -> ... -> 完整Response""")


@app.cell
def __(mo, prompt_text):
    mo.md(
        f"""给定 Prompt：*{prompt_text}*，\n
    LLM 生成下一个标记：**' the'** \n
    之后我们将 Prompt 更改为：*{prompt_text}***{" the"}**\n
    LLM 生成下一个标记：**' most'** \n
    之后我们将 Prompt 更改为：*{prompt_text}***{" the most"}**\n
    ... ... \n
    最后，我们得到了之前显示的完整 Response！
    """,
    )


@app.cell
def __(mo):
    mo.md(r"""## 3. 细节：Prompt -> Response中的第一个标记""")


@app.cell
def __(mo):
    mo.md(r"""### 3.1 Prompt文本 -> Prompt token id""")


@app.cell
def __(gpt2_tokenizer, mo, prompt_text):
    prompt_tokens = gpt2_tokenizer.encode(prompt_text)

    mo.md(f"Prompt 标记（长度={len(prompt_tokens)}）：**{prompt_tokens}**")
    return (prompt_tokens,)


@app.cell
def __(gpt2_tokenizer, mo, prompt_tokens):
    subwords = [gpt2_tokenizer.decode([token]) for token in prompt_tokens]

    mo.md(
        f"""从 Tokens 我们可以反向获取到 Prompt 的 subwords（由 BPE 生成）：**{subwords}**
        \n注意 **theorized** 被拆分为 **theor** 和 **ized**。所以我们在 Prompt 中得到了 10 个 subwords 和 9 个单词。
        (还要注意一些子词前的空格）""",
    )
    return (subwords,)


@app.cell
def __(mo, prompt_tokens, text_generator, torch):
    prompt_tokens_tensor = torch.tensor(prompt_tokens).unsqueeze(0)[:, -text_generator.context_length :]

    mo.md(
        f"""将 Prompt 标记转换为 torch 张量并按上下文长度（**{text_generator.context_length}**）截断。\n
    然后我们得到了形状为 **{prompt_tokens_tensor.shape}** 的 prompt_tokens_tensor **{prompt_tokens_tensor}**""",
    )
    return (prompt_tokens_tensor,)


@app.cell
def __(mo):
    mo.md(r"""### 3.2 Prompt标记 -> 下一个标记的 logits""")


@app.cell
def __(mo, prompt_tokens_tensor, text_generator, torch):
    with torch.inference_mode():
        logits = text_generator.gpt_model(prompt_tokens_tensor.to(text_generator.gpt_model.device))

    mo.md(
        f"将 `prompt_tokens_tensor` 输入到 GPT2 神经网络中，我们得到了 **logits**，一个形状为 **{logits.shape}** 的张量",
    )
    return (logits,)


@app.cell
def __(logits, mo):
    logits_last = logits[:, -1, :]

    mo.md(
        f"""仅选择最后的 logits（shape 为 **{logits_last.shape}**），这是词表（包含 50,527 个 Token）上下一个标记的 logits（可以视为未归一化的概率）""",
    )
    return (logits_last,)


@app.cell
def __(mo):
    mo.md(r"""### 3.3 下一个标记的 logits -> 下一个标记（Response中的第一个标记）""")


@app.cell
def __(logits_last, mo, text_generator, torch):
    logits_last_with_top_k = text_generator._logits_top_k_filter(logits_last, top_k=10)
    next_token_id = torch.argmax(logits_last_with_top_k, dim=-1, keepdim=True)

    mo.md(
        f"""对最后的 logit 应用 topK 采样和 Temperature 缩放，我们得到了下一个标记 id：
    **{next_token_id}**""",
    )
    return logits_last_with_top_k, next_token_id


@app.cell
def __(gpt2_tokenizer, mo, next_token_id):
    next_token = gpt2_tokenizer.decode(next_token_id.squeeze(0).tolist())

    mo.md(f"""使用 gpt2 tokenizer 解码，我们得到了下一个标记：**'{next_token}'**""")
    return (next_token,)


@app.cell
def __(mo):
    mo.md(r"""### 3.4 下一个Token（Response中的第一个Token） -> 完整Response""")


@app.cell
def __(mo):
    mo.md(
        """添加 Response 中的第一个标记，我们得到了一个新的 Prompt，将其输入到 llm 中，我们得到了一个新的下一个标记，...，最后我们将得到开头显示的完整 Response！""",
    )


@app.cell
def __(mo):
    mo.md(
        """由 [ToyLLM](https://github.com/ai-glimpse/toyllm) 提供支持 | 作者：[MathewShen](https://github.com/shenxiangzhuang)""",
    ).center()


@app.cell
def __():
    return


if __name__ == "__main__":
    app.run()
