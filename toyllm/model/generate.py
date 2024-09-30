import logging
from typing import Optional

import jaxtyping
import tiktoken
import torch
from typeguard import typechecked as typechecker

from toyllm.device import current_device
from toyllm.model.gpt import GPTModel
from toyllm.tokenizer import gpt2_tokenizer, text_to_token_ids, token_ids_to_text

logger = logging.getLogger(__name__)


class TextGenerator:
    def __init__(
        self,
        gpt_model: GPTModel,
        tokenizer: tiktoken.Encoding = gpt2_tokenizer,
        seed: int = 42,
    ):
        self.tokenizer = tokenizer
        self.seed = seed
        self.gpt_model = self.__load_gpt_model(gpt_model)

    def __load_gpt_model(self, gpt_model: GPTModel) -> GPTModel:
        torch.manual_seed(self.seed)
        # disable dropout and so on
        gpt_model.eval()
        if gpt_model.device != current_device:
            gpt_model.to(current_device)
        return gpt_model

    @property
    def context_length(self) -> int:
        return self.gpt_model.config.ctx_len

    def generate(
        self,
        prompt_text: str,
        max_gen_tokens: int = 10,
        top_k: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """

        :param prompt_text: prompt text
        :param max_gen_tokens: maximum number of tokens to generate
        :param top_k: only keep `top_k`(logits) candidate tokens to select from.
        A little `top_k` will reduce the randomness of generated output.
        `top_k` must be greater than 0, like 5, 10 and so on.
        :param temperature: "Temperatures greater than 1 will result in more uniformly distributed token probabilities
        after applying the softmax; temperatures smaller than 1 will result in
        more confident (sharper or more peaky) distributions after applying the softmax"
        (https://github.com/rasbt/LLMs-from-scratch/blob/main/ch05/01_main-chapter-code/ch05.ipynb)
        The default temperature value is 0.6 in llama2.

        """
        # prompt text to tokens: (1, n_tokens)
        prompt_tokens = text_to_token_ids(prompt_text, self.tokenizer).to(self.gpt_model.device)

        for _ in range(max_gen_tokens):
            # Crop current context if it exceeds the supported context size(ctx_len)
            # E.g., if LLM supports only 5 tokens, and the context size is 10
            # then only the last 5 tokens are used as context

            # (batch, n_tokens) --(crop context)--> (batch, n_tokens' = min(ctx_len, n_tokens))
            context_text_token_ids = prompt_tokens[:, -self.context_length :]

            # Get the predictions
            # use `inference_mode` rather than `no_grad`(https://stackoverflow.com/questions/74191070)
            with torch.inference_mode():
                # (batch, n_token') --(forward)--> (batch, n_token', vocab_size)
                logits = self.gpt_model(context_text_token_ids)

            # Focus only on the last time step
            # (batch, n_tokens', vocab_size) --(keep last time step token)--> (batch, vocab_size)
            logits = logits[:, -1, :]

            # logits filter & scale
            if top_k is not None:
                logits = self._logits_top_k_filter(logits, top_k)
            if temperature is not None:
                probs = self._logits_temperature_scale(logits, temperature)
                # Sample from the scaled multinomial distribution
                # (batch, vocab_size)--(keep the max prob token)--> (batch, 1)
                next_token_id = torch.multinomial(probs, num_samples=1)
            else:
                # Get the idx of the vocab entry with the highest logits value
                # (batch, vocab_size)--(keep the max prob token)--> (batch, 1)
                next_token_id = torch.argmax(logits, dim=-1, keepdim=True)

            # Append sampled index to the running sequence
            # (batch, n_tokens') --(append next token)--> (batch, n_tokens' + 1)
            prompt_tokens = torch.cat((prompt_tokens, next_token_id), dim=1)

        generate_text = token_ids_to_text(prompt_tokens)

        return generate_text

    @jaxtyping.jaxtyped(typechecker=typechecker)
    @staticmethod
    def _logits_top_k_filter(
        logits: jaxtyping.Float[torch.Tensor, "batch_size vocab_size"],
        top_k: int,
    ) -> jaxtyping.Float[torch.Tensor, "batch_size vocab_size"]:
        """
        ref1: https://github.com/rasbt/LLMs-from-scratch/blob/62fb11d5e0449a6d49bda7337d6cfa5a735718da/ch05/01_main-chapter-code/generate.py#L166-L185
        ref2: https://github.com/huggingface/transformers/blob/c4d4e8bdbd25d9463d41de6398940329c89b7fb6/src/transformers/generation_utils.py#L903-L941
        ref3: https://github.com/meta-llama/llama/blob/main/llama/generation.py#L188-L192
        """
        top_k = min(top_k, logits.size(-1))  # make sure top_k <= vocab size
        top_k_logits, _top_k_indexes = torch.topk(logits, k=top_k, dim=-1)
        min_logit = top_k_logits[:, -1]
        logits = torch.where(logits < min_logit, torch.tensor(float("-inf")).to(logits.device), logits)
        return logits

    @jaxtyping.jaxtyped(typechecker=typechecker)
    @staticmethod
    def _logits_temperature_scale(
        logits: jaxtyping.Float[torch.Tensor, "batch_size vocab_size"],
        temperature: float,
    ) -> jaxtyping.Float[torch.Tensor, "batch_size vocab_size"]:
        logits = logits / temperature
        probs = torch.softmax(logits, dim=-1)
        return probs


if __name__ == "__main__":
    from toyllm.model.config import GPTModelSize

    model_file_path = "./models/gpt_124m.pt"
    gpt = GPTModel(GPTModelSize.SMALL).load(model_file_path)
    text_generator = TextGenerator(gpt_model=gpt)

    prompt_text = "Alan Turing theorized that computers would one day become"
    generate_text = text_generator.generate(
        prompt_text=prompt_text,
        max_gen_tokens=40,
        top_k=10,
        temperature=1.5,
    )
    print(generate_text)
