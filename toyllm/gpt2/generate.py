import logging

import tiktoken
import torch

from toyllm.core import GenerationConfig, logits_temperature_scale, logits_top_k_filter
from toyllm.device import current_device
from toyllm.gpt2.gpt import GPTModel
from toyllm.gpt2.tokenizer import gpt2_tokenizer, text_to_token_ids, token_ids_to_text

logger = logging.getLogger(__name__)


class GPTTextGenerator:
    def __init__(
        self,
        gpt_model: GPTModel,
        tokenizer: tiktoken.Encoding = gpt2_tokenizer,
        seed: int = 42,
    ) -> None:
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
        prompt: str,
        config: GenerationConfig | None = None,
    ) -> str:
        """The GPT2 model generates text based on the given prompt.

        Args:
            prompt: prompt text
            config: generation config
        """
        if config is None:
            config = GenerationConfig()

        # prompt text to tokens: (1, n_tokens)
        prompt_tokens = text_to_token_ids(prompt, self.tokenizer).to(self.gpt_model.device)

        for _ in range(config.max_new_tokens):
            # Crop current context if it exceeds the supported context size(ctx_len)
            # E.g., if LLM supports only 5 tokens, and the context size is 10,
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
            if config.top_k is not None:
                logits = logits_top_k_filter(logits, config.top_k)
            if config.temperature is not None:
                logits = logits_temperature_scale(logits, config.temperature)
                probs = torch.softmax(logits, dim=-1)
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
        return generate_text  # type: ignore[no-any-return]
