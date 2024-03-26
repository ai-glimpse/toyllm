import tiktoken
import torch

from toyllm.device import get_device
from toyllm.model.config import GPT_CONFIG_124M, GPTModelConfig
from toyllm.model.gpt import GPTModel
from toyllm.tokenizer import gpt2_tokenizer, text_to_token_ids, token_ids_to_text


class TextGenerator:
    def __init__(
        self,
        model_config: GPTModelConfig,
        tokenizer: tiktoken.Encoding = gpt2_tokenizer,
        seed: int = 42,
    ):
        self.model_config = model_config
        self.tokenizer = tokenizer
        self.device = get_device()
        self.seed = seed

        self.gpt_model = self.get_gpt_model()

    def get_gpt_model(self) -> GPTModel:
        torch.manual_seed(self.seed)

        model = GPTModel(self.model_config)
        # disable dropout and so on
        model.eval()
        model.to(self.device)
        return model

    @property
    def context_length(self) -> int:
        return self.model_config.ctx_len

    def generate_text_from_prompt(
        self,
        prompt_text: str,
        max_gen_tokens: int = 10,
    ) -> str:
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

            # Get the idx of the vocab entry with the highest logits value
            # (batch, vocab_size)--(keep the max prob token)--> (batch, 1)
            next_token_id = torch.argmax(logits, dim=-1, keepdim=True)

            # Append sampled index to the running sequence
            # (batch, n_tokens') --(append next token)--> (batch, n_tokens' + 1)
            prompt_tokens = torch.cat((prompt_tokens, next_token_id), dim=1)

        generate_text = token_ids_to_text(prompt_tokens)

        return generate_text


if __name__ == "__main__":
    text_generator = TextGenerator(model_config=GPT_CONFIG_124M)

    prompt_text = "Hello, I am"
    generate_text = text_generator.generate_text_from_prompt(prompt_text=prompt_text)
    # Hello, I amulf Kai cog Portugal paStudio THE APR lie therapeutic
    print(generate_text)
