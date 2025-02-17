from dataclasses import dataclass
from typing import Optional

import numpy as np
import tiktoken
import torch

from toyllm.sps.models import BaseSpsModel


@dataclass
class TextGenerator:
    tokenizer: tiktoken.Encoding
    target_model: BaseSpsModel
    draft_model: BaseSpsModel
    lookahead: int = 5
    seed: int = 42

    def __post_init__(self):
        np.random.seed(self.seed)

    def generate(
        self,
        prompt_text: str,
        min_gen_tokens: int = 100,
        temperature: Optional[float] = None,
    ) -> str:
        context_length = self.target_model.get_context_length()

        # prompt text to tokens: (1, n_tokens)
        text_id_list = self.tokenizer.encode(prompt_text)
        prompt_tokens = torch.tensor(text_id_list).unsqueeze(0).to(self.target_model.device())  # add batch dimension
        generated_tokens: list[torch.Tensor] = []

        while len(generated_tokens) < min_gen_tokens:
            # Get the predictions
            # use `inference_mode` rather than `no_grad`(https://stackoverflow.com/questions/74191070)
            with torch.inference_mode():
                # Draft model lookahead: (prompt_tokens, next_token_id, next_token_logits)
                lookahead_tuples = []
                draft_prompt_tokens = prompt_tokens.clone()
                for _ in range(self.lookahead):
                    draft_next_token_ids, draft_next_token_logits = self._get_logits_and_next_token_id(
                        draft_prompt_tokens,
                        context_length,
                        self.draft_model,
                        temperature,
                    )
                    draft_next_token_id, draft_next_token_logits = draft_next_token_ids[0], draft_next_token_logits[0]
                    lookahead_tuples.append((draft_prompt_tokens.clone(), draft_next_token_id, draft_next_token_logits))
                    # Append sampled index to the running sequence
                    # (batch, n_tokens') --(append next token)--> (batch, n_tokens' + 1)
                    draft_prompt_tokens = torch.cat((draft_prompt_tokens, draft_next_token_ids), dim=1)

                # Target model logits
                target_model_probs = self._get_latest_n_token_probs(
                    draft_prompt_tokens, self.lookahead + 1, context_length, self.target_model, temperature
                ).squeeze(0)

                all_accept = True
                # Compare the target model's next token with the draft model's lookahead
                for t in range(self.lookahead):
                    draft_next_token_id, draft_next_token_logits = lookahead_tuples[t][1], lookahead_tuples[t][2]
                    target_next_token_probs = target_model_probs[t, :]

                    r = np.random.rand()
                    if r < min(
                        1.0,
                        (target_next_token_probs[draft_next_token_id] / draft_next_token_logits[draft_next_token_id])
                        .cpu()
                        .item(),
                    ):
                        print("Accept!")
                        next_generated_token = draft_next_token_id
                        generated_tokens.append(next_generated_token)
                        # Update prompt tokens
                        prompt_tokens = torch.cat((prompt_tokens, next_generated_token.unsqueeze(0)), dim=1)
                    else:
                        all_accept = False
                        print("Reject!")
                        prob_diff = target_next_token_probs - draft_next_token_logits
                        # element-wise max to 0
                        prob_diff = torch.clamp(prob_diff, min=0)
                        prob_diff = prob_diff / torch.sum(prob_diff)
                        next_generated_token = torch.multinomial(prob_diff, num_samples=1)
                        generated_tokens.append(next_generated_token)
                        # Update prompt tokens
                        prompt_tokens = torch.cat((prompt_tokens, next_generated_token.unsqueeze(0)), dim=1)
                        break
                if all_accept:
                    print("All accept!")
                    next_generated_token = torch.multinomial(target_model_probs[-1], num_samples=1)
                    generated_tokens.append(next_generated_token)
                    # Update prompt tokens
                    prompt_tokens = torch.cat((prompt_tokens, next_generated_token.unsqueeze(0)), dim=1)

        generate_text = self.tokenizer.decode(prompt_tokens.squeeze(0).tolist())
        return generate_text

    def _get_logits_and_next_token_id(
        self,
        prompt_tokens: torch.Tensor,
        context_length: int,
        model: BaseSpsModel,
        temperature: Optional[float] = None,
        eps: float = 1e-8,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Crop current context if it exceeds the supported context size(ctx_len)
        # E.g., if LLM supports only 5 tokens, and the context size is 10
        # then only the last 5 tokens are used as context

        # (batch, n_tokens) --(crop context)--> (batch, n_tokens' = min(ctx_len, n_tokens))
        context_text_token_ids = prompt_tokens[:, -context_length:]
        logits = model.get_next_token_logits(prompt_token=context_text_token_ids)
        logits = logits[:, -1, :]

        if temperature is not None:
            logits = logits / (temperature + eps)
            probs = torch.softmax(logits, dim=-1)
            next_token_id = torch.multinomial(probs, num_samples=1)
        else:
            next_token_id = torch.argmax(logits, dim=-1, keepdim=True)
            probs = torch.softmax(logits, dim=-1)

        return next_token_id, probs

    def _get_latest_n_token_probs(
        self,
        prompt_tokens: torch.Tensor,
        n: int,
        context_length: int,
        model: BaseSpsModel,
        temperature: Optional[float] = None,
        eps: float = 1e-8,
    ) -> torch.Tensor:
        # Crop current context if it exceeds the supported context size(ctx_len)
        # E.g., if LLM supports only 5 tokens, and the context size is 10
        # then only the last 5 tokens are used as context

        # (batch, n_tokens) --(crop context)--> (batch, n_tokens' = min(ctx_len, n_tokens))
        context_text_token_ids = prompt_tokens[:, -context_length:]
        logits = model.get_next_token_logits(prompt_token=context_text_token_ids)
        logits = logits[:, -n:, :]

        if temperature is not None:
            logits = logits / (temperature + eps)
        probs = torch.softmax(logits, dim=-1)
        return probs


if __name__ == "__main__":
    import time

    from toyllm.gpt2.tokenizer import get_gpt2_tokenizer
    from toyllm.sps.models import DraftModelGPT2, TargetModelGPT2

    text_generator = TextGenerator(
        tokenizer=get_gpt2_tokenizer(),
        target_model=TargetModelGPT2(),
        draft_model=DraftModelGPT2(),
        lookahead=4,
    )

    start_time = time.time()
    prompt_text = "Alan Turing theorized that computers would one day become"
    generate_text = text_generator.generate(
        prompt_text=prompt_text,
        min_gen_tokens=40,
        temperature=0,
    )
    print(generate_text)
    end_time = time.time()
    print("Time elapsed: {:.2f}s".format(end_time - start_time))
