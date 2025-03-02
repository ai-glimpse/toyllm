from dataclasses import dataclass

import numpy as np
import tiktoken
import torch

from toyllm.sps.models import BaseSpsModel


@dataclass
class SpsTextGenerator:
    tokenizer: tiktoken.Encoding
    target_model: BaseSpsModel
    draft_model: BaseSpsModel
    lookahead: int = 5
    seed: int = 42

    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed)

    def generate(
        self,
        prompt: str,
        min_gen_tokens: int = 100,
        temperature: None | float = None,
    ) -> str:
        # prompt text to tokens: (1, n_tokens)
        text_id_list = self.tokenizer.encode(prompt)
        prompt_tokens = torch.tensor(text_id_list).unsqueeze(0).to(self.target_model.device())  # add batch dimension
        generated_tokens: list[torch.Tensor] = []

        while len(generated_tokens) <= min_gen_tokens:
            with torch.inference_mode():
                # Draft model lookahead: (prompt_tokens, next_token_id, next_token_logits)
                lookahead_tuples = []
                draft_prompt_tokens = prompt_tokens.clone()
                for _ in range(self.lookahead):
                    draft_next_token_id, draft_next_token_probs = self._get_draft_next_token_id_and_probs(
                        draft_prompt_tokens,
                        self.draft_model,
                        temperature,
                    )
                    lookahead_tuples.append((draft_next_token_id, draft_next_token_probs))
                    # Append sampled index to the running sequence
                    # (batch, n_tokens') --(append next token)--> (batch, n_tokens' + 1)
                    draft_prompt_tokens = torch.cat((draft_prompt_tokens, draft_next_token_id.unsqueeze(0)), dim=1)

                # Target model logits
                target_model_probs = self._get_target_latest_n_token_probs(
                    draft_prompt_tokens, self.lookahead + 1, self.target_model, temperature
                ).squeeze(0)

                all_accept = True
                # Compare the target model's next token with the draft model's lookahead
                for t in range(self.lookahead):
                    draft_next_token_id, draft_next_token_probs = lookahead_tuples[t][0], lookahead_tuples[t][1]
                    target_next_token_probs = target_model_probs[t, :]

                    r = self.rng.random()
                    if r < min(
                        1.0,
                        (target_next_token_probs[draft_next_token_id] / draft_next_token_probs[draft_next_token_id])
                        .cpu()
                        .item(),
                    ):
                        # print("Accept!")
                        next_generated_token = draft_next_token_id
                        generated_tokens.append(next_generated_token)
                        # Update prompt tokens
                        prompt_tokens = torch.cat((prompt_tokens, next_generated_token.unsqueeze(0)), dim=1)
                    else:
                        all_accept = False
                        # print("Reject!")
                        prob_diff = target_next_token_probs - draft_next_token_probs
                        # element-wise max to 0
                        prob_diff = torch.clamp(prob_diff, min=0)
                        prob_diff = prob_diff / torch.sum(prob_diff)
                        next_generated_token = torch.multinomial(prob_diff, num_samples=1)
                        generated_tokens.append(next_generated_token)
                        # Update prompt tokens
                        prompt_tokens = torch.cat((prompt_tokens, next_generated_token.unsqueeze(0)), dim=1)
                        break
                if all_accept:
                    # print("All accept!")
                    next_generated_token = torch.multinomial(target_model_probs[-1], num_samples=1)
                    generated_tokens.append(next_generated_token)
                    # Update prompt tokens
                    prompt_tokens = torch.cat((prompt_tokens, next_generated_token.unsqueeze(0)), dim=1)

        generate_text = self.tokenizer.decode(prompt_tokens.squeeze(0).tolist())
        return generate_text

    def _get_draft_next_token_id_and_probs(
        self,
        prompt_tokens: torch.Tensor,
        model: BaseSpsModel,
        temperature: None | float = None,
        eps: float = 1e-8,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Crop current context if it exceeds the supported context size(ctx_len)
        # (batch, n_tokens) --(crop context)--> (batch, n_tokens' = min(ctx_len, n_tokens))
        context_length = self.target_model.get_context_length()
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
        # The batch dimension is 1(in this example), so we select the first element
        assert next_token_id.shape[0] == 1 and probs.shape[0] == 1
        return next_token_id[0], probs[0]

    def _get_target_latest_n_token_probs(
        self,
        prompt_tokens: torch.Tensor,
        n: int,
        model: BaseSpsModel,
        temperature: None | float = None,
        eps: float = 1e-8,
    ) -> torch.Tensor:
        context_length = self.target_model.get_context_length()

        # (batch, n_tokens) --(crop context)--> (batch, n_tokens' = min(ctx_len, n_tokens))
        context_text_token_ids = prompt_tokens[:, -context_length:]
        logits = model.get_next_token_logits(prompt_token=context_text_token_ids)
        logits = logits[:, -n:, :]

        if temperature is not None:
            logits = logits / (temperature + eps)
        probs = torch.softmax(logits, dim=-1)
        return probs
