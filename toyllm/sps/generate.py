from dataclasses import dataclass

import numpy as np
import tiktoken
import torch

from toyllm.sps.models import BaseSpsModel


@dataclass
class SpsTextGenerator:
    tokenizer: tiktoken.Encoding
    target_model: BaseSpsModel  # auto-regressive target model
    draft_model: BaseSpsModel  # auto-regressive draft model
    lookahead: int = 5  # K in sps paper
    seed: int = 42

    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed)

    def generate(
        self,
        prompt: str,  # given initial prompt
        target_seq_len: int = 100,
        temperature: None | float = None,
    ) -> str:
        """
        Args:
            prompt: Prompt text
            target_seq_len: The `T` in paper.
            The target sequence length to generate.
            temperature: Used to control the randomness of the generated text.
        """

        # sequence x0, x1, ..., xt
        text_id_list = self.tokenizer.encode(prompt)
        prompt_tokens = torch.tensor(text_id_list).to(self.target_model.device())

        # Initialise n <- t
        n = prompt_tokens.size(0)
        # while n < T do
        while n < target_seq_len:
            with torch.inference_mode():
                draft_prompt_tokens = prompt_tokens.clone()
                # for t = 1:K do
                #   Sample draft auto-regressively x'_t ~ p(x|x1, ..., xn, x'_1, ..., x'_{t-1})
                for _ in range(self.lookahead):
                    draft_model_probs = self.draft_model.forward(draft_prompt_tokens, temperature)
                    next_token_id = torch.multinomial(draft_model_probs[-1], num_samples=1)
                    draft_prompt_tokens = torch.cat([draft_prompt_tokens, next_token_id], dim=0)

                # 仅考虑 draft model 生成的 token
                draft_generate_tokens = draft_prompt_tokens[-self.lookahead :]
                draft_model_probs = draft_model_probs[-self.lookahead :, :]

                # compute k+1 sets of logits from drafts x'_1, ..., x'_K
                # target 模型另外考虑最后一个 token 的预测，用于 All accept 时推断出下一个 token
                target_model_probs = self.target_model.forward(draft_prompt_tokens, temperature)
                target_model_probs = target_model_probs[-(self.lookahead + 1) :, :]

                all_accept = True
                # for t = 1:K do
                for t in range(self.lookahead):
                    # Sample r ~ U(0, 1) from a uniform distribution
                    r = self.rng.random()

                    x = draft_generate_tokens[t]
                    px = draft_model_probs[t, x]
                    qx = target_model_probs[t, x]

                    # if r < min(1, q(x) / p(x)), then (accept x)
                    if r < min(1.0, (qx / px).cpu().item()):
                        next_generated_token = x
                        prompt_tokens = torch.cat([prompt_tokens, next_generated_token.unsqueeze(0)], dim=0)
                        n += 1
                    else:
                        all_accept = False
                        prob_diff = target_model_probs[t] - draft_model_probs[t]
                        # element-wise max to 0
                        prob_diff = torch.clamp(prob_diff, min=0)
                        prob_diff = prob_diff / torch.sum(prob_diff)
                        next_generated_token = torch.multinomial(prob_diff, num_samples=1)
                        prompt_tokens = torch.cat((prompt_tokens, next_generated_token), dim=0)
                        n += 1
                        break

                if all_accept:
                    next_generated_token = torch.multinomial(target_model_probs[-1], num_samples=1)
                    prompt_tokens = torch.cat([prompt_tokens, next_generated_token], dim=0)
                    n += 1
        generate_text = self.tokenizer.decode(prompt_tokens.tolist())
        return generate_text
