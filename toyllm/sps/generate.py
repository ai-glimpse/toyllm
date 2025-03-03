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
        text_id_list = self.tokenizer.encode(prompt)
        prompt_tokens = torch.tensor(text_id_list).to(self.target_model.device())

        n = prompt_tokens.size(0)
        while n < target_seq_len:
            with torch.inference_mode():
                draft_prompt_tokens = prompt_tokens.clone()
                for _ in range(self.lookahead):
                    draft_model_probs = self.draft_model.forward(draft_prompt_tokens, temperature)
                    next_token_id = torch.multinomial(draft_model_probs[-1], num_samples=1)
                    # Append sampled index to the running sequence
                    draft_prompt_tokens = torch.cat([draft_prompt_tokens, next_token_id], dim=0)

                target_model_probs = self.target_model.forward(draft_prompt_tokens, temperature)

                # 仅仅考虑最后 (lookahead + 1) 步的 token 预测概率
                draft_generate_tokens = draft_prompt_tokens[-self.lookahead :]
                draft_model_probs = draft_model_probs[-self.lookahead :, :]
                # target 模型另外考虑最后一个 token 的预测，用于 All accept 时推断出下一个 token
                target_model_probs = target_model_probs[-(self.lookahead + 1) :, :]

                all_accept = True
                # Compare the target model's next token with the draft model's lookahead
                for t in range(self.lookahead):
                    x = draft_generate_tokens[t]
                    px = draft_model_probs[t, x]
                    qx = target_model_probs[t, x]

                    r = self.rng.random()
                    if r < min(1.0, (qx / px).cpu().item()):
                        print("Accept!")
                        next_generated_token = x
                        # Update prompt tokens
                        prompt_tokens = torch.cat([prompt_tokens, next_generated_token.unsqueeze(0)], dim=0)
                    else:
                        all_accept = False
                        print("Reject!")
                        prob_diff = target_model_probs[t] - draft_model_probs[t]
                        # element-wise max to 0
                        prob_diff = torch.clamp(prob_diff, min=0)
                        prob_diff = prob_diff / torch.sum(prob_diff)
                        next_generated_token = torch.multinomial(prob_diff, num_samples=1)
                        # Update prompt tokens
                        prompt_tokens = torch.cat((prompt_tokens, next_generated_token), dim=0)
                        break
                    n += 1

                if all_accept:
                    print("All accept!")
                    next_generated_token = torch.multinomial(target_model_probs[-1], num_samples=1)
                    # Update prompt tokens
                    prompt_tokens = torch.cat([prompt_tokens, next_generated_token], dim=0)
                    n += 1
        generate_text = self.tokenizer.decode(prompt_tokens.tolist())
        return generate_text
