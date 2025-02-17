from abc import ABC, abstractmethod

import torch
from toyllm.gpt2.gpt import GPTModel

class BaseSpsModel(ABC):
    @abstractmethod
    def get_next_token_logits(self, prompt_token: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
    
    @abstractmethod
    def get_context_length(self) -> int:
        raise NotImplementedError


class TargetModelGPT2(BaseSpsModel):
    def get_next_token_logits(self, prompt_token: torch.Tensor) -> torch.Tensor:
        gpt = GPTModel("355M").load("../../models/gpt_355m.pt")
        logits = gpt.forward(prompt_token)
        return logits
   
    
    def get_context_length(self) -> int:
        return 1024


class DraftModelGPT2(BaseSpsModel):
    def get_next_token_logits(self, prompt_token: torch.Tensor) -> torch.Tensor:
        gpt = GPTModel("124M").load("../../models/gpt_124m.pt")
        logits = gpt.forward(prompt_token)
        return logits

    def get_context_length(self) -> int:
        return 1024
