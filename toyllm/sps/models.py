from abc import ABC, abstractmethod

import torch

from toyllm.device import current_device
from toyllm.gpt2.gpt import GPTModel


class BaseSpsModel(ABC):
    @abstractmethod
    def get_next_token_logits(self, prompt_token: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def get_context_length(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def device(self) -> torch.device:
        raise NotImplementedError


class GPTSpsModel(BaseSpsModel):
    def __init__(self, model_name: str):
        self.gpt_model = GPTModel(model_name).load(f"../../models/gpt_{model_name.lower()}.pt")
        self.gpt_model.eval()
        if self.gpt_model.device != current_device:
            self.gpt_model.to(current_device)

    def get_next_token_logits(self, prompt_token: torch.Tensor) -> torch.Tensor:
        logits = self.gpt_model.forward(prompt_token)
        return logits

    def get_context_length(self) -> int:
        return 1024

    def device(self) -> torch.device:
        return self.gpt_model.device
