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


class TargetModelGPT2(BaseSpsModel):
    def __init__(self):
        self.gpt_model = GPTModel("1558M").load("../../models/gpt_1558m.pt")
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


class DraftModelGPT2(BaseSpsModel):
    def __init__(self):
        self.gpt_model = GPTModel("124M").load("../../models/gpt_124m.pt")
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
