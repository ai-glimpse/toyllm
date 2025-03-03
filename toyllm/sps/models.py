from abc import ABC, abstractmethod

import torch

from toyllm.device import current_device
from toyllm.gpt2.gpt import GPTModel, GPTModelSize


class BaseSpsModel(ABC):
    @abstractmethod
    def forward(
        self,
        prompt_token: torch.Tensor,
        temperature: None | float = None,
        eps: float = 1e-8,
    ) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def get_context_length(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def device(self) -> torch.device:
        raise NotImplementedError


class GPTSpsModel(BaseSpsModel):
    def __init__(self, model_size: GPTModelSize):
        self.gpt_model = GPTModel(model_size).load()
        self.gpt_model.eval()
        if self.gpt_model.device != current_device:
            self.gpt_model.to(current_device)

    def forward(
        self,
        prompt_token: torch.Tensor,
        temperature: None | float = None,
        eps: float = 1e-8,
    ) -> torch.Tensor:
        logits = self.gpt_model.forward(prompt_token.unsqueeze(0)).squeeze(0)

        if temperature is not None:
            logits = logits / (temperature + eps)
        probs = torch.softmax(logits, dim=-1)
        return probs

    def get_context_length(self) -> int:
        return 1024

    def device(self) -> torch.device:
        return self.gpt_model.device
