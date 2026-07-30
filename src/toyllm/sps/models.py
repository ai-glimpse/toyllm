from abc import ABC, abstractmethod

import torch

from toyllm.core import GenerationConfig, logits_temperature_scale, logits_top_k_filter
from toyllm.device import current_device
from toyllm.gpt2 import GPTModel, GPTModelSize


class BaseSpsModel(ABC):
    @abstractmethod
    def inference(
        self,
        prompt_token: torch.Tensor,
        config: GenerationConfig,
    ) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def device(self) -> torch.device:
        raise NotImplementedError

    @staticmethod
    def logits_to_probs(logits: torch.Tensor, config: GenerationConfig) -> torch.Tensor:
        if config.top_k is not None:
            logits = logits_top_k_filter(logits, top_k=config.top_k)
        if config.temperature is not None:
            logits = logits_temperature_scale(logits, temperature=config.temperature)
        return torch.softmax(logits, dim=-1)


class GPTSpsModel(BaseSpsModel):
    def __init__(self, model_size: GPTModelSize) -> None:
        self.gpt_model = GPTModel(model_size).load()
        self.gpt_model.eval()
        if self.gpt_model.device != current_device:
            self.gpt_model.to(current_device)

    def inference(
        self,
        prompt_token: torch.Tensor,
        config: GenerationConfig,
    ) -> torch.Tensor:
        logits = self.gpt_model.forward(prompt_token.unsqueeze(0)).squeeze(0)
        return self.logits_to_probs(logits, config)

    def device(self) -> torch.device:
        return self.gpt_model.device
