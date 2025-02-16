from abc import ABC, abstractmethod

import torch


class BaseModel(ABC):
    @abstractmethod
    def get_next_token_logits(self, prompt: str) -> torch.Tensor:
        raise NotImplementedError


class TargetModel(BaseModel):
    # TODO: Implement the get_next_token_logits method
    def get_next_token_logits(self, prompt: str) -> torch.Tensor:
        return torch.rand(1, 50256)


class DraftModel(BaseModel):
    # TODO: Implement the get_next_token_logits method
    def get_next_token_logits(self, prompt: str) -> torch.Tensor:
        return torch.rand(1, 50256)
