from dataclasses import dataclass
from enum import StrEnum  # type: ignore[attr-defined]


class GPTModelSize(StrEnum):
    """
    Enum class for GPT model size
    """

    SMALL = "124M"
    MEDIUM = "355M"
    LARGE = "774M"
    XLARGE = "1558M"


@dataclass
class GPTModelConfig:
    """
    GPT Model Architecture Config
    """

    name: str
    """The name of the model"""
    vocab_size: int
    """The size of the vocabulary"""
    ctx_len: int
    """The length of the context/block"""
    emb_dim: int
    """The embedding size"""
    n_heads: int
    """The number of attention heads"""
    n_layers: int
    """The number of transformer layers"""
    drop_rate: float
    """The dropout rate"""
    qkv_bias: bool
    """The query key value bias terms"""


GPT_124M_MODEL_CONFIG = GPTModelConfig(
    name="gpt_124m",
    vocab_size=50257,
    ctx_len=1024,
    emb_dim=768,
    n_heads=12,
    n_layers=12,
    drop_rate=0.0,
    qkv_bias=True,
)

GPT_355M_MODEL_CONFIG = GPTModelConfig(
    name="gpt_355m",
    vocab_size=50257,
    ctx_len=1024,
    emb_dim=1024,
    n_heads=16,
    n_layers=24,
    drop_rate=0.0,
    qkv_bias=True,
)

GPT_774M_MODEL_CONFIG = GPTModelConfig(
    name="gpt_774m",
    vocab_size=50257,
    ctx_len=1024,
    emb_dim=1280,
    n_heads=20,
    n_layers=36,
    drop_rate=0.0,
    qkv_bias=True,
)


GPT_1558M_MODEL_CONFIG = GPTModelConfig(
    name="gpt_1558m",
    vocab_size=50257,
    ctx_len=1024,
    emb_dim=1600,
    n_heads=25,
    n_layers=48,
    drop_rate=0.0,
    qkv_bias=True,
)


def get_model_config(model_size: str | GPTModelSize) -> GPTModelConfig:
    if model_size == GPTModelSize.SMALL:
        return GPT_124M_MODEL_CONFIG
    elif model_size == GPTModelSize.MEDIUM:
        return GPT_355M_MODEL_CONFIG
    elif model_size == GPTModelSize.LARGE:
        return GPT_774M_MODEL_CONFIG
    elif model_size == GPTModelSize.XLARGE:
        return GPT_1558M_MODEL_CONFIG
    else:
        raise ValueError(f"Invalid model size: {model_size}")


@dataclass
class GPTTrainingConfig:
    """
    GPT training config: hyperparameters for GPT model training
    """

    learning_rate: float = 5e-4
    num_epochs: int = 10
    batch_size: int = 2
    weight_decay: float = 0.1
