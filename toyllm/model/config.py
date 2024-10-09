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

    :param vocab_size: vocabulary size
    :param ctx_len: length of context/block
    :param emb_dim: embedding size
    :param n_heads: number of attention heads
    :param n_layers: number of transformer layers
    :param drop_rate: dropout rate
    :param qkv_bias: query key value bias terms
    """

    name: str
    vocab_size: int
    ctx_len: int
    emb_dim: int
    n_heads: int
    n_layers: int
    drop_rate: float
    qkv_bias: bool


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


@dataclass
class GPTTrainingConfig:
    """
    GPT training config: hyperparameters for GPT model training
    """

    learning_rate: float = 5e-4
    num_epochs: int = 10
    batch_size: int = 2
    weight_decay: float = 0.1
