import jaxtyping
import torch
from typeguard import typechecked as typechecker


@jaxtyping.jaxtyped(typechecker=typechecker)
def logits_top_k_filter(
    logits: jaxtyping.Float[torch.Tensor, "batch_size vocab_size"],
    top_k: int | None = None,
) -> jaxtyping.Float[torch.Tensor, "batch_size vocab_size"]:
    """The top-k filtering method is used to prevent the model from considering tokens with low probabilities.

    ref1: https://github.com/rasbt/LLMs-from-scratch/blob/62fb11d5e0449a6d49bda7337d6cfa5a735718da/ch05/01_main-chapter-code/generate.py#L166-L185
    ref2: https://github.com/huggingface/transformers/blob/c4d4e8bdbd25d9463d41de6398940329c89b7fb6/src/transformers/generation_utils.py#L903-L941
    ref3: https://github.com/meta-llama/llama/blob/main/llama/generation.py#L188-L192.
    """
    if top_k is None:
        return logits
    top_k = min(top_k, logits.size(-1))  # make sure top_k <= vocab size
    top_k_logits, _ = torch.topk(logits, k=top_k, dim=-1)
    min_logit = top_k_logits[:, -1]
    logits = torch.where(logits < min_logit, torch.tensor(float("-inf")).to(logits.device), logits)
    return logits


@jaxtyping.jaxtyped(typechecker=typechecker)
def logits_temperature_scale(
    logits: jaxtyping.Float[torch.Tensor, "batch_size vocab_size"],
    temperature: float | None = None,
    eps: float = 1e-10,
) -> jaxtyping.Float[torch.Tensor, "batch_size vocab_size"]:
    """The temperature scaling method is used to control the randomness of the generated text."""
    if temperature is None:
        return logits
    logits = logits / torch.clip(torch.tensor(temperature, dtype=logits.dtype), min=eps, max=1.0)
    return logits
