import jaxtyping
import tiktoken
import torch
from typeguard import typechecked as typechecker


def get_gpt2_tokenizer() -> tiktoken.Encoding:
    tokenizer = tiktoken.get_encoding("gpt2")
    return tokenizer


GPT2Tokenizer = get_gpt2_tokenizer()


@jaxtyping.jaxtyped(typechecker=typechecker)
def text_to_token_ids(
    text: str,
    tokenizer: tiktoken.Encoding = GPT2Tokenizer,
) -> jaxtyping.Int[torch.Tensor, "1 num_tokens"]:
    """
    >>> text_to_token_ids("Hello World", get_gpt2_tokenizer())
    tensor([[15496,  2159]])
    """
    text_id_list = tokenizer.encode(text)
    text_id_tensor = torch.tensor(text_id_list).unsqueeze(0)  # add batch dimension
    return text_id_tensor


@jaxtyping.jaxtyped(typechecker=typechecker)
def token_ids_to_text(
    text_id_tensor: jaxtyping.Int[torch.Tensor, "1 num_tokens"],
    tokenizer: tiktoken.Encoding = GPT2Tokenizer,
) -> str:
    """
    >>> token_ids_to_text(torch.tensor([[15496,  2159]]), get_gpt2_tokenizer())
    'Hello World'
    """
    text_id_list = text_id_tensor.squeeze(0)  # remove batch dimension
    text = tokenizer.decode(text_id_list.tolist())
    return text


if __name__ == "__main__":
    import doctest

    doctest.testmod(verbose=True)
