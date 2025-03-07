import dataclasses


@dataclasses.dataclass
class GenerationConfig:
    """The GenerationConfig class is used to configure the text generation process."""

    max_new_tokens: int = 20
    """The maximum number of new tokens to generate."""
    top_k: int | None = None
    """The top-k value to use for the top-k filtering method."""
    temperature: float = 1.0
    """The temperature value to use for the temperature scaling method."""

    def __post_init__(self) -> None:
        if self.max_new_tokens <= 0:
            msg = "max_new_tokens must be a positive integer."
            raise ValueError(msg)
        if self.top_k is not None and self.top_k <= 0:
            msg = "top_k must be a positive integer."
            raise ValueError(msg)
        if self.temperature <= 0:
            msg = "temperature must be a positive float."
            raise ValueError(msg)
