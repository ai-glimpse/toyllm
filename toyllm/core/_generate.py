import dataclasses


@dataclasses.dataclass
class GenerationConfig:
    """The GenerationConfig class is used to configure the text generation process."""

    max_new_tokens: int = 20
    """The maximum number of new tokens to generate."""
    top_k: int | None = None
    """The top-k value to use for the top-k filtering method."""
    temperature: float | None = None
    """The temperature value to use for the temperature scaling method."""
