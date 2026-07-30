# ToyLLM: Learning LLM from Scratch

A hands-on educational project for understanding and implementing Large Language Models (LLMs) from scratch. This project provides implementations of GPT-2 and related techniques, making it an excellent resource for learning about transformer architectures and modern language models.

## Features

### GPT-2 Implementation
A clean, educational implementation of GPT-2 with type hints, supporting both training and inference.

### Speculative Sampling
An implementation of speculative sampling for faster inference, featuring configurable draft models and performance benchmarking.

### KV Cache Optimization
A memory-efficient GPT-2 implementation with KV cache optimization for handling longer sequences.

## Quick Start

### Prerequisites

- Python 3.14
- Git and Git LFS (for model files)
- UV (recommended package manager)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/ai-glimpse/toyllm.git
   cd toyllm
   ```

2. Set up the environment:
   ```bash
   # Create the environment and install the project
   uv sync
   ```

3. Download model files:
   ```bash
   # Install Git LFS if not already installed
   git lfs install

   # Download model files
   git clone https://huggingface.co/MathewShen/toyllm-gpt2 models
   ```

   Alternatively, you can manually download the model files from [Hugging Face](https://huggingface.co/MathewShen/toyllm-gpt2/tree/main) and place them in the `models` directory.

### Usage Examples

#### Basic GPT-2 Inference
```bash
uv run python -m toyllm.cli.run_gpt2 --help  # View available options
uv run python -m toyllm.cli.run_gpt2         # Run with default settings
```

#### KV Cache Optimized GPT-2
```bash
uv run python -m toyllm.cli.run_gpt2_kv --help  # View available options
uv run python -m toyllm.cli.run_gpt2_kv         # Run with default settings
```

#### Speculative Sampling
```bash
uv run python -m toyllm.cli.run_speculative_sampling --help  # View available options
uv run python -m toyllm.cli.run_speculative_sampling         # Run with default settings
```

#### Benchmarking
```bash
uv run python -m toyllm.cli.benchmark.bench_gpt2kv --help  # View available options
uv run python -m toyllm.cli.benchmark.bench_gpt2kv         # Run benchmarks
```

## Project Structure

```
src/toyllm/
├── cli/                    # Command-line interface modules
├── core/                   # Shared generation primitives
├── gpt2/                   # GPT-2 specific implementations
├── gpt2kv/                 # KV-cache optimized GPT-2
├── sps/                    # Speculative sampling implementations
└── util/                   # Utility functions
```

## Acknowledgements

This project is inspired by and builds upon the following excellent resources:

- [rasbt/LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch)
- [neelnanda-io/TransformerLens](https://github.com/neelnanda-io/TransformerLens)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
