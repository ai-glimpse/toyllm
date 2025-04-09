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

- Python 3.11 or 3.12
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
   # Create and activate virtual environment
   uv venv -p 3.12
   source .venv/bin/activate

   # Install toyllm
   uv pip install toyllm
   ```

3. Download model files:
   ```bash
   # Install Git LFS if not already installed
   git lfs install

   # Download model files
   git clone https://huggingface.co/MathewShen/toyllm-gpt2 models
   ```

   Alternatively, you can manually download the model files from [Hugging Face](https://huggingface.co/MathewShen/toyllm-gpt2/tree/main) and place them in the `toyllm/models` directory.

### Usage Examples

#### Basic GPT-2 Inference
```bash
python toyllm/cli/run_gpt2.py --help  # View available options
python toyllm/cli/run_gpt2.py         # Run with default settings
```

#### KV Cache Optimized GPT-2
```bash
python toyllm/cli/run_gpt2_kv.py --help  # View available options
python toyllm/cli/run_gpt2_kv.py         # Run with default settings
```

#### Speculative Sampling
```bash
python toyllm/cli/run_speculative_sampling.py --help  # View available options
python toyllm/cli/run_speculative_sampling.py         # Run with default settings
```

#### Benchmarking
```bash
python toyllm/cli/benchmark/bench_gpt2kv.py --help  # View available options
python toyllm/cli/benchmark/bench_gpt2kv.py         # Run benchmarks
```

## Project Structure

```
toyllm/
├── cli/                    # Command-line interface scripts
├── gpt2/                   # GPT-2 specific implementations
├── gpt2_kv/                # KV-cache optimized GPT-2
├── sps/                    # Speculative sampling implementations
├── util/                   # Utility functions
└── models/                 # Model weights and configurations
```

## Acknowledgements

This project is inspired by and builds upon the following excellent resources:

- [rasbt/LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch)
- [neelnanda-io/TransformerLens](https://github.com/neelnanda-io/TransformerLens)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

[Add your license information here]
