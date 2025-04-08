import gc
import time
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
import typer
from rich import box
from rich.console import Console
from rich.table import Table

from toyllm.core import GenerationConfig
from toyllm.gpt2 import GPTModel, GPTModelSize
from toyllm.gpt2 import GPTTextGenerator as NaiveGPTTextGenerator
from toyllm.gpt2_kv import GPTKVModel
from toyllm.gpt2_kv import GPTKVTextGenerator

# User and timestamp information
CURRENT_TIME = "2025-04-08 15:44:45"
CURRENT_USER = "shenxiangzhuang"

console = Console()
console.print("[bold]GPT-2 KV Cache Benchmark[/bold]")
console.print(f"Date: {CURRENT_TIME} UTC")
console.print(f"User: {CURRENT_USER}")


def cleanup_gpu_memory():
    """Clean up GPU memory to avoid OOM errors."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        console.print("[green]GPU memory cleaned up[/green]")


def benchmark_generation(
    model_size: GPTModelSize,
    max_new_tokens: int,
    prompt: str = "Alan Turing theorized that computers would one day become",
) -> tuple[float, float]:
    """Benchmark both implementations and return their execution times."""
    try:
        # Benchmark naive implementation
        console.print(
            f"[yellow]Running naive implementation for {model_size.name} with {max_new_tokens} tokens...[/yellow]"
        )
        naive_model = GPTModel(model_size).load()
        naive_generator = NaiveGPTTextGenerator(gpt_model=naive_model)

        start_time = time.time()
        naive_generator.generate(
            prompt=prompt,
            config=GenerationConfig(
                max_new_tokens=max_new_tokens,
                top_k=None,
                temperature=None,
            ),
        )
        naive_time = time.time() - start_time
        console.print(f"[green]Naive implementation completed in {naive_time:.2f} seconds[/green]")

        # Cleanup after naive implementation
        del naive_generator
        del naive_model
        cleanup_gpu_memory()

        # Benchmark KV implementation
        console.print(
            f"[yellow]Running KV implementation for {model_size.name} with {max_new_tokens} tokens...[/yellow]"
        )
        kv_model = GPTKVModel(model_size).load()
        kv_generator = GPTKVTextGenerator(gpt_model=kv_model)

        start_time = time.time()
        kv_generator.generate(
            prompt=prompt,
            config=GenerationConfig(
                max_new_tokens=max_new_tokens,
                top_k=None,
                temperature=None,
            ),
        )
        kv_time = time.time() - start_time
        console.print(f"[green]KV implementation completed in {kv_time:.2f} seconds[/green]")

        # Cleanup after KV implementation
        del kv_generator
        del kv_model
        cleanup_gpu_memory()

        return naive_time, kv_time
    except Exception as e:
        console.print(f"[red]Error during benchmark: {e}[/red]")
        cleanup_gpu_memory()  # Ensure cleanup even on error
        raise


def plot_results(csv_path: Path, output_dir: Path) -> None:
    """Create consolidated plots with all model sizes by reading from CSV file."""
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Read the benchmark results from CSV
    console.print(f"Reading benchmark results from {csv_path}")
    df = pd.read_csv(csv_path)

    # Get unique model sizes and tokens
    model_sizes = df["model_size"].unique()
    tokens = sorted(df["tokens"].unique())

    console.print(f"Creating plots for model sizes: {model_sizes}")
    console.print(f"Token sizes: {tokens}")

    # Set up colors and markers for different model sizes
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    markers = ["o", "s", "^", "D"]

    # 1. Create the time comparison plot with all model sizes
    plt.figure(figsize=(12, 8))

    for i, model_size in enumerate(model_sizes):
        model_data = df[df["model_size"] == model_size]

        # Sort by tokens
        model_data = model_data.sort_values(by="tokens")

        # Plot naive implementation
        plt.plot(
            model_data["tokens"],
            model_data["naive_time_seconds"],
            label=f"{model_size} - Naive",
            color=colors[i % len(colors)],
            marker=markers[i % len(markers)],
            linestyle="-",
            linewidth=2,
        )

        # Plot KV implementation
        plt.plot(
            model_data["tokens"],
            model_data["kv_time_seconds"],
            label=f"{model_size} - KV",
            color=colors[i % len(colors)],
            marker=markers[i % len(markers)],
            linestyle="--",
            linewidth=2,
        )

    plt.xlabel("Number of Tokens", fontsize=12)
    plt.ylabel("Time (seconds)", fontsize=12)
    plt.title("Generation Time Comparison - All Model Sizes", fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True)
    plt.tight_layout()

    time_plot_path = output_dir / "benchmark_time_comparison.png"
    plt.savefig(time_plot_path, dpi=300)
    plt.close()
    console.print(f"[green]Time comparison plot saved to {time_plot_path}[/green]")

    # 2. Create the speedup plot with all model sizes
    plt.figure(figsize=(12, 8))

    for i, model_size in enumerate(model_sizes):
        model_data = df[df["model_size"] == model_size]

        # Sort by tokens
        model_data = model_data.sort_values(by="tokens")

        plt.plot(
            model_data["tokens"],
            model_data["speedup"],
            label=f"{model_size}",
            color=colors[i % len(colors)],
            marker=markers[i % len(markers)],
            linewidth=2,
        )

    plt.xlabel("Number of Tokens", fontsize=12)
    plt.ylabel("Speedup (x)", fontsize=12)
    plt.title("Speedup of KV Implementation - All Model Sizes", fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True)
    plt.tight_layout()

    speedup_plot_path = output_dir / "benchmark_speedup.png"
    plt.savefig(speedup_plot_path, dpi=300)
    plt.close()
    console.print(f"[green]Speedup plot saved to {speedup_plot_path}[/green]")


def main(
    model_sizes: list[GPTModelSize] = [
        GPTModelSize.XLARGE,
        GPTModelSize.LARGE,
        GPTModelSize.MEDIUM,
        GPTModelSize.SMALL,
    ],  # Largest to smallest
    max_new_tokens_list: list[int] = list(range(1000, 99, -100)),  # Largest to smallest: 1000 to 100 in steps of 100
    output_dir: str = "benchmark/gpt2kv",
) -> None:
    """Run benchmarks comparing naive and KV-cache implementations."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Format for CSV column names: lowercase with underscores
    column_names = {
        "model_size": "model_size",  # Model size (SMALL, MEDIUM, LARGE, XLARGE)
        "tokens": "tokens",  # Number of tokens generated
        "naive_time_seconds": "naive_time_seconds",  # Time for naive implementation in seconds
        "kv_time_seconds": "kv_time_seconds",  # Time for KV implementation in seconds
        "speedup": "speedup",  # Speedup ratio (naive/kv)
        "timestamp": "timestamp",  # When the test was run
    }

    # Initialize results list
    results = []

    # Clean up GPU memory before starting benchmarks
    cleanup_gpu_memory()

    # Save benchmark configuration to a file
    with open(output_path / "benchmark_config.txt", "w") as f:
        f.write("# GPT-2 KV Cache Benchmark\n")
        f.write(f"# Date: {CURRENT_TIME} UTC\n")
        f.write(f"# User: {CURRENT_USER}\n\n")
        f.write(f"Model sizes: {[size.name for size in model_sizes]}\n")
        f.write(f"Token lengths: {max_new_tokens_list}\n")
        f.write(f"Benchmark started: {CURRENT_TIME}\n")

    # Summary table to be updated after each pair
    summary_table = Table(title="Benchmark Results Summary", box=box.ROUNDED)
    summary_table.add_column("Model Size", style="cyan")
    summary_table.add_column("Tokens", style="magenta")
    summary_table.add_column("Naive (s)", style="red")
    summary_table.add_column("KV (s)", style="green")
    summary_table.add_column("Speedup", style="yellow")
    summary_table.add_column("Timestamp", style="blue")

    for model_size in model_sizes:
        for max_new_tokens in max_new_tokens_list:
            console.print(
                f"\n[bold cyan]===== Benchmarking {model_size.name} with {max_new_tokens} tokens =====[/bold cyan]"
            )

            # Run the benchmark
            naive_time, kv_time = benchmark_generation(model_size, max_new_tokens)
            speedup = naive_time / kv_time
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # Print the pair results
            result_table = Table(title=f"Result: {model_size.name} with {max_new_tokens} tokens", box=box.SIMPLE)
            result_table.add_column("Implementation", style="cyan")
            result_table.add_column("Time (s)", style="green")
            result_table.add_row("Naive", f"{naive_time:.2f}")
            result_table.add_row("KV Cache", f"{kv_time:.2f}")
            result_table.add_row("Speedup", f"{speedup:.2f}x")
            console.print(result_table)

            # Add to results list with consistent column names
            results.append(
                {
                    column_names["model_size"]: model_size.name,
                    column_names["tokens"]: max_new_tokens,
                    column_names["naive_time_seconds"]: naive_time,
                    column_names["kv_time_seconds"]: kv_time,
                    column_names["speedup"]: speedup,
                    column_names["timestamp"]: timestamp,
                }
            )

            # Add to summary table
            summary_table.add_row(
                model_size.name,
                str(max_new_tokens),
                f"{naive_time:.2f}",
                f"{kv_time:.2f}",
                f"{speedup:.2f}x",
                timestamp,
            )

            # Print updated summary table
            console.print("\n[bold]Summary of all benchmarks so far:[/bold]")
            console.print(summary_table)

            # Save intermediate results after each benchmark
            interim_df = pd.DataFrame(results)
            interim_df.to_csv(output_path / "benchmark_results_interim.csv", index=False)
            console.print(f"[blue]Intermediate results saved to {output_path / 'benchmark_results_interim.csv'}[/blue]")

    # Convert results to DataFrame and save to CSV with proper column names
    df = pd.DataFrame(results)
    csv_path = output_path / "benchmark_results.csv"
    df.to_csv(csv_path, index=False)

    # Also save as TSV for easier viewing in spreadsheet software
    tsv_path = output_path / "benchmark_results.tsv"
    df.to_csv(tsv_path, sep="\t", index=False)

    console.print("\n[bold green]Benchmark complete![/bold green]")
    console.print("Results saved to:")
    console.print(f"- CSV: {csv_path}")
    console.print(f"- TSV: {tsv_path}")

    # Create and save plots
    plot_results(csv_path, output_path)
    console.print(f"Plots saved to {output_path}")


# Add a standalone plotting function that can be called directly
def create_plots_from_csv(csv_path: str, output_dir: str = None):
    """Standalone function to create plots from an existing CSV file."""
    csv_path = Path(csv_path)

    # If output directory not specified, use same directory as CSV
    if output_dir is None:
        output_dir = csv_path.parent
    else:
        output_dir = Path(output_dir)

    plot_results(csv_path, output_dir)
    return 0


if __name__ == "__main__":
    # Define a command group using typer
    app = typer.Typer()

    @app.command()
    def benchmark(
        model_sizes: list[GPTModelSize] = [
            GPTModelSize.XLARGE,
            GPTModelSize.LARGE,
            GPTModelSize.MEDIUM,
            GPTModelSize.SMALL,
        ],
        max_new_tokens_list: list[int] = list(range(1000, 99, -100)),
        output_dir: str = "benchmark/gpt2kv",
    ):
        """Run the full benchmark suite."""
        main(model_sizes, max_new_tokens_list, output_dir)

    @app.command()
    def plot(
        csv_file: str = typer.Argument(..., help="Path to the benchmark results CSV file"),
        output_dir: str = typer.Option(None, help="Output directory for plots (default: same as CSV)"),
    ):
        """Generate plots from an existing benchmark results CSV file."""
        return create_plots_from_csv(csv_file, output_dir)

    app()
