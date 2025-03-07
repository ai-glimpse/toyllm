# Copyright 2024 Xiangzhuang Shen
# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch


import matplotlib.pyplot as plt
import tiktoken
import torch
from torch.utils.data import DataLoader

from toyllm.core import GenerationConfig
from toyllm.device import current_device
from toyllm.gpt2.config import GPTModelSize, GPTTrainingConfig
from toyllm.gpt2.dataset import GPTDataloader
from toyllm.gpt2.generate import GPTTextGenerator
from toyllm.gpt2.gpt import GPTModel
from toyllm.gpt2.tokenizer import get_gpt2_tokenizer


def get_data_loaders(
    text: str,
    gpt_data_loader: GPTDataloader,
    train_ratio: float = 0.9,
) -> tuple[DataLoader, DataLoader]:  # type: ignore[type-arg]
    # set train/validation split index by train_ratio
    split_idx = int(train_ratio * len(text))
    train_loader = gpt_data_loader.create_dataloader(text=text[:split_idx], drop_last=True, shuffle=True)
    validation_loader = gpt_data_loader.create_dataloader(text=text[split_idx:], drop_last=False, shuffle=False)
    return train_loader, validation_loader


def calc_loss_batch(input_batch: torch.Tensor, target_batch: torch.Tensor, model: GPTModel) -> torch.Tensor:
    input_batch, target_batch = input_batch.to(current_device), target_batch.to(current_device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
    return loss


def calc_loss_loader(
    data_loader: DataLoader,  # type: ignore[type-arg]
    model: GPTModel,
    num_batches: int | None = None,
) -> float:
    total_loss = 0.0
    num_batches = len(data_loader) if num_batches is None else min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches


def evaluate_model(
    model: GPTModel,
    train_loader: DataLoader,  # type: ignore[type-arg]
    val_loader: DataLoader,  # type: ignore[type-arg]
    eval_iter: int,
) -> tuple[float, float]:
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss


def generate_and_print_sample(model: GPTModel, tokenizer: tiktoken.Encoding, start_context: str) -> None:
    model.eval()
    text_generate = GPTTextGenerator(gpt_model=model, tokenizer=tokenizer)
    generate_text = text_generate.generate(
        prompt=start_context,
        config=GenerationConfig(
            max_new_tokens=50,
            temperature=0.9,
            top_k=10,
        ),
    )
    print(generate_text.replace("\n", " "))  # Compact print format
    model.train()


def train_model_simple(
    model: GPTModel,
    train_loader: DataLoader,  # type: ignore[type-arg]
    val_loader: DataLoader,  # type: ignore[type-arg]
    optimizer: torch.optim.Optimizer,  # type: ignore[name-defined]
    num_epochs: int,
    eval_freq: int,
    eval_iter: int,
    start_context: str,
) -> tuple[list[float], list[float], list[int]]:
    # Initialize lists to track losses and tokens seen
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen = 0
    global_step = -1

    # Main training loop
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode

        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()  # Reset loss gradients from previous epoch
            loss = calc_loss_batch(input_batch, target_batch, model)
            # Calculate loss gradients
            loss.backward()  # type: ignore[no-untyped-call]
            optimizer.step()  # Update model weights using loss gradients
            tokens_seen += input_batch.numel()
            global_step += 1

            # Optional evaluation step
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(model, train_loader, val_loader, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Ep {epoch + 1} (Step {global_step:06d}): Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

        # Print a sample text after each epoch
        generate_and_print_sample(model, train_loader.dataset.tokenizer, start_context)  # type: ignore[attr-defined]

    return train_losses, val_losses, track_tokens_seen


def plot_losses(
    epochs_seen: torch.Tensor,
    tokens_seen: list[int],
    train_losses: list[float],
    val_losses: list[float],
) -> None:
    fig, ax1 = plt.subplots()

    # Plot training and validation loss against epochs
    ax1.plot(epochs_seen, train_losses, label="Training loss")
    ax1.plot(epochs_seen, val_losses, linestyle="-.", label="Validation loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")

    # Create a second x-axis for tokens seen
    ax2 = ax1.twiny()  # Create a second x-axis that shares the same y-axis
    ax2.plot(tokens_seen, train_losses, alpha=0)  # Invisible plot for aligning ticks
    ax2.set_xlabel("Tokens seen")

    fig.tight_layout()  # Adjust layout to make room
    plt.show()


def main(
    text: str,
    gpt_size: GPTModelSize,
    training_config: GPTTrainingConfig,
) -> tuple[list[float], list[float], list[int], GPTModel]:
    torch.manual_seed(123)
    # Initialize model
    model = GPTModel(gpt_size)
    model.to(current_device)
    optimizer = torch.optim.AdamW(  # type: ignore[attr-defined]
        model.parameters(),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )
    tokenizer = get_gpt2_tokenizer()

    gpt_data_loader = GPTDataloader(
        tokenizer=tokenizer,
        max_length=model.config.ctx_len,
        stride=model.config.ctx_len,
        batch_size=training_config.batch_size,
    )
    train_loader, val_loader = get_data_loaders(text, gpt_data_loader=gpt_data_loader, train_ratio=0.9)

    # Train model
    train_losses, val_losses, tokens_seen = train_model_simple(
        model,
        train_loader,
        val_loader,
        optimizer,
        num_epochs=training_config.num_epochs,
        eval_freq=5,
        eval_iter=1,
        start_context="Every effort moves you",
    )

    return train_losses, val_losses, tokens_seen, model


if __name__ == "__main__":
    from toyllm.gpt2.config import GPTModelSize, GPTTrainingConfig
    from toyllm.gpt2.dataset import read_simple_text_file

    training_config = GPTTrainingConfig(learning_rate=5e-4, num_epochs=40, batch_size=2, weight_decay=0.1)

    text = read_simple_text_file()
    # Initiate training
    train_losses, val_losses, tokens_seen, model = main(text, GPTModelSize.SMALL, training_config)

    # After training

    # Plot results
    epochs_tensor = torch.linspace(0, training_config.num_epochs, len(train_losses))
    plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)
    plt.savefig("loss.pdf")

    # Save and load model
    # torch.save(model.state_dict(), "model.pth")
    # model = GPTModel(gpt_config_124_m)
    # model.load_state_dict(torch.load("model.pth"))
