import logging
import pathlib
import urllib.request

import tiktoken
import torch
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)


def get_dataset_dir() -> pathlib.Path:
    return pathlib.Path(__file__).parents[1] / "dataset"


class GPTDataset(Dataset):
    def __init__(self, txt: str, tokenizer: tiktoken.Encoding, max_length: int, stride: int) -> None:
        """The GPTDataset class is used to create a PyTorch dataset from a text file.

        Args:
            txt: txt data
            tokenizer: tokenizer object
            max_length: max length
            stride: stride size.
        """
        self.tokenizer = tokenizer
        self.input_ids = []
        self.target_ids = []

        # Tokenize the entire text
        token_ids = tokenizer.encode(txt)

        # Use a sliding window to chunk the book into overlapping sequences of max_length
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i : i + max_length]
            target_chunk = token_ids[i + 1 : i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor]:
        return self.input_ids[idx], self.target_ids[idx]


class GPTDataloader:
    def __init__(
        self,
        tokenizer: tiktoken.Encoding,
        max_length: int,
        stride: int,
        batch_size: int,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.stride = stride
        self.batch_size = batch_size

    def create_dataloader(self, text: str, shuffle=True, drop_last=True) -> DataLoader:
        # Create dataset
        dataset = GPTDataset(text, self.tokenizer, self.max_length, self.stride)

        # Create dataloader
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle, drop_last=drop_last)

        return dataloader


def read_simple_text_file() -> str:
    file_name = "the-verdict.txt"
    url = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt"

    file_path = get_dataset_dir() / file_name

    # if not exists, download it first
    if not file_path.exists():
        logger.info(f"Downloading {url} to {file_path}")
        with urllib.request.urlopen(url) as response:  # noqa: S310
            text_data = response.read().decode("utf-8")
        with pathlib.Path.open(file_path, "w", encoding="utf-8") as file:
            file.write(text_data)

        logger.info(f"Saved {file_path}")
    # open the file
    with pathlib.Path.open(file_path, encoding="utf-8") as file:
        text_data = file.read()
    return text_data


if __name__ == "__main__":
    text = read_simple_text_file()
    print(len(text))
