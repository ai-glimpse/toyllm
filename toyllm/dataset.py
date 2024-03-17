import tiktoken
import torch

from torch.utils.data import Dataset, DataLoader


class GPTDatasetV1(Dataset):
    def __init__(
        self, txt: str, tokenizer: tiktoken.Encoding, max_length: int, stride: int
    ):
        """
        :param txt: txt data
        :param tokenizer: tokenizer object
        :param max_length: max length
        :param stride: stride size
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

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader_v1(
    txt, batch_size=4, max_length=256, stride=128, shuffle=True, drop_last=True
):
    # Initialize the tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")

    # Create dataset
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

    # Create dataloader
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last
    )

    return dataloader
