import lzma
from typing import List
import torch
from torch.utils.data import Dataset


class LMTextDataset(Dataset):
    """
    Simple language modeling dataset.
    Each sample: input_ids ([:-1]), target_ids ([1:])
    """

    def __init__(
        self,
        path: str,
        tokenizer,
        max_len: int = 128,
        limit_lines: int | None = None,
    ):
        self.samples: List[torch.Tensor] = []
        self.tokenizer = tokenizer
        self.max_len = max_len

        is_xz = path.endswith(".xz")

        def line_iter():
            if is_xz:
                with lzma.open(path, "rt", encoding="utf-8") as f:
                    for i, line in enumerate(f):
                        if limit_lines is not None and i >= limit_lines:
                            break
                        yield line.strip()
            else:
                with open(path, "r", encoding="utf-8") as f:
                    for i, line in enumerate(f):
                        if limit_lines is not None and i >= limit_lines:
                            break
                        yield line.strip()

        for line in line_iter():
            if not line:
                continue
            ids = self.tokenizer.encode(line, add_special_tokens=True)
            if len(ids) < 3:
                continue
            if len(ids) > max_len:
                ids = ids[:max_len]
            self.samples.append(torch.tensor(ids, dtype=torch.long))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ids = self.samples[idx]
        x = ids[:-1]
        y = ids[1:]
        return x, y


def collate_batch(batch, pad_id: int = 0):
    xs, ys = zip(*batch)
    max_len = max(x.size(0) for x in xs)
    B = len(xs)

    x_batch = torch.full((B, max_len), pad_id, dtype=torch.long)
    y_batch = torch.full((B, max_len), -100, dtype=torch.long)

    for i, (x, y) in enumerate(zip(xs, ys)):
        L = x.size(0)
        x_batch[i, :L] = x
        y_batch[i, :L] = y

    return x_batch, y_batch