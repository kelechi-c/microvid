from torch.utils.data import IterableDataset, DataLoader
from datasets import load_dataset


class Text2VideoDataset(IterableDataset):
    def __init__(self, split=512):
        super().__init__()
        self.split = split
        self.dataset = load_dataset(
            "tensorkelechi/tiny_webvid_latents",
            streaming=True,
            split="train",
            trust_remote_code=True,
        ).take(self.split)

    def __len__(self):
        return self.split

    def __iter__(self):
        for sample in self.dataset:
            latents = sample["video_latents"]  # type: ignore

            caption = sample["text_encoded"]  # type: ignore

            yield latents, caption