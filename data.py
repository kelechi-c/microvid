from torch.utils.data import IterableDataset, DataLoader
from datasets import load_dataset

data_id = "tensorkelechi/tiny_webvid_latents"

class Text2VideoDataset(IterableDataset):
    def __init__(self, split=64):
        super().__init__()
        self.split = split
        self.dataset = load_dataset(
            data_id,
            streaming=True,
            split="train",
            trust_remote_code=True,
        ).take(
            self.split
        )  # Haven't preprocessed enough videos yet

    def __len__(self):
        return self.split

    def __iter__(self):
        for sample in self.dataset:
            latents = torch.tensor(sample["hyv_latents"], dtype=torch.bfloat16)  # type: ignore

            caption = torch.tensor(sample["text_encoded"], dtype=torch.bfloat16)  # type: ignore
            caption = caption[:, :10, :]

            yield latents, caption
