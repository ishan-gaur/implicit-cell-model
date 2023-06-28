import torch
from torch.utils.data import Dataset, DataLoader, random_split
from lightning import LightningDataModule

class SimpleDataset(Dataset):
    def __init__(self, tensor) -> None:
        self.tensor = tensor

    def __getitem__(self, index):
        return self.tensor[index]

    def __len__(self):
        return self.tensor.size(0)


class ImageChannelDataset(Dataset):
    def __init__(self, dataset_dir, channel_name, color=None):
        self.dataset_dir = dataset_dir
        self.channel_name = channel_name
        self.images = torch.load(self.dataset_dir / f"{self.channel_name}.pt")

    def __len__(self):
        return len(self.images) 

    def __getitem__(self, idx):
        return self.images[idx]


class MultiModalDataModule(LightningDataModule):
    """
    Data module to present multiple image channels simultaneously.
    The behavior can be customized by setting the mode. The modes are:
    - paired: the images are paired by index.
    """
    def __init__(self, dataset_dirs, channel_names, colors, mode, split, batch_size, num_workers):
        super().__init__()
        # self.datasets = [torch.load(dir / f"{channel_name}.pt") for dir, channel_name in zip(dataset_dirs, channel_names)]
        self.datasets = []
        for dir, channel_name in zip(dataset_dirs, channel_names):
            num_shards = len(list(dir.glob(f"{channel_name}_*.pt")))
            dataset = []
            for i in range(num_shards):
                dataset.append(torch.load(dir / f"{channel_name}_{i}.pt"))
            self.datasets.append(torch.cat(dataset))
        self.channel_names = channel_names
        self.colors = colors
        self.mode = mode
        self.split = split
        self.batch_size = batch_size
        self.num_workers = num_workers
    
        if self.mode == "paired":
            if not all([len(self.datasets[0]) == len(d) for d in self.datasets]):
                raise ValueError("All datasets must have the same length. Got lengths: ", [len(d) for d in self.datasets])
            # stack should give us (modalities, samples, ...)
            # then swapaxes should give us (samples, modalities, ...) so that they are paired
            self.dataset = torch.stack(self.datasets).swapaxes(0, 1)
        elif self.mode == "unpaired":
            # stack should give us (modalities, samples, ...)
            # then [:, None, ...] should give us an empty channel index
            # for images this would be a 5D tensor
            self.dataset = torch.stack(self.datasets)[:, None, ...]
            raise NotImplementedError("Unpaired mode not implemented yet.")
        elif self.mode == "combined":
            # just combines all the samples into one big tensor
            # then add an empty "channel" channel
            self.dataset = torch.cat(self.datasets)[:, None, ...]
        else:
            raise ValueError(f"Mode must be one of {self.modes()}. Got {mode}.")

        self.dataset = SimpleDataset(self.dataset)

        if len(self.split) != 3:
            raise ValueError("split must be a tuple of length 3")
        self.data_train, self.data_val, self.data_test = random_split(self.dataset, self.split)

    def modes(self):
        return ["paired", "unpaired", "combined"]

    def __shared_dataloader(self, dataset, shuffle=True):
        return DataLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=True)

    def train_dataloader(self):
        return self.__shared_dataloader(self.data_train)
    
    def val_dataloader(self):
        return self.__shared_dataloader(self.data_val, shuffle=False)

    def test_dataloader(self):
        return self.__shared_dataloader(self.data_test, shuffle=False)

    def predict_dataloader(self):
        return super().predict_dataloader()

    def get_channels(self):
        return self.channel_names