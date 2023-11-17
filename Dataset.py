import torch
from torch.utils.data import Dataset, DataLoader, random_split
from lightning import LightningDataModule
from lightning.pytorch.utilities.combined_loader import CombinedLoader

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


import math
import warnings
from torch import default_generator, randperm
from torch._utils import _accumulate
from torch.utils.data import Subset

def random_split_indices(dataset, lengths, generator=default_generator):
    if math.isclose(sum(lengths), 1) and sum(lengths) <= 1:
        subset_lengths = []
        for i, frac in enumerate(lengths):
            if frac < 0 or frac > 1:
                raise ValueError(f"Fraction at index {i} is not between 0 and 1")
            n_items_in_split = int(
                math.floor(len(dataset) * frac)  # type: ignore[arg-type]
            )
            subset_lengths.append(n_items_in_split)
        remainder = len(dataset) - sum(subset_lengths)  # type: ignore[arg-type]
        # add 1 to all the lengths in round-robin fashion until the remainder is 0
        for i in range(remainder):
            idx_to_add_at = i % len(subset_lengths)
            subset_lengths[idx_to_add_at] += 1
        lengths = subset_lengths
        for i, length in enumerate(lengths):
            if length == 0:
                warnings.warn(f"Length of split at index {i} is 0. "
                              f"This might result in an empty dataset.")

    # Cannot verify that dataset is Sized
    if sum(lengths) != len(dataset):    # type: ignore[arg-type]
        raise ValueError("Sum of input lengths does not equal the length of the input dataset!")

    indices = randperm(sum(lengths), generator=generator).tolist()  # type: ignore[call-overload]
    return indices, lengths

def splits_from_indices(dataset, indices, lengths):
    return [Subset(dataset, indices[offset - length : offset]) for offset, length in zip(_accumulate(lengths), lengths)]


class MultiModalDataModule(LightningDataModule):
    """
    Data module to present multiple image channels simultaneously.
    The behavior can be customized by setting the mode. The modes are:
    - paired: the images are paired by index.
    """
    def __init__(self, dataset_dirs, channel_names, colors, mode, split, batch_size, num_workers):
        super().__init__()
        # self.datasets = [torch.load(dir / f"{channel_name}.pt") for dir, channel_name in zip(dataset_dirs, channel_names)]
        self.datasets = {}
        for dir, channel_name in zip(dataset_dirs, channel_names):
            num_shards = len(list(dir.glob(f"{channel_name}_*.pt")))
            dataset = []
            for i in range(num_shards):
                dataset.append(torch.load(dir / f"{channel_name}_{i}.pt"))
            self.datasets[channel_name] = torch.cat(dataset)
        self.channel_names = channel_names
        self.colors = colors
        self.mode = mode
        if len(split) != 3:
            raise ValueError("split must be a tuple of length 3")
        self.split = split
        self.batch_size = batch_size
        self.num_workers = num_workers
    
        if self.mode == "combined":
            # just combines all the samples into one big tensor
            # then add an empty "channel" channel
            self.dataset = torch.cat(list(self.datasets.values()))[:, None, ...]
            self.dataset = SimpleDataset(self.dataset)
            self.data_train, self.data_val, self.data_test = random_split(self.dataset, self.split)
        elif self.mode == "paired":
            if not all([len(self.datasets[self.channel_names[0]]) == len(d) for _, d in self.datasets.items()]):
                raise ValueError("All datasets must have the same length. Got lengths: ", [len(d) for d in self.datasets])
            indices, lengths = random_split_indices(self.datasets[self.channel_names[0]], self.split)
            self.data_train, self.data_val, self.data_test = {}, {}, {}
            for channel, dataset in self.datasets.items():
                c = channel
                self.data_train[c], self.data_val[c], self.data_test[c] = splits_from_indices(dataset, indices, lengths)
        elif self.mode == "unpaired":
            # stack should give us (modalities, samples, ...)
            # then [:, None, ...] should give us an empty channel index
            # for images this would be a 5D tensor
            # self.dataset = torch.stack(self.datasets)[:, None, ...]
            raise NotImplementedError("Unpaired mode not implemented yet.")
        else:
            raise ValueError(f"Mode must be one of {self.modes()}. Got {mode}.")

    def modes(self):
        return ["paired", "unpaired", "combined"]

    def __shared_dataloader(self, datasets, shuffle=True):
        if self.mode == "combined":
            # in this case datasets is only a single dataset
            return DataLoader(datasets, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=True, shuffle=shuffle)
        elif self.mode == "paired":
            loader_template = lambda dataset: DataLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=True, shuffle=shuffle)
            self.dataloaders = {channel: loader_template(dataset) for channel, dataset in datasets.items()}
            return CombinedLoader(self.dataloaders, "max_size")
        elif self.mode == "unpaired":
            raise NotImplementedError("Unpaired mode not implemented yet.")
        else:
            raise ValueError(f"Mode must be one of {self.modes()}. Got {self.mode}.")

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