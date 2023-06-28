import os
import argparse
from pathlib import Path
import torch
from torch.utils.data import Dataset, random_split, DataLoader
import imageio as iio
import numpy as np
import hpacellseg.cellsegmentator as cellsegmentator
from hpacellseg.utils import label_cell
import warnings
from packaging import version
import time
import kornia.augmentation as K
import kornia.geometry.transform as T
from kornia.utils import image_to_tensor

def cell_masks(dapi, gamma_tubulin, suppress_warnings=True):
    if version.parse(torch.__version__) >= version.parse("1.10.0"):
        raise ValueError(f"HPA Cell Segmentator is not compatible with torch >= 1.10.0.\nTorch {torch.__version__} detected. Are you using the 'cell-seg' conda environment?")
    if version.parse(np.__version__) >= version.parse("1.20.0"):
        raise ValueError(f"HPA Cell Segmentator is not compatible with torch >= 1.10.0.\nTorch {torch.__version__} detected. Are you using the 'cell-seg' conda environment?")
    if suppress_warnings:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return cell_masks_warn(dapi, gamma_tubulin)
    else:
        return cell_masks_warn(dapi, gamma_tubulin)


def cell_masks_warn(dapi, gamma_tubulin):
    """
    Segment the image into the cell and background
    Both arguments are lists of numpy arrays of the individual channels
    """
    segmentator = cellsegmentator.CellSegmentator(
        scale_factor=0.25,
        device="cuda",
        padding=True,
        multi_channel_model=False
    )

    # For nuclei
    nuc_segmentations = segmentator.pred_nuclei(dapi)

    # For full cells
    cell_segmentations = segmentator.pred_cells([
        gamma_tubulin,
        None,
        dapi
    ])

    # post-processing
    cell_masks = []
    for i in range(len(nuc_segmentations)):
        nuclei_mask, cell_mask = label_cell(nuc_segmentations[0], cell_segmentations[0])
        cell_masks.append(cell_mask)

    assert len(cell_masks) == len(dapi)

    return cell_masks


# In order to use a random_split, we can't have an IterableDataset
# Instead of loading in all the images we'll just load in the filenames
# With the current dataset folder structure we have split -> experiment -> image channel
# TODO: currently not adding a way to access the original split structure
class FUCCIDataset(Dataset):
    """
    Images are in DAPI, gamma-tubulin, Geminin, CDT1 order
    Geminin is a green fluorescent protein that is present during the S and G2 phases of the cell cycle
    CDT1 is a red fluorescent protein that is present during the G1 and S phases of the cell cycle
    DAPI is a blue fluorescent protein that binds to DNA
    gamma-tubulin is a red fluorescent protein that binds to microtubules
    """
    def __init__(self, data_dir, imsize, rebuild=False, verbose=False,
                 cmap=["pure_blue", "pure_yellow", "pure_green", "pure_red"],
                 transform=None):
        super().__init__()

        self.verbose = verbose
        self.transform = transform

        if not isinstance(data_dir, Path):
            data_dir = Path(data_dir)
        if not data_dir.exists():
            raise ValueError(f"{data_dir} not found")
        if not data_dir.is_dir():
            raise ValueError("data_dir must be a directory")
        self.data_dir = data_dir

        self.cmap = cmap

        if rebuild:
            clean_dir(self.data_dir)

        im_shape = [2048, 2048] # TODO: make this not hard-coded but without starting processing first
        if im_shape[0] < imsize:
            raise ValueError(f"imsize ({imsize}) must be less than or equal to the image height ({im_shape[0]})")
        if im_shape[1] < imsize:
            raise ValueError(f"imsize ({imsize}) must be less than or equal to the image width ({im_shape[1]})")
        
        self.imsize = imsize

        self.dataset = []
        cell_index = 0
        exp_count = 0

        for split in self.data_dir.iterdir():
            if not split.is_dir():
                if "batch" in split.stem:
                    continue
                if "index" in split.stem:
                    continue
                if "indices" in split.stem:
                    continue
                if "prediction" in split.stem:
                    continue
                if "color" in split.stem:
                    continue
                if "mu" in split.stem:
                    continue
                if "logvar" in split.stem:
                    continue
                if "var" in split.stem:
                    continue
                if "embedding" in split.stem:
                    continue
                if ".pt" == split.suffix:
                    continue
                warnings.warn(f"Training split {split} is not a directory")
                continue

            for experiment in split.iterdir():
                if not experiment.is_dir():
                    raise ValueError(f"Experiment {experiment} is not a directory")

                if experiment / "mask.png" not in experiment.iterdir():
                    image = self.__image_from_experiment(experiment)
                    mask = cell_masks(dapi=[image[:, :, 0]], gamma_tubulin=[image[:, :, 1]])[0]
                    iio.imwrite(experiment / "mask.png", mask)

                if experiment / "com.npy" not in experiment.iterdir():
                    num_cells = np.max(iio.imread(experiment / "mask.png"))
                    com = np.zeros((num_cells, 2))
                    for cell in range(num_cells):
                        com[cell] = np.mean(np.argwhere(iio.imread(experiment / "mask.png") == cell + 1), axis=0)
                    np.save(experiment / "com", com)

                self.dataset.append((cell_index, experiment))
                num_cells = np.max(iio.imread(experiment / "mask.png"))

                if experiment / "cells.npy" not in experiment.iterdir():
                    cell_images = []
                    for cell in range(num_cells):
                        cell_images.append(self.__get_single_item(cell_index + cell).numpy())
                    cell_images = np.array(cell_images)
                    np.save(experiment / "cells", cell_images)  

                cell_index += np.max(num_cells)

                exp_count += 1
                if self.verbose and exp_count % 10 == 0:
                    print(f"[{time.strftime('%m/%d/%Y @ %H:%M')}] Loaded image {exp_count}")
        
        self.channels = ["DAPI", "gamma-tubulin", "Geminin", "CDT1"]
        self.len = cell_index
        self.num_exp = exp_count

    
    def __image_from_experiment(self, experiment_dir):
        dapi_names = ["dapi", "nuclei", "nucleus", "dna", "nuclear"]
        tubulin_names = ["gamma-tubulin", "microtubule", "microtubules", "tubulin"]
        geminin_names = ["geminin", "gem"]
        cdt1_names = ["cdt1"]

        def channel_from_dir(names, dir):
            # TODO should provide a better guess for which channel is which
            # maybe just throw an error and ask the user to provide a specific name or function
            return min(list(filter(lambda x: any(name in x.stem.lower() for name in names), dir.iterdir())))
            
        dapi_file = channel_from_dir(dapi_names, experiment_dir)
        tubulin_file = channel_from_dir(tubulin_names, experiment_dir)
        geminin_file = channel_from_dir(geminin_names, experiment_dir)
        cdt1_file = channel_from_dir(cdt1_names, experiment_dir)

        dapi = iio.imread(dapi_file)
        gamma_tubulin = iio.imread(tubulin_file)
        geminin = iio.imread(geminin_file)
        cdt1 = iio.imread(cdt1_file)

        image_channels = [dapi, gamma_tubulin, geminin, cdt1]
        return np.stack(image_channels, axis=-1).astype(np.float32)

    def __len__(self):
        return self.len

    def image_exp_index(self, idx):
        exp_index = 0
        while exp_index < len(self.dataset) and self.dataset[exp_index][0] <= idx:
            exp_index += 1
        experiment_entry = self.dataset[exp_index - 1]
        cell_index = idx - experiment_entry[0]
        return experiment_entry, cell_index

    def __get_single_item(self, idx):
        experiment_entry, cell_index = self.image_exp_index(idx)
        image = self.__image_from_experiment(experiment_entry[1])
        mask = (iio.imread(experiment_entry[1] / "mask.png") == 1 + cell_index)
        cell_image = image * np.expand_dims(mask, axis=2)
        com = np.load(experiment_entry[1] / "com.npy")[cell_index]
        offset = (np.asarray(cell_image.shape[:-1]) / 2 - com).astype(int)
        centered = np.roll(cell_image, offset, axis=(0, 1))
        centered = image_to_tensor(centered, keepdim=True)
        cropped = K.CenterCrop(centered.shape[1] // 2, keepdim=True)(centered)
        img_small = T.resize(cropped, self.imsize)
        if self.transform is not None:
            return self.transform(img_small)
        return img_small
        # return cropped
        # centered = np.moveaxis(centered, -1, 0)
        # return centered
        # else:
        #     cell_images = np.load(experiment_entry[1] / "cells.npy")
        #     return image_to_tensor(cell_images[cell_index], keepdim=True)
    
    def __getitem__(self, idx):
        if isinstance(idx, int):
            return self.__get_single_item(idx)
        elif isinstance(idx, slice):
            return torch.stack([self.__get_single_item(i) for i in range(*idx.indices(self.len))])
        elif isinstance(idx, list):
            return torch.stack([self.__get_single_item(i) for i in idx])
        else:
            raise TypeError(f"Invalid argument type {type(idx)} must be int, slice, or list of ints.") 
    
    def get_experiment_cells(self, idx):
        if idx < 0 or idx > self.num_exp:
            raise IndexError(f"Index {idx} out of range for dataset of length {self.len}")
        cell_images = np.load(self.dataset[idx][1] / "cells.npy")
        return cell_images

    def get_original_image(self, idx):
        return self.__image_from_experiment(self.dataset[idx][1])

    def get_num_experiments(self):
        return len(self.dataset)

    def channel_colors(self):
        return self.cmap

    def get_channel_names(self):
        return self.channels
        

class ReferenceChannelDataset(FUCCIDataset):
    def __getitem__(self, idx):
        full_image = super().__getitem__(idx)
        return full_image[..., :2, :, :]

    def channel_colors(self):
        if len(self.cmap) > 2:
            return self.cmap[:2]
        else:
            return self.cmap

    def get_channel_names(self):
        return self.channels[:2]
     

class FUCCIChannelDataset(FUCCIDataset):
    def __getitem__(self, idx):
        full_image = super().__getitem__(idx)
        return full_image[..., 2:, :, :]

    def channel_colors(self):
        if len(self.cmap) > 2:
            return self.cmap[2:]
        else:
            return self.cmap

    def get_channel_names(self):
        return self.channels[2:]


class FUCCIDatasetInMemory(FUCCIDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dataset_images = []
        for i in range(self.num_exp):
            self.dataset_images.append(super().get_experiment_cells(i))
        self.dataset_images = np.concatenate(self.dataset_images, axis=0)
        self.dataset_images = torch.Tensor(self.dataset_images)
    
    def __getitem__(self, idx):
        images = self.dataset_images[idx]
        if self.transform is not None:
            images = torch.roll(images, shifts=1, dims=0)
            # images_temp = images.clone()
            # images[..., 0, :, :] = images_temp[..., 1, :, :]
            # images[..., 1, :, :] = images_temp[..., 0, :, :]
        return images


class ReferenceChannelDatasetInMemory(FUCCIDatasetInMemory):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dataset_images = self.dataset_images[..., :2, :, :]

    def channel_colors(self):
        if len(self.cmap) > 2:
            return self.cmap[:2]
        else:
            return self.cmap
        
    def get_channel_names(self):
        return self.channels[:2]
    

class FUCCIChannelDatasetInMemory(FUCCIDatasetInMemory):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dataset_images = self.dataset_images[..., 2:, :, :]

    def channel_colors(self):
        if len(self.cmap) > 2:
            return self.cmap[2:]
        else:
            return self.cmap

    def get_channel_names(self):
        return self.channels[2:]


class ChannelDatasetInMemory(FUCCIDatasetInMemory):
    def __init__(self, channel_slice=[0, 1, 2, 3], *args, **kwargs):
        if min(channel_slice) < 0 or max(channel_slice) > 3 or len(channel_slice) > 4:
            raise ValueError("Channels must be a list of at most 4 integers in the range [0, 3]")
        super().__init__(*args, **kwargs)
        self.dataset_images = self.dataset_images[..., channel_slice, :, :]
        self.channel_slice = channel_slice

    def channel_colors(self):
        return [self.cmap[c] for c in self.channel_slice]

    def get_channel_names(self):
        return [self.channels[c] for c in self.channel_slice]
    

class TotalDatasetInMemory(FUCCIDatasetInMemory):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        shape = self.dataset_images.shape
        self.dataset_images = self.dataset_images.reshape((-1, 1, shape[-2], shape[-1]))
        print(shape, self.dataset_images.shape)


class GemininDatasetInMemory(ChannelDatasetInMemory):
    def __init__(self, *args, **kwargs):
        super().__init__(channel_slice=[2], *args, **kwargs)
    
    def __getitem__(self, idx):
        return super().__getitem__(idx).clamp(min=200)


class CDT1DatasetInMemory(ChannelDatasetInMemory):
    def __init__(self, *args, **kwargs):
        super().__init__(channel_slice=[3], *args, **kwargs)
    
    def __getitem__(self, idx):
        return super().__getitem__(idx).clamp(min=200)


def clean_dir(data_dir):
    if not isinstance(data_dir, Path):
        data_dir = Path(data_dir)
    remove = lambda x, dir: os.remove(dir / x) if dir / x in dir.iterdir() else None
    for split in data_dir.iterdir():
        for experiment in split.iterdir():
            remove("mask.png", experiment)
            remove("com.npy", experiment)
            remove("cells.npy", experiment)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FUCCI Dataset Tool",
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-v", "--verbose", action="store_true", help="print verbose output")
    parser.add_argument("-c", "--clean", action="store_true", help="remove all mask files")
    parser.add_argument("-d", "--data", required=True, help="path to dataset")
    args = parser.parse_args()
    dataset_location = Path(args.data)

    if args.clean:
        clean_dir(dataset_location)
        exit(0)

    dataset = FUCCIDataset(dataset_location, verbose=args.verbose)
    print(f"Dataset channels: {dataset.channels}")
    print(f"Dataset total segmented cells count (dataset length): {len(dataset)}")
    print(f"Dataset original images count: {dataset.get_num_experiments()}")