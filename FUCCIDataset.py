import os
# import sys
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
# from bisect import bisect_right

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
    Images are is DAPI, gamma-tubulin, Geminin, and CDT1 order
    Geminin is a green fluorescent protein that is present during the S and G2 phases of the cell cycle
    CDT1 is a red fluorescent protein that is present during the G1 and S phases of the cell cycle
    DAPI is a blue fluorescent protein that binds to DNA
    gamma-tubulin is a red fluorescent protein that binds to microtubules
    """
    def __init__(self, data_dir, imsize, rebuild=False, verbose=False, cmap=["pure_blue", "pure_yellow", "pure_green", "pure_red"]):
        super().__init__()

        self.verbose = verbose

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

        im_shape = iio.imread(next(next(next(self.data_dir.iterdir()).iterdir()).iterdir())).shape
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
                raise ValueError(f"Training split {split} is not a directory")

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
                cell_index += np.max(iio.imread(experiment / "mask.png"))

                exp_count += 1
                if self.verbose and exp_count % 10 == 0:
                    print(f"[{time.strftime('%m/%d/%Y @ %H:%M')}] Loaded image {exp_count}")
        
        self.channels = ["DAPI", "gamma-tubulin", "Geminin", "CDT1"]
        self.len = cell_index
    
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

    def __dataset_to_exp_index(self, idx):
        # experiment_entry = next(filter(lambda experiment: experiment[0] <= idx, self.dataset))
        # exp_index = bisect_right(self.dataset, idx, key=lambda x: x[0])
        exp_index = 0
        while exp_index < len(self.dataset) and self.dataset[exp_index][0] <= idx:
            exp_index += 1
        experiment_entry = self.dataset[exp_index - 1]
        cell_index = idx - experiment_entry[0]
        return experiment_entry, cell_index

    def __get_single_item(self, idx):
        experiment_entry, cell_index = self.__dataset_to_exp_index(idx)
        image = self.__image_from_experiment(experiment_entry[1])
        mask = (iio.imread(experiment_entry[1] / "mask.png") == 1 + cell_index)
        cell_image = image * np.expand_dims(mask, axis=2)
        com = np.load(experiment_entry[1] / "com.npy")[cell_index]
        offset = (np.asarray(cell_image.shape[:-1]) / 2 - com).astype(int)
        centered = np.roll(cell_image, offset, axis=(0, 1))
        centered = image_to_tensor(centered, keepdim=True)
        cropped = K.CenterCrop(centered.shape[1] // 2, keepdim=True)(centered)
        img_small = T.resize(cropped, self.imsize)
        return img_small
        # return cropped
        # centered = np.moveaxis(centered, -1, 0)
        # return centered

    def __getitem__(self, idx):
        if isinstance(idx, int):
            return self.__get_single_item(idx)
        elif isinstance(idx, slice):
            return torch.stack([self.__get_single_item(i) for i in range(*idx.indices(self.len))])
        elif isinstance(idx, list):
            return torch.stack([self.__get_single_item(i) for i in idx])
        else:
            raise TypeError(f"Invalid argument type {type(idx)} must be int, slice, or list of ints.") 
    
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


def clean_dir(data_dir):
    if not isinstance(data_dir, Path):
        data_dir = Path(data_dir)
    for split in data_dir.iterdir():
        for experiment in split.iterdir():
            # remove mask.png if it exists
            if experiment / "mask.png" in experiment.iterdir():
                os.remove(experiment / "mask.png")

            if experiment / "com.npy" in experiment.iterdir():
                os.remove(experiment / "com.npy")

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