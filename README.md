# implicit-cell-model

If you get just the raw FUCCI Dataset, you will need to prep it with cell masks and center-of-mass measurements so the centered single-cell images can be extracted from the data upon request.

This can be done from the command-line by running `python FUCCIDataset.py -d {dataset_path}`. Upon completion it will also give a small summary of the data. If you need to recompile these files and want to clean the dataset directory for any reason, just pass the `-c` option (clean).

Running these commands requires using the `data-prep` conda environment. This can be setup from the `HPA-Cell-Segmentation` submodule of this repo. The reason for having a separate conda environment is due to the saved segmentation model's use of an old version of pytorch and some deprecated changes it uses from previous versions of numpy.

Once in the `HPA-Cell-Segmentation` directory, run `conda env create -f environment.yml`, `conda activate data-prep`, and then `sh install.sh`. It's important to not try to install each of the dependencies individually as it's very easy for the wrong versions of packages to get installed when added on sequentially. The resulting `data-prep` environment is only required for generating the `mask.png` and `com.npy` files for the dataset preprocessing.

To start using the actual project, you will also need to install the segmentation libraries just to use the FUCCIDataset file. From the implicit-cell-model directory, run `conda env create -f environment.yml`, `conda activate implicit`, and then `cd HPA-Cell-Segmentation && sh install.sh`.