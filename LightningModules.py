from pathlib import Path

import torch
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision.utils import make_grid

import lightning.pytorch as pl
from lightning.pytorch.callbacks import Callback

from kornia import tensor_to_image
from microfilm.colorify import multichannel_to_rgb
import numpy as np
import wandb

from FUCCIDataset import FUCCIDataset, ReferenceChannelDataset, FUCCIChannelDataset
from FUCCIDataset import FUCCIDatasetInMemory, ReferenceChannelDatasetInMemory, FUCCIChannelDatasetInMemory
from models import Encoder, Decoder

class FUCCIDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, dataset, batch_size, num_workers, imsize=1024, split=(0.64, 0.16, 0.2), in_memory=True):
        super().__init__()
        if not isinstance(data_dir, Path):
            data_dir = Path(data_dir)
        self.data_dir = data_dir

        self.imsize = imsize

        if not in_memory:
            if dataset == "total":
                self.dataset = FUCCIDataset(self.data_dir, imsize=self.imsize)
            if dataset == "reference":
                self.dataset = ReferenceChannelDataset(self.data_dir, imsize=self.imsize)
            if dataset == "fucci":
                self.dataset = FUCCIChannelDataset(self.data_dir, imsize=self.imsize)
        else:
            if dataset == "total":
                self.dataset = FUCCIDatasetInMemory(self.data_dir, imsize=self.imsize)
            if dataset == "reference":
                self.dataset = ReferenceChannelDatasetInMemory(self.data_dir, imsize=self.imsize)
            if dataset == "fucci":
                self.dataset = FUCCIChannelDatasetInMemory(self.data_dir, imsize=self.imsize)

        self.split = split
        if len(self.split) != 3:
            raise ValueError("split must be a tuple of length 3")
        self.data_train, self.data_val, self.data_test = random_split(self.dataset, self.split)
        
        self.batch_size = batch_size
        self.num_workers = num_workers

    def _shared_dataloader(self, dataset, num_workers=None):
        if num_workers is None:
            num_workers = self.num_workers
        return DataLoader(dataset, batch_size=self.batch_size, num_workers=num_workers, persistent_workers=True)
        # return DataLoader(dataset, batch_size=self.batch_size, num_workers=num_workers)

    def train_dataloader(self):
        return self._shared_dataloader(self.data_train)

    def val_dataloader(self):
        return self._shared_dataloader(self.data_val)

    def test_dataloader(self):
        return self._shared_dataloader(self.data_test, num_workers=self.num_workers)

    def get_channels(self):
        return self.dataset.get_channel_names()


class AutoEncoder(pl.LightningModule):
    def __init__(self,
        nc=1,
        nf=128,
        ch_mult=(1, 2, 4, 8, 8, 8),
        imsize=256,
        latent_dim=512,
        lr=1e-6,
        patience=4,
        channels=None
    ):

        super().__init__()
        self.save_hyperparameters()

        self.lr = lr
        self.encoder = Encoder(nc, nf, ch_mult, imsize, latent_dim)
        self.decoder = Decoder(nc, nf, ch_mult[::-1], imsize, latent_dim)
        self.patience = patience
        self.channels = channels

    def reparameterized_sampling(self, mu, logvar):
        # check if mu or logvar is nan
        if torch.isnan(mu).any():
            print("mu is nan")
        if torch.isnan(logvar).any():
            print("logvar is nan")
        std = torch.exp(0.5 * logvar)
        if torch.isnan(std).any():
            print("std is nan")
        eps = torch.randn_like(mu)
        if torch.isnan(eps).any():
            print("eps is nan")
        sample = eps.mul(std).add_(mu)
        if torch.isnan(sample).any():
            print("sample is nan")
        return sample

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterized_sampling(mu, logvar)
        x_hat = self.decoder(z)
        return x_hat

    def _shared_step(self, batch):
        x = batch
        mu, logvar = self.encoder(x)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(mu)
        z = eps.mul(std).add_(mu)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        loss_val = loss
        if self.channels is not None:
            loss_dict = {}
            loss_dict["total"] = loss
            for i, channel in enumerate(self.channels):
                loss_dict[channel] = F.mse_loss(x_hat[:,i,:,:], x[:,i,:,:])
            loss = loss_dict
        # print debegging info if loss is nan
        if torch.isnan(loss_val):
            # print("x:", x)
            # print("max x:", torch.max(x))
            # print("min x:", torch.min(x))
            # print("x_hat:", x_hat)
            # print("max x_hat:", torch.max(x_hat))
            # print("min x_hat:", torch.min(x_hat))
            # print("mu:", mu)
            # print("logvar:", logvar)
            # print("std:", std)
            # print("eps:", eps)
            # print("z:", z)
            # raise ValueError("loss is nan")
            self.log("val/nan_x", torch.sum(torch.isnan(x)), on_step=True, on_epoch=False, reduce_fx=torch.sum, sync_dist=False)
            self.log("val/nan_x_hat", torch.sum(torch.isnan(x_hat)), on_step=True, on_epoch=False, reduce_fx=torch.sum, sync_dist=False)
            self.log("val/nan_mu", torch.sum(torch.isnan(mu)), on_step=True, on_epoch=False, reduce_fx=torch.sum, sync_dist=False)
            self.log("val/nan_std", torch.sum(torch.isnan(std)), on_step=True, on_epoch=False,reduce_fx=torch.sum, sync_dist=False)
            self.log("val/nan_z", torch.sum(torch.isnan(z)), on_step=True, on_epoch=False, reduce_fx=torch.sum, sync_dist=False)
            self.log("val/nan_test", 1, on_step=True, on_epoch=False, reduce_fx=torch.sum, sync_dist=False)
        return loss, x_hat

    def training_step(self, batch, batch_idx):
        loss, x_hat = self._shared_step(batch)
        if isinstance(loss, dict):
            for channel in loss:
                if channel == "total":
                    self.log(f"train/loss", loss[channel], on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
                else:
                    self.log(f"train/loss_{channel}", loss[channel], on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        else:
            self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss['total']

    def validation_step(self, batch, batch_idx):
        loss, x_hat = self._shared_step(batch)
        if isinstance(loss, dict):
            for channel in loss:
                if channel == "total":
                    self.log(f"val/loss", loss[channel], on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
                else:
                    self.log(f"val/loss_{channel}", loss[channel], on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        else:
            self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss['total']
    
    def test_step(self, batch, batch_idx):
        loss, x_hat = self._shared_step(batch)
        if isinstance(loss, dict):
            for channel in loss:
                if channel == "total":
                    self.log(f"test/loss", loss[channel], on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
                else:
                    self.log(f"test/loss_{channel}", loss[channel], on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        else:
            self.log("test/loss", loss, sync_dist=True)
        return loss['total']

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        scheduler = ReduceLROnPlateau(
            optimizer,
            patience=self.patience,
            threshold=0.01
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "train/loss"
        }

    def lr_scheduler_step(self, scheduler, metric):
        scheduler.step(metric)

class ReconstructionVisualization(Callback):
    def __init__(self, num_images=8, every_n_epochs=1, channels=None):
        super().__init__()
        self.num_images = num_images
        self.every_n_epochs = every_n_epochs
        self.channels = channels

    def __shared_reconstruction_step(self, input_imgs, pl_module, cmap):
        # Reconstruct images
        input_imgs = input_imgs.to(pl_module.device)
        with torch.no_grad():
            pl_module.eval()
            reconst_imgs = pl_module(input_imgs)
            pl_module.train()
        # print(f"input: {input_imgs.shape}, reconst: {reconst_imgs.shape}")
        # Plot and add to wandb
        grid = ReconstructionVisualization.make_reconstruction_grid(input_imgs, reconst_imgs)
        grid = tensor_to_image(grid)
        grid = np.moveaxis(grid, -1, 0)
        rgb_grid, _, _, _ = multichannel_to_rgb(grid, cmaps=cmap)
        return rgb_grid, grid

    def __shared_logging_step(self, input_imgs, pl_module, cmap, trainer):
        rgb_grid, grid = self.__shared_reconstruction_step(input_imgs, pl_module, cmap)
        trainer.logger.experiment.log({
            f"{trainer.state.stage}/reconstruction_samples": wandb.Image(rgb_grid,
                caption="Original and reconstructed images")
        })

        if self.channels is not None:
            if len(self.channels) != grid.shape[0]:
                raise ValueError(f"Number of channels ({len(self.channels)}) does not match number of images ({grid.shape[0]})")
            for i, channel in enumerate(self.channels):
                trainer.logger.experiment.log({
                    f"{trainer.state.stage}/reconstruction_samples_{channel}": wandb.Image(grid[i],
                        caption=f"Original and reconstructed {channel} images")
                })

    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch % self.every_n_epochs == 0:
            input_imgs = trainer.datamodule.train_dataloader().dataset[:self.num_images]
            cmap = trainer.datamodule.dataset.channel_colors()
            self.__shared_logging_step(input_imgs, pl_module, cmap, trainer)

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch % self.every_n_epochs == 0:
            input_imgs = trainer.datamodule.val_dataloader().dataset[:self.num_images]
            cmap = trainer.datamodule.dataset.channel_colors()
            self.__shared_logging_step(input_imgs, pl_module, cmap, trainer)
            
    def on_test_end(self, trainer, pl_module):
        input_imgs = trainer.datamodule.test_dataloader().dataset[:self.num_images]
        cmap = trainer.datamodule.dataset.channel_colors()
        self.__shared_logging_step(input_imgs, pl_module, cmap, trainer)

    def image_list_normalization(image_list):
        for i in range(len(image_list)):
            image_list[i] = image_list[i] - torch.min(image_list[i])
            image_list[i] = image_list[i] / (torch.max(image_list[i]) - torch.min(image_list[i]))
            image_list[i] = image_list[i] * 2 - 1
        return image_list

    def make_reconstruction_grid(input_imgs, reconst_imgs):
        imgs = torch.stack([input_imgs, reconst_imgs], dim=1).flatten(0, 1)
        grid = make_grid(imgs, nrow=2, normalize=True)
        return grid