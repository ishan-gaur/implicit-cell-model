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

    def _shared_dataloader(self, dataset):
        return DataLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=True)

    def train_dataloader(self):
        return self._shared_dataloader(self.data_train)

    def val_dataloader(self):
        return self._shared_dataloader(self.data_val)

    def test_dataloader(self):
        return self._shared_dataloader(self.data_test)

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
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(mu)
        sample = eps.mul(std).add_(mu)
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
        if self.channels is not None:
            loss_dict = {}
            loss_dict["total"] = loss
            for i, channel in enumerate(self.channels):
                loss_dict[channel] = F.mse_loss(x_hat[:,i,:,:], x[:,i,:,:])
            loss = loss_dict
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
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "train/loss"
        }

    def lr_scheduler_step(self, scheduler, metric):
        scheduler.step(metric)

class ReconstructionVisualization(Callback):
    def __init__(self, num_images=8, every_n_epochs=1):
        super().__init__()
        self.num_images = num_images
        self.every_n_epochs = every_n_epochs

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
        grid, _, _, _ = multichannel_to_rgb(grid, cmaps=cmap)
        return grid

    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch % self.every_n_epochs == 0:
            input_imgs = trainer.datamodule.train_dataloader().dataset[:self.num_images]
            cmap = trainer.datamodule.dataset.channel_colors()
            grid = self.__shared_reconstruction_step(input_imgs, pl_module, cmap)
            trainer.logger.experiment.log({
                "training_samples": wandb.Image(grid,
                    caption="Original and reconstructed images")
            })

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch % self.every_n_epochs == 0:
            input_imgs = trainer.datamodule.val_dataloader().dataset[:self.num_images]
            cmap = trainer.datamodule.dataset.channel_colors()
            grid = self.__shared_reconstruction_step(input_imgs, pl_module, cmap)
            trainer.logger.experiment.log({
                "validation_samples": wandb.Image(grid,
                    caption="Original and reconstructed images")
            })
            
    def on_test_end(self, trainer, pl_module):
        input_imgs = trainer.datamodule.test_dataloader().dataset[:self.num_images]
        cmap = trainer.datamodule.dataset.channel_colors()
        grid = self.__shared_reconstruction_step(input_imgs, pl_module, cmap)
        trainer.logger.experiment.log({
            "testing_samples": wandb.Image(grid,
                caption="Original and reconstructed images")
        })

    def image_list_normalization(image_list):
        for i in range(len(image_list)):
            image_list[i] = image_list[i] - torch.min(image_list[i])
            image_list[i] = image_list[i] / (torch.max(image_list[i]) - torch.min(image_list[i]))
            image_list[i] = image_list[i] * 2 - 1
        return image_list

    def make_reconstruction_grid(input_imgs, reconst_imgs):
        imgs = torch.stack([input_imgs, reconst_imgs], dim=1).flatten(0, 1)
        grid = make_grid(imgs, nrow=2, normalize=True, range=(-1, 1))
        return grid