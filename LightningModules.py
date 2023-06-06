from pathlib import Path
from lightning.pytorch.utilities.types import EVAL_DATALOADERS

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
    def __init__(self, data_dir, dataset, batch_size, num_workers, 
                 imsize=1024, split=(0.64, 0.16, 0.2), in_memory=True,
                 permutation=None):
        super().__init__()
        if not isinstance(data_dir, Path):
            data_dir = Path(data_dir)
        self.data_dir = data_dir

        self.imsize = imsize

        # not the right thing to do...
        if permutation is not None:
            transform = lambda x: x.permute(permutation)
        else:
            transform = None

        if not in_memory:
            if dataset == "total":
                self.dataset = FUCCIDataset(self.data_dir, imsize=self.imsize, transform=transform)
            if dataset == "reference":
                self.dataset = ReferenceChannelDataset(self.data_dir, imsize=self.imsize, transform=transform)
            if dataset == "fucci":
                self.dataset = FUCCIChannelDataset(self.data_dir, imsize=self.imsize, transform=transform)
        else:
            if dataset == "total":
                self.dataset = FUCCIDatasetInMemory(self.data_dir, imsize=self.imsize, transform=transform)
            if dataset == "reference":
                self.dataset = ReferenceChannelDatasetInMemory(self.data_dir, imsize=self.imsize, transform=transform)
            if dataset == "fucci":
                self.dataset = FUCCIChannelDatasetInMemory(self.data_dir, imsize=self.imsize, transform=transform)

        self.split = split
        if len(self.split) != 3:
            raise ValueError("split must be a tuple of length 3")
        self.data_train, self.data_val, self.data_test = random_split(self.dataset, self.split)
        
        self.batch_size = batch_size
        self.num_workers = num_workers

    def _shared_dataloader(self, dataset, shuffle=True):
        return DataLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=True, shuffle=shuffle)

    def train_dataloader(self):
        return self._shared_dataloader(self.data_train)

    def val_dataloader(self):
        return self._shared_dataloader(self.data_val)

    def test_dataloader(self):
        return self._shared_dataloader(self.data_test)

    def predict_dataloader(self):
        return self._shared_dataloader(self.dataset, shuffle=False)

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
        channels=None,
        eps=1e-8,
        factor=0.1,
    ):

        super().__init__()
        self.save_hyperparameters()

        self.lr = lr
        self.encoder = Encoder(nc, nf, ch_mult, imsize, latent_dim)
        self.decoder = Decoder(nc, nf, ch_mult[::-1], imsize, latent_dim)
        self.patience = patience
        self.channels = channels
        self.eps = eps
        self.factor = factor

    def reparameterized_sampling(self, mu, var):
        std = torch.exp(0.5 * torch.log(var))
        eps = torch.randn_like(mu)
        sample = eps.mul(std).add_(mu)
        return sample

    def forward(self, x):
        mu, var = self.encoder(x)
        z = self.reparameterized_sampling(mu, var)
        x_hat = self.decoder(z)
        return x_hat

    def forward_embedding(self, x):
        mu, var = self.encoder(x)
        return mu, var

    def _shared_step(self, batch):
        x = batch
        x_hat = self.forward(x)
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
                    self.log(f"validate/loss", loss[channel], on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
                else:
                    self.log(f"validate/loss_{channel}", loss[channel], on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        else:
            self.log("validate/loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
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

    def list_predict_modes(self):
        return ["forward", "embedding"]

    def set_predict_mode(self, mode):
        if mode not in self.list_predict_modes():
            raise ValueError(f"Predict mode {mode} not supported. Must be one of {self.list_predict_modes()}")
        self.predict_mode = mode

    def predict_step(self, batch, batch_idx):
        # returns embeddings w two channels, first is mu, second is logvar
        if self.predict_mode is None or self.predict_mode == "forward":
            return self.forward(batch)
        elif self.predict_mode == "embedding":
            return self.forward_embedding(batch)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        scheduler = ReduceLROnPlateau(
            optimizer,
            patience=self.patience,
            eps=self.eps,
            factor=self.factor,
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
        input_imgs = input_imgs.to(pl_module.device)
        with torch.no_grad():
            pl_module.eval()
            reconst_imgs = pl_module(input_imgs)
            pl_module.train()
        grid = ReconstructionVisualization.make_reconstruction_grid(input_imgs, reconst_imgs)
        grid = tensor_to_image(grid)
        grid = np.moveaxis(grid, -1, 0)
        rgb_grid, _, _, _ = multichannel_to_rgb(grid, cmaps=cmap)
        return rgb_grid, grid

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


class EmbeddingLogger(Callback):
    def __init__(self, num_images=200, every_n_epochs=5):
        super().__init__()
        self.num_images = num_images
        self.every_n_epochs = every_n_epochs

    def __x_logging_step(self, x, trainer, name):
        trainer.logger.experiment.log({
            f"{trainer.state.stage}/embedding_{name}_mean": x.mean()
        })
        trainer.logger.experiment.log({
            f"{trainer.state.stage}/embedding_{name}_min": x.min()
        })
        trainer.logger.experiment.log({
            f"{trainer.state.stage}/embedding_{name}_max": x.max()
        })

    def __shared_logging_step(self, input_imgs, pl_module, trainer):
        input_imgs = input_imgs.to(pl_module.device)
        with torch.no_grad():
            pl_module.eval()
            mu, var = pl_module.forward_embedding(input_imgs)
            pl_module.train()
        self.__x_logging_step(mu, trainer, "mu")
        self.__x_logging_step(var, trainer, "var")

    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch % self.every_n_epochs == 0:
            input_imgs = trainer.datamodule.train_dataloader().dataset[:self.num_images]
            self.__shared_logging_step(input_imgs, pl_module, trainer)

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch % self.every_n_epochs == 0:
            input_imgs = trainer.datamodule.val_dataloader().dataset[:self.num_images]
            self.__shared_logging_step(input_imgs, pl_module, trainer)
            
    def on_test_end(self, trainer, pl_module):
        input_imgs = trainer.datamodule.test_dataloader().dataset[:self.num_images]
        self.__shared_logging_step(input_imgs, pl_module, trainer)