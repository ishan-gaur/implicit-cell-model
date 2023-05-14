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

class FUCCIDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, dataset, batch_size, num_workers, imsize=1024, split=(0.64, 0.16, 0.2)):
        super().__init__()
        if not isinstance(data_dir, Path):
            data_dir = Path(data_dir)
        self.data_dir = data_dir

        self.imsize = imsize

        if dataset == "total":
            self.dataset = FUCCIDataset(self.data_dir, imsize=self.imsize)
        if dataset == "reference":
            self.dataset = ReferenceChannelDataset(self.data_dir, imsize=self.imsize)
        if dataset == "fucci":
            self.dataset = FUCCIChannelDataset(self.data_dir, imsize=self.imsize)

        self.split = split
        if len(self.split) != 3:
            raise ValueError("split must be a tuple of length 3")
        self.data_train, self.data_val, self.data_test = random_split(self.dataset, self.split)
        
        self.batch_size = batch_size
        self.num_workers = min(num_workers, self.batch_size)

    def train_dataloader(self):
        return DataLoader(self.data_train, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.data_val, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.data_test, batch_size=self.batch_size, num_workers=self.num_workers)


class AutoEncoder(pl.LightningModule):
    def __init__(self, encoder, decoder, lr=1e-6):
        super().__init__()
        self.save_hyperparameters(ignore=["encoder", "decoder"])

        self.lr = lr
        self.encoder = encoder
        self.decoder = decoder

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
        return loss, x_hat

    def training_step(self, batch, batch_idx):
        loss, x_hat = self._shared_step(batch)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, x_hat = self._shared_step(batch)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        loss, x_hat = self._shared_step(batch)
        self.log("test/loss", loss, sync_dist=True)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        scheduler = ReduceLROnPlateau(
            optimizer,
            patience=4,
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
        print(f"input: {input_imgs.shape}, reconst: {reconst_imgs.shape}")
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

    def make_reconstruction_grid(input_imgs, reconst_imgs):
        imgs = torch.stack([input_imgs, reconst_imgs], dim=1).flatten(0, 1)
        grid = make_grid(imgs, nrow=2, normalize=True, range=(-1, 1))
        return grid