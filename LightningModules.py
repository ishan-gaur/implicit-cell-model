import random
from typing import List, Tuple
from pathlib import Path
from lightning.pytorch.utilities.types import EVAL_DATALOADERS

import torch
from torch import optim
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision.utils import make_grid

import lightning.pytorch as pl
from lightning.pytorch.callbacks import Callback

from kornia import tensor_to_image
from kornia.losses import ssim_loss
from microfilm.colorify import multichannel_to_rgb
import numpy as np
import wandb

from FUCCIDataset import FUCCIDataset, ReferenceChannelDataset, FUCCIChannelDataset
from FUCCIDataset import FUCCIDatasetInMemory, ReferenceChannelDatasetInMemory, FUCCIChannelDatasetInMemory, TotalDatasetInMemory
from models import Encoder, ImageEncoder, Decoder, ImageDecoder, MapperIn, MapperOut

class FUCCIDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, dataset, batch_size, num_workers, 
                 imsize=256, split=(0.64, 0.16, 0.2), in_memory=True,
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
                raise NotImplementedError("total dataset not implemented")
            if dataset == "reference":
                self.dataset = ReferenceChannelDataset(self.data_dir, imsize=self.imsize, transform=transform)
            if dataset == "fucci":
                self.dataset = FUCCIChannelDataset(self.data_dir, imsize=self.imsize, transform=transform)
            if dataset == "all":
                self.dataset = FUCCIDataset(self.data_dir, imsize=self.imsize, transform=transform)
        else:
            if dataset == "total":
                self.dataset = TotalDatasetInMemory(self.data_dir, imsize=self.imsize, transform=transform)
            if dataset == "reference":
                self.dataset = ReferenceChannelDatasetInMemory(self.data_dir, imsize=self.imsize, transform=transform)
            if dataset == "fucci":
                self.dataset = FUCCIChannelDatasetInMemory(self.data_dir, imsize=self.imsize, transform=transform)
            if dataset == "all":
                self.dataset = FUCCIDatasetInMemory(self.data_dir, imsize=self.imsize, transform=transform)
                

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
        return self._shared_dataloader(self.data_val, shuffle=False)

    def test_dataloader(self):
        return self._shared_dataloader(self.data_test, shuffle=False)

    def predict_dataloader(self):
        return self._shared_dataloader(self.dataset, shuffle=False)

    def get_channels(self):
        return self.dataset.get_channel_names()
    
    def channel_colors(self):
        return self.dataset.channel_colors()


class Bundle(nn.Module):
    def __init__(self, maps: List[Tuple[nn.Module, nn.Module]]):
        super().__init__()
        self.maps = maps


class MultiModalAutoencoder(pl.LightningModule):
    def __init__(self,
        nc: int = 1, # number of channels
        nf: int = 128, # number of filters
        ch_mult: Tuple[int, ...] = (1, 2, 4, 8, 8, 8), # multiple of filters for each layer, each layer downsamples by 2
        imsize: int = 256,
        latent_dim: int = 512,
        lr: float = 1e-6,
        lr_eps: float = 1e-10, # minimum change in learning rate allowed (no update after this)
        factor: float = 0.5, # factor to reduce lr by
        patience: int = 4, # number of epochs to wait before reducing lr
        channels: List[str] = [], # names of the channels for each mapping
        map_widths: Tuple[int, ...] = (2, 2), # width multiplier for each layer
        lambda_kl: float = 1.0, # weight for kl divergence loss
    ):

        super().__init__()
        self.save_hyperparameters()

        self.encoder = torch.compile(ImageEncoder(nc, nf, ch_mult, imsize, latent_dim))
        self.decoder = torch.compile(ImageDecoder(nc, nf, ch_mult[::-1], imsize, latent_dim))
        self.ae_latent = latent_dim

        self.lr = lr
        self.lr_eps = lr_eps
        self.factor = factor
        self.patience = patience
        self.lambda_kl = lambda_kl

        self.map_widths = map_widths
        self.maps_in = nn.ModuleList()
        self.maps_out = nn.ModuleList()
        self.channels = []
        for channel in channels:
            self.add_mapping(channel)

        self.latent_dim = latent_dim * self.map_widths[-1]

    def add_mapping(self, channel):
        print(f"adding channel {channel}")
        self.channels.append(channel)
        self.maps_in.append(torch.compile(MapperIn(self.ae_latent, self.map_widths)))
        self.maps_out.append(torch.compile(MapperOut(self.ae_latent, self.map_widths)))

    def reparameterized_sampling(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def encode(self, x, channel_idx):
        if channel_idx >= len(self.channels):
            raise ValueError(f"channel_idx {channel_idx} is out of range. Please add_mapping or check request.")
        image_z = self.encoder(x)
        mu, logvar = self.maps_in[channel_idx](image_z)
        return mu, logvar
    
    def decode(self, z, channel_idx):
        if channel_idx >= len(self.channels):
            raise ValueError(f"channel_idx {channel_idx} is out of range. Please add_mapping or check request.")
        image_z = self.maps_out[channel_idx](z)
        image = self.decoder(image_z)
        return image

    def forward(self, x):
        # x is N x 4 x 256 x 256 (N four-channel images)
        x = x.transpose(0, 1) # 4 x N x 256 x 256
        x = x[:, :, None, ...] # 4 x N x 1 x 256 x 256
        x_hat = torch.zeros_like(x)
        for channel_idx in range(len(self.channels)):
            mu, logvar = self.encode(x[channel_idx], channel_idx)
            z = self.reparameterized_sampling(mu, logvar)
            x_hat[channel_idx] = self.decode(z, channel_idx)
        x_hat = x_hat.transpose(0, 1).squeeze() # N x 4 x 256 x 256
        return x_hat

    def forward_embedding(self, x):
        # x is N x 4 x 256 x 256 (N four-channel images)
        x = x.transpose(0, 1)
        x = x[:, :, None, ...]
        mu, logvar = [], []
        for channel_idx in range(len(self.channels)):
            m, l = self.encode(x[channel_idx], channel_idx)
            mu.append(m)
            logvar.append(l)
        return mu, logvar

    def kl_loss(self, mu, logvar):
        var = torch.exp(logvar)
        covar = torch.diag_embed(var)
        return torch.sum(logvar) + torch.sum(torch.linalg.inv(covar)) + torch.sum(mu.pow(2))
    
    def __shared_step(self, batch, stage):
        # put all through encoding and calculate regularization term against prior
        # for each channel select a random output channel and calculate regularization loss
        embeddings = torch.zeros((batch.shape[1], batch.shape[0], self.latent_dim), device=self.device) # 4 x N x latent_dim
        # batch is N x 4 x 256 x 256
        batch = batch.transpose(0, 1) # 4 x N x 256 x 256
        batch = batch[:, :, None, ...] # 4 x N x 1 x 256 x 256
        loss_kl = 0
        for channel, channel_batch in enumerate(batch):
            mu, logvar = self.encode(channel_batch, channel)
            kl = self.kl_loss(mu, logvar)
            self.log(f'{stage}/kl_loss_{channel}', kl, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
            loss_kl += kl
            embeddings[channel] = self.reparameterized_sampling(mu, logvar)

        output_channels = random.sample(range(len(self.channels)), len(self.channels))
        x_hat = torch.stack([self.decode(embeddings[i], j) for i, j in enumerate(output_channels)])

        x_target = torch.stack([batch[channel] for channel in output_channels])
        mse_loss = F.mse_loss(x_hat, x_target) # TODO: log by input-output channel pairs too and aggregate over epoch
        loss = mse_loss + self.lambda_kl * loss_kl

        self.log(f'{stage}/mse_loss', mse_loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log(f'{stage}/loss', loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.__shared_step(batch, "train")
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss = self.__shared_step(batch, "validate")
        return loss
    
    def test_step(self, batch, batch_idx):
        loss = self.__shared_step(batch, "test")
        return loss
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
    #     scheduler = ReduceLROnPlateau(
    #         optimizer,
    #         patience=self.patience,
    #         eps=self.lr_eps,
    #         factor=self.factor,
    #     )
    #     return {
    #         "optimizer": optimizer,
    #         "lr_scheduler": scheduler,
    #         "monitor": "train/loss"
    #     }

    # def lr_scheduler_step(self, scheduler, metric):
    #     scheduler.step(metric)
        # self.log("lr", scheduler._last_lr[0], on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)


class CrossModalAutoencoder(pl.LightningModule):
    def __init__(self,
        nc: int = 1, # number of channels
        nf: int = 128, # number of filters
        ch_mult: Tuple[int, ...] = (1, 2, 4, 8, 8, 8), # multiple of filters for each layer, each layer downsamples by 2
        imsize: int = 256,
        latent_dim: int = 512,
        lr: float = 1e-6,
        lr_eps: float = 1e-10, # minimum change in learning rate allowed (no update after this)
        patience: int = 4, # number of epochs to wait before reducing lr
        channels: List[str] = [], # names of the channels for each mapping
        map_widths: Tuple[int, ...] = (2, 2), # width multiplier for each layer
    ):

        super().__init__()
        self.save_hyperparameters()

        # self.encoder = Encoder(nc, nf, ch_mult, imsize, latent_dim, estimate_var=False)
        self.encoder = Encoder(nc, nf, ch_mult, imsize, latent_dim)
        self.decoder = Decoder(nc, nf, ch_mult[::-1], imsize, latent_dim)
        self.ae_latent = latent_dim

        self.lr = lr
        self.lr_eps = lr_eps
        self.patience = patience

        self.map_widths = map_widths
        self.maps = {}
        self.channels = []
        for channel in channels:
            self.add_mapping(channel)

    def add_mapping(self, channel):
        print(f"adding channel {channel}")
        self.channels.append(channel)
        self.maps[channel] = {"encode": MapperIn(self.ae_latent, self.map_widths),
                                "decode": MapperOut(self.ae_latent, self.map_widths)}

    def reparameterized_sampling(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def encode(self, x, channel):
        if channel not in self.channels:
            raise ValueError(f"channel {channel} not in {self.channels}. Please add_mapping or check spelling.")
        # image_z = self.encoder(x)
        image_z, _ = self.encoder(x)
        mu, logvar = self.maps[channel]["encode"](image_z)
        return mu, logvar
    
    def decode(self, z, channel):
        if channel not in self.channels:
            raise ValueError(f"channel {channel} not in {self.channels}. Please add_mapping or check spelling.")
        image_z = self.maps[channel]["decode"](z)
        image = self.decoder(image_z)
        return image

    def forward(self, x):
        x_hat = {}
        for channel, images in x.items():
            mu, logvar = self.encode(images, channel)
            z = self.reparameterized_sampling(mu, logvar)
            x_hat[channel] = self.decode(z, channel)
        return x_hat
    
    def __shared_step(self, batch, stage):
        # put all through encoding and calculate regularization term against prior
        # for each channel select a random output channel and calculate regularization loss
        embeddings = {}
        for channel, images in batch.items():
            mu, logvar = self.encode(images, channel)
            embeddings[channel] = self.reparameterized_sampling(mu, logvar)
        output_channels = random.sample(self.channels, len(self.channels))
        x_hat = {channel: self.decode(embeddings[channel], output_channels[i]) for i, channel in enumerate(self.channels)}
        x_target = {channel: batch[channel] for channel in output_channels}
        x_hat = torch.cat([x_hat[channel] for channel in self.channels], dim=1)
        x_target = torch.cat([x_target[channel] for channel in self.channels], dim=1)
        loss = F.mse_loss(x_hat, x_target)
        self.log(f"{stage}/loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.__shared_step(batch, "train")
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss = self.__shared_step(batch, "val")
        return loss
    
    def test_step(self, batch, batch_idx):
        loss = self.__shared_step(batch, "test")
        return loss
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)


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
        lambda_kl=0.1,
    ):

        super().__init__()
        self.save_hyperparameters()

        self.lr = lr
        self.encoder = torch.compile(Encoder(nc, nf, ch_mult, imsize, latent_dim))
        self.decoder = torch.compile(Decoder(nc, nf, ch_mult[::-1], imsize, latent_dim))
        self.patience = patience
        self.channels = channels
        self.eps = eps
        self.factor = factor
        self.lambda_kl = lambda_kl

    def reparameterized_sampling(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(mu)
        if torch.isnan(std).any():
            print("std is nan")
            std = torch.clamp(std, min=self.eps, max=1e8)
        sample = eps.mul(std).add_(mu)
        return sample

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterized_sampling(mu, logvar)
        x_hat = self.decoder(z)
        return x_hat

    def forward_embedding(self, x):
        mu, logvar = self.encoder(x)
        return mu, logvar

    def forward_decoding(self, z_batch):
        return self.decoder(z_batch)

    def loss_function(self, x, x_hat, mu, logvar):
        mse_loss = F.mse_loss(x_hat, x)
        # var = torch.clamp(torch.exp(logvar), min=self.eps, max=1e8)
        # kl_loss = torch.sum(logvar) + torch.sum(torch.inverse(covar)) + torch.sum(mu.pow(2))
        # var = torch.exp(logvar)
        # covar = torch.diag_embed(var)
        # kl_loss = torch.sum(logvar) + torch.sum(torch.linalg.inv(covar)) + torch.sum(mu.pow(2))
        # loss = mse_loss + self.lambda_kl * kl_loss
        loss = mse_loss
        # return loss, mse_loss, kl_loss
        return loss, mse_loss, 0


    def _shared_step(self, batch):
        x = batch
        mu, logvar = self.encoder(x)
        z = self.reparameterized_sampling(mu, logvar)
        x_hat = self.decoder(z)
        loss, mse_loss, kl_loss = self.loss_function(x, x_hat, mu, logvar)
        loss_dict = {"total": loss, "mse": mse_loss, "kl": kl_loss}
        if self.channels is not None:
            for i, channel in enumerate(self.channels):
                loss_dict[channel] = F.mse_loss(x_hat[:,i,:,:], x[:,i,:,:])
        loss = loss_dict
        return loss, x_hat

    def _shared_logging(self, loss, stage):
        if isinstance(loss, dict):
            for channel in loss:
                if channel == "total":
                    self.log(f"{stage}/loss", loss[channel], on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
                else:
                    self.log(f"{stage}/loss_{channel}", loss[channel], on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        else:
            self.log(f"{stage}/loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)

    def training_step(self, batch, batch_idx):
        loss, x_hat = self._shared_step(batch)
        self._shared_logging(loss, "train")
        return loss["total"] if isinstance(loss, dict) else loss

    def validation_step(self, batch, batch_idx):
        loss, x_hat = self._shared_step(batch)
        self._shared_logging(loss, "validate")
        return loss["total"] if isinstance(loss, dict) else loss
    
    def test_step(self, batch, batch_idx):
        loss, x_hat = self._shared_step(batch)
        self._shared_logging(loss, "test")
        return loss["total"] if isinstance(loss, dict) else loss

    def list_predict_modes(self):
        return ["forward", "embedding", "sampling"]

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
        elif self.predict_mode == "sampling":
            return self.forward_decoding(batch)

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
        # self.log("lr", scheduler._last_lr[0], on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)


class ReconstructionVisualization(Callback):
    def __init__(self, num_images=8, every_n_epochs=5, channels=None, mode="single"):
        super().__init__()
        self.num_images = num_images
        self.every_n_epochs = every_n_epochs
        self.channels = channels
        self.mode = mode

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

    @staticmethod
    def image_list_normalization(image_list):
        for i in range(len(image_list)):
            image_list[i] = image_list[i] - torch.min(image_list[i])
            image_list[i] = image_list[i] / (torch.max(image_list[i]) - torch.min(image_list[i]))
            image_list[i] = image_list[i] * 2 - 1
        return image_list

    @staticmethod
    def make_reconstruction_grid(input_imgs, reconst_imgs):
        imgs = torch.stack([input_imgs, reconst_imgs], dim=1).flatten(0, 1)
        grid = make_grid(imgs, nrow=2, normalize=True)
        return grid

    def __shared_logging_step(self, input_imgs, pl_module, cmap, trainer, channel=""):
        rgb_grid, grid = self.__shared_reconstruction_step(input_imgs, pl_module, cmap)
        if self.mode == "single":
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

    def __shared_logging_dispatch(self, dataloader, pl_module, trainer):
        if self.mode == "multi":
            for channel, data in dataloader.iterables.items():
                input_imgs = data[:self.num_images]
                cmap = None
                self.__shared_logging_step(input_imgs, pl_module, cmap, trainer, channel)
        else:
            input_imgs = dataloader.dataset[:self.num_images]
            cmap = trainer.datamodule.channel_colors() if input_imgs.shape[-3] > 1 else None
            self.__shared_logging_step(input_imgs, pl_module, cmap, trainer)

    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch % self.every_n_epochs == 0:
            self.__shared_logging_dispatch(trainer.datamodule.train_dataloader(), pl_module, trainer)

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch % self.every_n_epochs == 0:
            self.__shared_logging_dispatch(trainer.datamodule.val_dataloader(), pl_module, trainer)
    def on_test_end(self, trainer, pl_module):
        self.__shared_logging_dispatch(trainer.datamodule.test_dataloader(), pl_module, trainer)

class EmbeddingLogger(Callback):
    def __init__(self, num_images=200, every_n_epochs=5, mode="single", channels=[]):
        super().__init__()
        self.num_images = num_images
        self.every_n_epochs = every_n_epochs
        self.mode = mode
        self.channels = channels

    def __x_logging_step(self, x, trainer, name, channel=""):
        trainer.logger.experiment.log({
            f"{trainer.state.stage}/{channel}_embedding_{name}_mean": x.mean()
        })
        trainer.logger.experiment.log({
            f"{trainer.state.stage}/{channel}_embedding_{name}_min": x.min()
        })
        trainer.logger.experiment.log({
            f"{trainer.state.stage}/{channel}_embedding_{name}_max": x.max()
        })

    def __shared_logging_step(self, input_imgs, pl_module, trainer, channel=""):
        input_imgs = input_imgs.to(pl_module.device)
        with torch.no_grad():
            pl_module.eval()
            mu, logvar = pl_module.forward_embedding(input_imgs)
            pl_module.train()
        return mu, logvar

    def __shared_logging_dispatch(self, dataloader, pl_module, trainer):
        if self.mode == "multi":
            for channel, data in dataloader.iterables.items():
                input_imgs = data[:self.num_images]
                mu, logvar = self.__shared_logging_step(input_imgs, pl_module, trainer, channel)
                self.__x_logging_step(mu, trainer, "mu", channel)
                self.__x_logging_step(logvar, trainer, "logvar", channel)
        elif self.mode == "all":
            input_imgs = dataloader.dataset[:self.num_images]
            mu, logvar = self.__shared_logging_step(input_imgs, pl_module, trainer)
            for m, l, c in zip(mu, logvar, self.channels):
                self.__x_logging_step(m, trainer, "mu", c)
                self.__x_logging_step(l, trainer, "logvar", c)
        else:
            input_imgs = dataloader.dataset[:self.num_images]
            mu, logvar = self.__shared_logging_step(input_imgs, pl_module, trainer)
            self.__x_logging_step(mu, trainer, "mu")
            self.__x_logging_step(logvar, trainer, "logvar")

    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch % self.every_n_epochs == 0:
            self.__shared_logging_dispatch(trainer.datamodule.train_dataloader(), pl_module, trainer)

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch % self.every_n_epochs == 0:
            self.__shared_logging_dispatch(trainer.datamodule.val_dataloader(), pl_module, trainer)
            
    def on_test_end(self, trainer, pl_module):
        self.__shared_logging_dispatch(trainer.datamodule.test_dataloader(), pl_module, trainer)