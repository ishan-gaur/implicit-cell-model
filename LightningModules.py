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
import lightning.pytorch as pl
import numpy as np


from FUCCIDataset import FUCCIDataset, ReferenceChannelDataset, FUCCIChannelDataset
from FUCCIDataset import FUCCIDatasetInMemory, ReferenceChannelDatasetInMemory, FUCCIChannelDatasetInMemory, TotalDatasetInMemory
from models import Encoder, ImageEncoder, Decoder, ImageDecoder, MapperIn, MapperOut, Discriminator

class FUCCIDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, dataset, batch_size, num_workers, 
                 imsize=256, split=(0.64, 0.16, 0.2), in_memory=True,
                 permutation=None, deterministic=True):
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
        if deterministic:
<<<<<<< HEAD
            generator = torch.Generator().manual_seed(420)
=======
            generator = torch.Generator().manual_seed(42)
>>>>>>> pretrained model for fucci prediction and deterministic splits
            self.data_train, self.data_val, self.data_test = random_split(self.dataset, self.split, generator=generator)
        else:
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


class FUCCIModel(pl.LightningModule):
    def __init__(self,
        ae_checkpoint: Path,
        # nc: int = 1, # number of channels
        # nf: int = 128, # number of filters
        # ch_mult: Tuple[int, ...] = (1, 2, 4, 8, 8, 8), # multiple of filters for each layer, each layer downsamples by 2
        # imsize: int = 256,
        latent_dim: int = 512, # TODO this should be accessible from the autoencoder
        lr: float = 1e-6,
        map_widths: Tuple[int, ...] = (2, 2), # width multiplier for each layer
        train_all: bool = False,
        train_encoder: bool = False,
        train_decoder: bool = False,
    ):

        super().__init__()
        self.save_hyperparameters()
        self.eps = 1e-6 # just above e^-20

        ae = AutoEncoder.load_from_checkpoint(ae_checkpoint)
        self.encoder = ae.encoder
        self.decoder = ae.decoder
        self.ae_latent = latent_dim

        self.lr = lr

        self.map_widths = map_widths
        self.maps_in = nn.ModuleList()
        self.maps_out = nn.ModuleList()
        self.channels_in = ["DAPI", "gamma-tubulin"]
        self.channels_out = ["Geminin", "CDT1"]
        for _ in self.channels_in:
            self.maps_in.append(torch.compile(MapperIn(self.ae_latent, self.map_widths)))
        for _ in self.channels_out:
            self.maps_out.append(torch.compile(MapperOut(self.ae_latent, self.map_widths)))

        self.latent_dim = latent_dim * self.map_widths[-1]

        self.train_all = train_all
        self.train_encoder = train_encoder
        self.train_decoder = train_decoder

    def reparameterized_sampling(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def encode(self, x): # x is B x C(2) x H x W
        # predicted variance for the autoencoder checkpoint is 0
        image_mu, _ = self.encoder(x.reshape(-1, 1, x.shape[-2], x.shape[-1]))
        image_z = image_mu # B x latent_dim, channels are interleaved
        mus, logvars = [], []
        for c, map_in in enumerate(self.maps_in):
            mu, logvar = map_in(image_z.view(x.shape[0], len(self.channels_in), -1)[:, c])
            mus.append(mu)
            logvars.append(logvar)
        return mus, logvars # each is B x latent_dim, channels are separated

    def sample_average(self, mus, logvars): # mus and logvars are lists of B x latent_dim for each channel
        mus = torch.stack(mus, dim=1) # B x c x latent_dim
        mu = mus.mean(dim=1) # B x latent_dim
        vars = torch.stack(logvars, dim=1).exp()
        var = (vars + mus.pow(2)).mean(dim=1) - mus.mean(dim=1).pow(2)
        logvar = var.log()
        return mu, logvar
    
    def decode(self, z):
        image_zs = []
        for map_out in self.maps_out:
            image_zs.append(map_out(z)[:, None, ...])
        image_zs = torch.cat(image_zs, dim=1)
        image = self.decoder(image_zs.view(-1, image_zs.shape[-2], image_zs.shape[-1]))
        image = image.reshape(image_zs.shape[0], len(self.channels_out), image.shape[-2], image.shape[-1])
        return image

    def forward(self, x):
        # x is N x 4 x 256 x 256 (N four-channel images)
        # x = x[:, :2]
        mus, logvars = self.encode(x)
        mu_z, logvar_z = self.sample_average(mus, logvars)
        z = self.reparameterized_sampling(mu_z, logvar_z)
        x_hat = self.decode(z)
        return x_hat

    def forward_embedding(self, x):
        # x is N x 4 x 256 x 256 (N four-channel images)
        x = x[:, :2]
        mus, logvars = self.encode(x)
        mu_z, logvar_z = self.sample_average(mus, logvars)
        return mu_z, logvar_z

    def kl_loss(self, mu, logvar):
        var = torch.exp(logvar)
        covar = torch.diag_embed(var)
        return torch.sum(logvar) + torch.sum(torch.linalg.inv(covar)) + torch.sum(mu.pow(2))
        # return torch.sum(logvar) + torch.sum(torch.exp(-logvar + self.eps)) + torch.sum(mu.pow(2))

    def sigma_reconstruction_loss(self, x, x_hat):
        log_sigma = ((x_hat - x) ** 2).mean(list(range(len(x.shape))), keepdim=True).sqrt().log()
        min = np.log(self.eps) # just above e^-20 so min is -20
        log_sigma = min + F.softplus(log_sigma - min)
        gaussian_nll = 0.5 * torch.pow((x_hat - x) / log_sigma.exp(), 2) + log_sigma
        return gaussian_nll.sum()

    def reg_mse_loss(self, x, x_hat):
        x = x[:, 2:] # comparing only to the target channels
        return F.mse_loss(x, x_hat)

    def __shared_step(self, batch, stage):
        # batch is N x 4 x 256 x 256
        x = batch[:, :2]
        x_target = batch[:, 2:]

        mus, logvars = self.encode(x)
        mu, logvar = self.sample_average(mus, logvars)
        z = self.reparameterized_sampling(mu, logvar)
        x_hat = self.decode(z)

        loss_kl = self.kl_loss(mu, logvar)
        recon_loss = self.sigma_reconstruction_loss(x_target, x_hat)
        loss = recon_loss + loss_kl # optimal sigma VAE

        self.log(f'{stage}/kl_loss', loss_kl, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log(f'{stage}/recon_loss', recon_loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log(f'{stage}/loss', loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)

        return loss
    
    def configure_optimizers(self):
        # only want to train the maps
        for param in self.encoder.parameters():
            param.requires_grad = self.train_all or self.train_encoder
        for param in self.decoder.parameters():
            param.requires_grad = self.train_all or self.train_decoder

        parameters = []
        for c in range(len(self.channels_in)):
            parameters.extend(self.maps_in[c].parameters())
        for c in range(len(self.channels_out)):
            parameters.extend(self.maps_out[c].parameters())
        print("\tTraining input and output maps")
        if self.train_encoder or self.train_all:
            parameters.extend(self.encoder.parameters())
            print("\tTraining encoder")
        if self.train_decoder or self.train_all:
            parameters.extend(self.decoder.parameters())
            print("\tTraining decoder")
        optimizer = optim.Adam(parameters, lr=self.lr)

        return optimizer

    def training_step(self, batch, batch_idx):
        loss = self.__shared_step(batch, "train")
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss = self.__shared_step(batch, "validate")
        return loss
    
    def test_step(self, batch, batch_idx):
        loss = self.__shared_step(batch, "test")
        return loss

    def get_predict_modes(self):
        return ["embedding"]

    def set_predict_mode(self, mode):
        if mode not in self.get_predict_modes():
            raise ValueError(f"mode must be one of {self.get_predict_modes()}, got {mode}")
        self.predict_mode = mode

    def predict_step(self, batch, batch_idx):
        if self.predict_mode == "embedding":
            with torch.inference_mode(mode=False):
                return self.forward_embedding(batch)
        else:
            raise ValueError(f"mode must be one of {self.get_predict_modes()}, got {self.predict_mode}")


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
        lambda_kl: float = 0.5, # weight for kl divergence loss
        lambda_div: float = 0.5, # weight for the discriminator
        alt_training: bool = False, # whether to alternate training of the discriminator and the autoencoder
        alt_batch_ct: int = 4 * 4465, # number of batches to alternate over
        train_generator: bool = True,
    ):

        super().__init__()
        self.save_hyperparameters()
        self.eps = 1e-6 # just above e^-20

        self.encoder = torch.compile(ImageEncoder(nc, nf, ch_mult, imsize, latent_dim))
        self.decoder = torch.compile(ImageDecoder(nc, nf, ch_mult[::-1], imsize, latent_dim))
        self.ae_latent = latent_dim

        self.lr = lr
        self.lr_eps = lr_eps
        self.factor = factor
        self.patience = patience
        self.lambda_kl = lambda_kl
        self.lambda_div = lambda_div
        self.alt_training = alt_training
        self.alt_batch_ct = alt_batch_ct
        self.wait_periods = 0
        self.wait_threshold = 20
        self.train_generator = train_generator

        self.map_widths = map_widths
        self.maps_in = nn.ModuleList()
        self.maps_out = nn.ModuleList()
        self.channels = []
        for channel in channels:
            self.add_mapping(channel)

        self.latent_dim = latent_dim * self.map_widths[-1]

        self.discriminator = torch.compile(Discriminator(num_classes=len(self.channels), input_dim=self.latent_dim))
        # self.discriminator = None
        if self.discriminator is not None:
            self.automatic_optimization = False

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
        # var = torch.exp(logvar)
        # covar = torch.diag_embed(var)
        # return torch.sum(logvar) + torch.sum(torch.linalg.inv(covar)) + torch.sum(mu.pow(2))
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        # return torch.sum(logvar) + torch.sum(torch.exp(-logvar + self.eps)) + torch.sum(mu.pow(2))

    def sigma_reconstruction_loss(self, x, x_hat):
        log_sigma = ((x_hat - x) ** 2).mean(list(range(len(x.shape))), keepdim=True).sqrt().log()
        min = np.log(self.eps)
        log_sigma = min + F.softplus(log_sigma - min)
        gaussian_nll = 0.5 * torch.pow((x_hat - x) / log_sigma.exp(), 2) + log_sigma
        return gaussian_nll.sum()

    def reg_mse_loss(self, embeddings, batch):
        x_hat = torch.stack([self.decode(embeddings[i], i) for i in range(len(self.channels))])
        x_target = torch.stack([batch[channel] for channel in range(len(self.channels))])
        mse_loss = F.mse_loss(x_hat, x_target) # TODO: log by input-output channel pairs too and aggregate over epoch
        return mse_loss, x_hat, x_target

    def perm_mse_loss(self, embeddings, batch):
        output_channels = random.sample(range(len(self.channels)), len(self.channels))
        perm_x_hat = torch.stack([self.decode(embeddings[i], j) for i, j in enumerate(output_channels)])
        perm_x_target = torch.stack([batch[channel] for channel in output_channels])
        perm_loss = F.mse_loss(perm_x_hat, perm_x_target)
        return perm_loss, perm_x_hat, perm_x_target

    def impute_mse_loss(self, embeddings, batch):
        # select a random channel per image in batch
        output_channels = random.choices(range(len(self.channels)), batch.shape[1])
        # create a tensor that is the same shape as embeddings but with the selected channel zeroed out
        # use this to get the average embedding
        # decode this average embedding and compare to the original image
        raise NotImplementedError
    
    def discriminator_loss(self, embeddings):
        # embeddings is 4 x N x latent_dim
        flat_embeddings = torch.clone(embeddings).flatten(0, 1)
        target_channels = torch.tensor([[i for _ in range(embeddings.shape[1])] for i in range(len(self.channels))], device=self.device)
        target_channels = target_channels.flatten()
        predicted_channels = self.discriminator(flat_embeddings)
        loss = F.cross_entropy(predicted_channels, target_channels)
        return loss

    def __shared_opt_step(self, loss, optimizer, clip_val=0.5, retain_graph=False):
        self.toggle_optimizer(optimizer)
        optimizer.zero_grad()
        # self.manual_backward(loss, retain_graph=retain_graph)
        self.manual_backward(loss)
        self.clip_gradients(optimizer, gradient_clip_val=clip_val, gradient_clip_algorithm='norm')
        optimizer.step()
        self.untoggle_optimizer(optimizer)
    
    def __shared_step(self, batch, stage, batch_idx=None):
        # put all through encoding and calculate regularization term against prior
        # for each channel select a random output channel and calculate regularization loss
        embeddings = torch.zeros((batch.shape[1], batch.shape[0], self.latent_dim), device=self.device) # 4 x N x latent_dim
        opt_g, opt_d = self.optimizers()
        # batch is N x 4 x 256 x 256
        batch = batch.transpose(0, 1) # 4 x N x 256 x 256
        batch = batch[:, :, None, ...] # 4 x N x 1 x 256 x 256
        # loss_kl = 0
        for channel, channel_batch in enumerate(batch):
            mu, logvar = self.encode(channel_batch, channel)
            # kl = self.kl_loss(mu, logvar)
            # self.log(f'{stage}/kl_loss_{self.channels[channel]}', kl, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
            # loss_kl += kl
            embeddings[channel] = self.reparameterized_sampling(mu, logvar)

        _, x_hat, x_target = self.reg_mse_loss(embeddings, batch)
        # _, perm_x_hat, perm_x_target = self.perm_mse_loss(embeddings, batch)
        reg_recon_loss = self.sigma_reconstruction_loss(x_target, x_hat)
        # perm_recon_loss = self.sigma_reconstruction_loss(perm_x_target, perm_x_hat)
        discrim_loss = self.discriminator_loss(embeddings.detach())
        # self.impute_mse_loss(embeddings, batch)
        # loss = mse_loss + self.lambda_kl * loss_kl
        # loss = mse_loss + perm_loss + self.lambda_kl * loss_kl
        # loss = recon_loss + loss_kl
        # loss = recon_loss + loss_kl + perm_loss
        # loss = 0.5 * (reg_recon_loss + perm_recon_loss) + loss_kl - self.discriminator_loss(embeddings)
        # loss = (reg_recon_loss + loss_kl) - self.lambda_div * self.discriminator_loss(embeddings)
        loss = reg_recon_loss - self.lambda_div * self.discriminator_loss(embeddings)
        if stage == 'train':
            if self.alt_training:
                if ((batch_idx / self.alt_batch_ct) > self.wait_threshold and
                    (batch_idx // self.alt_batch_ct) % 2 == 0):
                    self.__shared_opt_step(reg_recon_loss, opt_d)
            else:
                self.__shared_opt_step(discrim_loss, opt_d)

            if self.alt_training:
                if ((batch_idx // self.alt_batch_ct) % 2 == 1 or
                    (batch_idx / self.alt_batch_ct) <= self.wait_threshold):
                    self.__shared_opt_step(loss, opt_g)
            else:
                self.__shared_opt_step(loss, opt_g)

        # self.log(f'{stage}/mse_loss', mse_loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        # self.log(f'{stage}/perm_loss', perm_recon_loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log(f'{stage}/recon_loss', reg_recon_loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log(f'{stage}/discrim_loss', discrim_loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log(f'{stage}/loss', loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)

        # if stage == "train":
        #     opt_g, opt_d = self.optimizers()
        #     # self.__shared_opt_step(loss, opt_g, retain_graph=True)
        #     self.__shared_opt_step(discrim_loss, opt_d)
        #     self.__shared_opt_step(loss, opt_g)

        return loss

    def training_step(self, batch, batch_idx):
        loss = self.__shared_step(batch, "train", batch_idx)
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss = self.__shared_step(batch, "validate")
        return loss
    
    def test_step(self, batch, batch_idx):
        loss = self.__shared_step(batch, "test")
        return loss

    def get_predict_modes(self):
        return ["embedding"]

    def set_predict_mode(self, mode):
        if mode not in self.get_predict_modes():
            raise ValueError(f"mode must be one of {self.get_predict_modes()}, got {mode}")
        self.predict_mode = mode

    def predict_step(self, batch, batch_idx):
        if self.predict_mode == "embedding":
            return self.forward_embedding(batch)
        else:
            raise ValueError(f"mode must be one of {self.get_predict_modes()}, got {self.predict_mode}")
    
    def configure_optimizers(self):
        for param in self.encoder.parameters():
            param.requires_grad = self.train_generator
        for param in self.decoder.parameters():
            param.requires_grad = self.train_generator
        parameters = []
        for i in range(len(self.channels)):
            parameters.extend(self.maps_in[i].parameters())
            parameters.extend(self.maps_out[i].parameters())
        generator_parameters = []
        if self.train_generator:
            generator_parameters += list(self.encoder.parameters()) + list(self.decoder.parameters())
        for c in range(len(self.channels)):
            generator_parameters += list(self.maps_in[c].parameters())
            generator_parameters += list(self.maps_out[c].parameters())
        opt_g = optim.Adam(generator_parameters, lr=lr)
        opt_d = optim.Adam(self.discriminator.parameters, lr=self.lr)
        return [opt_g, opt_d], []
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
