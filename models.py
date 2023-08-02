from typing import Tuple
import torch
from torch.autograd import Variable
from torch import nn
import lightning.pytorch as pl


class MapperIn(nn.Module):
    def __init__(self, 
        input_dim: int = 512, 
        width_mult: Tuple[int, ...] = (2, 2)
    ):
        super().__init__()
        self.layers = nn.ModuleList()
        in_dim = input_dim
        for i in range(len(width_mult)):
            out_dim = int(input_dim * width_mult[i])
            self.layers.append(torch.nn.Linear(in_dim, out_dim))
            if i != len(width_mult) - 1:
                self.layers.append(torch.nn.LeakyReLU(inplace=True))
            in_dim = out_dim

        latent_dim = input_dim * width_mult[-1]
        self.fc_mu = torch.nn.Linear(latent_dim, latent_dim)
        self.fc_logvar = torch.nn.Linear(latent_dim, latent_dim)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.fc_mu(x), self.fc_logvar(x)


class MapperOut(nn.Module):
    def __init__(self,
        input_dim: int = 512,
        width_mult: Tuple[int, ...] = (2, 2)
    ):
        super().__init__()
        self.layers = nn.ModuleList()
        out_dim = input_dim
        for i in range(len(width_mult) - 1, -1, -1):
            in_dim = int(input_dim * width_mult[i])
            self.layers.insert(0, torch.nn.Linear(in_dim, out_dim))
            if i != 0:
                self.layers.insert(0, torch.nn.LeakyReLU(inplace=True))
            else:
                self.layers.insert(0, torch.nn.Sigmoid())
            out_dim = in_dim

    def forward(self, z):
        for layer in self.layers:
            z = layer(z)
        return z


class Discriminator(nn.Module):
    def __init__(self,
        num_classes: int = 2,
        input_dim: int = 512,
    ):
        super().__init__()
        assert num_classes > 1
        self.num_classes = num_classes
        self.layers = nn.ModuleList()
        self.layers.append(torch.nn.Linear(input_dim, input_dim))
        self.layers.append(torch.nn.LeakyReLU(0.2, inplace=True))
        self.layers.append(torch.nn.Linear(input_dim, input_dim))
        self.layers.append(torch.nn.LeakyReLU(0.2, inplace=True))
        self.layers.append(torch.nn.Linear(input_dim, self.num_classes))
        self.softmax = torch.nn.Softmax(dim=1)
        self.cross_entropy = torch.nn.CrossEntropyLoss()

    def forward(self, z):
        for layer in self.layers:
            z = layer(z)
        return self.softmax(z)

    def discriminator_loss(self, inputs, targets):
        preds = self.forward(inputs)
        return self.cross_entropy(preds, targets)

    def generator_loss(self, embeddings):
        preds = self.forward(embeddings)
        return self.cross_entropy(preds, torch.ones_like(preds) / self.num_classes)


# to predict pseudotime from latent space
# note that the model is quite limited, this is to encourage a latent space
# where the desired feature is obviously engineered in
class PseudoTimeRegressor(nn.Module):
    def __init__(self,
        input_dim: int = 512,
        output_dim: int = 1
    ):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(torch.nn.Linear(input_dim, output_dim))
        self.layers.append(torch.nn.Sigmoid())
    
    def forward(self, z):
        for layer in self.layers:
            z = layer(z)
        return z

    def loss(self, inputs, targets):
        preds = self.forward(inputs)
        return torch.nn.MSELoss(preds, targets)


class Encoder(nn.Module):
    # def __init__(self, nc=1, nf=128, ch_mult=(1, 2, 4, 8, 8, 8), imsize=256, latent_dim=512, estimate_var=True):
    def __init__(self, nc=1, nf=128, ch_mult=(1, 2, 4, 8, 8, 8), imsize=256, latent_dim=512):
        """
        x should be CHW
        nc: number of channels in input
        nf: number of discriminator filters per input channel
        imsize: size of the input image (assumed square)
        latent_variable_size: size of the latent space
        batchnorm: use batch normalization
        """
        super().__init__()

        if imsize < 2 ** len(ch_mult):
            raise ValueError("Image size not large enough to accommodate the number of downsampling layers: len(ch_mult).")

        if latent_dim > imsize ** 2:
            raise ValueError("Latent dimension larger than the number of pixels in the image.")
 
        self.nc = nc
        self.nf = nf * nc
        self.latent_dim = latent_dim
        self.ch_mult = ch_mult
 
        self.layers = nn.ModuleList()
        in_ch = nc
        for depth in range(len(ch_mult)):
            out_ch = self.nf * self.ch_mult[depth]
            self.layers.append(
                nn.Sequential(
                    nn.BatchNorm2d(in_ch),
                    nn.Conv2d(in_ch, out_ch, 4, 2, 1),
                    nn.LeakyReLU(0.2, inplace=True)
                ) 
            )
            in_ch = out_ch


        state_width = imsize // (2 ** len(self.ch_mult))
        state_area = state_width ** 2
        self.fc_input_size = self.nf * ch_mult[-1] * state_area

        self.fc_mu = nn.Linear(self.fc_input_size, self.latent_dim)
        # if estimate_var:
        self.fc_logvar = nn.Linear(self.fc_input_size, self.latent_dim)
 
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = x.view(-1, self.fc_input_size)
        return self.fc_mu(x), self.fc_logvar(x)


class ImageEncoder(nn.Module):
    # def __init__(self, nc=1, nf=128, ch_mult=(1, 2, 4, 8, 8, 8), imsize=256, latent_dim=512, estimate_var=True):
    def __init__(self, nc=1, nf=128, ch_mult=(1, 2, 4, 8, 8, 8), imsize=256, latent_dim=512):
        """
        x should be CHW
        nc: number of channels in input
        nf: number of discriminator filters per input channel
        imsize: size of the input image (assumed square)
        latent_variable_size: size of the latent space
        batchnorm: use batch normalization
        """
        super().__init__()

        if imsize < 2 ** len(ch_mult):
            raise ValueError("Image size not large enough to accommodate the number of downsampling layers: len(ch_mult).")

        if latent_dim > imsize ** 2:
            raise ValueError("Latent dimension larger than the number of pixels in the image.")
 
        self.nc = nc
        self.nf = nf * nc
        self.latent_dim = latent_dim
        self.ch_mult = ch_mult
 
        self.layers = nn.ModuleList()
        in_ch = nc
        for depth in range(len(ch_mult)):
            out_ch = self.nf * self.ch_mult[depth]
            self.layers.append(
                nn.Sequential(
                    nn.BatchNorm2d(in_ch),
                    nn.Conv2d(in_ch, out_ch, 4, 2, 1),
                    nn.LeakyReLU(inplace=True)
                ) 
            )
            in_ch = out_ch


        state_width = imsize // (2 ** len(self.ch_mult))
        state_area = state_width ** 2
        self.fc_input_size = self.nf * ch_mult[-1] * state_area

        self.fc_mu = nn.Linear(self.fc_input_size, self.latent_dim)
 
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = x.reshape((-1, self.fc_input_size))
        return self.fc_mu(x)


class Decoder(nn.Module):
    def __init__(self, nc=1, nf=128, ch_mult=(8, 8, 8, 4, 2, 1), imsize=256, latent_dim=512):
        super().__init__()

        if imsize < 2 ** len(ch_mult):
            raise ValueError("Image size not large enough to accommodate the number of downsampling layers: len(ch_mult).")

        if latent_dim > imsize ** 2:
            raise ValueError("Latent dimension larger than the number of pixels in the image.")
 
        self.nc = nc
        self.nf = nf * nc
        self.latent_dim = latent_dim
        self.ch_mult = ch_mult

        self.state_width = imsize // (2 ** len(ch_mult))
        state_area = self.state_width ** 2
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, self.nf * ch_mult[0] * state_area),
            nn.LeakyReLU(inplace=True)
        )

        self.layers = nn.ModuleList()
        out_ch = nc
        for depth in range(len(ch_mult), 0, -1):
            in_ch = self.nf * ch_mult[depth - 1]
            self.layers.insert(0,
                nn.Sequential(
                    nn.BatchNorm2d(in_ch),
                    nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1),
                    nn.LeakyReLU(0.2, inplace=True) if depth > 1 else nn.Tanh()
                )
            )
            out_ch = in_ch

    def forward(self, z):
        z = self.fc(z)
        z = z.view(-1, self.nf * self.ch_mult[0], self.state_width, self.state_width)
        for layer in self.layers:
            z = layer(z)
        return z


class ImageDecoder(nn.Module):
    def __init__(self, nc=1, nf=128, ch_mult=(8, 8, 8, 4, 2, 1), imsize=256, latent_dim=512):
        super().__init__()

        if imsize < 2 ** len(ch_mult):
            raise ValueError("Image size not large enough to accommodate the number of downsampling layers: len(ch_mult).")

        if latent_dim > imsize ** 2:
            raise ValueError("Latent dimension larger than the number of pixels in the image.")
 
        self.nc = nc
        self.nf = nf * nc
        self.latent_dim = latent_dim
        self.ch_mult = ch_mult

        self.state_width = imsize // (2 ** len(ch_mult))
        state_area = self.state_width ** 2
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, self.nf * ch_mult[0] * state_area),
            nn.LeakyReLU(inplace=True)
        )

        self.layers = nn.ModuleList()
        out_ch = nc
        for depth in range(len(ch_mult), 0, -1):
            in_ch = self.nf * ch_mult[depth - 1]
            self.layers.insert(0,
                nn.Sequential(
                    nn.BatchNorm2d(in_ch),
                    nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1),
                    nn.LeakyReLU(inplace=True) if depth > 1 else nn.Tanh()
                )
            )
            out_ch = in_ch

    def forward(self, z):
        z = self.fc(z)
        z = z.view(-1, self.nf * self.ch_mult[0], self.state_width, self.state_width)
        for layer in self.layers:
            z = layer(z)
        return z


# class SigmaAutoencoder(nn.Module):