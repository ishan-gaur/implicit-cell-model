import torch
from torch.autograd import Variable
from torch import nn
import lightning.pytorch as pl

class Encoder(nn.Module):
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
        # self.save_hyperparameters()

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
        # to retrieve mean and variance of predicted latent distribution
        # self.fc = nn.Sequential(
        #     nn.Dropout(0.5),
        #     nn.Linear(self.fc_input_size, 2 * self.latent_dim),
        #     nn.LeakyReLU(inplace=True),
        # )

        self.fc_mu = nn.Linear(self.fc_input_size, self.latent_dim)
        # self.fc_mu = nn.Sequential
        #     # nn.Dropout(0.5),
        #     # nn.BatchNorm1d(2 * self.latent_dim),
        #     nn.Linear(self.fc_input_size, self.latent_dim),
        # )

        self.fc_logvar = nn.Linear(self.fc_input_size, self.latent_dim)
        # self.fc_logvar = nn.Sequential(
        #     # nn.Dropout(0.5),
        #     # nn.BatchNorm1d(2 * self.latent_dim),
        #     nn.Linear(self.fc_input_size, self.latent_dim),
        #     # nn.ReLU()
        #     # nn.Softplus()
        # )
 
    def forward(self, x):
        # for i in range(len(self.layers)):
        #     x = self.layers[i](x)
        for layer in self.layers:
            x = layer(x)
        x = x.view(-1, self.fc_input_size)
        return self.fc_mu(x), self.fc_logvar(x)


class Decoder(nn.Module):
    def __init__(self, nc=1, nf=128, ch_mult=(8, 8, 8, 4, 2, 1), imsize=256, latent_dim=512):
        super().__init__()
        # self.save_hyperparameters()

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
                    nn.LeakyReLU(inplace=True)
                )
            )
            out_ch = in_ch

    def forward(self, z):
        z = self.fc(z)
        z = z.view(-1, self.nf * self.ch_mult[0], self.state_width, self.state_width)
        # for i in range(len(self.layers)):
        #     z = self.layers[i](z)
        for layer in self.layers:
            z = layer(z)
        return z