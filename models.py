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
        self.fc_mu = nn.Sequential(
            nn.Linear(self.fc_input_size, self.latent_dim),
            # nn.Sigmoid()
        )
        self.fc_var = nn.Sequential(
            nn.Linear(self.fc_input_size, self.latent_dim),
            nn.ReLU()
        )
 
    def forward(self, x):
        for i in range(len(self.layers)):
            x = self.layers[i](x)
        x = x.view(-1, self.fc_input_size)
        return self.fc_mu(x), self.fc_var(x)


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
        for i in range(len(self.layers)):
            z = self.layers[i](z)
        return z


class VAE(nn.Module):
    def __init__(self, nc=1, ngf=128, ndf=128, latent_variable_size=128, imsize=64, batchnorm=False):
        """
        x should be CHW
        nc: number of channels in input
        ngf: number of generator filters
        ndf: number of discriminator filters
        latent_variable_size: size of the latent space
        imsize: size of the input image (assumed square)
        batchnorm: use batch normalization
        """
        super().__init__()
 
        self.nc = nc
        self.ngf = ngf
        self.ndf = ndf
        self.imsize = imsize
        self.latent_variable_size = latent_variable_size
        self.batchnorm = batchnorm
 
        self.encoder = nn.Sequential(
            # input is 3 x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 2 x 2
        )
 
        self.fc1 = nn.Linear(ndf*8*2*2, latent_variable_size)
        self.fc2 = nn.Linear(ndf*8*2*2, latent_variable_size)
 
        # decoder
 
        self.d1 = nn.Sequential(
            nn.Linear(latent_variable_size, ngf*8*2*2),
            nn.ReLU(inplace=True),
        )

        self.decoder = nn.Sequential(
            # input is Z, going into a convolution
            # state size. (ngf*8) x 2 x 2
            nn.ConvTranspose2d(ngf * 8, ngf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2,     ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(    ngf,      nc, 4, 2, 1, bias=False),
            nn.Sigmoid(),
            # state size. (nc) x 64 x 64
        )
 
        self.bn_mean = nn.BatchNorm1d(latent_variable_size)
 
    def encode(self, x):
        h = self.encoder(x)
        h = h.view(-1, self.ndf*8*2*2)
        if self.batchnorm:
            return self.bn_mean(self.fc1(h)), self.fc2(h)
        else:
            return self.fc1(h), self.fc2(h)
 
    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        if torch.cuda.is_available():
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)
 
    def decode(self, z):
        h = self.d1(z)
        h = h.view(-1, self.ngf*8, 2, 2)
        return self.decoder(h)
 
    def get_latent_var(self, x):
        mu, logvar = self.encode(x.view(-1, self.nc, self.imsize, self.imsize))
        z = self.reparametrize(mu, logvar)
        return z
 
    def generate(self, z):
        res = self.decode(z)
        return res
 
    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, self.nc, self.imsize, self.imsize))
        z = self.reparametrize(mu, logvar)
        res = self.decode(z)
        return res, z, mu, logvar