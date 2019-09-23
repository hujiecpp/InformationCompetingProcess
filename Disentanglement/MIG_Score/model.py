"""model.py"""

import torch
import torch.nn as nn
#import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable

def reparametrize(mu, logvar):
    std = logvar.div(2).exp()
    eps = Variable(std.data.new(std.size()).normal_())
    return mu + std*eps

class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)

class ICP_Encoder(nn.Module):
    def __init__(self, z_dim, y_dim, nc):
        super(ICP_Encoder, self).__init__()

        self.nc = nc
        self.z_dim = z_dim
        self.y_dim = y_dim

        sequence = [nn.Conv2d(nc, 32, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(64, 256, 4, 1),
            nn.ReLU(True),
            View((-1, 256*1*1))]

        sequence_z = sequence + [nn.Linear(256, z_dim * 2)]
        sequence_y = sequence + [nn.Linear(256, y_dim)]

        self.encoder_z = nn.Sequential(*sequence_z)
        self.encoder_y = nn.Sequential(*sequence_y)

    def forward(self, x):
        distributions = self.encoder_z(x)
        mu = distributions[:, : self.z_dim]
        logvar = distributions[:, self.z_dim : ]
        z = reparametrize(mu, logvar)
        y = self.encoder_y(x)

        return z, mu, logvar, y

class ICP_Decoder(nn.Module):
    def __init__(self, dim, nc):
        super(ICP_Decoder, self).__init__()

        self.nc = nc
        self.dim = dim

        self.decoder = nn.Sequential(
                nn.Linear(dim, 256),
                View((-1, 256, 1, 1)),
                nn.ReLU(True),
                nn.ConvTranspose2d(256, 64, 4),
                nn.ReLU(True),
                nn.ConvTranspose2d(64, 64, 4, 2, 1),
                nn.ReLU(True),
                nn.ConvTranspose2d(64, 32, 4, 2, 1),
                nn.ReLU(True),
                nn.ConvTranspose2d(32, 32, 4, 2, 1),
                nn.ReLU(True),
                nn.ConvTranspose2d(32, nc, 4, 2, 1)
            )

    def forward(self, zy):

        x_rec = self.decoder(zy)

        return x_rec

class Discriminator(nn.Module):
    def __init__(self, y_dim):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(y_dim * 2, y_dim),
            nn.LeakyReLU(0.2, True),
            nn.Linear(y_dim, y_dim),
            nn.LeakyReLU(0.2, True),
            nn.Linear(y_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, y):
        return self.net(y).squeeze()

class MLP(nn.Module):
    def __init__(self, s_dim, t_dim):
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(s_dim, s_dim),
            nn.LeakyReLU(0.2, True),
            nn.Linear(s_dim, t_dim),
            nn.LeakyReLU(0.2, True),
            nn.Linear(t_dim, t_dim),
            nn.ReLU()
        )

    def forward(self, s):
        t = self.net(s)
        return t
