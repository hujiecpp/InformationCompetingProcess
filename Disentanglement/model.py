import numpy as np
import torch.nn as nn
import torch.nn.init as init

def reparametrize(mu, logvar):
    std = logvar.div(2).exp()
    eps = std.data.new(std.size()).normal_()
    return mu + std*eps


class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)


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

# -------------- For dSprites, 3D faces datasets --------------
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

# -------------- For CelebA dataset --------------
class ICP_Encoder_Big(nn.Module):
    def __init__(self, z_dim, y_dim, nc, img_dim = 128):
        super(ICP_Encoder_Big, self).__init__()
        
        self.img_dim = img_dim
        self.nc = nc
        self.z_dim = z_dim
        self.y_dim = y_dim
        self.layer_num = int(np.log2(img_dim)) - 3
        self.max_channel_num = img_dim * 8
        self.f_num = 4

        # Encoder
        sequence = []
        for i in range(self.layer_num + 1):
            now_channel_num = self.max_channel_num // 2**(self.layer_num - i)

            if i == 0:
                sequence += [nn.Conv2d(in_channels = self.nc, out_channels = now_channel_num, 
                                        kernel_size = 4, stride = 2, padding = 1),]
            else:
                sequence += [nn.Conv2d(in_channels = pre_channel_num, out_channels = now_channel_num, 
                                        kernel_size = 4, stride = 2, padding = 1),
                             nn.BatchNorm2d(now_channel_num),]
            sequence += [nn.LeakyReLU(0.2, True)]

            pre_channel_num = now_channel_num

        sequence_z = sequence + [View((-1, self.f_num * self.f_num * self.max_channel_num)),
                     nn.Linear(self.f_num * self.f_num * self.max_channel_num, self.z_dim * 2)]

        sequence_y = sequence + [View((-1, self.f_num * self.f_num * self.max_channel_num)),
                     nn.Linear(self.f_num * self.f_num * self.max_channel_num, self.y_dim)]

        self.encoder_z = nn.Sequential(*sequence_z)
        self.encoder_y = nn.Sequential(*sequence_y)

    def forward(self, x):
        # Encode
        distributions = self.encoder_z(x)
        mu = distributions[:, : self.z_dim]
        logvar = distributions[:, self.z_dim : ]
        z = reparametrize(mu, logvar)
        y = self.encoder_y(x)
       
        return z, mu, logvar, y

class ICP_Decoder_Big(nn.Module):
    def __init__(self, dim, nc, img_dim = 128):
        super(ICP_Decoder_Big, self).__init__()

        self.img_dim = img_dim
        self.nc = nc
        self.dim = dim
        self.layer_num = int(np.log2(img_dim)) - 3
        self.max_channel_num = img_dim * 8
        self.f_num = 4

         # Decoder
        sequence = [nn.Linear(self.dim, self.f_num * self.f_num * self.max_channel_num),
                    nn.ReLU(True),
                    View((-1, self.max_channel_num, self.f_num, self.f_num)),]
        pre_channel_num = self.max_channel_num
        for i in range(self.layer_num):
            now_channel_num = self.max_channel_num // 2**(i + 1)

            sequence += [nn.ConvTranspose2d(in_channels = pre_channel_num, out_channels = now_channel_num,
                                kernel_size = 4, stride = 2, padding = 1),
                         nn.BatchNorm2d(now_channel_num),
                         nn.ReLU(True),]

            pre_channel_num = now_channel_num

        sequence += [nn.ConvTranspose2d(in_channels = now_channel_num, out_channels = self.nc,
                                kernel_size = 4, stride = 2, padding = 1),
                     # nn.Tanh()
                    ]
        self.decoder = nn.Sequential(*sequence)

    def forward(self, z):
        x_rec = self.decoder(z)

        return x_rec
