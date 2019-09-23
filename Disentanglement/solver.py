import os
import torch
import torch.optim as optim
import torch.nn.functional as F
from torchvision.utils import save_image

from utils import cuda, grid2gif
from model import ICP_Encoder, ICP_Decoder, Discriminator, MLP, ICP_Encoder_Big, ICP_Decoder_Big
from dataset import return_data
import numpy as np
import random
import torchvision.utils as vutils

def reconstruction_loss(x, x_recon, distribution):
    batch_size = x.size(0)
    assert batch_size != 0

    if distribution == 'bernoulli':
        recon_loss = F.binary_cross_entropy_with_logits(x_recon, x, reduction='sum').div(batch_size)
    elif distribution == 'gaussian':
        recon_loss = F.mse_loss(x_recon, x, reduction='sum').div(batch_size)
    else:
        recon_loss = None

    return recon_loss


def kl_divergence(mu, logvar):
    batch_size = mu.size(0)
    assert batch_size != 0
    if mu.data.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1))
    if logvar.data.ndimension() == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))

    klds = -0.5*(1 + logvar - mu.pow(2) - logvar.exp())
    total_kld = klds.sum(1).mean(0, True)
    dimension_wise_kld = klds.mean(0)
    mean_kld = klds.mean(1).mean(0, True)

    return total_kld, dimension_wise_kld, mean_kld

class Solver(object):
    def __init__(self, args):
        self.use_cuda = args.cuda and torch.cuda.is_available()
        self.max_iter = args.max_iter
        self.global_iter = args.global_iter

        self.z_dim = args.z_dim
        self.y_dim = args.y_dim

        self.gamma = args.gamma  # MLP
        self.alpha = args.alpha  # DIS
        self.beta = args.beta    # KL
        self.rec = args.rec      # REC

        self.image_size = args.image_size
        self.lr = args.lr

        if args.dataset.lower() == 'dsprites' or args.dataset.lower() == 'faces':
            self.nc = 1
            self.decoder_dist = 'bernoulli'

            self.netE = cuda(ICP_Encoder(z_dim = self.z_dim, y_dim = self.y_dim, nc = self.nc), self.use_cuda)
            self.netG_Less = cuda(ICP_Decoder(dim = self.z_dim, nc = self.nc), self.use_cuda)
            self.netG_More = cuda(ICP_Decoder(dim = self.y_dim, nc = self.nc), self.use_cuda)
            self.netG_Total = cuda(ICP_Decoder(dim = self.z_dim + self.y_dim, nc = self.nc), self.use_cuda)
            self.netD = cuda(Discriminator(y_dim = self.y_dim), self.use_cuda)
            self.netZ2Y = cuda(MLP(s_dim = self.z_dim, t_dim = self.y_dim), self.use_cuda)
            self.netY2Z = cuda(MLP(s_dim = self.y_dim, t_dim = self.z_dim), self.use_cuda)

        elif args.dataset.lower() == 'celeba':
            self.nc = 3
            self.decoder_dist = 'gaussian'

            self.netE = cuda(ICP_Encoder_Big(z_dim = self.z_dim, y_dim = self.y_dim, nc = self.nc), self.use_cuda)
            self.netG_Less = cuda(ICP_Decoder_Big(dim = self.z_dim, nc = self.nc), self.use_cuda)
            self.netG_More = cuda(ICP_Decoder_Big(dim = self.y_dim, nc = self.nc), self.use_cuda)
            self.netG_Total = cuda(ICP_Decoder_Big(dim = self.z_dim + self.y_dim, nc = self.nc), self.use_cuda)
            self.netD = cuda(Discriminator(y_dim = self.y_dim), self.use_cuda)
            self.netZ2Y = cuda(MLP(s_dim = self.z_dim, t_dim = self.y_dim), self.use_cuda)
            self.netY2Z = cuda(MLP(s_dim = self.y_dim, t_dim = self.z_dim), self.use_cuda)
        else:
            raise NotImplementedError

        self.optimizerG = optim.Adam([{'params' : self.netE.parameters()},
                         {'params' : self.netG_Less.parameters()},
                         {'params' : self.netG_More.parameters()},
                         {'params' : self.netG_Total.parameters()}], lr = self.lr, betas = (0.9, 0.999))
        self.optimizerD = optim.Adam(self.netD.parameters(), lr = self.lr, betas = (0.9, 0.999))
        self.optimizerMLP = optim.Adam([{'params' : self.netZ2Y.parameters()},
                           {'params' : self.netY2Z.parameters()}], lr = self.lr, betas = (0.9, 0.999))


        self.save_name = args.save_name
        self.ckpt_dir = os.path.join('./checkpoints/', args.save_name)
        if not os.path.exists(self.ckpt_dir):
            os.makedirs(self.ckpt_dir, exist_ok=True)
        self.ckpt_name = args.ckpt_name
        if args.train and self.ckpt_name is not None:
            self.load_checkpoint(self.ckpt_name)
        self.save_output = args.save_output
        self.output_dir = os.path.join('./trainvis/', args.save_name)
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)

        self.display_step = args.display_step
        self.save_step = args.save_step

        self.dataset = args.dataset
        self.batch_size = args.batch_size
        self.data_loader = return_data(args)

    def train(self):
        self.net_mode(train=True)
        out = False
        ones = cuda(torch.ones(self.batch_size, dtype = torch.float), self.use_cuda)
        zeros = cuda(torch.zeros(self.batch_size, dtype = torch.float), self.use_cuda)

        while not out:
            for x in self.data_loader:
                self.global_iter += 1

                ### Train ###
                x = cuda(x, self.use_cuda)

                ## Update MLP
                z, mu, logvar, y = self.netE(x)
                rec_y = self.netZ2Y(z)
                rec_z = self.netY2Z(y)

                # loss MLP
                loss_MLP = self.gamma * (F.mse_loss(rec_z, z.detach(), reduction='sum').div(self.batch_size) \
                                        + F.mse_loss(rec_y, y.detach(), reduction='sum').div(self.batch_size))
                self.optimizerMLP.zero_grad()
                loss_MLP.backward()
                self.optimizerMLP.step()

                # loss D
                index = np.arange(x.size()[0])
                np.random.shuffle(index)
                y_shuffle = y.clone()
                y_shuffle = y_shuffle[index, :]

                real_score = self.netD(torch.cat([y.detach(), y.detach()], dim=1))
                fake_score = self.netD(torch.cat([y.detach(), y_shuffle.detach()], dim=1))

                loss_D = self.alpha * (F.binary_cross_entropy(real_score, ones, reduction='sum').div(self.batch_size) \
                                     + F.binary_cross_entropy(fake_score, zeros, reduction='sum').div(self.batch_size))
          
                self.optimizerD.zero_grad()
                loss_D.backward()
                self.optimizerD.step()
                
                ## Update G
                z, mu, logvar, y = self.netE(x)

                rec_y = self.netZ2Y(z)
                rec_z = self.netY2Z(y)

                rec_less_x = self.netG_Less(z)
                rec_more_x = self.netG_More(y)
                rec_x = self.netG_Total(torch.cat([z, y], dim=1))

                # loss MLP
                loss_MLP = (F.mse_loss(rec_z, z.detach(), reduction='sum').div(self.batch_size) \
                            + F.mse_loss(rec_y, y.detach(), reduction='sum').div(self.batch_size))

                # loss D
                index = np.arange(x.size()[0])
                np.random.shuffle(index)
                y_shuffle = y.clone()
                y_shuffle = y_shuffle[index, :]

                real_score = self.netD(torch.cat([y, y.detach()], dim=1))
                fake_score = self.netD(torch.cat([y, y_shuffle.detach()], dim=1))

                loss_D = (F.binary_cross_entropy(real_score, ones, reduction='sum').div(self.batch_size) \
                                     + F.binary_cross_entropy(fake_score, zeros, reduction='sum').div(self.batch_size))

                # loss KL
                loss_kl, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)

                # loss Rec
                loss_less_rec = reconstruction_loss(x, rec_less_x, self.decoder_dist)
                loss_more_rec = reconstruction_loss(x, rec_more_x, self.decoder_dist)
                loss_total_rec = reconstruction_loss(x, rec_x, self.decoder_dist)
                loss_rec = loss_total_rec + loss_less_rec + loss_more_rec

                # total Loss
                loss_total = self.rec * loss_rec + self.beta * loss_kl + self.alpha * loss_D - self.gamma * loss_MLP

                self.optimizerG.zero_grad()
                loss_total.backward()
                self.optimizerG.step()


                ### Test ###
                if self.global_iter % 50 == 0:
                    print('[Iter-{}]: loss_total: {:.5f}, loss_MLP: {:.5f}, loss_D: {:.5f}, loss_kl: {:.5f}, loss_less_rec: {:.5f}, loss_more_rec: {:.5f}, loss_total_rec: {:.5f}'.format(
                        self.global_iter, loss_total.item(), loss_MLP.item(), loss_D.item(), loss_kl.item(),
                        loss_less_rec.item(), loss_more_rec.item(), loss_total_rec.item()))

                if self.global_iter % self.display_step == 0:
                    if self.save_output:
                        self.viz_traverse()

                if self.global_iter % self.save_step == 0:
                    self.save_checkpoint('last')
                    print('Saved checkpoint(iter:{})'.format(self.global_iter))

                if self.global_iter % 50000 == 0:
                    self.save_checkpoint(str(self.global_iter))

                if self.global_iter >= self.max_iter:
                    out = True
                    break

    def test(self, epoch = 'last'):
        if not os.path.exists("./testvis/"):
            os.mkdir("./testvis/")

        if not os.path.exists("./testvis/{}/".format(self.save_name)):
            os.mkdir("./testvis/{}/".format(self.save_name))

        self.load_checkpoint(epoch)
        self.net_mode(train = False)
        
        limit = 3
        inter = 0.2
        interpolation = torch.arange(-limit, limit + inter, inter)
        print("testing: ", interpolation)
        # Random
        n_dsets = len(self.data_loader.dataset)
        fixed_idxs = [random.randint(1, n_dsets-1),random.randint(1, n_dsets-1),
                      random.randint(1, n_dsets-1),random.randint(1, n_dsets-1),
                      random.randint(1, n_dsets-1),random.randint(1, n_dsets-1),
                      random.randint(1, n_dsets-1),random.randint(1, n_dsets-1),
                      random.randint(1, n_dsets-1),random.randint(1, n_dsets-1),
                      random.randint(1, n_dsets-1),random.randint(1, n_dsets-1)]

        for i, fixed_idx in enumerate(fixed_idxs):
            fixed_img = cuda(self.data_loader.dataset.__getitem__(fixed_idx), self.use_cuda).unsqueeze(0)
            vutils.save_image(fixed_img.cpu().data[:1, ], './testvis/{}/{}.jpg'.format(self.save_name, fixed_idx))
            _, z, _, y = self.netE(fixed_img)

            x_less = self.netG_Less(z)
            x_more = self.netG_More(y)
            x_rec = self.netG_Total(torch.cat([z, y], dim=1))
            if self.dataset == 'faces':
                x_less = torch.sigmoid(x_less)
                x_more = torch.sigmoid(x_more)
                x_rec = torch.sigmoid(x_rec)
            vutils.save_image(x_less.cpu().data[:1, ], './testvis/{}/{}_less.png'.format(self.save_name, fixed_idx))
            vutils.save_image(x_more.cpu().data[:1, ], './testvis/{}/{}_more.png'.format(self.save_name, fixed_idx))
            vutils.save_image(x_rec.cpu().data[:1, ], './testvis/{}/{}_rec.png'.format(self.save_name, fixed_idx))

            for row in range(self.z_dim):
                z_tmp = z.clone()
                for j, val in enumerate(interpolation):
                    z_tmp[:, row] = val
                    sample = self.netG_Total(torch.cat([z_tmp, y], dim=1))
                    if self.dataset == 'faces':
                        sample = torch.sigmoid(sample)
                    vutils.save_image(sample.cpu().data[:1, ], './testvis/{}/{}_{}_{}.jpg'.format(self.save_name, fixed_idx, row, j))

            fixed_img = cuda(self.data_loader.dataset.__getitem__(fixed_idx), self.use_cuda).unsqueeze(0)
            vutils.save_image(fixed_img.cpu().data[:1, ], './testvis/{}/{}.jpg'.format(self.save_name, fixed_idx))
            _, z, _, y = self.netE(fixed_img)


            for row in range(self.y_dim):
                y_tmp = y.clone()
                for j, val in enumerate(interpolation):
                    y_tmp[:, row] = val
                    sample = self.netG_Total(torch.cat([z, y_tmp], dim=1))
                    if self.dataset == 'faces':
                        sample = torch.sigmoid(sample)
                    vutils.save_image(sample.cpu().data[:1, ], './testvis/{}/{}_{}_{}.jpg'.format(self.save_name, fixed_idx, row + self.z_dim, j))
        print("done!")

    def viz_traverse(self, limit = 3, inter = 2/3, loc = -1):
        self.net_mode(train = False)
        
        decoder = self.netG_Total
        encoder = self.netE
        interpolation = torch.arange(-limit, limit+0.1, inter)

        n_dsets = len(self.data_loader.dataset)
        rand_idx = random.randint(1, n_dsets-1)

        random_img = self.data_loader.dataset.__getitem__(rand_idx)
        random_img = cuda(random_img, self.use_cuda).unsqueeze(0)
        
        _, random_img_z, _, random_img_y = encoder(random_img)

        random_z = cuda(torch.rand(1, self.z_dim), self.use_cuda)

        if self.dataset == 'dsprites':
            fixed_idx1 = 87040 # square
            fixed_idx2 = 332800 # ellipse
            fixed_idx3 = 578560 # heart

            fixed_img1 = self.data_loader.dataset.__getitem__(fixed_idx1)
            fixed_img1 = cuda(fixed_img1, self.use_cuda).unsqueeze(0)
            _, fixed_img_z1, _, fixed_img_y1 = encoder(fixed_img1)

            fixed_img2 = self.data_loader.dataset.__getitem__(fixed_idx2)
            fixed_img2 = cuda(fixed_img2, self.use_cuda).unsqueeze(0)
            _, fixed_img_z2, _, fixed_img_y2 = encoder(fixed_img2)

            fixed_img3 = self.data_loader.dataset.__getitem__(fixed_idx3)
            fixed_img3 = cuda(fixed_img3, self.use_cuda).unsqueeze(0)
            _, fixed_img_z3, _, fixed_img_y3 = encoder(fixed_img3)

            Z = {'fixed_square':fixed_img_z1, 'fixed_ellipse':fixed_img_z2,
                 'fixed_heart':fixed_img_z3, 'random_img':random_img_z}

            Y = {'fixed_square':fixed_img_y1, 'fixed_ellipse':fixed_img_y2,
                 'fixed_heart':fixed_img_y3, 'random_img':random_img_y}
        else:
            fixed_idx = 0
            fixed_img = self.data_loader.dataset.__getitem__(fixed_idx)
            fixed_img = cuda(fixed_img, self.use_cuda).unsqueeze(0)
            _, fixed_img_z, _, fixed_img_y = encoder(fixed_img)

            Z = {'fixed_img':fixed_img_z, 'random_img':random_img_z}
            Y = {'fixed_img':fixed_img_y, 'random_img':random_img_y}

        gifs = []
        for key in Z.keys():
            z_ori = Z[key]
            y = Y[key]
            samples = []
            for row in range(self.z_dim):
                if loc != -1 and row != loc:
                    continue
                z = z_ori.clone()
                for val in interpolation:
                    z[:, row] = val
                    sample = decoder(torch.cat([z, y], dim=1)).date
                    if self.dataset == 'faces':
                        sample = torch.sigmoid(sample)
                    samples.append(sample)
                    gifs.append(sample)
            samples = torch.cat(samples, dim=0).cpu()
            title = '{}_latent_traversal(iter:{})'.format(key, self.global_iter)

        if self.save_output:
            output_dir = os.path.join(self.output_dir, str(self.global_iter))
            os.makedirs(output_dir, exist_ok=True)
            gifs = torch.cat(gifs)
            gifs = gifs.view(len(Z), self.z_dim, len(interpolation), self.nc, self.image_size, self.image_size).transpose(1, 2)
            for i, key in enumerate(Z.keys()):
                for j, val in enumerate(interpolation):
                    save_image(tensor=gifs[i][j].cpu(),
                               filename=os.path.join(output_dir, '{}_{}.jpg'.format(key, j)),
                               nrow=self.z_dim, pad_value=1)

                grid2gif(os.path.join(output_dir, key+'*.jpg'),
                         os.path.join(output_dir, key+'.gif'), delay=10)

        self.net_mode(train = True)
        
    def net_mode(self, train):
        if not isinstance(train, bool):
            raise('Only bool type is supported. True or False')

        if train:
            self.netE.train()
            self.netG_Less.train()
            self.netG_More.train()
            self.netG_Total.train()
            self.netD.train()
            self.netZ2Y.train()
            self.netY2Z.train()
        else:
            self.netE.eval()
            self.netG_Less.eval()
            self.netG_More.eval()
            self.netG_Total.eval()
            self.netD.eval()
            self.netZ2Y.eval()
            self.netY2Z.eval()

    def save_checkpoint(self, epoch):
        netE_path = "checkpoints/{}/netE_{}.pth".format(self.save_name, epoch)
        netG_Less_path = "checkpoints/{}/netG_Less_{}.pth".format(self.save_name, epoch)
        netG_More_path = "checkpoints/{}/netG_More_{}.pth".format(self.save_name, epoch)
        netG_Total_path = "checkpoints/{}/netG_Total_{}.pth".format(self.save_name, epoch)
        netD_path = "checkpoints/{}/netD_{}.pth".format(self.save_name, epoch)
        netZ2Y_path = "checkpoints/{}/netZ2Y_{}.pth".format(self.save_name, epoch)
        netY2Z_path = "checkpoints/{}/netY2Z_{}.pth".format(self.save_name, epoch)

        if not os.path.exists("checkpoints"):
            os.mkdir("checkpoints")

        if not os.path.exists("checkpoints/{}".format(self.save_name)):
            os.mkdir("checkpoints/{}".format(self.save_name))

        torch.save(self.netE.state_dict(), netE_path)
        torch.save(self.netG_Less.state_dict(), netG_Less_path)
        torch.save(self.netG_More.state_dict(), netG_More_path)
        torch.save(self.netG_Total.state_dict(), netG_Total_path)
        torch.save(self.netD.state_dict(), netD_path)
        torch.save(self.netZ2Y.state_dict(), netZ2Y_path)
        torch.save(self.netY2Z.state_dict(), netY2Z_path)
    
    def load_checkpoint(self, epoch):
        netE_path = "checkpoints/{}/netE_{}.pth".format(self.save_name, epoch)
        netG_Less_path = "checkpoints/{}/netG_Less_{}.pth".format(self.save_name, epoch)
        netG_More_path = "checkpoints/{}/netG_More_{}.pth".format(self.save_name, epoch)
        netG_Total_path = "checkpoints/{}/netG_Total_{}.pth".format(self.save_name, epoch)
        netD_path = "checkpoints/{}/netD_{}.pth".format(self.save_name, epoch)
        netZ2Y_path = "checkpoints/{}/netZ2Y_{}.pth".format(self.save_name, epoch)
        netY2Z_path = "checkpoints/{}/netY2Z_{}.pth".format(self.save_name, epoch)

        if os.path.isfile(netE_path):
            self.netE.load_state_dict(torch.load(netE_path))
            self.netG_Less.load_state_dict(torch.load(netG_Less_path))
            self.netG_More.load_state_dict(torch.load(netG_More_path))
            self.netG_Total.load_state_dict(torch.load(netG_Total_path))
            self.netD.load_state_dict(torch.load(netD_path))
            self.netZ2Y.load_state_dict(torch.load(netZ2Y_path))
            self.netY2Z.load_state_dict(torch.load(netY2Z_path))
            print("=> loaded checkpoint '{} (iter {})'".format(netE_path, self.global_iter))
        else:
            print("=> no checkpoint found at '{}'".format(netE_path))
