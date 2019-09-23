import argparse

import numpy as np
import torch

from solver import Solver
from utils import str2bool

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


def main(args):
    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

    net = Solver(args)

    if args.train:
        net.train()
    else:
        net.test()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Information Competing Process (ICP)')

    parser.add_argument('--train', default=True, type=str2bool, help='train or test')
    parser.add_argument('--seed', default=2, type=int, help='random seed')
    parser.add_argument('--cuda', default=True, type=str2bool, help='enable cuda')
    parser.add_argument('--max_iter', default=1e6, type=float, help='maximum training iteration')
    parser.add_argument('--batch_size', default=64, type=int, help='batch size')

    parser.add_argument('--z_dim', default=10, type=int, help='dimension of the representation z')
    parser.add_argument('--y_dim', default=2, type=int, help='dimension of the representation y')
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')

    parser.add_argument('--gamma', default=1, type=float, help='Compete - MLP')
    parser.add_argument('--alpha', default=1, type=float, help='Max - DIS')
    parser.add_argument('--beta', default=4, type=float, help='MIN - KL')
    parser.add_argument('--rec', default=1, type=float, help='Synergy - REC')
    
    parser.add_argument('--dset_dir', default='data', type=str, help='dataset directory')
    parser.add_argument('--dataset', default='CelebA', type=str, help='dataset name: [CelebA, faces, dsprites]')
    parser.add_argument('--image_size', default=64, type=int, help='image size. [64, 128].')
    parser.add_argument('--num_workers', default=4, type=int, help='dataloader num_workers')

    parser.add_argument('--save_name', default='main', type=str, help='output name.')
    parser.add_argument('--save_output', default=True, type=str2bool, help='save traverse images and gif.')

    parser.add_argument('--display_step', default=10000, type=int, help='number of iterations after which loss data is printed.')
    parser.add_argument('--save_step', default=10000, type=int, help='number of iterations after which a checkpoint is saved.')

    parser.add_argument('--ckpt_name', default='last', type=str, help='load previous checkpoint. insert checkpoint filename.')
    parser.add_argument('--global_iter', default=0, type=float, help='number of iterations continue to train.')

    args = parser.parse_args()

    main(args)
