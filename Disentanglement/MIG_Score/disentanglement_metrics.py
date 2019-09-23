import math
import os
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

import lib.utils as utils
import lib.datasets as dset
# from metric_helpers.loader import load_model_and_dataset
from metric_helpers.mi_metric import compute_metric_shapes, compute_metric_faces

import lib.dist as dist

from model import *

def estimate_entropies(qz_samples, qz_params, q_dist, n_samples=10000, weights=None):
    """Computes the term:
        E_{p(x)} E_{q(z|x)} [-log q(z)]
    and
        E_{p(x)} E_{q(z_j|x)} [-log q(z_j)]
    where q(z) = 1/N sum_n=1^N q(z|x_n).
    Assumes samples are from q(z|x) for *all* x in the dataset.
    Assumes that q(z|x) is factorial ie. q(z|x) = prod_j q(z_j|x).

    Computes numerically stable NLL:
        - log q(z) = log N - logsumexp_n=1^N log q(z|x_n)

    Inputs:
    -------
        qz_samples (K, N) Variable
        qz_params  (N, K, nparams) Variable
        weights (N) Variable
    """

    # Only take a sample subset of the samples
    if weights is None:
        qz_samples = qz_samples.index_select(1, torch.randperm(qz_samples.size(1))[:n_samples].cuda())
    else:
        sample_inds = torch.multinomial(weights, n_samples, replacement=True)
        qz_samples = qz_samples.index_select(1, sample_inds)

    K, S = qz_samples.size()
    N, _, nparams = qz_params.size()
    assert(nparams == q_dist.nparams)
    assert(K == qz_params.size(1))

    if weights is None:
        weights = -math.log(N)
    else:
        weights = torch.log(weights.view(N, 1, 1) / weights.sum())

    entropies = torch.zeros(K).cuda()

    pbar = tqdm(total=S)
    k = 0
    while k < S:
        batch_size = min(10, S - k)
        logqz_i = q_dist.log_density(
            qz_samples.view(1, K, S).expand(N, K, S)[:, :, k:k + batch_size],
            qz_params.view(N, K, 1, nparams).expand(N, K, S, nparams)[:, :, k:k + batch_size])
        k += batch_size

        # computes - log q(z_i) summed over minibatch
        entropies += - utils.logsumexp(logqz_i + weights, dim=0, keepdim=False).data.sum(1)
        pbar.update(batch_size)
    pbar.close()

    entropies /= S

    return entropies


def mutual_info_metric_shapes(vae, shapes_dataset):
    dataset_loader = DataLoader(shapes_dataset, batch_size=1000, num_workers=1, shuffle=False)

    N = len(dataset_loader.dataset)  # number of data samples
    K = 10                    # number of latent variables
    nparams = dist.Normal().nparams
    vae.eval()

    print('Computing q(z|x) distributions.')
    qz_params = torch.Tensor(N, K, nparams)

    n = 0
    for xs in dataset_loader:
        batch_size = xs.size(0)
        xs = xs.view(batch_size, 1, 64, 64).cuda()

        z, mu, logvar, y = vae(xs)
        mu = mu.view(batch_size, K, 1)
        logvar = logvar.view(batch_size, K, 1) 
        target = torch.cat([mu, logvar], dim=2)

        qz_params[n:n + batch_size] = target.view(batch_size, K, nparams).data
        n += batch_size

    qz_params = qz_params.view(3, 6, 40, 32, 32, K, nparams).cuda()
    qz_samples = dist.Normal().sample(params=qz_params)

    print('Estimating marginal entropies.')
    # marginal entropies
    marginal_entropies = estimate_entropies(
        qz_samples.view(N, K).transpose(0, 1),
        qz_params.view(N, K, nparams),
        dist.Normal())

    marginal_entropies = marginal_entropies.cpu()
    cond_entropies = torch.zeros(4, K)

    print('Estimating conditional entropies for scale.')
    for i in range(6):
        qz_samples_scale = qz_samples[:, i, :, :, :, :].contiguous()
        qz_params_scale = qz_params[:, i, :, :, :, :].contiguous()

        cond_entropies_i = estimate_entropies(
            qz_samples_scale.view(N // 6, K).transpose(0, 1),
            qz_params_scale.view(N // 6, K, nparams),
            dist.Normal())

        cond_entropies[0] += cond_entropies_i.cpu() / 6

    print('Estimating conditional entropies for orientation.')
    for i in range(40):
        qz_samples_scale = qz_samples[:, :, i, :, :, :].contiguous()
        qz_params_scale = qz_params[:, :, i, :, :, :].contiguous()

        cond_entropies_i = estimate_entropies(
            qz_samples_scale.view(N // 40, K).transpose(0, 1),
            qz_params_scale.view(N // 40, K, nparams),
            dist.Normal())

        cond_entropies[1] += cond_entropies_i.cpu() / 40

    print('Estimating conditional entropies for pos x.')
    for i in range(32):
        qz_samples_scale = qz_samples[:, :, :, i, :, :].contiguous()
        qz_params_scale = qz_params[:, :, :, i, :, :].contiguous()

        cond_entropies_i = estimate_entropies(
            qz_samples_scale.view(N // 32, K).transpose(0, 1),
            qz_params_scale.view(N // 32, K, nparams),
            dist.Normal())

        cond_entropies[2] += cond_entropies_i.cpu() / 32

    print('Estimating conditional entropies for pox y.')
    for i in range(32):
        qz_samples_scale = qz_samples[:, :, :, :, i, :].contiguous()
        qz_params_scale = qz_params[:, :, :, :, i, :].contiguous()

        cond_entropies_i = estimate_entropies(
            qz_samples_scale.view(N // 32, K).transpose(0, 1),
            qz_params_scale.view(N // 32, K, nparams),
            dist.Normal())

        cond_entropies[3] += cond_entropies_i.cpu() / 32

    metric = compute_metric_shapes(marginal_entropies, cond_entropies)
    return metric, marginal_entropies, cond_entropies


def mutual_info_metric_faces(vae, shapes_dataset):
    dataset_loader = DataLoader(shapes_dataset, batch_size=1000, num_workers=1, shuffle=False)

    N = len(dataset_loader.dataset)  # number of data samples
    K = 10                    # number of latent variables
    nparams = dist.Normal().nparams
    vae.eval()

    print('Computing q(z|x) distributions.')
    qz_params = torch.Tensor(N, K, nparams)

    n = 0
    for xs in dataset_loader:
        batch_size = xs.size(0)
        xs = xs.view(batch_size, 1, 64, 64).cuda()
        
        z, mu, logvar, y = vae(xs)
        mu = mu.view(batch_size, K, 1)
        logvar = logvar.view(batch_size, K, 1) 
        target = torch.cat([mu, logvar], dim=2)

        qz_params[n:n + batch_size] = target.view(batch_size, K, nparams).data
        n += batch_size

    qz_params = qz_params.view(50, 21, 11, 11, K, nparams).cuda()
    qz_samples = dist.Normal().sample(params=qz_params)

    print('Estimating marginal entropies.')
    # marginal entropies
    marginal_entropies = estimate_entropies(
        qz_samples.view(N, K).transpose(0, 1),
        qz_params.view(N, K, nparams),
        dist.Normal())

    marginal_entropies = marginal_entropies.cpu()
    cond_entropies = torch.zeros(3, K)

    print('Estimating conditional entropies for azimuth.')
    for i in range(21):
        qz_samples_pose_az = qz_samples[:, i, :, :, :].contiguous()
        qz_params_pose_az = qz_params[:, i, :, :, :].contiguous()

        cond_entropies_i = estimate_entropies(
            qz_samples_pose_az.view(N // 21, K).transpose(0, 1),
            qz_params_pose_az.view(N // 21, K, nparams),
            dist.Normal())

        cond_entropies[0] += cond_entropies_i.cpu() / 21

    print('Estimating conditional entropies for elevation.')
    for i in range(11):
        qz_samples_pose_el = qz_samples[:, :, i, :, :].contiguous()
        qz_params_pose_el = qz_params[:, :, i, :, :].contiguous()

        cond_entropies_i = estimate_entropies(
            qz_samples_pose_el.view(N // 11, K).transpose(0, 1),
            qz_params_pose_el.view(N // 11, K, nparams),
            dist.Normal())

        cond_entropies[1] += cond_entropies_i.cpu() / 11

    print('Estimating conditional entropies for lighting.')
    for i in range(11):
        qz_samples_lighting = qz_samples[:, :, :, i, :].contiguous()
        qz_params_lighting = qz_params[:, :, :, i, :].contiguous()

        cond_entropies_i = estimate_entropies(
            qz_samples_lighting.view(N // 11, K).transpose(0, 1),
            qz_params_lighting.view(N // 11, K, nparams),
            dist.Normal())

        cond_entropies[2] += cond_entropies_i.cpu() / 11

    metric = compute_metric_faces(marginal_entropies, cond_entropies)
    return metric, marginal_entropies, cond_entropies

def setup_data_loaders(dataset, batch_size = 2048, use_cuda=True):
    if dataset == 'shapes':
        train_set = dset.Shapes()
    elif dataset == 'faces':
        train_set = dset.Faces()
    return train_set

img_dim = 64
nc = 1
z_dim = 10
y_dim = 2

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpt', required=True)
    parser.add_argument('--name', type=str, default='shapes')
    args = parser.parse_args()

    print(args)

    vae_path = "{}".format(args.checkpt)
    vae = ICP_Encoder(z_dim = z_dim, y_dim = y_dim, nc = 1).cuda()
    vae.load_state_dict(torch.load(vae_path))

    dataset = setup_data_loaders(args.name)

    metric, marginal_entropies, cond_entropies = eval('mutual_info_metric_' + args.name)(vae, dataset)
    
    print('MIG: {:.2f}'.format(metric))
