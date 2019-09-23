from model import *
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import os
import argparse
import numpy as np
from utils import progress_bar, str2bool
from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser(description='ICP CIFAR10/CIFAR100 Training.')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--train', default=True, type=str2bool, help='train or test')
parser.add_argument('--dataset', default='cifar10', type=str, help='dataset name: [cifar10, cifar100].')
parser.add_argument('--model', default='vgg16', type=str, help='model: [vgg16, googlenet, resnet20, densenet40].')
parser.add_argument('--epoch', default=90, type=int, help='the number of epoch.')
parser.add_argument('--lr_decay_epochs', default=[30,60], nargs='+', type=int, help='the epoch to decay the learning rate.')

parser.add_argument('--gamma', default=0.01, type=float, help='Compete - MLP')
parser.add_argument('--alpha', default=0.01, type=float, help='Max - DIS')
parser.add_argument('--beta', default=0.001, type=float, help='MIN - KL')
parser.add_argument('--rec', default=0.1, type=float, help='Synergy - REC')
#parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
args = parser.parse_args()
print(args)

if args.train:
    writer = SummaryWriter('runs/ICP_{}_{}'.format(args.dataset, args.model))

torch.manual_seed(233)
torch.cuda.manual_seed(233)
np.random.seed(233)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
if device == 'cuda':
    cudnn.benchmark = True

# Data
print('==> Preparing data..')
if args.dataset.lower() == 'cifar10':
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)#, drop_last = True

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

    # Model
    print('==> Building model..')
    netE, netG_Less, netG_More, netG_Total, netD, netZ2Y, netY2Z = \
                                get_model(args.dataset, args.model, device)

elif args.dataset.lower() == 'cifar100':
    transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343), (0.2673342858792401, 0.2564384629170883, 0.27615047132568404))])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343), (0.2673342858792401, 0.2564384629170883, 0.27615047132568404))])

    trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

    # Model
    print('==> Building model..')
    netE, netG_Less, netG_More, netG_Total, netD, netZ2Y, netY2Z = \
                                get_model(args.dataset, args.model, device)

else:
    raise NotImplementedError


criterion = nn.CrossEntropyLoss()
loss_MSE = nn.MSELoss()
def loss_KL(mu, logvar):
    kld = torch.mean(-0.5*(1+logvar-mu**2-torch.exp(logvar)).sum(1))
    return kld

optimizerG = optim.SGD([{'params' : netE.parameters()},
                        {'params' : netG_Less.parameters()},
                        {'params' : netG_More.parameters()},
                        {'params' : netG_Total.parameters()}], lr=args.lr, momentum=0.9, weight_decay=5e-4)
optimizerD = optim.SGD(netD.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
optimizerMLP = optim.SGD([{'params' : netZ2Y.parameters()},
                          {'params' : netY2Z.parameters()}], lr=args.lr, momentum=0.9, weight_decay=5e-4)

schedulerG = optim.lr_scheduler.MultiStepLR(optimizerG, milestones = args.lr_decay_epochs, gamma=0.1)
schedulerD = optim.lr_scheduler.MultiStepLR(optimizerD, milestones = args.lr_decay_epochs, gamma=0.1)
schedulerMLP = optim.lr_scheduler.MultiStepLR(optimizerMLP, milestones = args.lr_decay_epochs, gamma=0.1)

# hyper-parameters
gamma = args.gamma
alpha = args.alpha
beta = args.beta
rec = args.rec

# Train
def train(epoch):
    print('\nEpoch: %d' % epoch)
    netE.train()
    netG_Less.train()
    netG_More.train()
    netG_Total.train()
    netD.train()
    netZ2Y.train()
    netY2Z.train()

    train_loss = 0
    train_MLP = 0
    train_D  = 0
    train_kl = 0
    train_less_rec = 0
    train_more_rec = 0
    train_total_rec = 0

    correct = 0
    total = 0
    for batch_idx, (x, targets) in enumerate(trainloader):
        x, targets = x.to(device), targets.to(device)

        ## Update MLP
        z, mu, logvar, y = netE(x)
        rec_y = netZ2Y(z)
        rec_z = netY2Z(y)

        # loss MLP
        # or: (to prevent gradient explosion - nan)
        # gamma * (F.mse_loss(rec_z, z.detach(), reduction='sum').div(self.batch_size) \
        #        + F.mse_loss(rec_y, y.detach(), reduction='sum').div(self.batch_size))
        loss_MLP = gamma * (loss_MSE(rec_z, z.detach()) + loss_MSE(rec_y, y.detach())) #/ args.batch_size
        optimizerMLP.zero_grad()
        loss_MLP.backward()
        optimizerMLP.step()

        # loss D
        index = np.arange(x.size()[0])
        np.random.shuffle(index)
        y_shuffle = y.clone()
        y_shuffle = y_shuffle[index, :]

        real_score = netD(torch.cat([y.detach(), y.detach()], dim=1))
        fake_score = netD(torch.cat([y.detach(), y_shuffle.detach()], dim=1))

        # or: (to prevent gradient explosion - nan)
        # alpha * (F.binary_cross_entropy(real_score, ones, reduction='sum').div(self.batch_size) \
        #       + F.binary_cross_entropy(fake_score, zeros, reduction='sum').div(self.batch_size))
        loss_D = -alpha * torch.mean(torch.log(real_score) + torch.log(1 - fake_score))

        optimizerD.zero_grad()
        loss_D.backward()
        optimizerD.step()

        ## Update G
        z, mu, logvar, y = netE(x)

        rec_y = netZ2Y(z)
        rec_z = netY2Z(y)

        rec_less_x = netG_Less(z)
        rec_more_x = netG_More(y)
        rec_x = netG_Total(torch.cat([z, y], dim=1))

        # loss MLP
        # or: (to prevent gradient explosion - nan)
        #  (F.mse_loss(rec_z, z.detach(), reduction='sum').div(self.batch_size) \
        # + F.mse_loss(rec_y, y.detach(), reduction='sum').div(self.batch_size))
        loss_MLP = (loss_MSE(rec_z, z.detach()) + loss_MSE(rec_y, y.detach())) #/ args.batch_size

        # loss D
        index = np.arange(x.size()[0])
        np.random.shuffle(index)
        y_shuffle = y.clone()
        y_shuffle = y_shuffle[index, :]

        real_score = netD(torch.cat([y, y.detach()], dim=1))
        fake_score = netD(torch.cat([y, y_shuffle.detach()], dim=1))

        # or: (to prevent gradient explosion - nan)
        # alpha * (F.binary_cross_entropy(real_score, ones, reduction='sum').div(self.batch_size) \
        #       + F.binary_cross_entropy(fake_score, zeros, reduction='sum').div(self.batch_size))
        loss_D = -torch.mean(torch.log(real_score) + torch.log(1 - fake_score))

        # loss KL
        loss_kl = loss_KL(mu, logvar)

        # loss Rec
        loss_less_rec = criterion(rec_less_x, targets)
        loss_more_rec = criterion(rec_more_x, targets)
        loss_total_rec = criterion(rec_x, targets)
        loss_rec = loss_less_rec + loss_more_rec + loss_total_rec

        # total Loss
        loss_total = rec * loss_rec + beta * loss_kl + alpha * loss_D - gamma * loss_MLP

        optimizerG.zero_grad()
        loss_total.backward()
        optimizerG.step()

        train_loss += loss_total.item()
        train_MLP += loss_MLP.item()
        train_D  += loss_D.item()
        train_kl += loss_kl.item()
        train_less_rec += loss_less_rec.item()
        train_more_rec += loss_more_rec.item()
        train_total_rec += loss_total_rec.item()

        _, predicted = rec_x.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        # print('Train Epoch: {} [{}/{} ({:.0f}%)]\tloss_total: {:.5f}, loss_MLP: {:.5f}, loss_D: {:.5f}, loss_kl: {:.5f}, loss_less_rec: {:.5f}, loss_more_rec: {:.5f}, loss_total_rec: {:.5f}'.format(
        #             epoch, batch_idx * len(x), len(trainloader.dataset),
        #             100. * batch_idx / len(trainloader),
        #             loss_total.item(), loss_MLP.item(), loss_D.item(), loss_kl.item(),
        #             loss_less_rec.item(), loss_more_rec.item(), loss_total_rec.item()))

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

    writer.add_scalar('train_loss', train_loss, epoch)
    writer.add_scalar('train_MLP', train_MLP, epoch)
    writer.add_scalar('train_D', train_D, epoch)
    writer.add_scalar('train_kl', train_kl, epoch)
    writer.add_scalar('train_less_rec', train_less_rec, epoch)
    writer.add_scalar('train_more_rec', train_more_rec, epoch)
    writer.add_scalar('train_total_rec', train_total_rec, epoch)

# Test
def test(epoch = 0):
    global best_acc
    netE.eval()
    netG_Less.eval()
    netG_More.eval()
    netG_Total.eval()
    netD.eval()
    netZ2Y.eval()
    netY2Z.eval()

    # Total
    test_loss = 0
    correct_total = 0
    correct_less = 0
    correct_more = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (x, targets) in enumerate(testloader):
            x, targets = x.to(device), targets.to(device)
            z, mu, logvar, y = netE(x)
            outputs = netG_Total(torch.cat([mu, y], dim=1))
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct_total += predicted.eq(targets).sum().item()

            outputs_less = netG_Less(mu)
            _, predicted_less = outputs_less.max(1)
            correct_less += predicted_less.eq(targets).sum().item()

            outputs_more = netG_More(y)
            _, predicted_more = outputs_more.max(1)
            correct_more += predicted_more.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct_total/total, correct_total, total))

    # Save checkpoint.
    acc_total = 100.*correct_total/total

    acc_less = 100.*correct_less/total
    acc_more = 100.*correct_more/total

    print('Error Rate: ', 100.0 - acc_total)

    if args.train:
        writer.add_scalar('accuracy', acc_total, epoch)
        writer.add_scalar('accuracy_less', acc_less, epoch)
        writer.add_scalar('accuracy_more', acc_more, epoch)

        if acc_total > best_acc:
            print('Saving..')
            state = {
                'netE': netE.state_dict(),
                'netG_Less': netG_Less.state_dict(),
                'netG_More': netG_More.state_dict(),
                'netG_Total': netG_Total.state_dict(),
                'netD': netD.state_dict(),
                'netZ2Y': netZ2Y.state_dict(),
                'netY2Z': netY2Z.state_dict(),

                'acc_total': acc_total,
                'acc_less': acc_less,
                'acc_more': acc_more,
                'epoch': epoch,
            }
            if not os.path.isdir('checkpoints'):
                os.mkdir('checkpoints')
            torch.save(state, './checkpoints/ICP_{}_{}.t7'.format(args.dataset, args.model))
            best_acc = acc_total


if __name__ == '__main__':
    
    if args.train:
        for epoch in range(start_epoch, start_epoch + args.epoch):
            schedulerG.step()
            schedulerD.step()
            schedulerMLP.step()
            train(epoch)
            test(epoch)
    else:
        print('==> Resuming from checkpoint..')
        assert os.path.isdir('checkpoints'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load('./checkpoints/ICP_{}_{}.t7'.format(args.dataset, args.model))
        netE.load_state_dict(checkpoint['netE'])
        netG_Less.load_state_dict(checkpoint['netG_Less'])
        netG_More.load_state_dict(checkpoint['netG_More'])
        netG_Total.load_state_dict(checkpoint['netG_Total'])
        netD.load_state_dict(checkpoint['netD'])
        netZ2Y.load_state_dict(checkpoint['netZ2Y'])
        netY2Z.load_state_dict(checkpoint['netY2Z'])
        print("=> loaded checkpoint ICP_{}_{}.".format(args.dataset, args.model))
        test()
