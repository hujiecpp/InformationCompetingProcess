import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def reparametrize(mu, logvar):
    std = logvar.mul(0.5).exp_()
    eps = std.data.new(std.size()).normal_()
    return mu + std * eps

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
            nn.Linear(t_dim, t_dim),
            nn.LeakyReLU(0.2, True),
            nn.Linear(t_dim, t_dim),
            nn.ReLU()
        )

    def forward(self, s):
        t = self.net(s)
        return t

## ---------------- CIFAR10 ----------------
# ----------------- VGG -----------------
cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

class VGG_FeatureExtractor(nn.Module):
    def __init__(self, vgg_name):
        super(VGG_FeatureExtractor, self).__init__()
        self.z_dim = 512
        self.y_dim = 512

        self.feature_z = self._make_layers(cfg[vgg_name], self.z_dim * 2)
        self.feature_y = self._make_layers(cfg[vgg_name], self.y_dim)
        

    def forward(self, x):
        distributions = self.feature_z(x)
        distributions = distributions.view(distributions.size(0), -1)
        mu = distributions[:, : self.z_dim]
        logvar = distributions[:, self.z_dim : ]
        z = reparametrize(mu, logvar)

        y = self.feature_y(x)
        y = y.view(y.size(0), -1)
        return z, mu, logvar, y

    def _make_layers(self, cfg, dim):
        layers = []
        in_channels = 3
        for x in cfg[:-2]:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x

        layers += [nn.Conv2d(in_channels, dim, kernel_size=3, padding=1),
                               nn.BatchNorm2d(dim),
                               nn.ReLU(inplace=True)]
        layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

class VGG_Classifier(nn.Module):
    def __init__(self, dim = 512, num_classes = 10):
        super(VGG_Classifier, self).__init__()
        self.dim = dim
        self.classifier = nn.Linear(self.dim, num_classes)

    def forward(self, z):
        out = self.classifier(z)
        return out

# ----------------- GoogleNet -----------------
class Inception(nn.Module):
    def __init__(self, in_planes, n1x1, n3x3red, n3x3, n5x5red, n5x5, pool_planes):
        super(Inception, self).__init__()
        # 1x1 conv branch
        self.b1 = nn.Sequential(
            nn.Conv2d(in_planes, n1x1, kernel_size=1),
            nn.BatchNorm2d(n1x1),
            nn.ReLU(True),
        )

        # 1x1 conv -> 3x3 conv branch
        self.b2 = nn.Sequential(
            nn.Conv2d(in_planes, n3x3red, kernel_size=1),
            nn.BatchNorm2d(n3x3red),
            nn.ReLU(True),
            nn.Conv2d(n3x3red, n3x3, kernel_size=3, padding=1),
            nn.BatchNorm2d(n3x3),
            nn.ReLU(True),
        )

        # 1x1 conv -> 5x5 conv branch
        self.b3 = nn.Sequential(
            nn.Conv2d(in_planes, n5x5red, kernel_size=1),
            nn.BatchNorm2d(n5x5red),
            nn.ReLU(True),
            nn.Conv2d(n5x5red, n5x5, kernel_size=3, padding=1),
            nn.BatchNorm2d(n5x5),
            nn.ReLU(True),
            nn.Conv2d(n5x5, n5x5, kernel_size=3, padding=1),
            nn.BatchNorm2d(n5x5),
            nn.ReLU(True),
        )

        # 3x3 pool -> 1x1 conv branch
        self.b4 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            nn.Conv2d(in_planes, pool_planes, kernel_size=1),
            nn.BatchNorm2d(pool_planes),
            nn.ReLU(True),
        )

    def forward(self, x):
        y1 = self.b1(x)
        y2 = self.b2(x)
        y3 = self.b3(x)
        y4 = self.b4(x)
        return torch.cat([y1,y2,y3,y4], 1)

class GoogLeNet_FeatureExtractor(nn.Module):
    def __init__(self):
        super(GoogLeNet_FeatureExtractor, self).__init__()
        self.pre_layers_z = nn.Sequential(
            nn.Conv2d(3, 192, kernel_size=3, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(True),
        )
        self.a3_z = Inception(192,  64,  96, 128, 16, 32, 32)
        self.b3_z = Inception(256, 128, 128, 192, 32, 96, 64)

        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

        self.a4_z = Inception(480, 192,  96, 208, 16,  48,  64)
        self.b4_z = Inception(512, 160, 112, 224, 24,  64,  64)
        self.c4_z = Inception(512, 128, 128, 256, 24,  64,  64)
        self.d4_z = Inception(512, 112, 144, 288, 32,  64,  64)
        self.e4_z = Inception(528, 256, 160, 320, 32, 128, 128)

        self.a5_z = Inception(832, 256, 160, 320, 32, 128, 128)
        self.b5_z_mu = Inception(832, 384, 192, 384, 48, 128, 128)
        self.b5_z_logvar = Inception(832, 384, 192, 384, 48, 128, 128)

        self.pre_layers_y = nn.Sequential(
            nn.Conv2d(3, 192, kernel_size=3, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(True),
        )
        self.a3_y = Inception(192,  64,  96, 128, 16, 32, 32)
        self.b3_y = Inception(256, 128, 128, 192, 32, 96, 64)
        self.a4_y = Inception(480, 192,  96, 208, 16,  48,  64)
        self.b4_y = Inception(512, 160, 112, 224, 24,  64,  64)
        self.c4_y = Inception(512, 128, 128, 256, 24,  64,  64)
        self.d4_y = Inception(512, 112, 144, 288, 32,  64,  64)
        self.e4_y = Inception(528, 256, 160, 320, 32, 128, 128)
        self.a5_y = Inception(832, 256, 160, 320, 32, 128, 128)
        self.b5_y = Inception(832, 384, 192, 384, 48, 128, 128)

        self.avgpool = nn.AvgPool2d(8, stride=1)

    def forward(self, x):
        out_z = self.pre_layers_z(x)
        out_z = self.a3_z(out_z)
        out_z = self.b3_z(out_z)
        out_z = self.maxpool(out_z)
        out_z = self.a4_z(out_z)
        out_z = self.b4_z(out_z)
        out_z = self.c4_z(out_z)
        out_z = self.d4_z(out_z)
        out_z = self.e4_z(out_z)
        out_z = self.maxpool(out_z)
        out_z = self.a5_z(out_z)

        mu = self.b5_z_mu(out_z)
        mu = self.avgpool(mu)
        mu = mu.view(mu.size(0), -1)
        
        logvar = self.b5_z_logvar(out_z)
        logvar = self.avgpool(logvar)
        logvar = logvar.view(logvar.size(0), -1)
        z = reparametrize(mu, logvar)

        out_y = self.pre_layers_y(x)
        out_y = self.a3_y(out_y)
        out_y = self.b3_y(out_y)
        out_y = self.maxpool(out_y)
        out_y = self.a4_y(out_y)
        out_y = self.b4_y(out_y)
        out_y = self.c4_y(out_y)
        out_y = self.d4_y(out_y)
        out_y = self.e4_y(out_y)
        out_y = self.maxpool(out_y)
        out_y = self.a5_y(out_y)
        out_y = self.b5_y(out_y)
        out_y = self.avgpool(out_y)
        y = out_y.view(out_y.size(0), -1)

        return z, mu, logvar, y

class GoogLeNet_Classifier(nn.Module):
    def __init__(self, dim = 1024, num_classes=10):
        super(GoogLeNet_Classifier, self).__init__()
        self.classifier = nn.Linear(dim, num_classes)

    def forward(self, z):
        out = self.classifier(z)
        return out

# ----------------- ResNet -----------------
class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)

class BasicBlock_Res(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlock_Res, self).__init__()

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2],
                                                  (0, 0, 0, 0, (planes - x.shape[1])//2, (planes - x.shape[1])//2), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        self.x = out
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet_FeatureExtractor(nn.Module):
    def __init__(self, block = BasicBlock_Res, num_blocks = [3, 3, 3]):
        super(ResNet_FeatureExtractor, self).__init__()

        self.in_planes = 16
        self.layers_z = [nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False),
                            nn.BatchNorm2d(16)] + \
                        self._make_layer(block, 16, num_blocks[0], stride=1) + \
                        self._make_layer(block, 32, num_blocks[1], stride=2) + \
                        self._make_layer(block, 64 * 2, num_blocks[2], stride=2)

        self.in_planes = 16
        self.layers_y = [nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False),
                            nn.BatchNorm2d(16)] + \
                        self._make_layer(block, 16, num_blocks[0], stride=1) + \
                        self._make_layer(block, 32, num_blocks[1], stride=2) + \
                        self._make_layer(block, 64, num_blocks[2], stride=2)
        self.feature_z = nn.Sequential(*self.layers_z)
        self.feature_y = nn.Sequential(*self.layers_y)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return layers

    def forward(self, x):
        distributions = self.feature_z(x)
        distributions = F.avg_pool2d(distributions, 4)
        distributions = distributions.view(distributions.size(0), -1)
        mu = distributions[:, : 256]
        logvar = distributions[:, 256 : ]
        z = reparametrize(mu, logvar)

        y = self.feature_y(x)
        y = F.avg_pool2d(y, 4)
        y = y.view(y.size(0), -1)
        return z, mu, logvar, y

class ResNet_Classifier(nn.Module):
    def __init__(self, dim = 256, num_classes=10):
        super(ResNet_Classifier, self).__init__()
        self.classifier = nn.Linear(dim, num_classes)

    def forward(self, z):
        out = self.classifier(z)
        return out

# ----------------- DenseNet -----------------
class Bottleneck(nn.Module):
    def __init__(self, inplanes, expansion=4, growthRate=12, dropRate=0):
        super(Bottleneck, self).__init__()
        planes = expansion * growthRate
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, growthRate, kernel_size=3,
                               padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.dropRate = dropRate

    def forward(self, x):
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        if self.dropRate > 0:
            out = F.dropout(out, p=self.dropRate, training=self.training)

        out = torch.cat((x, out), 1)

        return out

class BasicBlock_Dense(nn.Module):
    def __init__(self, inplanes, expansion=1, growthRate=12, dropRate=0):
        super(BasicBlock_Dense, self).__init__()
        planes = expansion * growthRate
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = nn.Conv2d(inplanes, growthRate, kernel_size=3,
                               padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.dropRate = dropRate

    def forward(self, x):
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        if self.dropRate > 0:
            out = F.dropout(out, p=self.dropRate, training=self.training)

        out = torch.cat((x, out), 1)

        return out

class Transition(nn.Module):
    def __init__(self, inplanes, outplanes):
        super(Transition, self).__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = nn.Conv2d(inplanes, outplanes, kernel_size=1,
                               bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = F.avg_pool2d(out, 2)
        return out

class DenseNet_FeatureExtractor(nn.Module):
    def __init__(self, depth=22, block=Bottleneck, dropRate=0, num_classes=10, growthRate=12, compressionRate=2):
        super(DenseNet_FeatureExtractor, self).__init__()

        assert (depth - 4) % 3 == 0, 'depth should be 3n+4'
        n = (depth - 4) // 3 if block == BasicBlock_Dense else (depth - 4) // 6

        self.growthRate = growthRate
        self.dropRate = dropRate

        # self.inplanes is a global variable used across multiple
        # helper functions
        # z
        self.inplanes = growthRate * 2
        self.conv1_z = nn.Conv2d(3, self.inplanes, kernel_size=3, padding=1,
                               bias=False)
        self.dense1_z = self._make_denseblock(block, n)
        self.trans1_z = self._make_transition(compressionRate)
        self.dense2_z = self._make_denseblock(block, n)
        self.trans2_z = self._make_transition(compressionRate)
        tmp = self.inplanes
        self.dense3_z_mu = self._make_denseblock(block, n)
        self.inplanes = tmp
        self.dense3_z_logvar = self._make_denseblock(block, n)
        self.bn_z_mu = nn.BatchNorm2d(self.inplanes)
        self.bn_z_logvar = nn.BatchNorm2d(self.inplanes)
        self.relu_z = nn.ReLU(inplace=True)
        self.avgpool_z = nn.AvgPool2d(8)

        # y
        self.growthRate = growthRate
        self.inplanes = growthRate * 2
        self.conv1_y = nn.Conv2d(3, self.inplanes, kernel_size=3, padding=1,
                               bias=False)
        self.dense1_y = self._make_denseblock(block, n)
        self.trans1_y = self._make_transition(compressionRate)
        self.dense2_y = self._make_denseblock(block, n)
        self.trans2_y = self._make_transition(compressionRate)
        self.dense3_y = self._make_denseblock(block, n)
        self.bn_y = nn.BatchNorm2d(self.inplanes)
        self.relu_y = nn.ReLU(inplace=True)
        self.avgpool_y = nn.AvgPool2d(8)

    def _make_denseblock(self, block, blocks):
        layers = []
        for i in range(blocks):
            # Currently we fix the expansion ratio as the default value
            layers.append(block(self.inplanes, growthRate=self.growthRate, dropRate=self.dropRate))
            self.inplanes += self.growthRate

        return nn.Sequential(*layers)

    def _make_transition(self, compressionRate):
        inplanes = self.inplanes
        outplanes = int(math.floor(self.inplanes // compressionRate))
        self.inplanes = outplanes
        return Transition(inplanes, outplanes)

    def forward(self, x):
        out_z = self.conv1_z(x)

        out_z = self.trans1_z(self.dense1_z(out_z))
        out_z = self.trans2_z(self.dense2_z(out_z))
        mu = self.dense3_z_mu(out_z)
        mu = self.bn_z_mu(mu)
        mu = self.relu_z(mu)
        mu = self.avgpool_z(mu)
        mu = mu.view(mu.size(0), -1)

        logvar = self.dense3_z_logvar(out_z)
        logvar = self.bn_z_logvar(logvar)
        logvar = self.relu_z(logvar)
        logvar = self.avgpool_z(logvar)
        logvar = logvar.view(logvar.size(0), -1)

        # print(mu.shape, logvar.shape)
        z = reparametrize(mu, logvar)

        out_y = self.conv1_y(x)

        out_y = self.trans1_y(self.dense1_y(out_y))
        out_y = self.trans2_y(self.dense2_y(out_y))
        out_y = self.dense3_y(out_y)
        out_y = self.bn_y(out_y)
        out_y = self.relu_y(out_y)
        out_y = self.avgpool_y(out_y)
        y = out_y.view(out_y.size(0), -1)
        # print(z.shape,y.shape)
        return z, mu, logvar, y

class DenseNet_Classifier(nn.Module):
    def __init__(self, dim = 456, num_classes=10):
        super(DenseNet_Classifier, self).__init__()
        self.classifier = nn.Linear(dim, num_classes)

    def forward(self, z):
        out = self.classifier(z)
        return out
# -------------------- Get Model --------------------
def get_model(dataset_name, model_name, device):
    if dataset_name.lower() == 'cifar10':
        num_classes = 10
    elif dataset_name.lower() == 'cifar100':
        num_classes = 100
    else:
        raise NotImplementedError

    if model_name.lower() == 'vgg16':
        netE = VGG_FeatureExtractor('VGG16').to(device)
        netG_Less = VGG_Classifier(dim = 512, num_classes = num_classes).to(device)
        netG_More = VGG_Classifier(dim = 512, num_classes = num_classes).to(device)
        netG_Total = VGG_Classifier(dim = 512 * 2, num_classes = num_classes).to(device)
        netD = Discriminator(512).to(device)
        netZ2Y = MLP(s_dim = 512, t_dim = 512).to(device)
        netY2Z = MLP(s_dim = 512, t_dim = 512).to(device)

    elif model_name.lower() == 'googlenet':
        netE = GoogLeNet_FeatureExtractor(num_classes = num_classes).to(device)
        netG_Less = GoogLeNet_Classifier(dim = 1024, num_classes = num_classes).to(device)
        netG_More = GoogLeNet_Classifier(dim = 1024, num_classes = num_classes).to(device)
        netG_Total = GoogLeNet_Classifier(dim = 1024 * 2, num_classes = num_classes).to(device)

        netD = Discriminator(1024).to(device)
        netZ2Y = MLP(s_dim = 1024, t_dim = 1024).to(device)
        netY2Z = MLP(s_dim = 1024, t_dim = 1024).to(device)
        if device == 'cuda':
            netE = torch.nn.DataParallel(netE)
            netG_Less = torch.nn.DataParallel(netG_Less)
            netG_More = torch.nn.DataParallel(netG_More)
            netG_Total = torch.nn.DataParallel(netG_Total)
            netD = torch.nn.DataParallel(netD)
            netZ2Y = torch.nn.DataParallel(netZ2Y)
            netY2Z = torch.nn.DataParallel(netY2Z)

    elif model_name.lower() == 'resnet20':
        netE = ResNet_FeatureExtractor().to(device)
        netG_Less = ResNet_Classifier(dim = 256, num_classes = num_classes).to(device)
        netG_More = ResNet_Classifier(dim = 256, num_classes = num_classes).to(device)
        netG_Total = ResNet_Classifier(dim = 256 * 2, num_classes = num_classes).to(device)

        netD = Discriminator(256).to(device)
        netZ2Y = MLP(s_dim = 256, t_dim = 256).to(device)
        netY2Z = MLP(s_dim = 256, t_dim = 256).to(device)

    elif model_name.lower() == 'densenet40':
        netE = DenseNet_FeatureExtractor(block=BasicBlock_Dense, depth=40, dropRate=0, num_classes=num_classes, growthRate=12, compressionRate=1).to(device)
        netG_Less = DenseNet_Classifier(dim = 456, num_classes = num_classes).to(device)
        netG_More = DenseNet_Classifier(dim = 456, num_classes = num_classes).to(device)
        netG_Total = DenseNet_Classifier(dim = 456 * 2, num_classes = num_classes).to(device)

        netD = Discriminator(456).to(device)
        netZ2Y = MLP(s_dim = 456, t_dim = 456).to(device)
        netY2Z = MLP(s_dim = 456, t_dim = 456).to(device)

    else:
        raise NotImplementedError

    return netE, netG_Less, netG_More, netG_Total, netD, netZ2Y, netY2Z
