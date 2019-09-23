# Prerequisites
- python3
- pytorch 1.0
- tensorboardX

# Prepare Datasets
The datasets of cifar10 and cifar100 will be downloaded automatically while runing the codes.

# Training
To train a model such as vgg16 on cifar10 with ICP:
> chmod +x ./scripts/Train_cifar10_vgg16.sh
> sh ./scripts/Train_cifar10_vgg16.sh

# Testing
To test a trained model such as vgg16 on cifar10:
> chmod +x ./scripts/Test_cifar10_vgg16.sh
> sh ./scripts/Test_cifar10_vgg16.sh

# Results and Logs
The error rates of ICP on cifar-10:

|          | VGG16    | GoogLeNet | ResNet20 | DenseNet40 |
|   :---:  |:--------:|:--------: |:-------: |:-------:   |
|Baseline  |6.67      |4.92       |7.63      |5.83        |
|ICP-ALL   |6.97      |4.76       |6.47      |6.13        |
|ICP-COM   |6.59      |4.67       |7.33      |5.63        |
|ICP       |6.10      |4.26       |6.01      |4.99        |

The error rates of ICP on cifar-100:

|          | VGG16    | GoogLeNet | ResNet20 | DenseNet40 |
|   :---:  |:--------:|:--------: |:-------: |:-------:   |
|Baseline  |26.41     |20.68      |31.91     |27.55       |
|ICP-ALL   |26.73     |20.90      |28.35     |27.51       |
|ICP-COM   |26.37     |20.81      |32.76     |26.85       |
|ICP       |24.54     |18.55      |28.13     |24.52       |

Baseline denotes the performance of original model, ICP-ALL denotes the result of ICP without all the information constraints, ICP-COM denotes the results of ICP without the competing constraints.

The logs of getting our paper's results such as vgg16 on cifar10 with ICP can be shown by:
> tensorboard --logdir runs/ICP_cifar10_vgg

# Trained Models
The trained models of getting our paper's results can be download by [Baidu Netdisk](https://pan.baidu.com/s/1JLQrOvVWbWIXzu_A2l4Ccw) (Password: vd3i), or [Google Drive](https://drive.google.com/drive/folders/19mBHxAVYALPzIQLvvL0uU9-XMLEttBc6?usp=sharing).
