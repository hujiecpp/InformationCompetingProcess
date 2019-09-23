This is the project page of our paper:

"Information Competing Process for Learning Diversified Representations." Hu, J., Ji, R., Zhang, S., Sun, X., Ye, Q., Lin, C. W., & Tian, Q. In *NeurIPS 2019.* [paper]

**Our paper is revising now, and the camera-ready version will be uploaded soon.**

If you have any problem, please feel free to contact us.

# 1. Supervised Setting: Classification Task
The codes, usages, models and results for classification task can be found in: [./Classification/](https://github.com/hujiecpp/InformationCompetingProcess/tree/master/Classification).

We implement ICP to train VGG16, GoogLeNet, ResNet20 and DenseNet40 on Cifar10 and Cifar100 datasets.

Our codes for the classification task are based on [pytorch-cifar](https://github.com/kuangliu/pytorch-cifar) and the models from [KSE](https://github.com/yuchaoli/KSE/tree/master/model).

# 2. Self-Supervised Setting: Disentanglement Task
The codes, usages, models and results for disentanglement task can be found in: [./Disentanglement/](https://github.com/hujiecpp/InformationCompetingProcess/tree/master/Disentanglement).

We implement ICP to train Beta-VAE on dSprites, 3D Faces and CelebA datasets.

Our codes for the disentanglement task are based on [Beta-VAE](https://github.com/1Konny/Beta-VAE).

The evaluation metric (MIG) for disentanglement are from [beta-tcvae](https://github.com/rtqichen/beta-tcvae), and we thank Ricky for the help of using the 3D Faces dataset.

# 3. Citation
If our paper helps your research, please cite it in your publications:
```
@inproceedings{hu2019information,
  title={Information Competing Process for Learning Diversified Representations},
  author={Hu, Jie and Ji, Rongrong and Zhang, ShengChuan and Sun, Xiaoshuai and Ye, Qixiang and Lin, Chia-Wen and Tian, Qi},
  booktitle={Advances in Neural Information Processing Systems},
  year={2019}
}
```