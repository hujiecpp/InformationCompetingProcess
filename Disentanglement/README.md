# Prerequisites
- python3
- pytorch 1.0
- tensorboardX

# Prepare Datasets
The preparing of data is the same as [FactorVAE](https://github.com/1Konny/FactorVAE).

1. For dSprites Dataset:

> chmod +x ./scripts/prepare_data.sh

> sh scripts/prepare_data.sh dsprites

2. For CelebA Dataset([download](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)):

> download img_align_celeba.zip into ./data/

> sh scripts/prepare_data.sh CelebA

3. For 3D Faces Dataset, We cannot publicly distribute this due to the [license](https://faces.dmi.unibas.ch/bfm/main.php?nav=1-2&id=downloads).

# Training
To train a model on dSprites with ICP:
> chmod +x ./scripts/Train_dsprites.sh

> sh ./scripts/Train_dsprites.sh

# Testing
To test a trained model on dSprites:
> chmod +x ./scripts/Test_dsprites.sh

> sh ./scripts/Test_dsprites.sh

# MIG Score
To evaluate the MIG Score on the dSprites and 3D Faces datasets:
> chmod +x ./scripts/MIG_dsprites.sh ./scripts/MIG_faces.sh

> sh ./scripts/MIG_dsprites.sh

> sh ./scripts/MIG_faces.sh

# Results
The MIG score of ICP on dSprites and 3D Faces:

|          | dSprites | 3D Faces  |
|   :---:  |:--------:|:--------: |
|Beta-VAE  |0.22      |0.54       |
|Beta-TCVAE|0.38      |0.62       |
|ICP-ALL   |0.33      |0.26       |
|ICP-COM   |0.20      |0.57       |
|**ICP**   |**0.48**  |**0.73**   |

ICP-ALL denotes the result of ICP without all the information constraints, ICP-COM denotes the results of ICP without the competing constraints.

# Trained Models
The trained models of getting our paper's results can be download by [Baidu Netdisk](https://pan.baidu.com/s/1JLQrOvVWbWIXzu_A2l4Ccw) (Password: vd3i), or [Google Drive](https://drive.google.com/drive/folders/19mBHxAVYALPzIQLvvL0uU9-XMLEttBc6?usp=sharing).
