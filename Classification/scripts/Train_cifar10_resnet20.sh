CUDA_VISIBLE_DEVICES=0 python main.py \
--dataset cifar10 \
--model resnet20 \
--epoch 200 \
--lr_decay_epochs 100 150
