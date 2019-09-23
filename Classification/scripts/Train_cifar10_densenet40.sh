CUDA_VISIBLE_DEVICES=0 python main.py \
--dataset cifar10 \
--model densenet40 \
--epoch 300 \
--lr_decay_epochs 150 225
