CUDA_VISIBLE_DEVICES=0 python main.py \
--dataset cifar10 \
--model vgg16 \
--epoch 90 \
--lr_decay_epochs 30 60
