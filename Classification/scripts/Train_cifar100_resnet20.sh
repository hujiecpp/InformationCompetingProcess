CUDA_VISIBLE_DEVICES=0 python main.py \
--dataset cifar100 \
--model resnet20 \
--epoch 200 \
--gamma 0.01 --alpha 0.01 --beta 0.001 --rec 0.1 \
--lr_decay_epochs 100 150
