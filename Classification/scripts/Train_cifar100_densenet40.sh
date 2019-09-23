CUDA_VISIBLE_DEVICES=0 python main.py \
--dataset cifar100 \
--model densenet40 \
--epoch 300 \
--gamma 0.001 --alpha 0.001 --beta 0.0001 --rec 0.1 \
--lr_decay_epochs 150 225
