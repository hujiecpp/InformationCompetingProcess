CUDA_VISIBLE_DEVICES=1,2 python main.py \
--dataset cifar100 \
--model googlenet \
--epoch 200 \
--gamma 0.01 --alpha 0.01 --beta 0.001 --rec 0.1 \
--lr_decay_epochs 100 150
