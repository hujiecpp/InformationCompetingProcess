CUDA_VISIBLE_DEVICES=1,2 python main.py \
--dataset cifar10 \
--model googlenet \
--epoch 200 \
--lr_decay_epochs 100 150
