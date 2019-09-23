#
CUDA_VISIBLE_DEVICES=0 python main.py \
--dataset dsprites \
--lr 5e-4 \
--batch_size 64 \
--z_dim 10 --y_dim 2 \
--max_iter 2.5e5 \
--gamma 1 --alpha 1 --beta 10 --rec 0.5 \
--save_name dsprites_ICP

CUDA_VISIBLE_DEVICES=0 python main.py \
--dataset dsprites \
--lr 5e-4 \
--batch_size 64 \
--z_dim 10 --y_dim 2 \
--max_iter 3.5e5 \
--gamma 1 --alpha 1 --beta 10 --rec 1 \
--global_iter 2.5e5 \
--ckpt_name 250000 \
--save_name dsprites_ICP