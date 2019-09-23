#
CUDA_VISIBLE_DEVICES=0 python main.py \
--train False \
--dataset dsprites \
--z_dim 10 --y_dim 2 \
--save_name dsprites_ICP
