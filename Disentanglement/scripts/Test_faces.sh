#
CUDA_VISIBLE_DEVICES=0 python main.py \
--train False \
--dataset faces \
--z_dim 10 --y_dim 2 \
--save_name faces_ICP
