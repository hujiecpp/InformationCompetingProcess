#
CUDA_VISIBLE_DEVICES=0 python main.py \
--train False \
--dataset celebA \
--z_dim 32 --y_dim 32 \
--image_size 128 \
--save_name celebA_ICP
