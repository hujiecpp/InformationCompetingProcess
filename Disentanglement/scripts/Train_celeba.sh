#
CUDA_VISIBLE_DEVICES=0 python main.py \
--dataset celebA \
--lr 5e-4 \
--batch_size 64 \
--z_dim 32 --y_dim 32 \
--max_iter 1.5e6 \
--image_size 128 \
--gamma 1 --alpha 5 --beta 5 --rec 1 \
--save_name celebA_ICP