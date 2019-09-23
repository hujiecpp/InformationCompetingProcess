#
CUDA_VISIBLE_DEVICES=0 python main.py \
--dataset faces \
--lr 5e-4 \
--batch_size 64 \
--z_dim 10 --y_dim 2 \
--max_iter 1e5 \
--gamma 1 --alpha 1 --beta 4 --rec 1 \
--save_name faces_ICP