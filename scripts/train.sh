CUDA_VISIBLE_DEVICES=0 python3 src/train.py \
    --domain_name cheetah \
    --task_name run \
    --action_repeat 8 \
    --mode train \
    --use_inv \
    --num_shared_layers 8 \
    --seed 0 \
    --work_dir logs/cheetah_run/vision \
    --save_model