CUDA_VISIBLE_DEVICES=0 python3 src/eval.py \
    --domain_name cheetah \
    --task_name run \
    --action_repeat 8 \
    --mode train \
    --use_inv \
    --num_shared_layers 8 \
    --seed 0 \
    --work_dir logs/cheetah_run/state_vector/ \
    --pad_num_episodes 10 \
    --pad_checkpoint 500k \
    --use_state_vector \
    --save_video