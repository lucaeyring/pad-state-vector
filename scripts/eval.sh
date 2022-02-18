CUDA_VISIBLE_DEVICES=0 python3 src/eval.py \
    --domain_name cartpole \
    --task_name swingup \
    --action_repeat 8 \
    --mode cartpole_damping \
    --use_inv \
    --num_shared_layers 8 \
    --seed 0 \
    --work_dir logs/cartpole_swingup/inv/state_vector \
    --pad_num_episodes 10 \
    --pad_checkpoint 250k \
    --use_state_vector \
    --save_video