# export TORCH_DISTRIBUTED_DEBUG=DETAIL

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

# You MUST provide --checkpoint_path
# And --arkit if you want to evaluate ARKitScenes dataset

python3 -m torch.distributed.launch --nproc_per_node 1 --master_port `expr 12452 + $CUDA_VISIBLE_DEVICES` \
    eval.py\
    --pc_loss \
    --ema \
    --batch_size 16 \
    $@
