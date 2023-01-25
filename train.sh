# export TORCH_DISTRIBUTED_DEBUG=DETAIL
# export CUDA_LAUNCH_BLOCKING=1

export checkpoint_path=pretrained_model/T10-base.pth
export rate=0.10

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

# For semi-supervised training, you need to specify the checkpoint path of supervised version to resume
# And you need to specify the end_proportion to control the amount of labeled data

python3 -m torch.distributed.launch --nproc_per_node 1 --master_port `expr 23333 + $CUDA_VISIBLE_DEVICES` \
    train.py\
    --pc_loss \
    --max_epoch 1200 \
    --batch_size 3 \
    --optimizer adamW \
    --start_proportion 0.0 \
    --weight_decay 0.0005 \
    --end_proportion $rate \
    --checkpoint_path $checkpoint_path \
    --ema \
    --gamma_mixture \
    --learning_rate 2e-3 \
    --decoder_learning_rate 1e-4 \
    --lambda_metric_normal 0.0005 \
    --lambda_metric_vertical 0.0005 \
    --lambda_metric_size 0.0005 \
    --lambda_metric_score 0.0005 \
    --lambda_arkit_pc_loss 0.0 \
    --consistency_weight 0.05 \
    $@
