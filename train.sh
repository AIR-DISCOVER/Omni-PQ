# export TORCH_DISTRIBUTED_DEBUG=DETAIL
# export CUDA_LAUNCH_BLOCKING=1

export TA=./pretrained_model/ckpt_epoch_last_downloaded.pth
export T05=./log/wise-dawn-368/pq-transformer/scannet_wise-dawn-368/ckpt_epoch_best.pth
export T1=./log/cool-shadow-159/pq-transformer/scannet_cool-shadow-159/ckpt_epoch_best.pth
export T2=./log/expert-eon-302/pq-transformer/scannet_expert-eon-302/ckpt_epoch_best.pth
export T3=./log/crimson-valley-303/pq-transformer/scannet_crimson-valley-303/ckpt_epoch_best.pth
export T4=./log/cool-butterfly-304/pq-transformer/scannet_cool-butterfly-304/ckpt_epoch_best.pth
export T7=./log/sparkling-serenity-325/pq-transformer/scannet_sparkling-serenity-325/ckpt_epoch_best.pth

export TA_rate=1.00
export T05_rate=0.05
export T1_rate=0.10
export T2_rate=0.20
export T3_rate=0.30
export T4_rate=0.40
export T7_rate=0.70

export CUDA_VISIBLE_DEVICES=2
export WANDB_MODE=disabled

../.conda/envs/pqt/bin/python3.6 -m torch.distributed.launch --nproc_per_node 1 --master_port `expr 12452 + $CUDA_VISIBLE_DEVICES` \
    train_point_loss.py\
    --pc_loss \
    --max_epoch 1200 \
    --batch_size 3 \
    --optimizer adamW \
    --start_proportion 0.0 \
    --weight_decay 0.0005 \
    --end_proportion $T2_rate \
    --checkpoint_path $T2 \
    --ema \
    --gamma_mixture \
    --learning_rate 2e-3 \
    --decoder_learning_rate 1e-4 \
    --lambda_weak_quad_scores 0.0000 \
    --lambda_weak_distance 0.00000 \
    --lambda_metric_normal 0.500 \
    --lambda_metric_vertical 0.500 \
    --lambda_metric_size 0.500 \
    --lambda_metric_score 0.0500 \
    --lambda_arkit_pc_loss 0.0 \
    --consistency_weight 0.05 \
    $@


# --arkit \
#     --checkpoint_path ./pretrained_model/ckpt_epoch_last_downloaded.pth \

# 
# 