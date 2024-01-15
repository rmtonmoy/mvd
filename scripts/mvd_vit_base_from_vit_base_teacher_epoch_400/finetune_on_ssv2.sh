#!/bin/bash
GPUS=`nvidia-smi -L | wc -l`
OUTPUT_DIR='/home/myid/rm54254/mvd/OUTPUT/finetune_on_ssv2'
MODEL_PATH='/data/NuanceNet/Datafiles/mvd_b_from_b_ckpt_399.pth'
DATA_PATH='/data/NuanceNet/Datafiles/ssv2_anno_v4'
DATA_ROOT='/data/NuanceNet/Datafiles'

# train on 16 V100 GPUs (2 nodes x 8 GPUs)
OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=${GPUS} \
    run_class_finetuning.py \
    --model vit_base_patch16_224 \
    --data_set SSV2 --nb_classes 174 \
    --data_path ${DATA_PATH} \
    --data_root ${DATA_ROOT} \
    --finetune ${MODEL_PATH} \
    --log_dir ${OUTPUT_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --input_size 224 --short_side_size 224 \
    --opt adamw --opt_betas 0.9 0.999 --weight_decay 0.05 \
    --batch_size 24 --update_freq 1 --num_sample 2 \
    --save_ckpt_freq 5 --no_save_best_ckpt \
    --num_frames 16 \
    --lr 5e-4 --epochs 30 \
    --dist_eval --test_num_segment 2 --test_num_crop 3 \
    --use_checkpoint \
    --enable_deepspeed
