#!/usr/bin/env bash
clear
for optimizer in adamw
do
    export CUDA_VISIBLE_DEVICES=4
    export OMP_NUM_THREADS=4
    python ./src/run_experiment.py \
        --problem cv \
        --dataset cifar10 \
        --eval_runs 1 \
        --n_epoches_train 1 \
        --optimizer $optimizer \
        --verbose \
        --lr 1e-3 \
        --wandb # --use_old_tune_params --momentum 0.95 --weight_decay 1e-5 --tune \\
done