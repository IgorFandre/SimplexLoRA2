clear
#for dataset in cola mnli mrpc qnli qqp rte sst2 stsb
CUDA_VISIBLE_DEVICES=6 python ./src/run_experiment.py \
    --dataset sst2 \
    --model microsoft/deberta-v3-base \
    --optimizer simplex_adamw \
    --batch_size 32 \
    --gradient_accumulation_steps 1 \
    --lr_scheduler_type linear \
    --lr 1e-4 \
    --weight_decay_w 1e-7 \
    --learning_rate_w 5e0 \
    --warmup_ratio 0.1 \
    --max_train_steps 5 \
    --eval_strategy epoch \
    --save_strategy no \
    --ft_strategy SimplexLoRA \
    --simplex_step 5 \
    --max_simplex_steps 2 \
    --lora_r 1 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --wandb \