clear
#for dataset in cola mnli mrpc qnli qqp rte sst2 stsb
CUDA_VISIBLE_DEVICES=6 python ./src/run_experiment.py \
    --dataset sst2 \
    --model microsoft/deberta-v3-base \
    --optimizer weight_adamw \
    --batch_size 32 \
    --gradient_accumulation_steps 1 \
    --lr 1e-4 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.1 \
    --max_train_steps 5 \
    --eval_strategy epoch \
    --save_strategy no \
    --ft_strategy WeightLoRA \
    --max_fat_steps 5 \
    --lora_r 1 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --wandb \