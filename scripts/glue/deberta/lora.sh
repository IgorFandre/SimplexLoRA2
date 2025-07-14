clear
#for task_name in cola mnli mrpc qnli qqp rte sst2 stsb
CUDA_VISIBLE_DEVICES=3 python ./src/run_experiment.py \
    --problem fine-tuning \
    --dataset glue \
    --task_name mnli \
    --model microsoft/deberta-v3-base \
    --use_fast_tokenizer \
    --optimizer adamw \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 32 \
    --gradient_accumulation_steps 6 \
    --lr 1e-4 \
    --lr_scheduler_type linear \
    --warmup_steps 100 \
    --max_train_steps 10 \
    --eval_strategy epoch \
    --save_strategy no \
    --ft_strategy LoRA \
    --lora_r 1 \
    --lora_alpha 32 \
    --lora_dropout 0.05
