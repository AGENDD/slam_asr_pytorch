CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --num_processes 1 --mixed_precision fp16 finetune.py \
    --dataset temp_datasets/covost_en2zh-CN \
    --split clean \
    --output_dir ./output/covost_slam_asr \
    --logging_steps 10 \
    --save_strategy epoch \
    --data_seed 42 \
    --save_total_limit 3 \
    --eval_dataset_size 10 \
    --max_eval_samples 1000 \
    --per_device_eval_batch_size 1 \
    --max_new_tokens 32 \
    --dataloader_num_workers 8 \
    --group_by_length=True \
    --logging_strategy steps \
    --remove_unused_columns False \
    --do_train \
    --warmup_ratio 0.1 \
    --lr_scheduler_type linear \
    --source_max_len 16 \
    --target_max_len 512 \
    --per_device_train_batch_size 2 \
    --max_steps 0 \
    --num_train_epochs 50 \
    --learning_rate 1e-4 \
    --adam_beta2 0.999 \
    --max_grad_norm 1.0 \
    --weight_decay 0.0 \
    --seed 0 \
    --trust_remote_code \
    --report_to tensorboard \
    --gradient_accumulation_steps 8