CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --num_processes 4 --mixed_precision fp16 finetune.py \
    --dataset librispeech_asr \
    --split clean \
    --output_dir ./output/rwkv \
    --logging_steps 1 \
    --save_strategy steps \
    --save_steps 100 \
    --data_seed 42 \
    --save_total_limit 3 \
    --eval_dataset_size 10 \
    --max_eval_samples 1000 \
    --per_device_eval_batch_size 1 \
    --gradient_clip_val 30 \
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
<<<<<<< Updated upstream
    --per_device_train_batch_size 4 \
=======
    --per_device_train_batch_size 1 \
>>>>>>> Stashed changes
    --max_steps 0 \
    --num_train_epochs 50 \
    --learning_rate 1e-3 \
    --adam_beta2 0.999 \
    --max_grad_norm 1.0 \
    --weight_decay 0.0 \
    --seed 0 \
    --trust_remote_code \
    --report_to tensorboard \
<<<<<<< Updated upstream
    --ddp_find_unused_parameters False \
=======
    --gradient_accumulation_steps 8 
>>>>>>> Stashed changes
