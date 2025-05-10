python finetune.py \
    --base_model 'yahma/llama-7b-hf' \
    --data_path $3 \
    --output_dir $1 \
    --batch_size 16  --micro_batch_size 16 --num_epochs 3 \
    --learning_rate 2e-4 --cutoff_len 256 --val_set_size 120 \
    --eval_step 80 --save_step 80  --adapter_name $2 \
    --target_modules '["q_proj", "k_proj", "v_proj", "up_proj", "down_proj"]' \
    --lora_r 32 --lora_alpha 64 --use_gradient_checkpointing