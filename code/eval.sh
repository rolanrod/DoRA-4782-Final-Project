python eval.py \
    --adapter $2 \
    --dataset $3 \
    --base_model 'yahma/llama-7b-hf' \
    --batch_size 1 \
    --lora_weights $1|tee -a $1/boolq.txt