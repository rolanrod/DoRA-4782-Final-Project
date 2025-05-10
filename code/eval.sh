python commonsense_evaluate.py \
    --model LLaMA-7B \
    --adapter $3 \
    --dataset $2 \
    --base_model 'yahma/llama-7b-hf' \
    --batch_size 1 \
    --lora_weights $1|tee -a $1/boolq.txt