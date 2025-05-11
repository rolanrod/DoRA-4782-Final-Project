import copy
import json
import os
import re
import sys
import argparse
import fire
import torch

sys.path.append(os.path.join(os.getcwd(), "peft/src/"))
from peft import PeftModel
from tqdm import tqdm
from transformers import GenerationConfig, LlamaTokenizer, AutoModelForCausalLM

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

def main(
        base_model = "",
        lora_weights = "tloen/alpaca-lora-7b",
        share_gradio = False,
):
    args = parse_args()

    def evaluate(
            instructions,
            input=None,
            temperature=0.1,
            top_p=0.75,
            top_k=40,
            num_beams=4,
            max_new_tokens=32,
            **kwargs,
    ):
        prompts = [generate_prompt(instruction, input) for instruction in instructions]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True)
        input_ids = inputs["input_ids"].to(device)
        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            do_sample=True,
            num_beams=num_beams,
            **kwargs,
        )
        with torch.no_grad():
            generation_output = model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=max_new_tokens,
            )
        s = generation_output.sequences
        outputs = tokenizer.batch_decode(s, skip_special_tokens=True)
        # print(f"outputs: {outputs}")
        outputs = [o.split("### Response:")[-1].strip() for o in outputs]
        return outputs

    dataset = load_data(args)
    batches = create_batch(dataset, args.batch_size)
    tokenizer, model = load_model(args)

    if args.adapter in {"lora", "dora"}:
        print("Merging LoRA/DoRA weights into the original weights")

        for name, module in model.model.named_modules():
            if isinstance(model.peft_config.target_modules, str):
                is_target = re.fullmatch(model.peft_config.target_modules, name)
            else:
                is_target = any(name.endswith(target) for target in model.peft_config.target_modules)

            if is_target:
                module.merge_weights = True
                module.train(False)


    total = len(batches)
    correct = 0
    current = 0
    output_data = []
    pbar = tqdm(total=total)
    save_file = f'experiment/"Llama-7B-{args.adapter}-{args.dataset}.json'
    os.makedirs(os.path.dirname(save_file), exist_ok=True)
    for idx, batch in enumerate(batches):
        current += len(batch)
        instructions = [data.get('instruction') for data in batch]

        outputs = evaluate(instructions)

        for data, output in zip(batch, outputs):
            label = data.get('answer')
            prediction = extract_answer(args, output)
            is_correct = (label == prediction)

            if is_correct:
                correct += 1

            result = copy.deepcopy(data)
            result.update({
                'output_pred': output,
                'pred': prediction,
                'flag': is_correct
            })

            output_data.append(result)

        print('------------------------------')
        print(f'\rTest:{idx + 1}/{total} | Accuracy {correct}  {correct / current}')
        print('------------------------------')

        
        with open(save_file, 'w+') as f:
            json.dump(output_data, f, indent=4)
        pbar.update(1)

    pbar.close()
    print('\n')
    print('Test Completed!')

def generate_prompt(instruction, input=None):
    if input:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

                ### Instruction:
                {instruction}

                ### Input:
                {input}

                ### Response:
                """  
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request. 

                ### Instruction:
                {instruction}

                ### Response:
                """ 

# Load the dataset from a JSON file
def load_data(args) -> list:
    file_path = f'../data/{args.dataset}/test.json'
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"can not find dataset file : {file_path}")
    json_data = json.load(open(file_path, 'r'))
    return json_data

# Create batches of data
def create_batch(dataset, batch_size):
    batches = []
    num_batch = len(dataset)//batch_size if len(dataset) % batch_size == 0 else len(dataset)//batch_size + 1
    for i in range(num_batch):
        batch = dataset[i*batch_size: min((i+1)*batch_size, len(dataset))]
        batches.append(batch)
    return batches

# Parse command line arguments for different datasets, models, and parameters
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=["boolq", "hellaswag", "winogrande"],required=True)
    parser.add_argument('--adapter', choices=['lora', 'dora'],required=True)
    parser.add_argument('--base_model', required=True)
    parser.add_argument('--lora_weights', required=True)
    parser.add_argument('--batch_size', type=int, required=True)

    return parser.parse_args()

# Load the model and tokenizer
def load_model(args) -> tuple:
    
    base_model = args.base_model
    if not base_model:
        raise ValueError(f'can not find base model name by the value: {args.base_model}')
    lora_weights = args.lora_weights
    if not lora_weights:
        raise ValueError(f'can not find lora weight, the value is: {lora_weights}')

    tokenizer = LlamaTokenizer.from_pretrained(base_model)
    tokenizer.padding_side = "left"
    tokenizer.pad_token_id = (0)
    if device == "cuda":
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            offload_folder="offload" 
        ) 
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            torch_dtype=torch.float16,
            device_map={"":0}
        )
        model.eval()
    else:
        model = AutoModelForCausalLM.from_pretrained(
            base_model, device_map={"": device}, low_cpu_mem_usage=True
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            device_map={"": device},
        )

        model.config.pad_token_id = tokenizer.pad_token_id = 0
        model.config.bos_token_id = 1
        model.config.eos_token_id = 2

        model.eval()
        model = torch.compile(model)

    return tokenizer, model

# Extracts the answer from model's output based on the dataset
def extract_answer(args, sentence: str) -> float:
    dataset = args.dataset
    if dataset == 'boolq':
        sentence_ = sentence.strip()
        pred_answers = re.findall(r'true|false', sentence_)
        if not pred_answers:
            return ""
        return pred_answers[0]
    elif dataset == 'winogrande':
        sentence_ = sentence.strip()
        pred_answers = re.findall(r'option1|option2', sentence_)
        if not pred_answers:
            return ""
        return pred_answers[0]
    elif dataset == 'hellaswag':
        sentence_ = sentence.strip()
        pred_answers = re.findall(r'ending1|ending2|ending3|ending4', sentence_)
        if not pred_answers:
            return ""
        return pred_answers[0]


if __name__ == "__main__":
    main()