# Introduction

This project is an attempt to replicate the results of DoRA, a parameter-efficient fine-tuning (PEFT) method that improved on the previous standard (LoRA) by decomposing pre-trained weight into magnitude and direction. The work by Shih-Yang Liu et. al consistently outperformed LoRA on fine-tuning tasks with several open source base models. This project was completed by Rolando Rodríguez, Chris Fernandes, Ishaan Nanal, and Gerardo Montemayor and all credit is given to the writers of the original paper for their work.

_DoRA: Weight-Decomposed Low-Rank Adaptation_, available at https://arxiv.org/abs/2402.

# Chosen Result

We were particularly interested in reproducing the superior performance of DoRA over LoRA on commonsense reasoning tasks. These are located in Table 1 of the original paper. 

# Github Contents

Refer to `./peft/src/peft/tuners/dora.py` for the implementation of DoRA.

Refer to `./finetune.py` for the LlaMA-7B finetuning script using either DoRA or LoRA.

Refer to `./commonsense_evaluate.py` for the evaluation script of the finetuned LlaMA-7B model.


# Re-implementation

## Setup
Create a conda environment as described below.
```bash
conda create -n dora_llama python=3.10
conda activate dora_llama
pip install -r requirements.txt
```

## Finetuning and Evaluation

### Finetuning (`./finetune.sh`)
This file contains the code to finetune LLaMA-7B using DoRA. User can specify different DoRA configuration for finetuning. To be specific, the first argument denotes the destination for saving the fine-tuned model and the second argument specifies whether you want to use lora or dora for finetuning.
 
For example, to finetune LLaMA-7B on BoolQ using DoRA, you could do:
```
sh finetune.sh ./finetuned_result/dora_r32 dora ./dataset/boolq
```

### Evaluation (`./eval.sh`)

This file contains the code to evaluate LLaMA-7B finetuned with DoRA or LoRA on any of the commonsense reasoning tasks. The first argument is the address of the model weights, the second argument specifies what dataset to evaluate on, and the last argument determines whether the weights are from DoRA or LoRA.

For example, to evaluate a LLaMA-7B model that was trained on BoolQ using DoRA, you could do:
```
sh eval.sh ./finetuned_result/dora_r32 ./data/boolq DoRA
```

# Results/Insights
Due to our limited compute, we were unable to train on the full dataset used in the paper (a combination of several commonsense reasoning datasets). Instead, we trained and then evaluated our model on BoolQ, HellaSwag, and Winogrande independently. Our results, and the original paper's results, are below. 

| PEFT Strategy | BoolQ | HellaSwag | Winogrande |
|----------|----------|----------|----------|
| Paper's LoRA| 67.5 | 83.4     | 80.4     |
| Our LoRA   | 63.5  | 41.2     | 74.7    |
| Paper's DoRA | 69.7 | 87.2    | 81.9 |
| Our DoRA   | 65.8| 51.2 | 79.5 |

TODO: Explain the our results in relation to the paper's. E.g. why is there a difference

# Conclusion

# References

# Acknoledgements

TODOS: 
- Empty sections
- Add data as described in handout
- TODO: restructure the repo and code
- LICENSE


