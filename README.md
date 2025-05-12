# Introduction

This project is an attempt to replicate the results of DoRA, a parameter-efficient fine-tuning (PEFT) method that improved on the previous standard (LoRA) by decomposing pre-trained weight into magnitude and direction. The work by Shih-Yang Liu et. al consistently outperformed LoRA on fine-tuning tasks with several open source base models. This project was completed by Rolando Rodríguez, Chris Fernandes, Ishaan Nanal, and Gerardo Montemayor and all credit is given to the writers of the original paper for their work.

# Chosen Result

We were particularly interested in reproducing the superior performance of DoRA over LoRA on commonsense reasoning tasks. These are located in Table 1 of the original paper. 

# Github Contents

Refer to `./code/peft/src/peft/tuners/dora.py` for the implementation of DoRA.

Refer to `./code/finetune.py` for the LlaMA-7B finetuning script using either DoRA or LoRA.

Refer to `.code/eval.py` for the evaluation script of the finetuned LlaMA-7B model.

# Re-implementation Details

DoRA is an extension of LoRA, so, like the authors of DoRA, we made use of the existing LoRA implementation to guide our implementation of DoRA. We struggled a lot with memory management, so we ultimately only tested Llama-7B on BoolQ, Winogrande, and Hellaswag. We compared accuracy after finetuning for LoRA vs DoRA. We were unable to train on the full dataset used in the paper (a combination of several commonsense reasoning datasets). Instead, we trained and then evaluated our model on BoolQ, HellaSwag, and Winogrande independently.

# Reproduction Steps
## Setup
Create a conda environment as described below.
```bash
conda create -n dora_llama python=3.10
conda activate dora_llama
pip install -r requirements.txt
```

Computational Requirements: A GPU with at least 40GB of VRAM. We used an A100 GPU.

## Finetuning and Evaluation

### Finetuning (`./finetune.sh`)
This file contains the code to finetune LLaMA-7B using DoRA. User can specify different DoRA configuration for finetuning. To be specific, the first argument denotes the destination for saving the fine-tuned model, the second argument specifies whether you want to use lora or dora for finetuning, and the third argument denotes the destination of the training data.
 
For example, to finetune LLaMA-7B on BoolQ using DoRA, you could do:
```
sh finetune.sh ./finetuned_result/dora_r32 dora ../data/boolq/train.json
```

### Evaluation (`./eval.sh`)

This file contains the code to evaluate LLaMA-7B finetuned with DoRA or LoRA on any of the commonsense reasoning tasks. The first argument is the directory path of the model weights, the second argument determines whether the weights are from DoRA or LoRA, and the third argument specifies what dataset to evaluate on.

For example, to evaluate a LLaMA-7B model that was trained on BoolQ using DoRA, you could do:
```
sh eval.sh ./finetuned_result/dora_r32 dora boolq
```

# Results/Insights
Due to our limited compute, we were unable to train on the full dataset used in the paper (a combination of several commonsense reasoning datasets). Instead, we trained and then evaluated our model on BoolQ, HellaSwag, and Winogrande independently. Our results, and the original paper's results, are below. 

| PEFT Strategy | BoolQ | HellaSwag | Winogrande |
|----------|----------|----------|----------|
| Paper's LoRA| 67.5 | 83.4     | 80.4     |
| Our LoRA   | 63.5  | 41.2     | 74.7    |
| Paper's DoRA | 69.7 | 87.2    | 81.9 |
| Our DoRA   | 65.8| 51.2 | 79.5 |

Our approach yielded similar accuracy on BoolQ and Winogrande, but on HellaSwag our accuracy was much lower. We suspect that this is due to HellaSwag containing a more complex task (correctly completing a sentence by choosing one of 4 offerred endings), whereas BoolQ and Winogrande are binary tasks (true/false and fill in the blank given only two options). Regardless, we still showed that DoRA was able to outperform LoRA on all three datasets.

# Conclusion
- DoRA outperforms LoRA while keeping marginal parameter count low.
- Decoupling direction and magnitude allows gradient updates to optimize both independently.
- Normalization as a remedy for instability can be extended to fine-tuning.

# References
- _DoRA: Weight-Decomposed Low-Rank Adaptation_, available at https://arxiv.org/abs/2402.
- [DoRA](https://github.com/NVlabs/DoRA)
- [LLM-Adapter](https://github.com/AGI-Edgerunners/LLM-Adapters)
- [PEFT](https://github.com/huggingface/peft)

# Acknoledgements
This was our final project for Cornell University's CS 4782: Intro to Deep Learning, taught by Kilian Q. Weinberger and Jennifer J. Sun.

