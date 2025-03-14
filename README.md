# Implementation of _DoRA: Towards Weight-Decomposed Low-Rank Adaptation_


This project is an attempt to replicate the results of DoRA, a parameter-efficient fine-tuning (PEFT) method that improved on the previous standard (LoRA) by decomposing pre-trained weight into magnitude and direction. The work by Shih-Yang Liu et. al consistently outperformed LoRA on fine-tuning tasks with several open source base models. This project was completed by Rolando Rodríguez, Chris Fernandes, Ishaan Nanal, and Gerardo Montemayor and all credit is given to the writers of the original paper for their work.

_DoRA: Weight-Decomposed Low-Rank Adaptation_, available at https://arxiv.org/abs/2402.09353