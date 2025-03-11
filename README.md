# cs-4782-final-project

## Ideas
https://www.reddit.com/r/MachineLearning/comments/16ij18f/d_the_ml_papers_that_rocked_our_world_20202023/

### NLP
- Small language model (probably still infeasible)
    - Language Models are Few-Shot Learners is safely a very infeasible paper to replicate
- Post-training LLaMA base model 
    - Grok says a 7B parameter model can be feasibly fine-tuned on a single GPU or trained from scratch on a small corpus with 2-4 GPUs. 

### Image
#### Classification
- ViT-Tiny, ViT-Small can be feasibly trained on smaller datasets on a single GPU
    - Fine-tuning tasks would be easier
- Vision Transformers Need Registers paper
- YOLOv4 --> feasible and trainable on a signle GPU
- Swin transformer --> hierarchical vision transformer using shifted windows

#### Diffusion
- Simplified image generation ("High-Resolution Image Synthesis with Latent Diffusion Models")

### Audio
- FastSpeech 2: Fast and High-Quality End-to-End Text to Speech
- VITS: Conditional Variational Autoencoder with Adversarial Learning for End-to-End Text-to-Speech - 2021
- Wav2Vec 2.0: A Framework for Self-Supervised Learning of Speech Representations - 2020

### Misc
- Multi-modal learning with CLIP

## Roly's favorite ideas:
- LLaMA post-training (or even pre-training, if it's on a small enough corpus)
- Wav2Vec and FastSpeech
- Multi-modal models like CLIP

## Chris's favorite ideas:
- ViT (on sign up sheet)
- A Watermark for Large Language Models (on sign up sheet)
- Images that Sound (on sign up sheet, but looks potentially challenging)
