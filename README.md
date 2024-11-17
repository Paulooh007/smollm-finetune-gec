# SmolLM-135M finetuned for Grammatical Error Correction (GEC) 

> This project was completed as part of the application process for the Cohere AI Research Scholar Program 2025 cohort.


## Overview
This project implements and fine-tunes the [SmolLM-135M](https://huggingface.co/HuggingFaceTB/SmolLM-135M) language model for grammatical error correction (GEC) using the Grammarly CoEdIT dataset. The implementation includes three main training approaches: Supervised Fine-Tuning (SFT), Direct Preference Optimization (DPO), and Contrastive Preference Optimization (CPO).

## Components

### 1. Base Model Implementation
- Implements custom SmolLM architecture with bug fixes
- Includes key components like RopeAttention, MLP, RMSNorm, and LlamaDecoder
- Built on PyTorch framework

### 2. Training Pipeline
The training process consists of three phases:

1. **Supervised Fine-Tuning (SFT)**
   - Fine-tunes SmolLM on the CoEdIT dataset
   - Uses filtered GEC-specific training data
   - Achieves BLEU score of ~0.48 after one epoch

2. **Direct Preference Optimization (DPO)**
   - Creates preference dataset using edit distance metric
   - Generates text variants using different temperature settings
   - Improves BLEU score to ~0.50

3. **Contrastive Preference Optimization (CPO)**
   - Implements memory-efficient alternative to DPO
   - Uses contrastive learning approach
   - Provides comparable performance to DPO with better resource utilization

## Dataset
- Uses Grammarly [CoEdIT](https://huggingface.co/datasets/grammarly/coedit) dataset
- Filtered to focus on GEC tasks
- Training set: ~19,823 samples
- Test set: ~485 samples

## Performance
The model shows progressive improvement through different training stages:
- Base SFT: BLEU score ~0.48
- After DPO: BLEU score ~0.50
- CPO provides comparable results with better efficiency

## Technical Details
- Model: SmolLM-135M
- Framework: PyTorch
- Key Libraries: transformers, trl, datasets
- Training Parameters:
  - Max sequence length: 350
  - Batch size: Variable (adjusted for memory constraints)
  - Learning rates: 7e-5 (SFT), 5e-7 (DPO/CPO)

## Implementation Environment
- All experiments were conducted using Google Colab's free tier
- Hardware: NVIDIA T4 GPU 
- This setup demonstrates the model's ability to achieve meaningful results with limited computational resources