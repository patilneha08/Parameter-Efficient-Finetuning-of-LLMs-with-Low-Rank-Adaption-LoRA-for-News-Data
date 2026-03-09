# Parameter-Efficient-Finetuning-of-LLMs-with-Low-Rank-Adaption-LoRA-for-News-Data
This repository contains a Jupyter notebook focused on fine-tuning a RoBERTa-base model for sequence classification using Low-Rank Adaptation (LoRA). The notebook is designed to run in a Kaggle environment and utilizes the AG News dataset to classify news articles into four categories.

### Project Overview
The project demonstrates a parameter-efficient fine-tuning (PEFT) approach. By using LoRA, the model significantly reduces the number of trainable parameters, making the training process faster and less memory-intensive while maintaining high performance.

### Key Technical Specifications:

<b>Base Model</b>: roberta-base.

<b>Adaptation Method</b>: LoRA (Low-Rank Adaptation) with a rank (r) of 4 and an alpha (α) of 16.

<b>Trainable Parameters</b>: Approximately 0.80% (999,172 out of ~125.6 million).

<b>Hardware</b>: Optimized for NVIDIA Tesla T4 GPU.

### Notebook Workflow
The notebook follows a standard machine learning pipeline:

#### 1. Environment Setup: 
Imports necessary libraries such as torch, transformers, datasets, and peft.

#### 2. Data Loading & Preprocessing:

Loads the ag_news dataset.

Performs text cleaning (removing newlines and HTML entities) and tokenizes the text with a maximum length of 128.

#### 3. Model Configuration:

Initializes roberta-base for sequence classification with 4 labels.

Applies LoRA to the query, value, and output.dense layers.

#### 4. Training:

Configures TrainingArguments with a cosine learning rate scheduler, weight decay, and mixed-precision training (bf16).

Uses the Hugging Face Trainer API for 3 epochs.

#### 5. Model Saving & Loading: 
Saves the fine-tuned adapter weights, tokenizer, and configuration to a saved_model directory.

#### 6. Inference:

Loads the saved model and tokenizer.

Generates predictions for a provided unlabelled test dataset (test_unlabelled.pkl).

#### 7. Results Visualization: 
Plots training vs. validation loss and validation accuracy curves.

### Requirements
To run this notebook, the following packages are required:

torch

transformers

datasets

peft

scikit-learn

pandas

numpy

matplotlib

### File Outputs
saved_model/: Directory containing the adapter weights and tokenizer.

submission.csv: Final prediction file containing ID and label for the test set.

### Performance Summary
During training, the model achieved the following metrics:

Epoch 1: Accuracy ~91.55%.

Epoch 2: Accuracy ~92.57%.

Epoch 3: Accuracy ~92.71%.
