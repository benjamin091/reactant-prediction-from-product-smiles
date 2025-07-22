# Reactant Prediction from Product SMILES

## Overview

This project addresses the challenge of predicting reactant molecules from the SMILES representation of product molecules in chemical reactions. The task is framed as a **sequence-to-sequence translation problem**, where a machine learning model learns to map from product SMILES to corresponding reactant SMILES.

To solve this task, we apply a **pretrained Transformer model**, which is fine-tuned on a curated dataset of reaction SMILES pairs. The pipeline includes data preprocessing, SMILES tokenization, model fine-tuning, and evaluation based on string accuracy and chemical validity.

---

## Objectives

- Train a model to predict reactant SMILES from product SMILES
- Leverage a pretrained Transformer architecture and apply transfer learning
- Optimize the model to generate valid, novel, and accurate reactant predictions

---

## Workflow

### 1. **Data Preprocessing**
- Raw data in the form `[Reactants SMILES] >> [Products SMILES]`
- Reactions are parsed and split into input-output pairs:  
  `X = Product SMILES`, `y = Reactant SMILES`
- The SMILES strings are standardized and cleaned

### 2. **Tokenization**
- SMILES strings are tokenized into meaningful substructures or character-level tokens
- Special tokens (e.g. `[START]`, `[END]`) are added to mark sequence boundaries
- A vocabulary is constructed and applied consistently across dataset splits

### 3. **Modeling**
- A **Transformer-based encoder-decoder model** is used (e.g. pretrained on chemical language modeling tasks)
- Fine-tuning is performed on the reaction prediction task using cross-entropy loss
- Teacher forcing is applied during training to improve convergence

### 4. **Training**
- The pretrained model is fine-tuned on the preprocessed dataset using tokenized inputs
- The training loop includes:
  - Batch-wise token generation
  - Masked attention for autoregressive decoding
  - Validation set monitoring

### 5. **Prediction & Evaluation**
- The trained model is used to generate reactants for unseen product SMILES
- Evaluation metrics include:
  - **Sequence accuracy**
  - **BLEU score**
  - **SMILES validity (via RDKit)**
  - **Levenshtein distance / Tanimoto similarity** to ground-truth
- Example predictions are visualized and analyzed

---

## Technologies Used

- Python 3.x  
- Hugging Face Transformers (or OpenNMT / custom Transformer)  
- RDKit (for chemical validation and SMILES operations)  
- PyTorch or TensorFlow (model training & fine-tuning)  
- pandas, numpy, matplotlib, seaborn



