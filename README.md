# Nanoparticle–Membrane Interaction Mechanism Classification

This project presents an end-to-end Natural Language Processing (NLP) pipeline for classifying nanoparticle–membrane interaction mechanisms from PubMed abstracts using transformer-based models.

The study compares the performance of:
- BERT
- SciBERT
- BioBERT
- PubMedBERT

## Overview

Understanding how nanoparticles interact with biological membranes is critical in drug delivery, nanomedicine, and cellular uptake. This project uses NLP techniques to automatically classify interaction mechanisms from biomedical literature.

### Target Classes
- Adhesion
- Wrapping
- Penetration
- Endocytosis

---

## Project Pipeline

1. Data collection from PubMed
2. LLM-assisted labeling
3. Manual review (optional)
4. Data preprocessing and splitting
5. Exploratory data analysis
6. Fine-tuning transformer models
7. Evaluation and comparison
8. Inference on new abstracts

---

## Project Structure

```text
.
├── config.py                         # Configuration (paths, parameters)
├── requirements.txt                 # Dependencies
├── prepare_dataset.ipynb            # Optional notebook workflow
├── pubmed_fetch.py                  # Fetch abstracts from PubMed
├── label_abstracts_llm.py           # LLM-based labeling
├── manual_review_prepare.py         # Prepare uncertain samples for review
├── preprocess_dataset.py            # Cleaning + splitting dataset
├── exploratory_analysis.py          # Basic EDA
├── exploratory_data_analysis.py     # Additional EDA plots
├── train_bert.py                    # Train BERT
├── train_scibert.py                 # Train SciBERT
├── train_biobert.py                 # Train BioBERT
├── train_pubmedbert.py              # Train PubMedBERT
├── infer_new_abstracts.py           # Inference script
