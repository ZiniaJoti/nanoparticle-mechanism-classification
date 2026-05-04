"""
config.py
=========
Central configuration for the Nanoparticle–Membrane Interaction Classification project.
All paths, hyperparameters, and model settings are defined here.
"""

import os
from dotenv import load_dotenv

load_dotenv()
ncbi_email = os.getenv("NCBI_EMAIL") 
ncbi_key = os.getenv("NCBI_API_KEY")
openai_key = os.getenv("OPENAI_API_KEY")
gemini_key=os.getenv("GEMINI_API_KEY")
# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_RAW_DIR       = os.path.join(BASE_DIR, "data", "raw")
DATA_LABELED_DIR   = os.path.join(BASE_DIR, "data", "labeled")
DATA_PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR         = os.path.join(BASE_DIR, "models")
RESULTS_DIR        = os.path.join(BASE_DIR, "results")
LOGS_DIR           = os.path.join(BASE_DIR, "logs")

# Create directories if they don't exist
for _dir in [DATA_RAW_DIR, DATA_LABELED_DIR, DATA_PROCESSED_DIR,
             MODELS_DIR, RESULTS_DIR, LOGS_DIR]:
    os.makedirs(_dir, exist_ok=True)

# ─────────────────────────────────────────────
# FILE NAMES
# ─────────────────────────────────────────────
RAW_CSV                 = os.path.join(DATA_RAW_DIR,       "abstracts_raw.csv")
RAW_JSONL               = os.path.join(DATA_RAW_DIR,       "abstracts_raw.jsonl")
LABELED_CSV             = os.path.join(DATA_LABELED_DIR,   "abstracts_labeled.csv")
MANUAL_REVIEW_CSV       = os.path.join(DATA_LABELED_DIR,   "manual_review.csv")
UNCERTAIN_LABELED_CSV   = os.path.join(DATA_LABELED_DIR,   "uncertain_labeled.csv")
PROCESSED_TRAIN         = os.path.join(DATA_PROCESSED_DIR, "train.csv")
PROCESSED_VAL           = os.path.join(DATA_PROCESSED_DIR, "val.csv")
PROCESSED_TEST          = os.path.join(DATA_PROCESSED_DIR, "test.csv")
PROCESSED_ALL           = os.path.join(DATA_PROCESSED_DIR, "all_clean.csv")

# ─────────────────────────────────────────────
# PUBMED SEARCH
# ─────────────────────────────────────────────
PUBMED_SEARCH_QUERY = (
    '('
    'nanoparticle[tiab] OR nanoparticles[tiab] OR nanocarrier[tiab] OR nanocarriers[tiab] '
    'OR nanomedicine[tiab] OR "drug delivery"[tiab] OR "drug carrier"[tiab]'
    ') '
    'AND '
    '('
    '"membrane wrapping"[tiab] OR wrapping[tiab] OR engulfment[tiab] OR engulfed[tiab] '
    'OR "partial wrapping"[tiab] OR "full wrapping"[tiab] OR "membrane deformation"[tiab] '
    'OR curvature[tiab] OR bending[tiab] OR '
    'adhesion[tiab] OR adhesive[tiab] OR "membrane adhesion"[tiab] OR attachment[tiab] OR binding[tiab] OR '
    'penetration[tiab] OR penetrate[tiab] OR penetrating[tiab] OR translocation[tiab] OR translocate[tiab] '
    'OR insertion[tiab] OR poration[tiab] OR pore[tiab] OR rupture[tiab] OR disruption[tiab] OR '
    'endocytosis[tiab] OR endocytic[tiab] OR internalization[tiab] OR internalisation[tiab] '
    'OR uptake[tiab] OR "cellular uptake"[tiab] OR "receptor-mediated"[tiab] '
    'OR "clathrin-mediated"[tiab] OR "caveolin-mediated"[tiab]'
    ') '
    'NOT '
    '('
    'review[pt] OR "systematic review"[tiab] OR "literature review"[tiab]'
    ')'
)

PUBMED_MAX_RESULTS  = 1600          # total abstracts to fetch
PUBMED_BATCH_SIZE   = 200           # PMIDs per request
PUBMED_RETRIES      = 3
PUBMED_RETRY_DELAY  = 5             # seconds between retries
PUBMED_EMAIL        = ncbi_email   
# ─────────────────────────────────────────────
# LLM LABELING
# ─────────────────────────────────────────────
# Supported backends: "gemini", "openai", "ollama", 
LLM_BACKEND         = "openai"      

# GEMINI settings  ← INSERT YOUR API KEY HERE
GEMINI_API_KEY      = gemini_key
GEMINI_MODEL        = "gemini-2.0-flash"

# OpenAI settings  ← INSERT YOUR API KEY HERE
OPENAI_API_KEY      = openai_key
OPENAI_MODEL        = "gpt-4o-mini"

# Ollama (local) settings – no key needed
OLLAMA_MODEL        = "llama3:latest"
OLLAMA_BASE_URL     = "http://localhost:11434"

LLM_TEMPERATURE     = 0.0           # deterministic labeling
LLM_MAX_TOKENS      = 300
LLM_BATCH_SAVE_EVERY = 50           # save progress every N abstracts
LLM_RETRIES         = 3
LLM_RETRY_DELAY     = 5             # seconds

# Confidence threshold below which a sample goes to manual review
UNCERTAIN_CONFIDENCE_THRESHOLD = 0.5

# ─────────────────────────────────────────────
# LABEL DEFINITIONS
# ─────────────────────────────────────────────
LABEL_NAMES = ["wrapping", "adhesion", "penetration", "endocytosis"]
LABEL2ID    = {label: idx for idx, label in enumerate(LABEL_NAMES)}
ID2LABEL    = {idx: label for label, idx in LABEL2ID.items()}
NUM_LABELS  = len(LABEL_NAMES)

# ─────────────────────────────────────────────
# TRAINING — SHARED HYPERPARAMETERS
# (keep identical for both models for fair comparison)
# ─────────────────────────────────────────────
RANDOM_SEED     = 42
MAX_LENGTH      = 256           # token limit per abstract
BATCH_SIZE      = 16
EVAL_BATCH_SIZE  = 32
LEARNING_RATE   = 2e-5
NUM_EPOCHS      = 5
WEIGHT_DECAY    = 0.01
WARMUP_RATIO    = 0.1
GRAD_ACCUM_STEPS = 1            # increase if GPU memory is limited

USE_CLASS_WEIGHTS = True
USE_FP16 = True

EARLY_STOPPING_PATIENCE = 2
SAVE_TOTAL_LIMIT = 2


TRAIN_RATIO     = 0.70
VAL_RATIO       = 0.15
TEST_RATIO      = 0.15          # must sum to 1.0

# =========================================================
# TRANSFORMER MODELS
# =========================================================
BERT_MODEL_NAME = "google-bert/bert-base-uncased"
SCIBERT_MODEL_NAME = "allenai/scibert_scivocab_uncased"
BIOBERT_MODEL_NAME = "dmis-lab/biobert-base-cased-v1.2"
PUBMEDBERT_MODEL_NAME = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"

MODEL_NAME_MAP = {
    "bert": BERT_MODEL_NAME,
    "scibert": SCIBERT_MODEL_NAME,
    "biobert": BIOBERT_MODEL_NAME,
    "pubmedbert": PUBMEDBERT_MODEL_NAME,
}

BERT_SAVE_DIR      = os.path.join(MODELS_DIR, "bert_finetuned")
SCIBERT_SAVE_DIR   = os.path.join(MODELS_DIR, "scibert_finetuned")
BIOBERT_SAVE_DIR   = os.path.join(MODELS_DIR, "biobert_finetuned")
PUBMEDBERT_SAVE_DIR= os.path.join(MODELS_DIR, "pubmedbert_finetuned")

MODEL_SAVE_DIR_MAP = {
    "bert": BERT_SAVE_DIR,
    "scibert": SCIBERT_SAVE_DIR,
    "biobert": BIOBERT_SAVE_DIR,
    "pubmedbert": PUBMEDBERT_SAVE_DIR,
}

# ─────────────────────────────────────────────
# RESULTS FILE NAMES
# ─────────────────────────────────────────────
BERT_METRICS_FILE         = os.path.join(RESULTS_DIR, "bert_metrics.json")
SCIBERT_METRICS_FILE      = os.path.join(RESULTS_DIR, "scibert_metrics.json")
BIOBERT_METRICS_FILE      = os.path.join(RESULTS_DIR, "biobert_metrics.json")
PUBMEDBERT_METRICS_FILE   = os.path.join(RESULTS_DIR, "pubmedbert_metrics.json")

COMPARISON_TABLE_CSV      = os.path.join(RESULTS_DIR, "model_comparison.csv")
COMPARISON_PLOT_FILE      = os.path.join(RESULTS_DIR, "model_comparison.png")
COMPARISON_PANELS_DIR     = os.path.join(RESULTS_DIR, "comparison_panels")

BERT_CONFUSION_MATRIX     = os.path.join(RESULTS_DIR, "bert_confusion_matrix.png")
SCIBERT_CONFUSION_MATRIX  = os.path.join(RESULTS_DIR, "scibert_confusion_matrix.png")
BIOBERT_CONFUSION_MATRIX  = os.path.join(RESULTS_DIR, "biobert_confusion_matrix.png")
PUBMEDBERT_CONFUSION_MATRIX = os.path.join(RESULTS_DIR, "pubmedbert_confusion_matrix.png")

BERT_TRAINING_CURVE       = os.path.join(RESULTS_DIR, "bert_training_curves.png")
SCIBERT_TRAINING_CURVE    = os.path.join(RESULTS_DIR, "scibert_training_curves.png")
BIOBERT_TRAINING_CURVE    = os.path.join(RESULTS_DIR, "biobert_training_curves.png")
PUBMEDBERT_TRAINING_CURVE = os.path.join(RESULTS_DIR, "pubmedbert_training_curves.png")


METRICS_FILE_MAP = {
    "bert": BERT_METRICS_FILE,
    "scibert": SCIBERT_METRICS_FILE,
    "biobert": BIOBERT_METRICS_FILE,
    "pubmedbert": PUBMEDBERT_METRICS_FILE,
}

CONFUSION_MATRIX_MAP = {
    "bert": BERT_CONFUSION_MATRIX,
    "scibert": SCIBERT_CONFUSION_MATRIX,
    "biobert": BIOBERT_CONFUSION_MATRIX,
    "pubmedbert": PUBMEDBERT_CONFUSION_MATRIX,
}

TRAINING_CURVE_MAP = {
    "bert": BERT_TRAINING_CURVE,
    "scibert": SCIBERT_TRAINING_CURVE,
    "biobert": BIOBERT_TRAINING_CURVE,
    "pubmedbert": PUBMEDBERT_TRAINING_CURVE,
}

EDA_PLOT_FILE             = os.path.join(RESULTS_DIR, "eda_plots.png")
