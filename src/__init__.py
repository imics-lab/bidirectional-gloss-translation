"""
Sign Language Translation Pipeline
Fine-tuning Pre-trained Language Models for Bidirectional Sign Language Translation

Supports:
- Models: T5, Flan-T5, mBART, LLaMA 3.1
- Datasets: RWTH-PHOENIX-14T, SIGNUM, ASLG-PC12
- Tasks: Gloss-to-Text (G2T) and Text-to-Gloss (T2G)
"""

from .pipeline import SignLanguageTranslationPipeline


__all__ = ["SignLanguageTranslationPipeline"]

# Model shortcuts
SUPPORTED_MODELS = {
    "t5-small": "t5-small",
    "flan-t5-small": "google/flan-t5-small", 
    "mbart-small": "facebook/mbart-large-50-many-to-many-mmt",
    "llama-8b": "meta-llama/Meta-Llama-3.1-8B"
}

# Dataset information
SUPPORTED_DATASETS = {
    "phoenix": "RWTH-PHOENIX-Weather 2014T",
    "signum": "SIGNUM", 
    "aslg": "ASLG-PC12"
}