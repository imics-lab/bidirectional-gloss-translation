## Bidirectional Sign Language Translation with Pre-trained Language Models

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the implementation for **"Finetuning Pre-trained Language Models for Bidirectional Sign Language Gloss to Text Translation"** - a comprehensive study comparing fine-tuned pre-trained language models against transformer models trained from scratch for sign language gloss translation.

## Overview

Our research demonstrates that fine-tuning large pre-trained language models significantly outperforms training from scratch for bidirectional sign language gloss translation tasks. We evaluate multiple PLMs across three benchmark datasets with state-of-the-art results.

### Supported Models

| Model | Parameters | Architecture | Performance Highlights |
|-------|------------|--------------|----------------------|
| **T5-small** | 220M | Encoder-Decoder | Strong baseline performance |
| **Flan-T5-small** | 220M | Encoder-Decoder | Instruction-tuned advantages |
| **mBART-small** | 125M | Encoder-Decoder | Multilingual denoising benefits |
| **LLaMA 3.1 8B** | 8B | Decoder-only | Best overall performance |


## Results

Fine-tuned PLMs achieve **74-130% relative BLEU-4 improvements** over baseline transformers. Results averaged over 10 runs:

### Gloss-to-Text Performance (BLEU-4)
| Dataset | Baseline | T5 | Flan-T5 | mBART | **LLaMA 8B** |
|---------|----------|----|---------| ------|-------------|
| PHOENIX-14T | 13.06 | 22.73 | 19.03 | 25.58 | **29.92** |
| SIGNUM | 34.24 | 52.87 | 50.72 | **67.60** | 65.78 |
| ASLG-PC12 | 62.81 | 68.69 | 65.40 | 79.58 | **83.10** |

### Text-to-Gloss Performance (BLEU-4)
| Dataset | Baseline | T5 | Flan-T5 | mBART | **LLaMA 8B** |
|---------|----------|----|---------| ------|-------------|
| PHOENIX-14T | 6.98 | 8.49 | 10.00 | 12.10 | **16.81** |
| SIGNUM | 10.30 | **34.66** | 32.44 | 25.43 | 29.74 |
| ASLG-PC12 | 14.50 | 21.73 | 18.51 | 27.68 | **55.21** |

**Key Findings:** LLaMA 8B excels on large datasets; mBART performs best on controlled vocabularies; T2G remains 30-60% more challenging than G2T.

## Installation

### Requirements
- Python 3.8+
- CUDA-capable GPU (8GB+ VRAM recommended, 16GB+ for LLaMA)
- 16GB+ system RAM (32GB+ recommended for LLaMA)

### Quick Setup
```bash
git clone https://github.com//imics-lab/bidirectional-gloss-translation.git
cd bidirectional-gloss-translation

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Quick Start
```bash
# Train T5-small on SIGNUM dataset for gloss-to-text
python train.py --model t5-small --dataset signum --task g2t

# Train LLaMA on PHOENIX for both directions
python train.py --model llama-8b --dataset phoenix --task both

# Train mBART on ASLG with custom settings
python train.py --model mbart-small --dataset aslg --epochs 10 --batch_size 4
```

### Programmatic Usage
```python
from src/pipeline import SignLanguageTranslationPipeline

# Initialize pipeline
pipeline = SignLanguageTranslationPipeline(
    model_name="t5-small",
    dataset_type="signum"
)

# Load and preprocess data
df = pipeline.load_dataset()
df = pipeline.preprocess_data(df, sample_size=1000)

# Prepare for training
ds, val_df = pipeline.prepare_data_for_training(df, task="g2t")

# Train model
tokenizer, model = pipeline.load_model_and_tokenizer()
tok_ds = pipeline.tokenize_data(ds)
trainer = pipeline.train_model(tok_ds)

# Generate translations
text = "I need help with my homework"
gloss = pipeline.generate_single_translation(text, task="t2g")
print(f"Generated gloss: {gloss}")
```

### Available Options

#### Models (`--model`)
- `t5-small`: T5-small (220M params)
- `flan-t5-small`: Flan-T5-small (220M params)  
- `mbart-small`: mBART-small (125M params)
- `llama-8b`: LLaMA 3.1 8B (8B params)

#### Datasets (`--dataset`)
- `signum`: SIGNUM dataset (DGS ↔ German)
- `phoenix`: RWTH-PHOENIX-14T (DGS ↔ German)
- `aslg`: ASLG-PC12 (ASL ↔ English)

#### Tasks (`--task`)
- `g2t`: Gloss-to-Text translation
- `t2g`: Text-to-Gloss translation  
- `both`: Train both directions sequentially


## Model Performance

### Hardware Requirements

| Model | Min VRAM | Recommended VRAM | Training Time* |
|-------|----------|------------------|----------------|
| T5-small | 4GB | 8GB | ~2 hours |
| Flan-T5-small | 4GB | 8GB | ~2 hours |
| mBART-small | 6GB | 8GB | ~2.5 hours |
| LLaMA 8B | 12GB | 16GB+ | ~8 hours |

*Approximate times for 1000 samples, 5 epochs on RTX 4090*

### Evaluation Metrics
- **BLEU-1/2/3/4**: N-gram precision scores
- **ROUGE-L**: Longest common subsequence
- **METEOR**: Alignment-based semantic evaluation  
- **WER**: Word Error Rate

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
