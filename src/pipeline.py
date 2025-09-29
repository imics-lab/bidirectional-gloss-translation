"""
Main pipeline for Signum Text-to-Gloss Translation
"""

import torch
import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from sklearn.model_selection import train_test_split
import evaluate
from tqdm import tqdm
import warnings
import os
import gc
import re
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
import nltk
import jiwer
from rouge_score import rouge_scorer

# Download required NLTK data
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)

# Disable external logging
os.environ["WANDB_DISABLED"] = "true"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SignLanguageTranslationPipeline:
    """
    Pipeline for bidirectional sign language gloss translation
    Supports multiple datasets and pre-trained models
    """
    
    SUPPORTED_MODELS = {
        "t5-small": "t5-small",
        "flan-t5-small": "google/flan-t5-small", 
        "mbart-small": "facebook/mbart-large-50-many-to-many-mmt",
        "llama-8b": "meta-llama/Meta-Llama-3.1-8B"
    }
    
    SUPPORTED_DATASETS = {
        "phoenix": "RWTH-PHOENIX-Weather 2014T",
        "signum": "SIGNUM", 
        "aslg": "ASLG-PC12"
    }
    
    def __init__(self, model_name="t5-small", dataset_type="signum"):
        if model_name in self.SUPPORTED_MODELS:
            self.model_name = self.SUPPORTED_MODELS[model_name]
        else:
            self.model_name = model_name  # Allow custom model names
            
        self.dataset_type = dataset_type
        self.tokenizer = None
        self.model = None
        self.peft_model = None
        self.bleu_metric = evaluate.load("bleu")
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=False)
        
        print(f"Initialized pipeline for {self.dataset_type} dataset with {model_name} model")

    def parse_dataset_file(self, filepath, file_type='gloss', dataset_format='auto'):
        """Parse dataset files with format detection"""
        data = []
        
        with open(filepath, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f):
                line = line.strip()
                if not line or line.startswith('#'):
                    continue

                # Try different delimiters based on dataset
                delimiters = ['\t', '|', '::'] if dataset_format == 'auto' else ['\t']
                
                for delimiter in delimiters:
                    if delimiter in line:
                        parts = line.split(delimiter, 1)
                        if len(parts) == 2:
                            data.append({
                                'id': parts[0].strip(),
                                file_type: parts[1].strip()
                            })
                        break
                else:
                    # Handle single column format (for some ASLG files)
                    if dataset_format == 'aslg' or self.dataset_type == 'aslg':
                        data.append({
                            'id': f"aslg_{line_num:06d}",
                            file_type: line.strip()
                        })

        return data

    def load_dataset(self, **kwargs):
        """Load dataset based on dataset type"""
        if self.dataset_type == "signum":
            return self.load_signum_dataset(**kwargs)
        elif self.dataset_type == "phoenix":
            return self.load_phoenix_dataset(**kwargs)
        elif self.dataset_type == "aslg":
            return self.load_aslg_dataset(**kwargs)
        else:
            raise ValueError(f"Unsupported dataset type: {self.dataset_type}")

    def load_signum_dataset(self, anno_path='data/signum_sents_anno_eng.txt',
                           trans_path='data/signum_sents_trans_eng.txt'):
        """Load and merge Signum dataset files"""
        gloss_data = self.parse_dataset_file(anno_path, 'gloss')
        trans_data = self.parse_dataset_file(trans_path, 'translation')

        gloss_df = pd.DataFrame(gloss_data)
        trans_df = pd.DataFrame(trans_data)
        df = pd.merge(gloss_df, trans_df, on='id', how='inner')

        print(f"Loaded {len(df)} parallel examples from SIGNUM dataset")
        return df
    
    def load_phoenix_dataset(self, train_path='data/phoenix2014T.train.gloss',
                           dev_path='data/phoenix2014T.dev.gloss',
                           test_path='data/phoenix2014T.test.gloss'):
        """Load RWTH-PHOENIX-Weather 2014T dataset"""
        print("Loading RWTH-PHOENIX-Weather 2014T dataset...")
        
        all_data = []
        for split, path in [('train', train_path), ('dev', dev_path), ('test', test_path)]:
            if os.path.exists(path):
                with open(path, 'r', encoding='utf-8') as f:
                    for i, line in enumerate(f):
                        parts = line.strip().split('|')
                        if len(parts) >= 3:
                            all_data.append({
                                'id': f"{split}_{i:06d}",
                                'gloss': parts[0].strip(),
                                'translation': parts[1].strip(),
                                'split': split
                            })
        
        df = pd.DataFrame(all_data)
        print(f"Loaded {len(df)} parallel examples from PHOENIX dataset")
        return df
    
    def load_aslg_dataset(self, gloss_path='data/aslg_pc12_glosses.txt',
                         text_path='data/aslg_pc12_text.txt'):
        """Load ASLG-PC12 dataset"""
        print("Loading ASLG-PC12 dataset...")
        
        # ASLG dataset might have parallel files or single combined file
        if os.path.exists(gloss_path) and os.path.exists(text_path):
            gloss_data = self.parse_dataset_file(gloss_path, 'gloss', 'aslg')
            trans_data = self.parse_dataset_file(text_path, 'translation', 'aslg')
            
            gloss_df = pd.DataFrame(gloss_data)
            trans_df = pd.DataFrame(trans_data)
            df = pd.merge(gloss_df, trans_df, on='id', how='inner')
        else:
            # Try to find combined file
            combined_path = 'data/aslg_pc12_combined.txt'
            if os.path.exists(combined_path):
                with open(combined_path, 'r', encoding='utf-8') as f:
                    data = []
                    for i, line in enumerate(f):
                        if '\t' in line:
                            parts = line.strip().split('\t')
                            if len(parts) >= 2:
                                data.append({
                                    'id': f"aslg_{i:06d}",
                                    'translation': parts[0].strip(),
                                    'gloss': parts[1].strip()
                                })
                df = pd.DataFrame(data)
            else:
                raise FileNotFoundError("ASLG dataset files not found")
        
        print(f"Loaded {len(df)} parallel examples from ASLG-PC12 dataset")
        return df

    def preprocess_signum_data(self, df, sample_size=None):
        """Preprocess Signum data for training"""
        # Clean text columns
        df['translation'] = df['translation'].apply(lambda x: x.strip().lower())
        df['gloss'] = df['gloss'].apply(lambda x: x.strip())  # Preserve case in glosses

        # Remove empty entries
        df = df[(df['translation'].str.len() > 0) & (df['gloss'].str.len() > 0)]

        # Sample if requested (for memory efficiency during testing)
        if sample_size and len(df) > sample_size:
            df = df.sample(sample_size, random_state=42).reset_index(drop=True)

        print(f"Preprocessed {len(df)} examples")
        return df

    def prepare_data_for_training(self, df, test_size=0.2, task="g2t"):
        """Prepare data with appropriate prompts for different models and tasks"""
        
        def create_prompt(text, gloss, task_direction="g2t"):
            is_llama = "llama" in self.model_name.lower()
            
            if is_llama:
                # LLaMA format with chat templates
                if task_direction == "g2t":  # Gloss to Text
                    return (
                        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
                        "You are an expert translator converting sign language gloss notation to natural language text. "
                        f"Follow {self.dataset_type.upper()} dataset conventions.\n\n"
                        "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
                        f"Convert this sign language gloss to text: {gloss}<|eot_id|>"
                        "<|start_header_id|>assistant<|end_header_id|>\n\n"
                        f"{text}<|eot_id|>"
                    )
                else:  # Text to Gloss
                    return (
                        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
                        "You are an expert translator converting natural language text to sign language gloss notation. "
                        f"Follow {self.dataset_type.upper()} gloss notation rules.\n\n"
                        "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
                        f"Convert this text to sign language gloss: {text}<|eot_id|>"
                        "<|start_header_id|>assistant<|end_header_id|>\n\n"
                        f"{gloss}<|eot_id|>"
                    )
            else:
                # T5/mBART format
                if task_direction == "g2t":
                    return f"translate {self.dataset_type} gloss to text: {gloss}"
                else:
                    return f"translate text to {self.dataset_type} gloss: {text}"

        # Create training prompts based on task
        if task == "g2t":
            df['training_text'] = df.apply(
                lambda r: create_prompt(r['translation'], r['gloss'], "g2t"), axis=1
            )
            df['target'] = df['translation']
        else:  # t2g
            df['training_text'] = df.apply(
                lambda r: create_prompt(r['translation'], r['gloss'], "t2g"), axis=1
            )
            df['target'] = df['gloss']

        # Split into train and validation
        train_df, val_df = train_test_split(
            df[['training_text', 'target', 'translation', 'gloss']],
            test_size=test_size,
            random_state=42
        )

        # Create datasets
        train_ds = Dataset.from_pandas(train_df[['training_text', 'target']].reset_index(drop=True))
        val_ds = Dataset.from_pandas(val_df[['training_text', 'target']].reset_index(drop=True))

        return DatasetDict({'train': train_ds, 'validation': val_ds}), val_df

    def load_model_and_tokenizer(self):
        """Load and configure model with appropriate settings for different architectures"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Model-specific configurations
        is_llama = "llama" in self.model_name.lower()
        is_mbart = "mbart" in self.model_name.lower()
        
        if is_llama:
            # LLaMA requires quantization for memory efficiency
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
            )

            base_model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=bnb_config,
                device_map='auto',
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                trust_remote_code=True
            )
            self.model = prepare_model_for_kbit_training(base_model)

            # LoRA configuration for LLaMA
            lora_cfg = LoraConfig(
                r=16,
                lora_alpha=32,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                              "gate_proj", "up_proj", "down_proj"],
                lora_dropout=0.1,
                bias="none",
                task_type=TaskType.CAUSAL_LM
            )
            self.peft_model = get_peft_model(self.model, lora_cfg)
            
        else:
            # T5, Flan-T5, mBART - encoder-decoder models
            from transformers import AutoModelForSeq2SeqLM
            
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map='auto' if torch.cuda.is_available() else None
            )
            
            # Lighter LoRA for smaller models
            lora_cfg = LoraConfig(
                r=8,
                lora_alpha=16,
                target_modules=["q", "v"] if not is_mbart else ["q_proj", "v_proj"],
                lora_dropout=0.1,
                bias="none",
                task_type=TaskType.SEQ_2_SEQ_LM
            )
            self.peft_model = get_peft_model(self.model, lora_cfg)

        print(f"Model loaded: {self.model_name}")
        if hasattr(self.peft_model, 'print_trainable_parameters'):
            self.peft_model.print_trainable_parameters()
        
        return self.tokenizer, self.peft_model

    def tokenize_data(self, ds, max_length=512):
        """Tokenize dataset for training - handles both encoder-decoder and decoder-only models"""
        is_llama = "llama" in self.model_name.lower()
        
        def tokenize_fn(ex):
            if is_llama:
                # Decoder-only model (LLaMA)
                tok = self.tokenizer(
                    ex['training_text'],
                    truncation=True,
                    padding='max_length',
                    max_length=max_length
                )
                tok['labels'] = tok['input_ids'].copy()
            else:
                # Encoder-decoder model (T5, mBART)
                inputs = self.tokenizer(
                    ex['training_text'],
                    truncation=True,
                    padding='max_length',
                    max_length=max_length//2
                )
                targets = self.tokenizer(
                    ex['target'],
                    truncation=True,
                    padding='max_length',
                    max_length=max_length//2
                )
                tok = {
                    'input_ids': inputs['input_ids'],
                    'attention_mask': inputs['attention_mask'],
                    'labels': targets['input_ids']
                }
            return tok

        return ds.map(
            tokenize_fn,
            batched=True,
            remove_columns=ds['train'].column_names,
            batch_size=32
        )

    def train_model(self, tok_ds, output_dir="./model_output", epochs=5, batch_size=2, learning_rate=2e-4):
        """Train the model with optimized settings for different architectures"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

        is_llama = "llama" in self.model_name.lower()
        
        if is_llama:
            # Decoder-only model training
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm=False,
                pad_to_multiple_of=8
            )
        else:
            # Encoder-decoder model training
            from transformers import DataCollatorForSeq2Seq
            data_collator = DataCollatorForSeq2Seq(
                tokenizer=self.tokenizer,
                model=self.peft_model,
                pad_to_multiple_of=8
            )

        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=4,
            eval_strategy='steps',
            eval_steps=100,
            save_steps=200,
            logging_steps=25,
            learning_rate=learning_rate,
            bf16=torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8,
            fp16=torch.cuda.is_available() and torch.cuda.get_device_capability()[0] < 8,
            save_total_limit=3,
            remove_unused_columns=True,
            gradient_checkpointing=True,
            warmup_ratio=0.1,
            weight_decay=0.01,
            load_best_model_at_end=True,
            metric_for_best_model="loss",
            report_to=None,  # Disable wandb
        )

        trainer = Trainer(
            model=self.peft_model,
            args=training_args,
            train_dataset=tok_ds['train'],
            eval_dataset=tok_ds['validation'],
            tokenizer=self.tokenizer,
            data_collator=data_collator
        )

        trainer.train()
        trainer.save_model()
        self.tokenizer.save_pretrained(output_dir)
        return trainer

    def preprocess_data(self, df, sample_size=None):
        """Preprocess data for training - works with all datasets"""
        # Clean text columns
        df['translation'] = df['translation'].apply(lambda x: x.strip().lower())
        df['gloss'] = df['gloss'].apply(lambda x: x.strip())  # Preserve case in glosses

        # Remove empty entries
        df = df[(df['translation'].str.len() > 0) & (df['gloss'].str.len() > 0)]

        # Sample if requested (for memory efficiency during testing)
        if sample_size and len(df) > sample_size:
            df = df.sample(sample_size, random_state=42).reset_index(drop=True)

        print(f"Preprocessed {len(df)} examples for {self.dataset_type} dataset")
        return df

    def generate_single_translation(self, text, max_new_tokens=50, task="g2t"):
        """Generate translation for a single input"""
        is_llama = "llama" in self.model_name.lower()
        
        if is_llama:
            # LLaMA chat format
            if task == "g2t":
                prompt = (
                    "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
                    f"You are an expert translator converting {self.dataset_type} sign language gloss to natural text.\n\n"
                    "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
                    f"Convert this sign language gloss to text: {text}<|eot_id|>"
                    "<|start_header_id|>assistant<|end_header_id|>\n\n"
                )
            else:
                prompt = (
                    "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
                    f"You are an expert translator converting text to {self.dataset_type} sign language gloss notation.\n\n"
                    "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
                    f"Convert this text to sign language gloss: {text}<|eot_id|>"
                    "<|start_header_id|>assistant<|end_header_id|>\n\n"
                )

            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=256
            ).to(self.peft_model.device)

            out = self.peft_model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                num_return_sequences=1,
                do_sample=False,
                temperature=0.1,
                top_p=0.9,
                early_stopping=True,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.eos_token_id,
                no_repeat_ngram_size=3
            )

            raw = self.tokenizer.decode(
                out[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            )

            # Extract just the translation part
            result = raw.split('\n')[0].strip()
            if '<|' in result:
                result = result.split('<|')[0].strip()
        
        else:
            # T5/mBART format
            if task == "g2t":
                prompt = f"translate {self.dataset_type} gloss to text: {text}"
            else:
                prompt = f"translate text to {self.dataset_type} gloss: {text}"
            
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=256
            ).to(self.peft_model.device)
            
            out = self.peft_model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                num_return_sequences=1,
                do_sample=False,
                early_stopping=True
            )
            
            result = self.tokenizer.decode(out[0], skip_special_tokens=True)
            # Remove the input prompt from output
            if prompt in result:
                result = result.replace(prompt, "").strip()

        return result

    def generate_translations(self, texts, max_new_tokens=50):
        """Generate glosses for multiple texts"""
        self.peft_model.eval()
        return [self.generate_single_translation(t, max_new_tokens)
                for t in tqdm(texts, desc="Generating glosses")]

    def calculate_metrics(self, predictions, references):
        """Calculate comprehensive metrics"""
        metrics = {
            'bleu_1': [], 'bleu_2': [], 'bleu_3': [], 'bleu_4': [],
            'rouge1_f': [], 'rouge2_f': [], 'rougeL_f': [],
            'wer': [], 'meteor': []
        }

        smoothie = SmoothingFunction().method1

        for pred, ref in zip(predictions, references):
            # BLEU scores
            ref_tokens = ref.split()
            pred_tokens = pred.split()

            weights = [(1,0,0,0), (0.5,0.5,0,0), (1/3,1/3,1/3,0), (0.25,0.25,0.25,0.25)]
            for i, w in enumerate(weights, 1):
                try:
                    score = sentence_bleu([ref_tokens], pred_tokens, weights=w, smoothing_function=smoothie)
                    metrics[f'bleu_{i}'].append(score)
                except:
                    metrics[f'bleu_{i}'].append(0.0)

            # ROUGE scores
            rouge_scores = self.rouge_scorer.score(ref, pred)
            metrics['rouge1_f'].append(rouge_scores['rouge1'].fmeasure)
            metrics['rouge2_f'].append(rouge_scores['rouge2'].fmeasure)
            metrics['rougeL_f'].append(rouge_scores['rougeL'].fmeasure)

            # WER
            try:
                wer = jiwer.wer(ref, pred)
                metrics['wer'].append(wer)
            except:
                metrics['wer'].append(1.0)

            # METEOR
            try:
                meteor = meteor_score([ref_tokens], pred_tokens)
                metrics['meteor'].append(meteor)
            except:
                metrics['meteor'].append(0.0)

        # Calculate averages
        avg_metrics = {k: sum(v)/len(v) if v else 0.0 for k, v in metrics.items()}
        return avg_metrics