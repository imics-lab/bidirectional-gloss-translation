
"""
Training script for Multi-Dataset Sign Language Translation
Supports PHOENIX, SIGNUM, ASLG-PC12 datasets and T5, Flan-T5, mBART, LLaMA models
"""

import argparse
import os
from src.pipeline import SignLanguageTranslationPipeline


def main():
    parser = argparse.ArgumentParser(description='Train Sign Language Translation Model')
    
    # Model arguments
    parser.add_argument('--model', type=str, default='t5-small',
                       choices=['t5-small', 'flan-t5-small', 'mbart-small', 'llama-8b'],
                       help='Pre-trained model to use')
    parser.add_argument('--custom_model_name', type=str, default=None,
                       help='Custom HuggingFace model name (overrides --model)')
    
    # Dataset arguments
    parser.add_argument('--dataset', type=str, default='signum',
                       choices=['signum', 'phoenix', 'aslg'],
                       help='Dataset to use for training')
    parser.add_argument('--data_dir', type=str, default='data/',
                       help='Directory containing dataset files')
    parser.add_argument('--sample_size', type=int, default=None,
                       help='Limit dataset size for testing (optional)')
    parser.add_argument('--test_size', type=float, default=0.2,
                       help='Validation split ratio')
    
    # Task arguments
    parser.add_argument('--task', type=str, default='g2t',
                       choices=['g2t', 't2g', 'both'],
                       help='Translation task: gloss-to-text, text-to-gloss, or both')
    
    # Training arguments
    parser.add_argument('--output_dir', type=str, default='./model_output',
                       help='Output directory for model checkpoints')
    parser.add_argument('--max_length', type=int, default=512,
                       help='Maximum sequence length')
    parser.add_argument('--epochs', type=int, default=5,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=2,
                       help='Per-device batch size')
    parser.add_argument('--learning_rate', type=float, default=2e-4,
                       help='Learning rate')
    
    # Evaluation arguments
    parser.add_argument('--eval_samples', type=int, default=100,
                       help='Number of samples for evaluation')
    parser.add_argument('--skip_training', action='store_true',
                       help='Skip training and only run evaluation')
    
    args = parser.parse_args()
    
    # Use custom model name if provided
    model_name = args.custom_model_name if args.custom_model_name else args.model
    
    # Initialize pipeline
    print(f"Initializing pipeline:")
    print(f"  Model: {model_name}")
    print(f"  Dataset: {args.dataset}")
    print(f"  Task: {args.task}")
    
    pipeline = SignLanguageTranslationPipeline(
        model_name=model_name,
        dataset_type=args.dataset
    )
    
    # Load dataset
    print("\nLoading dataset...")
    try:
        # Dataset-specific file paths
        if args.dataset == 'signum':
            df = pipeline.load_dataset(
                anno_path=os.path.join(args.data_dir, 'signum_sents_anno_eng.txt'),
                trans_path=os.path.join(args.data_dir, 'signum_sents_trans_eng.txt')
            )
        elif args.dataset == 'phoenix':
            df = pipeline.load_dataset(
                train_path=os.path.join(args.data_dir, 'phoenix2014T.train.gloss'),
                dev_path=os.path.join(args.data_dir, 'phoenix2014T.dev.gloss'),
                test_path=os.path.join(args.data_dir, 'phoenix2014T.test.gloss')
            )
        elif args.dataset == 'aslg':
            df = pipeline.load_dataset(
                gloss_path=os.path.join(args.data_dir, 'aslg_pc12_glosses.txt'),
                text_path=os.path.join(args.data_dir, 'aslg_pc12_text.txt')
            )
            
    except FileNotFoundError as e:
        print(f"Error: Dataset files not found!")
        print(f"Expected files in {args.data_dir}:")
        if args.dataset == 'signum':
            print("  - signum_sents_anno_eng.txt")
            print("  - signum_sents_trans_eng.txt")
        elif args.dataset == 'phoenix':
            print("  - phoenix2014T.train.gloss")
            print("  - phoenix2014T.dev.gloss") 
            print("  - phoenix2014T.test.gloss")
        elif args.dataset == 'aslg':
            print("  - aslg_pc12_glosses.txt")
            print("  - aslg_pc12_text.txt")
        return
    
    # Preprocess data
    df = pipeline.preprocess_data(df, sample_size=args.sample_size)
    
    def train_and_evaluate(task_name):
        print(f"\n{'='*50}")
        print(f"Training {task_name.upper()} model")
        print(f"{'='*50}")
        
        # Prepare training data for specific task
        ds, val_df = pipeline.prepare_data_for_training(df, test_size=args.test_size, task=task_name)
        
        if not args.skip_training:
            # Load model and tokenizer
            print("Loading model and tokenizer...")
            tokenizer, model = pipeline.load_model_and_tokenizer()
            
            # Tokenize data
            print("Tokenizing dataset...")
            tok_ds = pipeline.tokenize_data(ds, max_length=args.max_length)
            
            # Update output directory for this specific task
            task_output_dir = f"{args.output_dir}_{args.dataset}_{model_name.replace('/', '_')}_{task_name}"
            
            # Train model
            print("Starting training...")
            trainer = pipeline.train_model(
                tok_ds, 
                output_dir=task_output_dir,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate
            )
            print(f"Training completed! Model saved to {task_output_dir}")
        
        # Generate sample translations
        print(f"\n=== Sample {task_name.upper()} Translations ===")
        samples = df.sample(min(5, len(df)), random_state=42)
        
        if task_name == "g2t":
            sample_inputs = samples['gloss'].tolist()
            sample_refs = samples['translation'].tolist()
            input_label = "Gloss"
            ref_label = "Reference Text"
        else:
            sample_inputs = samples['translation'].tolist()
            sample_refs = samples['gloss'].tolist()
            input_label = "Text"
            ref_label = "Reference Gloss"
        
        if args.skip_training:
            print("Skipping sample generation (no trained model)")
        else:
            outputs = pipeline.generate_translations(sample_inputs)
            
            for inp, ref, pred in zip(sample_inputs, sample_refs, outputs):
                print(f"\n{input_label}: {inp}")
                print(f"{ref_label}: {ref}")
                print(f"Predicted: {pred}")
        
        # Evaluate on validation set
        print(f"\n=== {task_name.upper()} Validation Evaluation ===")
        eval_size = min(args.eval_samples, len(val_df))
        
        if task_name == "g2t":
            val_inputs = val_df['gloss'].tolist()[:eval_size]
            val_refs = val_df['translation'].tolist()[:eval_size]
        else:
            val_inputs = val_df['translation'].tolist()[:eval_size]
            val_refs = val_df['gloss'].tolist()[:eval_size]
        
        if args.skip_training:
            print("Skipping evaluation (no trained model)")
        else:
            val_preds = pipeline.generate_translations(val_inputs)
            
            # Calculate metrics
            metrics = pipeline.calculate_metrics(val_preds, val_refs)
            
            print(f"\n{task_name.upper()} Validation Metrics (n={eval_size}):")
            print(f"BLEU-1: {metrics['bleu_1']:.4f}")
            print(f"BLEU-2: {metrics['bleu_2']:.4f}")
            print(f"BLEU-3: {metrics['bleu_3']:.4f}")
            print(f"BLEU-4: {metrics['bleu_4']:.4f}")
            print(f"ROUGE-1 F: {metrics['rouge1_f']:.4f}")
            print(f"ROUGE-2 F: {metrics['rouge2_f']:.4f}")
            print(f"ROUGE-L F: {metrics['rougeL_f']:.4f}")
            print(f"WER: {metrics['wer']:.4f}")
            print(f"METEOR: {metrics['meteor']:.4f}")
            
            return metrics
        return None
    
    # Train and evaluate based on task selection
    if args.task == "both":
        g2t_metrics = train_and_evaluate("g2t")
        t2g_metrics = train_and_evaluate("t2g")
        
        if not args.skip_training:
            print(f"\n{'='*50}")
            print("FINAL COMPARISON")
            print(f"{'='*50}")
            print(f"G2T BLEU-4: {g2t_metrics['bleu_4']:.4f}")
            print(f"T2G BLEU-4: {t2g_metrics['bleu_4']:.4f}")
    else:
        train_and_evaluate(args.task)


if __name__ == "__main__":
    main()#!/usr/bin/env python3
"""
Training script for Signum Text-to-Gloss Translation
"""

import argparse
import os
from pipeline import SignumTextToGlossTranslationPipeline


def main():
    parser = argparse.ArgumentParser(description='Train Signum Text-to-Gloss Translation Model')
    
    # Model arguments
    parser.add_argument('--model_name', type=str, default='meta-llama/Meta-Llama-3-8B',
                       help='Pre-trained model name from HuggingFace')
    
    # Data arguments
    parser.add_argument('--anno_path', type=str, default='signum_sents_anno_eng.txt',
                       help='Path to gloss annotation file')
    parser.add_argument('--trans_path', type=str, default='signum_sents_trans_eng.txt',
                       help='Path to translation file')
    parser.add_argument('--sample_size', type=int, default=None,
                       help='Limit dataset size for testing (optional)')
    parser.add_argument('--test_size', type=float, default=0.2,
                       help='Validation split ratio')
    
    # Training arguments
    parser.add_argument('--output_dir', type=str, default='./signum_model_output',
                       help='Output directory for model checkpoints')
    parser.add_argument('--max_length', type=int, default=512,
                       help='Maximum sequence length')
    
    # Evaluation arguments
    parser.add_argument('--eval_samples', type=int, default=100,
                       help='Number of samples for evaluation')
    parser.add_argument('--skip_training', action='store_true',
                       help='Skip training and only run evaluation')
    
    args = parser.parse_args()
    
    # Initialize pipeline
    print(f"Initializing pipeline with model: {args.model_name}")
    pipeline = SignumTextToGlossTranslationPipeline(model_name=args.model_name)
    
    # Load dataset
    print("Loading Signum dataset...")
    if not os.path.exists(args.anno_path) or not os.path.exists(args.trans_path):
        print(f"Error: Dataset files not found!")
        print(f"Expected: {args.anno_path} and {args.trans_path}")
        print("Please ensure the Signum dataset files are in the current directory.")
        return
    
    df = pipeline.load_signum_dataset(
        anno_path=args.anno_path,
        trans_path=args.trans_path
    )
    
    # Preprocess data
    df = pipeline.preprocess_signum_data(df, sample_size=args.sample_size)
    
    # Prepare training data
    ds, val_df = pipeline.prepare_data_for_training(df, test_size=args.test_size)
    
    if not args.skip_training:
        # Load model and tokenizer
        print("Loading model and tokenizer...")
        tokenizer, model = pipeline.load_model_and_tokenizer()
        
        # Tokenize data
        print("Tokenizing dataset...")
        tok_ds = pipeline.tokenize_data(ds, max_length=args.max_length)
        
        # Train model
        print("Starting training...")
        trainer = pipeline.train_model(tok_ds, output_dir=args.output_dir)
        print(f"Training completed! Model saved to {args.output_dir}")
    
    # Generate sample translations
    print("\n=== Sample Translations ===")
    samples = df['translation'].sample(min(5, len(df)), random_state=42).tolist()
    sample_refs = df[df['translation'].isin(samples)]['gloss'].tolist()
    
    if args.skip_training:
        print("Skipping sample generation (no trained model)")
    else:
        outputs = pipeline.generate_translations(samples)
        
        for text, ref, pred in zip(samples, sample_refs, outputs):
            print(f"\nText: {text}")
            print(f"Reference: {ref}")
            print(f"Predicted: {pred}")
    
    # Evaluate on validation set
    print(f"\n=== Validation Set Evaluation ===")
    eval_size = min(args.eval_samples, len(val_df))
    val_texts = val_df['translation'].tolist()[:eval_size]
    val_refs = val_df['gloss'].tolist()[:eval_size]
    
    if args.skip_training:
        print("Skipping evaluation (no trained model)")
    else:
        val_preds = pipeline.generate_translations(val_texts)
        
        # Calculate metrics
        metrics = pipeline.calculate_metrics(val_preds, val_refs)
        
        print(f"\nValidation Metrics (n={eval_size}):")
        print(f"BLEU-1: {metrics['bleu_1']:.4f}")
        print(f"BLEU-2: {metrics['bleu_2']:.4f}")
        print(f"BLEU-3: {metrics['bleu_3']:.4f}")
        print(f"BLEU-4: {metrics['bleu_4']:.4f}")
        print(f"ROUGE-1 F: {metrics['rouge1_f']:.4f}")
        print(f"ROUGE-2 F: {metrics['rouge2_f']:.4f}")
        print(f"ROUGE-L F: {metrics['rougeL_f']:.4f}")
        print(f"WER: {metrics['wer']:.4f}")
        print(f"METEOR: {metrics['meteor']:.4f}")


if __name__ == "__main__":
    main()