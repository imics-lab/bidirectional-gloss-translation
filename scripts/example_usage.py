#!/usr/bin/env python3
"""
Comprehensive examples for Sign Language Translation Pipeline
Demonstrates usage across all datasets and models
"""

from src.pipeline import SignLanguageTranslationPipeline
import os


def example_all_models():
    """Test all supported models with SIGNUM dataset"""
    print("=== Testing All Supported Models ===")
    
    models = ["t5-small", "flan-t5-small", "mbart-small"]  # Skip llama-8b for demo
    
    for model_name in models:
        print(f"\n--- {model_name.upper()} Model ---")
        
        try:
            pipeline = SignLanguageTranslationPipeline(
                model_name=model_name,
                dataset_type="signum"
            )
            
            # Check if data exists
            if os.path.exists("data/signum_sents_anno_eng.txt"):
                df = pipeline.load_dataset()
                df = pipeline.preprocess_data(df, sample_size=50)  # Small sample
                print(f"✓ Loaded {len(df)} samples for {model_name}")
            else:
                print(f"⚠ Dataset not found for {model_name} demo")
                
        except Exception as e:
            print(f"✗ Error with {model_name}: {e}")


def example_all_datasets():
    """Test loading all supported datasets"""
    print("\n=== Testing All Supported Datasets ===")
    
    datasets = ["signum", "phoenix", "aslg"]
    
    for dataset_name in datasets:
        print(f"\n--- {dataset_name.upper()} Dataset ---")
        
        try:
            pipeline = SignLanguageTranslationPipeline(
                model_name="t5-small",
                dataset_type=dataset_name
            )
            
            # Try to load dataset
            df = pipeline.load_dataset()
            df = pipeline.preprocess_data(df, sample_size=10)
            
            print(f"✓ Successfully loaded {dataset_name}: {len(df)} samples")
            print(f"  Sample data structure: {list(df.columns)}")
            
            # Show sample
            if len(df) > 0:
                sample = df.iloc[0]
                print(f"  Sample gloss: {sample['gloss'][:50]}...")
                print(f"  Sample text: {sample['translation'][:50]}...")
                
        except FileNotFoundError:
            print(f"⚠ Dataset files not found for {dataset_name}")
        except Exception as e:
            print(f"✗ Error loading {dataset_name}: {e}")


def example_training_workflow():
    """Complete training workflow example"""
    print("\n=== Complete Training Workflow ===")
    
    # Use smallest model and dataset for demo
    pipeline = SignLanguageTranslationPipeline(
        model_name="t5-small",
        dataset_type="signum"
    )
    
    try:
        # 1. Load data
        print("1. Loading dataset...")
        df = pipeline.load_dataset()
        df = pipeline.preprocess_data(df, sample_size=100)  # Very small for demo
        
        # 2. Prepare for both tasks
        print("2. Preparing data for training...")
        
        # G2T task
        ds_g2t, val_df_g2t = pipeline.prepare_data_for_training(df, task="g2t")
        print(f"   G2T: {len(ds_g2t['train'])} train, {len(ds_g2t['validation'])} val")
        
        # T2G task  
        ds_t2g, val_df_t2g = pipeline.prepare_data_for_training(df, task="t2g")
        print(f"   T2G: {len(ds_t2g['train'])} train, {len(ds_t2g['validation'])} val")
        
        # 3. Show sample prompts
        print("3. Sample training prompts:")
        print(f"   G2T prompt: {ds_g2t['train'][0]['training_text'][:100]}...")
        print(f"   T2G prompt: {ds_t2g['train'][0]['training_text'][:100]}...")
        
        # 4. Tokenization (without actually loading model)
        print("4. Tokenization ready for model loading")
        
        # 5. Model loading would happen here
        print("5. Model loading step (skipped in demo)")
        # tokenizer, model = pipeline.load_model_and_tokenizer()
        # tok_ds = pipeline.tokenize_data(ds_g2t)
        # trainer = pipeline.train_model(tok_ds)
        
        print("✓ Workflow completed successfully!")
        
    except FileNotFoundError:
        print("⚠ Dataset files not found. Please ensure SIGNUM files are in data/ directory")
    except Exception as e:
        print(f"✗ Workflow error: {e}")


def example_inference_simulation():
    """Simulate inference without trained model"""
    print("\n=== Inference Simulation ===")
    
    pipeline = SignLanguageTranslationPipeline(
        model_name="t5-small",
        dataset_type="signum"
    )
    
    # Example inputs for different tasks
    example_texts = [
        "I need help with my homework",
        "The weather is nice today", 
        "How are you feeling?"
    ]
    
    example_glosses = [
        "I NEED HELP MY HOMEWORK",
        "WEATHER NICE TODAY",
        "HOW YOU FEEL"
    ]
    
    print("Text-to-Gloss examples:")
    for i, text in enumerate(example_texts):
        print(f"  {i+1}. Text: {text}")
        print(f"     Expected Gloss: {example_glosses[i]}")
        # Real inference: gloss = pipeline.generate_single_translation(text, task="t2g")
        
    print("\nGloss-to-Text examples:")
    for i, gloss in enumerate(example_glosses):
        print(f"  {i+1}. Gloss: {gloss}")
        print(f"     Expected Text: {example_texts[i]}")
        # Real inference: text = pipeline.generate_single_translation(gloss, task="g2t")
    
    print("\nNote: Load a trained model to see actual translations!")


def example_evaluation_metrics():
    """Demonstrate evaluation metrics calculation"""
    print("\n=== Evaluation Metrics Example ===")
    
    pipeline = SignLanguageTranslationPipeline(
        model_name="t5-small",
        dataset_type="signum"
    )
    
    # Example predictions vs references for different datasets
    examples = {
        "signum": {
            "predictions": [
                "I NEED HELP HOMEWORK",
                "WEATHER NICE TODAY", 
                "HOW YOU FEEL"
            ],
            "references": [
                "I NEED HELP MY HOMEWORK",
                "WEATHER NICE TODAY",
                "HOW YOU FEELING"
            ]
        },
        "phoenix": {
            "predictions": [
                "MORNING WEATHER SUNNY",
                "TEMPERATURE TWENTY DEGREE"
            ],
            "references": [
                "MORNING WEATHER WILL SUNNY",
                "TEMPERATURE TWENTY DEGREE CELSIUS"
            ]
        }
    }
    
    for dataset, data in examples.items():
        print(f"\n--- {dataset.upper()} Metrics ---")
        metrics = pipeline.calculate_metrics(data["predictions"], data["references"])
        
        print(f"Predictions: {data['predictions']}")
        print(f"References:  {data['references']}")
        print(f"Results:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")


def main():
    """Run all examples"""
    print("Sign Language Translation Pipeline - Comprehensive Examples")
    print("=" * 60)
    
    # Check data directory
    if not os.path.exists("data"):
        print("⚠ Warning: 'data/' directory not found.")
        print("  Create it and add dataset files for full functionality.")
        os.makedirs("data", exist_ok=True)
    
    # Run examples
    try:
        example_all_models()
        example_all_datasets()
        example_training_workflow()
        example_inference_simulation()
        example_evaluation_metrics()
        
        print("\n" + "=" * 60)
        print("Examples completed! For actual training:")
        print("  python train.py --model t5-small --dataset signum --task g2t")
        print("  python train.py --help  # for all options")
        
    except KeyboardInterrupt:
        print("\n\nExamples interrupted by user")
    except Exception as e:
        print(f"\nUnexpected error in examples: {e}")


if __name__ == "__main__":
    main()