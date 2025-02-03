from src.model import setup_training, prepare_training_data, train_model
from src.preprocess import load_medical_dataset, analyze_dataset
import torch

def main():
    print("Analyzing dataset...")
    df = load_medical_dataset()
    stats = analyze_dataset(df)
    print("\nDataset Statistics:")
    for key, value in stats.items():
        print(f"{key}: {value}")

    print("\nSetting up training...")
    model, tokenizer = setup_training()
    dataset_mapped = prepare_training_data()

    print("\nStarting training...")
    training_stats = train_model(model, tokenizer, dataset_mapped)
    
    print("\nTraining completed!")
    print(f"Training runtime: {training_stats.metrics['train_runtime']} seconds")
    used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    print(f"Peak GPU memory usage: {used_memory} GB")

if __name__ == "__main__":
    main()