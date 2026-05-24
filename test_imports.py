import sys
import traceback

print("Script started...")
try:
    print("Importing torch...")
    import torch
    print(f"Torch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    print("Importing transformers...")
    import transformers
    print(f"Transformers version: {transformers.__version__}")

    print("Importing peft...")
    import peft
    print(f"PEFT version: {peft.__version__}")

    print("Importing datasets...")
    import datasets
    print(f"Datasets version: {datasets.__version__}")

    print("Importing sklearn...")
    import sklearn
    print(f"Sklearn version: {sklearn.__version__}")

    print("Importing src.training.lora_finetune...")
    import src.training.lora_finetune
    print("lora_finetune imported successfully!")

    print("Calling lora_finetune.main('configs/lora.yaml')...")
    src.training.lora_finetune.main("configs/lora.yaml")
    print("lora_finetune.main() executed successfully!")

    print("All imports succeeded!")
except Exception as e:
    print(f"Error during import: {e}")
    traceback.print_exc()
print("Script finished.")
