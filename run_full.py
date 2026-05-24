import sys
import traceback

print("[Launcher] Pre-importing major frameworks for stability...")
try:
    import torch
    print(f"[Launcher] Torch loaded. Version: {torch.__version__}, CUDA: {torch.cuda.is_available()}")
    import transformers
    print(f"[Launcher] Transformers loaded. Version: {transformers.__version__}")
    import datasets
    print(f"[Launcher] Datasets loaded. Version: {datasets.__version__}")
    
    import os
    # 确保项目根目录在 sys.path 中
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    
    print("[Launcher] Importing Full Fine-tune module...")
    from src.training.full_finetune import main
    
    print("[Launcher] Starting Full Fine-tuning main training pipeline...", flush=True)
    main("configs/full_finetune.yaml")
    print("[Launcher] Full Fine-tuning completed successfully!", flush=True)
    
except Exception as e:
    print(f"[Launcher] Error encountered: {e}")
    traceback.print_exc()
    sys.exit(1)
