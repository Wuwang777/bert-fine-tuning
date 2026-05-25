import os
import sys
import traceback

def main():
    print("[Launcher] Pre-importing major frameworks for API service...")
    try:
        import torch
        print(f"[Launcher] Torch loaded. Version: {torch.__version__}, CUDA: {torch.cuda.is_available()}")
        import transformers
        print(f"[Launcher] Transformers loaded. Version: {transformers.__version__}")
        import fastapi
        print(f"[Launcher] FastAPI loaded. Version: {fastapi.__version__}")
        import uvicorn
        print(f"[Launcher] Uvicorn loaded. Version: {uvicorn.__version__}")

        # 确保项目根目录在 sys.path 中
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        
        print("[Launcher] Starting FastAPI BERT Sentiment Analysis service on http://localhost:8000 ...", flush=True)
        
        # 运行 uvicorn，关闭 reload 以避免 Windows 多进程 spawn 引导程序冲突
        uvicorn.run("src.api.app:app", host="0.0.0.0", port=8000, reload=False)
        
    except Exception as e:
        print(f"[Launcher] API Startup Error: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()
