"""
FastAPI 接口服务 — 中文情感分析推理服务

提供 RESTful API，方便前端或其他微服务调用。功能包括：
- 单条文本情感分析预测 (/predict)
- 批量文本情感分析预测 (/predict/batch)
- 查看服务运行状态及已训练模型 (/health)
- 支持模型热切换 (/model/switch)，动态切换全量微调和 LoRA 模型
- 支持 CORS 跨域请求，便于前端直接调用
"""

import os
import sys
import logging
import threading
from typing import List, Optional

# 兼容本地离线模式
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

# 确保项目根目录在 sys.path 中以加载 gui.predictor 和 src 模块
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from gui.predictor import SentimentPredictor

# ─── 日志配置 ───
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("bert-sentiment-api")

# ─── 全局参数与状态 ───
BASE_MODEL = "hfl/chinese-roberta-wwm-ext"
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

# 全局推理器与其访问锁（保证模型热切换及并发访问的线程安全）
predictor: Optional[SentimentPredictor] = None
current_method: Optional[str] = None
predictor_lock = threading.Lock()

# ─── FastAPI 应用初始化 ───
app = FastAPI(
    title="BERT 中文情感分析 API 服务",
    description="提供基于 BERT (RoBERTa) 微调模型的中文情感分析接口，支持全量微调 (Full Finetune) 和 LoRA 两种模型热切换。",
    version="1.0.0",
)

# ─── CORS 跨域配置 ───
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许前端任意域名/端口跨域访问
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── 静态文件服务与前端页面 ───
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
if os.path.exists(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# ─── Pydantic 数据验证模型 ───
class PredictRequest(BaseModel):
    text: str = Field(
        ..., 
        example="这家店的菜品味道很棒，服务员态度也超级好，强烈推荐！",
        description="待分析情感的中文文本内容，长度建议在 1 到 512 字符之间"
    )

class BatchPredictRequest(BaseModel):
    texts: List[str] = Field(
        ...,
        example=[
            "这家店的菜品味道很棒，服务员态度也超级好，强烈推荐！",
            "体验极差，菜品不新鲜，等了半个多小时都没上齐，以后再也不来了。"
        ],
        description="待分析情感的中文文本列表"
    )

class ModelSwitchRequest(BaseModel):
    method: str = Field(
        ...,
        example="lora",
        description="要切换的目标模型微调类型，仅支持 'full'（全量微调）或 'lora'（LoRA微调）"
    )

class PredictionResult(BaseModel):
    text: str
    label: str
    confidence: float
    probs: List[float]

class PredictResponse(BaseModel):
    status: str
    model_type: str
    result: PredictionResult

class BatchPredictResponse(BaseModel):
    status: str
    model_type: str
    results: List[PredictionResult]


# ─── 辅助函数：加载与切换模型 ───
def load_model(method: str) -> bool:
    """
    加载指定类型的微调模型。线程安全。
    
    Args:
        method: "full" 或 "lora"
        
    Returns:
        bool: 是否加载成功
    """
    global predictor, current_method
    
    model_dir = os.path.join(MODELS_DIR, method)
    if not os.path.exists(model_dir):
        logger.warning(f"无法加载模型：目录不存在 {model_dir}。请确保对应的微调训练已完成。")
        return False
        
    try:
        logger.info(f"正在加载 {method.upper()} 情感分析模型，目录: {model_dir}...")
        new_predictor = SentimentPredictor(
            model_dir=model_dir,
            base_model_name=BASE_MODEL,
            method=method
        )
        with predictor_lock:
            predictor = new_predictor
            current_method = method
        logger.info(f"🎉 {method.upper()} 情感分析模型加载成功！设备: {new_predictor.device}")
        return True
    except Exception as e:
        logger.error(f"加载模型 {method} 失败: {str(e)}", exc_info=True)
        return False


# ─── 生命周期管理 ───
@app.on_event("startup")
def startup_event():
    """服务启动时，默认加载可用的模型。优先加载 full-finetuning 模型，其次 LoRA"""
    logger.info("正在启动 BERT 中文情感分析 API 服务...")
    
    # 扫描可用模型
    full_exists = os.path.exists(os.path.join(MODELS_DIR, "full"))
    lora_exists = os.path.exists(os.path.join(MODELS_DIR, "lora"))
    
    if full_exists:
        success = load_model("full")
        if success:
            return
            
    if lora_exists:
        success = load_model("lora")
        if success:
            return
            
    logger.warning(
        "⚠️ 警告: 未能在 'models/' 目录下找到任何有效的模型！\n"
        "服务将正常启动，但调用预测接口时会返回错误。\n"
        "请先运行 'run_full.py' 或 'run_lora.py' 进行模型训练微调，"
        "或通过 POST /model/switch 接口动态加载已完成训练的模型。"
    )


# ─── 路由定义 ───

@app.get("/", tags=["系统"])
def read_root():
    """返回内置的测试前端网页"""
    index_file = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return {"message": "BERT Sentiment API is running. Please check /docs for OpenAPI definitions."}


@app.get("/health", tags=["系统"])
def health_check():
    """查看服务健康状态及可用模型"""
    full_exists = os.path.exists(os.path.join(MODELS_DIR, "full"))
    lora_exists = os.path.exists(os.path.join(MODELS_DIR, "lora"))
    
    return {
        "status": "online",
        "loaded_model": current_method if predictor else None,
        "device": str(predictor.device) if predictor else None,
        "available_models_on_disk": {
            "full": full_exists,
            "lora": lora_exists
        },
        "message": "API 服务运行正常。可用模型为 True 代表该微调模型已被训练且保存在本地。" if (full_exists or lora_exists) 
                   else "目前本地 models/ 目录下没有检测到已微调的模型，请先运行训练脚本。"
    }


@app.post("/predict", response_model=PredictResponse, tags=["预测"])
def predict_sentiment(req: PredictRequest):
    """
    分析单条文本情感倾向（正面/负面）
    """
    global predictor, current_method
    
    if not predictor:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="当前没有加载任何微调模型。请先进行模型训练，或者通过 /model/switch 接口加载。"
        )
        
    text = req.text.strip()
    if not text:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="输入文本不能为空"
        )
        
    try:
        with predictor_lock:
            res = predictor.predict(text)
            
        return {
            "status": "success",
            "model_type": current_method,
            "result": {
                "text": text,
                "label": res["label"],
                "confidence": res["confidence"],
                "probs": res["probs"]
            }
        }
    except Exception as e:
        logger.error(f"推理失败: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"模型推理失败: {str(e)}"
        )


@app.post("/predict/batch", response_model=BatchPredictResponse, tags=["预测"])
def predict_sentiment_batch(req: BatchPredictRequest):
    """
    批量分析文本列表的情感倾向
    """
    global predictor, current_method
    
    if not predictor:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="当前没有加载任何微调模型。请先进行模型训练，或者通过 /model/switch 接口加载。"
        )
        
    if not req.texts:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="输入文本列表不能为空"
        )
        
    cleaned_texts = [text.strip() for text in req.texts if text.strip()]
    if not cleaned_texts:
         raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="输入文本列表中不能全部为空字符串"
        )
         
    try:
        results = []
        with predictor_lock:
            for text in cleaned_texts:
                res = predictor.predict(text)
                results.append({
                    "text": text,
                    "label": res["label"],
                    "confidence": res["confidence"],
                    "probs": res["probs"]
                })
                
        return {
            "status": "success",
            "model_type": current_method,
            "results": results
        }
    except Exception as e:
        logger.error(f"批量推理失败: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"模型批量推理失败: {str(e)}"
        )


@app.post("/model/switch", tags=["管理"])
def switch_model(req: ModelSwitchRequest):
    """
    动态切换加载的微调模型类型 (full 或 lora)
    """
    method = req.method.lower().strip()
    if method not in ["full", "lora"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="切换参数错误。只能在 'full' (全量微调) 或 'lora' (LoRA微调) 之间进行选择。"
        )
        
    model_dir = os.path.join(MODELS_DIR, method)
    if not os.path.exists(model_dir):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"切换失败: 本地未找到 {method} 微调模型的保存文件。请确保已经运行了对应的训练脚本。"
        )
        
    success = load_model(method)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"加载模型权重发生异常，请检查服务器后台日志。"
        )
        
    return {
        "status": "success",
        "message": f"成功切换到 {method.upper()} 情感分析微调模型！",
        "current_model": current_method
    }


if __name__ == "__main__":
    import uvicorn
    # 支持从命令行运行，默认使用 8000 端口
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
