"""
推理封装模块

SentimentPredictor 类统一封装全量微调和 LoRA 两种模型的加载与推理。
加载模型时使用本地保存的 tokenizer 副本，无需联网。
"""

import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
import json
import torch
import torch.nn.functional as F
from transformers import BertTokenizer, AutoModel
from peft import PeftModel

# 导入路径兼容处理
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models.classifier import BertSentimentClassifier


LABEL_NAMES = {0: "负面 😞", 1: "正面 😊"}


class SentimentPredictor:
    """
    情感分析推理封装。

    统一支持全量微调和 LoRA 两种模型格式的加载与推理。

    Args:
        model_dir: 模型目录（如 ./models/full 或 ./models/lora）
        base_model_name: 基底模型名称或路径
        method: 加载方式 ("full" 或 "lora")
    """

    def __init__(
        self,
        model_dir: str,
        base_model_name: str = "hfl/chinese-roberta-wwm-ext",
        method: str = "full",
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.method = method
        self.model_dir = model_dir

        # 路径检查
        if not os.path.exists(model_dir):
            raise FileNotFoundError(
                f"模型目录不存在: {model_dir}\n"
                f"请先运行训练脚本生成模型文件。"
            )

        # 加载 tokenizer（使用本地副本，无需联网）
        tokenizer_dir = os.path.join(model_dir, "tokenizer")
        if os.path.exists(tokenizer_dir):
            print(f"[Predictor] 从本地加载 Tokenizer: {tokenizer_dir}")
            self.tokenizer = BertTokenizer.from_pretrained(tokenizer_dir)
        else:
            print(f"[Predictor] 本地 tokenizer 不存在，从模型名加载: {base_model_name}")
            self.tokenizer = BertTokenizer.from_pretrained(base_model_name)

        # 加载模型配置
        config_path = os.path.join(model_dir, "config.json")
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
        else:
            cfg = {"num_classes": 2, "dropout": 0.1}

        # 加载模型
        if method == "full":
            model_path = os.path.join(model_dir, "best_model.pt")
            if not os.path.exists(model_path):
                raise FileNotFoundError(
                    f"模型权重文件不存在: {model_path}\n"
                    f"请先运行全量微调训练。"
                )

            self.model = BertSentimentClassifier(
                model_name=cfg.get("model_name", base_model_name),
                num_classes=cfg.get("num_classes", 2),
                dropout=cfg.get("dropout", 0.1),
            )
            self.model.load_state_dict(
                torch.load(model_path, map_location=self.device, weights_only=True)
            )
            print(f"[Predictor] 全量微调模型已加载: {model_path}")

        elif method == "lora":
            adapter_path = os.path.join(model_dir, "adapter_model.safetensors")
            adapter_bin_path = os.path.join(model_dir, "adapter_model.bin")
            if not (os.path.exists(adapter_path) or os.path.exists(adapter_bin_path)):
                raise FileNotFoundError(
                    f"LoRA 适配器文件不存在: {model_dir}\n"
                    f"请先运行 LoRA 微调训练。"
                )

            base = BertSentimentClassifier(
                model_name=cfg.get("model_name", base_model_name),
                num_classes=cfg.get("num_classes", 2),
                dropout=cfg.get("dropout", 0.1),
            )
            self.model = PeftModel.from_pretrained(base, model_dir)

            # 显式加载保存的自定义分类头权重
            head_path = os.path.join(model_dir, "classifier_head.pt")
            if os.path.exists(head_path):
                base.classifier.load_state_dict(
                    torch.load(head_path, map_location=self.device, weights_only=True)
                )
                print(f"[Predictor] LoRA 自定义分类头权重已成功加载: {head_path}")
            else:
                print(f"[Predictor] 警告: 未找到 LoRA 自定义分类头权重 {head_path}，分类头将保持随机初始化！")

            print(f"[Predictor] LoRA 模型已加载: {model_dir}")

        else:
            raise ValueError(f"不支持的方法: {method}，请使用 'full' 或 'lora'")

        self.model.to(self.device)
        self.model.eval()
        print(f"[Predictor] 设备: {self.device}")

    def predict(self, text: str) -> dict:
        """
        对单条文本进行情感预测。

        Args:
            text: 原始中文文本

        Returns:
            dict: {
                "label": "正面 😊" 或 "负面 😞",
                "confidence": float (0~1),
                "probs": [负面概率, 正面概率]
            }
        """
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=128,
            truncation=True,
            padding="max_length",
        )

        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)

        with torch.no_grad():
            logits = self.model(input_ids, attention_mask)
            probs = F.softmax(logits, dim=-1).squeeze().cpu().tolist()

        # 处理单条输入时 probs 可能是标量列表
        if isinstance(probs, float):
            probs = [probs]

        pred_id = int(torch.tensor(probs).argmax())

        return {
            "label": LABEL_NAMES[pred_id],
            "confidence": probs[pred_id],
            "probs": probs,  # [负面概率, 正面概率]
        }

    def predict_batch(self, texts: list) -> list:
        """
        批量预测。

        Args:
            texts: 文本列表

        Returns:
            list[dict]: 每条文本的预测结果
        """
        return [self.predict(text) for text in texts]
