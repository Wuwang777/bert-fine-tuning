"""
统一训练器

SentimentTrainer 类支持全量微调和 LoRA 两种方法共用，包含：
- 差异化学习率（BERT 层 / 分类头）
- 线性 warmup + 线性衰减调度器
- fp16 混合精度训练
- 梯度裁剪
- 模型保存（full: state_dict, lora: save_pretrained）
- 训练日志记录
"""

import os
import json
import time
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from src.evaluation.evaluator import compute_metrics, print_report, format_metrics_summary
from src.utils.logger import TrainingLogger


class SentimentTrainer:
    """
    情感分析统一训练器。

    支持全量微调和 LoRA 两种训练方式，通过 config 参数控制行为差异。

    Args:
        model: 模型实例（BertSentimentClassifier 或 LoRA 包装后的模型）
        train_loader: 训练集 DataLoader
        dev_loader: 验证集 DataLoader
        config: 训练配置字典
        device: 训练设备
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        dev_loader: DataLoader,
        config: dict,
        device: torch.device,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.dev_loader = dev_loader
        self.config = config
        self.device = device

        # 初始化日志
        self.logger = TrainingLogger(config["output_dir"], config)

        # 构建优化器（差异化学习率）
        self.optimizer = self._build_optimizer()

        # 构建调度器（线性 warmup + 线性衰减）
        grad_accum = config.get("gradient_accumulation_steps", 1)
        total_steps = (len(train_loader) // grad_accum) * config["epochs"]
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=int(total_steps * 0.1),  # 前 10% steps warmup
            num_training_steps=total_steps,
        )

        # 损失函数
        self.loss_fn = nn.CrossEntropyLoss()

        # 混合精度
        self.scaler = torch.amp.GradScaler("cuda")
        self.grad_accum_steps = grad_accum

        # 记录最优指标
        self.best_acc = 0.0
        self.best_epoch = 0

    def _build_optimizer(self) -> AdamW:
        """
        构建 AdamW 优化器。

        全量微调：BERT 层和分类头使用差异化学习率。
        LoRA：所有可训练参数使用统一学习率。
        """
        method = self.config.get("method", "full")

        if method == "full":
            # 差异化学习率：BERT 层用 bert_lr，分类头用 head_lr
            bert_params = []
            head_params = []
            for name, param in self.model.named_parameters():
                if not param.requires_grad:
                    continue
                if "bert" in name:
                    bert_params.append(param)
                elif "classifier" in name:
                    head_params.append(param)
                else:
                    head_params.append(param)

            optimizer = AdamW([
                {"params": bert_params, "lr": self.config["bert_lr"]},
                {"params": head_params, "lr": self.config["head_lr"]},
            ], weight_decay=self.config.get("weight_decay", 0.01))

            print(f"[Trainer] 差异化学习率: BERT={self.config['bert_lr']}, "
                  f"Head={self.config['head_lr']}")
        else:
            # LoRA / 统一学习率
            lr = self.config.get("lr", self.config.get("bert_lr", 1e-4))
            trainable_params = [p for p in self.model.parameters() if p.requires_grad]
            optimizer = AdamW(trainable_params, lr=lr,
                              weight_decay=self.config.get("weight_decay", 0.01))
            print(f"[Trainer] 统一学习率: {lr}")

        return optimizer

    def train_one_epoch(self, epoch: int) -> tuple:
        """
        训练一个 epoch。

        Args:
            epoch: 当前 epoch 编号

        Returns:
            tuple: (平均 loss, 验证指标字典)
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        self.optimizer.zero_grad()

        for step, batch in enumerate(self.train_loader):
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)

            with torch.amp.autocast("cuda"):
                logits = self.model(input_ids, attention_mask)
                loss = self.loss_fn(logits, labels)
                loss = loss / self.grad_accum_steps  # 梯度累积缩放

            self.scaler.scale(loss).backward()

            if (step + 1) % self.grad_accum_steps == 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()
                self.optimizer.zero_grad()

            total_loss += loss.item() * self.grad_accum_steps
            num_batches += 1

        avg_loss = total_loss / num_batches

        # 验证
        metrics = self.evaluate(self.dev_loader)

        # 记录日志
        self.logger.log(epoch, avg_loss, metrics)

        return avg_loss, metrics

    def evaluate(self, data_loader: DataLoader) -> dict:
        """
        在给定数据集上评估模型。

        Args:
            data_loader: 评估用 DataLoader

        Returns:
            dict: 评估指标字典
        """
        self.model.eval()
        all_preds, all_labels = [], []

        with torch.no_grad():
            for batch in data_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)

                with torch.amp.autocast("cuda"):
                    logits = self.model(input_ids, attention_mask)

                preds = logits.argmax(dim=-1).cpu().tolist()
                all_preds.extend(preds)
                all_labels.extend(batch["labels"].tolist())

        return compute_metrics(all_labels, all_preds)

    def save_model(self, save_dir: str, tokenizer, method: str = "full"):
        """
        将最优模型和 tokenizer 保存至指定目录。

        Args:
            save_dir: 保存目录（如 ./models/full 或 ./models/lora）
            tokenizer: 分词器实例
            method: 保存模式 ("full" 或 "lora")
        """
        os.makedirs(save_dir, exist_ok=True)

        if method == "full":
            # 全量微调：保存完整 state_dict
            model_path = os.path.join(save_dir, "best_model.pt")
            torch.save(self.model.state_dict(), model_path)

            # 保存模型配置
            config_path = os.path.join(save_dir, "config.json")
            model_config = {
                "model_name": self.config.get("model_name", "hfl/chinese-roberta-wwm-ext"),
                "num_classes": self.config.get("num_classes", 2),
                "dropout": self.config.get("dropout", 0.1),
            }
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(model_config, f, ensure_ascii=False, indent=2)

            print(f"[Trainer] 全量微调模型已保存至 {save_dir}")

        elif method == "lora":
            # LoRA：保存适配器权重
            self.model.save_pretrained(save_dir)

            # 手工保存自定义分类头的状态字典
            head_path = os.path.join(save_dir, "classifier_head.pt")
            if hasattr(self.model, "base_model") and hasattr(self.model.base_model, "model"):
                torch.save(self.model.base_model.model.classifier.state_dict(), head_path)
            else:
                torch.save(self.model.classifier.state_dict(), head_path)
            print(f"[Trainer] LoRA 自定义分类头权重已保存至 {head_path}")

            # 也保存分类头配置
            config_path = os.path.join(save_dir, "config.json")
            model_config = {
                "model_name": self.config.get("model_name", "hfl/chinese-roberta-wwm-ext"),
                "num_classes": self.config.get("num_classes", 2),
                "dropout": self.config.get("dropout", 0.1),
            }
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(model_config, f, ensure_ascii=False, indent=2)

            print(f"[Trainer] LoRA 适配器已保存至 {save_dir}")

        # 保存 tokenizer 本地副本
        tokenizer_dir = os.path.join(save_dir, "tokenizer")
        tokenizer.save_pretrained(tokenizer_dir)
        print(f"[Trainer] Tokenizer 已保存至 {tokenizer_dir}")

    def train(self, tokenizer=None, method: str = "full", test_loader: DataLoader = None):
        """
        完整训练流程。

        Args:
            tokenizer: 分词器（用于保存）
            method: 训练方法 ("full" 或 "lora")
            test_loader: 测试集 DataLoader（训练结束后评估）

        Returns:
            dict: 训练结果摘要
        """
        save_dir = self.config.get("save_dir", f"./models/{method}")
        epochs = self.config["epochs"]

        print("\n" + "=" * 60)
        print(f"开始训练 — {method.upper()} 方法")
        print(f"Epochs: {epochs}, Device: {self.device}")
        print("=" * 60)

        # 重置峰值显存统计
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        start_time = time.time()

        for epoch in range(1, epochs + 1):
            epoch_start = time.time()
            loss, metrics = self.train_one_epoch(epoch)
            epoch_time = time.time() - epoch_start

            print(f"Epoch {epoch}/{epochs} | "
                  f"Loss: {loss:.4f} | "
                  f"Acc: {metrics['accuracy']:.4f} | "
                  f"Time: {epoch_time:.1f}s")

            # 保存最优模型
            if metrics["accuracy"] > self.best_acc:
                self.best_acc = metrics["accuracy"]
                self.best_epoch = epoch
                if tokenizer:
                    self.save_model(save_dir, tokenizer, method=method)
                print(f"  ★ 新的最优模型！Acc={self.best_acc:.4f}")

        total_time = time.time() - start_time
        training_time_min = total_time / 60

        # 峰值显存
        peak_memory_mb = 0
        if torch.cuda.is_available():
            peak_memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024

        # 可训练参数量
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        print("\n" + "=" * 60)
        print(f"训练完成！")
        print(f"  最优 Dev Acc: {self.best_acc:.4f} (Epoch {self.best_epoch})")
        print(f"  训练时长: {training_time_min:.1f} 分钟")
        print(f"  峰值显存: {peak_memory_mb:.0f} MB")
        print(f"  可训练参数: {trainable_params:,}")
        print(f"  模型保存: {save_dir}")
        print("=" * 60)

        # 在测试集上评估
        test_metrics = None
        if test_loader:
            print("\n在测试集上评估...")
            test_metrics = self.evaluate(test_loader)
            print(format_metrics_summary(test_metrics))
            # 简化：直接用 evaluate 的结果
            print(f"\nTest Accuracy: {test_metrics['accuracy']:.4f}")

        # 更新日志元信息
        self.logger.finalize({
            "training_time_min": round(training_time_min, 2),
            "peak_memory_mb": round(peak_memory_mb, 0),
            "trainable_params": trainable_params,
            "best_acc": self.best_acc,
            "best_epoch": self.best_epoch,
            "method": method,
            "test_metrics": test_metrics,
        })

        return {
            "best_acc": self.best_acc,
            "best_epoch": self.best_epoch,
            "training_time_min": training_time_min,
            "peak_memory_mb": peak_memory_mb,
            "trainable_params": trainable_params,
            "test_metrics": test_metrics,
        }
