"""
训练日志记录器

每 epoch 记录 loss / accuracy / F1 等指标，保存为 CSV 文件。
支持训练过程中的实时记录和训练结束后的日志读取。
"""

import os
import csv
import json
from datetime import datetime
from typing import Optional


class TrainingLogger:
    """
    训练日志记录器，将每个 epoch 的指标保存为 CSV 文件。

    CSV 列：epoch, train_loss, accuracy, f1_macro, f1_per_class, timestamp
    同时保存训练元信息（config, 开始时间等）到 meta.json
    """

    def __init__(self, output_dir: str, config: Optional[dict] = None):
        """
        Args:
            output_dir: 日志保存目录（如 ./results/full 或 ./results/lora）
            config: 训练配置字典，写入 meta.json
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # CSV 日志文件
        self.csv_path = os.path.join(output_dir, "training_log.csv")
        self.fieldnames = [
            "epoch", "train_loss", "accuracy", "f1_macro",
            "f1_per_class", "best_f1", "timestamp"
        ]

        # 初始化 CSV（写入表头）
        with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writeheader()

        # 保存训练元信息
        if config:
            meta_path = os.path.join(output_dir, "meta.json")
            meta = {
                "config": config,
                "start_time": datetime.now().isoformat(),
                "status": "running"
            }
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)

        self.best_f1 = 0.0
        print(f"[Logger] 日志将保存至 {self.csv_path}")

    def log(self, epoch: int, train_loss: float, metrics: dict) -> None:
        """
        记录一个 epoch 的训练结果。

        Args:
            epoch: 当前 epoch 编号
            train_loss: 训练平均损失
            metrics: 评估指标字典，需包含 accuracy, f1_macro, f1_per_class
        """
        if metrics["f1_macro"] > self.best_f1:
            self.best_f1 = metrics["f1_macro"]

        row = {
            "epoch": epoch,
            "train_loss": f"{train_loss:.6f}",
            "accuracy": f"{metrics['accuracy']:.6f}",
            "f1_macro": f"{metrics['f1_macro']:.6f}",
            "f1_per_class": str(metrics.get("f1_per_class", [])),
            "best_f1": f"{self.best_f1:.6f}",
            "timestamp": datetime.now().isoformat()
        }

        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow(row)

    def finalize(self, extra_info: Optional[dict] = None) -> None:
        """
        训练结束后更新元信息。

        Args:
            extra_info: 额外信息（如训练时长、峰值显存等）
        """
        meta_path = os.path.join(self.output_dir, "meta.json")
        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            meta["end_time"] = datetime.now().isoformat()
            meta["status"] = "completed"
            meta["best_f1"] = self.best_f1
            if extra_info:
                meta.update(extra_info)
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)


def load_training_log(log_path: str) -> list:
    """
    加载训练日志 CSV 文件。

    Args:
        log_path: CSV 文件路径

    Returns:
        包含每行记录的字典列表
    """
    records = []
    with open(log_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["epoch"] = int(row["epoch"])
            row["train_loss"] = float(row["train_loss"])
            row["accuracy"] = float(row["accuracy"])
            row["f1_macro"] = float(row["f1_macro"])
            row["best_f1"] = float(row["best_f1"])
            records.append(row)
    return records
