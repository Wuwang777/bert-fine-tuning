"""
评估模块

计算分类任务的核心指标：Accuracy、F1-macro、混淆矩阵等。
"""

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
)
from typing import List, Optional


def compute_metrics(labels: List[int], preds: List[int]) -> dict:
    """
    计算分类评估指标。

    Args:
        labels: 真实标签列表
        preds: 预测标签列表

    Returns:
        dict: {
            "accuracy": float,
            "f1_macro": float,
            "f1_per_class": list[float],
            "precision_macro": float,
            "recall_macro": float,
            "confusion_matrix": list[list[int]]
        }
    """
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro"),
        "f1_per_class": f1_score(labels, preds, average=None).tolist(),
        "precision_macro": precision_score(labels, preds, average="macro"),
        "recall_macro": recall_score(labels, preds, average="macro"),
        "confusion_matrix": confusion_matrix(labels, preds).tolist(),
    }


def print_report(
    labels: List[int],
    preds: List[int],
    class_names: Optional[List[str]] = None,
) -> str:
    """
    打印并返回详细的分类报告。

    Args:
        labels: 真实标签列表
        preds: 预测标签列表
        class_names: 类别名称列表

    Returns:
        str: 分类报告文本
    """
    if class_names is None:
        class_names = ["负面", "正面"]

    report = classification_report(labels, preds, target_names=class_names)
    print("\n" + "=" * 60)
    print("分类报告")
    print("=" * 60)
    print(report)
    return report


def format_metrics_summary(metrics: dict) -> str:
    """
    将指标字典格式化为可读文本。

    Args:
        metrics: compute_metrics 返回的指标字典

    Returns:
        str: 格式化后的指标摘要
    """
    lines = [
        f"  Accuracy:        {metrics['accuracy']:.4f}",
        f"  Precision-macro: {metrics['precision_macro']:.4f}",
        f"  Recall-macro:    {metrics['recall_macro']:.4f}",
    ]
    return "\n".join(lines)
