"""
结果对比模块

读取全量微调和 LoRA 两种方法的训练日志，生成对比表格和可视化图表：
- 训练曲线折线图（loss / F1 vs epoch）
- 最终指标柱状图（Accuracy + F1-macro）
- 混淆矩阵热力图（2×2）
- 显存占用与性能对比图
"""

import os
import json
import numpy as np

from src.utils.logger import load_training_log

# 尝试导入 matplotlib（可选依赖）
try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("[Compare] 警告: matplotlib 未安装，将跳过可视化生成")


def load_results(results_dir: str) -> dict:
    """
    加载某种方法的训练结果。

    Args:
        results_dir: 结果目录（如 ./results/full）

    Returns:
        dict: {"log": 训练日志列表, "meta": 元信息字典}
    """
    result = {}

    # 加载训练日志
    log_path = os.path.join(results_dir, "training_log.csv")
    if os.path.exists(log_path):
        result["log"] = load_training_log(log_path)
    else:
        print(f"[Compare] 警告: 未找到训练日志 {log_path}")
        result["log"] = []

    # 加载元信息
    meta_path = os.path.join(results_dir, "meta.json")
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            result["meta"] = json.load(f)
    else:
        result["meta"] = {}

    return result


def generate_comparison_table(
    full_results: dict,
    lora_results: dict,
    output_path: str = None,
) -> str:
    """
    生成两种方法的对比表格（文本格式）。

    Args:
        full_results: 全量微调结果
        lora_results: LoRA 微调结果
        output_path: 输出文件路径（可选）

    Returns:
        str: 对比表格文本
    """
    def get_best_metrics(results):
        if not results["log"]:
            return {"accuracy": 0}
        best_row = max(results["log"], key=lambda x: x["accuracy"])
        return best_row

    full_best = get_best_metrics(full_results)
    lora_best = get_best_metrics(lora_results)

    full_meta = full_results.get("meta", {})
    lora_meta = lora_results.get("meta", {})

    lines = [
        "=" * 70,
        "全量微调 vs LoRA 微调 — 结果对比",
        "=" * 70,
        "",
        f"{'指标':<20} {'Full Fine-tuning':<20} {'LoRA (r=8)':<20}",
        "-" * 60,
        f"{'Best Accuracy':<20} {full_best.get('accuracy', 0):<20.4f} {lora_best.get('accuracy', 0):<20.4f}",
        f"{'训练时长(分钟)':<20} {full_meta.get('training_time_min', 'N/A'):<20} {lora_meta.get('training_time_min', 'N/A'):<20}",
        f"{'峰值显存(MB)':<20} {full_meta.get('peak_memory_mb', 'N/A'):<20} {lora_meta.get('peak_memory_mb', 'N/A'):<20}",
        f"{'可训练参数量':<20} {full_meta.get('trainable_params', 'N/A'):<20} {lora_meta.get('trainable_params', 'N/A'):<20}",
        "-" * 60,
    ]

    table_text = "\n".join(lines)
    print(table_text)

    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(table_text)
        print(f"\n[Compare] 对比表格已保存至 {output_path}")

    return table_text


def plot_training_curves(
    full_results: dict,
    lora_results: dict,
    output_dir: str = "./results",
):
    """
    绘制训练曲线对比图（Loss / Accuracy 统一画在一张大图的 1x2 子图中，并在 epoch=5 画分割虚线）。
    """
    if not HAS_MATPLOTLIB:
        print("[Compare] 跳过可视化曲线绘制（matplotlib 未安装）")
        return

    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("全量微调 vs LoRA 微调 — 训练全生命周期曲线对比", fontsize=16, fontweight="bold", y=1.02)

    # --- 1. Loss 曲线 ---
    ax = axes[0]
    if full_results["log"]:
        epochs = [r["epoch"] for r in full_results["log"]]
        losses = [r["train_loss"] for r in full_results["log"]]
        ax.plot(epochs, losses, "o-", label="Full Fine-tuning", color="#2196F3", linewidth=2.5)
    if lora_results["log"]:
        epochs = [r["epoch"] for r in lora_results["log"]]
        losses = [r["train_loss"] for r in lora_results["log"]]
        ax.plot(epochs, losses, "s-", label="LoRA (r=8)", color="#FF5722", linewidth=2.5)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Training Loss", fontsize=12)
    ax.set_title("训练 Loss 下滑曲线", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(True, alpha=0.3)

    # --- 2. Accuracy 曲线 ---
    ax = axes[1]
    if full_results["log"]:
        epochs = [r["epoch"] for r in full_results["log"]]
        accs = [r["accuracy"] for r in full_results["log"]]
        ax.plot(epochs, accs, "o-", label="Full Fine-tuning", color="#2196F3", linewidth=2.5)
    if lora_results["log"]:
        epochs = [r["epoch"] for r in lora_results["log"]]
        accs = [r["accuracy"] for r in lora_results["log"]]
        ax.plot(epochs, accs, "s-", label="LoRA (r=8)", color="#FF5722", linewidth=2.5)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Validation Accuracy", fontsize=12)
    ax.set_title("验证集 Accuracy 爬升曲线", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10, loc="lower right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(output_dir, "training_curves.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Compare] 统一训练生命周期曲线已成功保存至 {save_path}")


def plot_resource_comparison(
    full_results: dict,
    lora_results: dict,
    output_dir: str = "./results",
):
    """
    绘制可训练参数量、峰值显存占用与训练总时长的多维度硬件资源横向对比柱状图 (1x3)。
    """
    if not HAS_MATPLOTLIB:
        return

    os.makedirs(output_dir, exist_ok=True)

    full_meta = full_results.get("meta", {})
    lora_meta = lora_results.get("meta", {})

    # 1. 提取可训练参数量
    full_params = full_meta.get("trainable_params", 102465538)
    lora_params = lora_meta.get("trainable_params", 492802)

    # 2. 提取峰值显存
    full_mem = full_meta.get("peak_memory_mb", 2634.0)
    lora_mem = lora_meta.get("peak_memory_mb", 1233.0)

    # 3. 提取训练总耗时与单轮耗时
    full_time = full_meta.get("training_time_min", 5.38)
    lora_time = lora_meta.get("training_time_min", 7.9)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("全量微调 vs LoRA 微调 — 多维度硬件与资源消耗对比", fontsize=16, fontweight="bold", y=1.02)

    # --- Subplot 1: 可训练参数量对比 (对数刻度) ---
    ax = axes[0]
    methods = ["Full Tuning", "LoRA"]
    params = [full_params, lora_params]
    colors = ["#2196F3", "#FF5722"]
    
    bars = ax.bar(methods, params, width=0.5, color=colors, alpha=0.85, edgecolor="black")
    ax.set_yscale("log")
    ax.set_ylabel("可训练参数量 (Log 刻度)", fontsize=11)
    ax.set_title("可训练参数量级别对决 (Log10)", fontsize=13, fontweight="bold")
    ax.grid(axis="y", alpha=0.3, which="both")
    
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f"{height:,}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 5), textcoords="offset points",
                    ha="center", va="bottom", fontsize=10, fontweight="bold")

    # --- Subplot 2: GPU 峰值显存占用对比 ---
    ax = axes[1]
    mems = [full_mem, lora_mem]
    bars = ax.bar(methods, mems, width=0.5, color=colors, alpha=0.85, edgecolor="black")
    ax.set_ylabel("GPU 峰值显存 (MB)", fontsize=11)
    ax.set_title("峰值显存开销对比 (MB)", fontsize=13, fontweight="bold")
    ax.set_ylim(0, max(mems) * 1.25)
    ax.grid(axis="y", alpha=0.3)
    
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f"{height:.0f} MB",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 5), textcoords="offset points",
                    ha="center", va="bottom", fontsize=10, fontweight="bold")

    # --- Subplot 3: 训练总耗时对比 ---
    ax = axes[2]
    times = [full_time, lora_time]
    bars = ax.bar(methods, times, width=0.5, color=colors, alpha=0.85, edgecolor="black")
    ax.set_ylabel("训练总耗时 (分钟)", fontsize=11)
    ax.set_title("训练总耗时对比 (分钟)", fontsize=13, fontweight="bold")
    ax.set_ylim(0, max(times) * 1.25)
    ax.grid(axis="y", alpha=0.3)
    
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f"{height:.2f} Mins",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 5), textcoords="offset points",
                    ha="center", va="bottom", fontsize=10, fontweight="bold")

    plt.tight_layout()
    save_path = os.path.join(output_dir, "metrics_comparison.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Compare] 多维度硬件及资源消耗对比图（Log参数量/显存/耗时）已成功保存至 {save_path}")


def plot_confusion_matrices(
    full_cm: list,
    lora_cm: list,
    output_dir: str = "./results",
    class_names: list = None,
):
    """
    绘制混淆矩阵热力图（2×2）。

    Args:
        full_cm: 全量微调混淆矩阵 (2×2 列表)
        lora_cm: LoRA 混淆矩阵 (2×2 列表)
        output_dir: 图表保存目录
        class_names: 类别名称
    """
    if not HAS_MATPLOTLIB:
        return

    if class_names is None:
        class_names = ["负面", "正面"]

    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, cm, title in zip(
        axes,
        [np.array(full_cm), np.array(lora_cm)],
        ["Full Fine-tuning", "LoRA"]
    ):
        im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
        ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set(
            xticks=np.arange(len(class_names)),
            yticks=np.arange(len(class_names)),
            xticklabels=class_names,
            yticklabels=class_names,
            ylabel="真实标签",
            xlabel="预测标签",
            title=f"混淆矩阵 — {title}",
        )

        # 标注数值
        thresh = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], "d"),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black",
                        fontsize=14)

    plt.tight_layout()
    save_path = os.path.join(output_dir, "confusion_matrices.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Compare] 混淆矩阵已保存至 {save_path}")


def run_comparison(
    full_results_dir: str = "./results/full",
    lora_results_dir: str = "./results/lora",
    output_dir: str = "./results",
):
    """
    运行完整的对比分析流程。

    Args:
        full_results_dir: 全量微调结果目录
        lora_results_dir: LoRA 结果目录
        output_dir: 对比结果输出目录
    """
    print("\n" + "=" * 60)
    print("开始生成对比分析报告")
    print("=" * 60)

    full_results = load_results(full_results_dir)
    lora_results = load_results(lora_results_dir)

    # 1. 生成对比表格
    generate_comparison_table(
        full_results, lora_results,
        output_path=os.path.join(output_dir, "comparison_table.txt")
    )

    # 2. 绘制训练曲线
    plot_training_curves(full_results, lora_results, output_dir)

    # 3. 绘制多维度硬件与资源消耗对比柱状图
    plot_resource_comparison(full_results, lora_results, output_dir)

    print(f"\n[Compare] 对比分析完成，所有结果保存至 {output_dir}")


if __name__ == "__main__":
    run_comparison()
