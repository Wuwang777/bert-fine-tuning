"""
全量微调入口脚本

加载配置、数据、模型，执行全量微调训练，保存最优模型至 ./models/full/。

用法：
    python -m src.training.full_finetune --config configs/full_finetune.yaml
"""

import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import sys
import argparse
import yaml
import torch
from transformers import AutoTokenizer

# 确保项目根目录在 sys.path 中
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.utils.seed import set_seed
from src.data.preprocess import preprocess_and_save
from src.data.dataset import create_dataloaders
from src.models.classifier import BertSentimentClassifier
from src.training.trainer import SentimentTrainer
from src.evaluation.evaluator import print_report


def load_config(config_path: str) -> dict:
    """加载 YAML 配置文件"""
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def main(config_path: str = "configs/full_finetune.yaml"):
    """
    全量微调主流程。

    1. 加载配置
    2. 固定随机种子
    3. 数据预处理（如尚未处理）
    4. 构建 DataLoader
    5. 构建模型
    6. 训练
    7. 测试集评估
    """
    # 1. 加载配置
    config = load_config(config_path)
    config["method"] = "full"
    print(f"[Full] 配置文件: {config_path}")
    print(f"[Full] 配置内容: {config}")

    # 2. 固定随机种子
    set_seed(config.get("seed", 42))

    # 3. 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Full] 训练设备: {device}")
    if torch.cuda.is_available():
        print(f"[Full] GPU: {torch.cuda.get_device_name(0)}")
        print(f"[Full] 显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    # 4. 数据预处理（如尚未处理）
    data_dir = config.get("data_dir", "./data/processed")
    if not os.path.exists(data_dir):
        print("[Full] 数据尚未预处理，开始预处理...")
        preprocess_and_save(
            dataset_name=config.get("dataset_name", "seamew/ChnSentiCorp"),
            output_dir=data_dir,
            seed=config.get("seed", 42),
        )

    # 5. 构建 Tokenizer 和 DataLoader
    model_name = config.get("model_name", "hfl/chinese-roberta-wwm-ext")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    dataloaders = create_dataloaders(
        data_dir=data_dir,
        tokenizer=tokenizer,
        max_length=config.get("max_length", 128),
        train_batch_size=config.get("batch_size", 16),
        eval_batch_size=config.get("eval_batch_size", 64),
    )

    # 6. 构建模型
    model = BertSentimentClassifier(
        model_name=model_name,
        num_classes=config.get("num_classes", 2),
        dropout=config.get("dropout", 0.1),
    )

    param_info = model.get_trainable_params_count()
    print(f"[Full] 模型参数: 总计 {param_info['total']:,}, "
          f"可训练 {param_info['trainable']:,} ({param_info['trainable_pct']})")

    # 7. 训练
    trainer = SentimentTrainer(
        model=model,
        train_loader=dataloaders["train"],
        dev_loader=dataloaders["validation"],
        config=config,
        device=device,
    )

    results = trainer.train(
        tokenizer=tokenizer,
        method="full",
        test_loader=dataloaders.get("test"),
    )

    # 8. 最终测试集评估 + 报告
    if dataloaders.get("test"):
        print("\n" + "=" * 60)
        print("最终测试集详细报告")
        print("=" * 60)
        test_metrics = trainer.evaluate(dataloaders["test"])

        # 收集所有预测
        all_preds, all_labels = [], []
        model.eval()
        with torch.no_grad():
            for batch in dataloaders["test"]:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                with torch.amp.autocast("cuda"):
                    logits = model(input_ids, attention_mask)
                preds = logits.argmax(dim=-1).cpu().tolist()
                all_preds.extend(preds)
                all_labels.extend(batch["labels"].tolist())

        print_report(all_labels, all_preds)

    print("\n[Full] 全量微调完成！")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BERT 全量微调入口")
    parser.add_argument(
        "--config", type=str, default="configs/full_finetune.yaml",
        help="配置文件路径"
    )
    args = parser.parse_args()
    main(args.config)
