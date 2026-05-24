# 基于 BERT 的中文情感分析微调对比研究

> 全量微调 / LoRA 两方法性能对比 · 含 GUI 推理应用

## 📋 项目概述

本项目以中文二分类情感分析为任务，基于 `hfl/chinese-roberta-wwm-ext` 预训练模型，对比**全量微调（Full Fine-tuning）**与 **LoRA** 两种微调方法的性能差异，并将最优模型封装为 tkinter GUI 推理应用。

### 核心特性

- 🔬 **系统对比**：全量微调 vs LoRA，统一实验框架下的公平对比
- 📊 **完整评估**：Accuracy、F1-macro、混淆矩阵、训练曲线等多维度分析
- 🖥️ **GUI 应用**：tkinter 推理界面，支持 full/lora 模型热切换
- 🔄 **可复现**：固定随机种子，配置文件驱动，训练日志完整记录

## 🏗️ 技术选型

| 组件 | 选型 | 说明 |
|------|------|------|
| 底座模型 | `hfl/chinese-roberta-wwm-ext` | 全词掩码 + RoBERTa，~102M 参数 |
| 数据集 | `seamew/ChnSentiCorp` | 中文情感二分类，~12K 条 |
| 分类头 | 两层 FFN | Linear→LayerNorm→GELU→Dropout→Linear |
| LoRA 库 | `peft >= 0.10` | 注入 query/value，rank=8 |
| GUI | tkinter | 零依赖，跨平台 |

## 📁 目录结构

```
bert-fine-tuning/
├── configs/
│   ├── full_finetune.yaml      # 全量微调配置
│   └── lora.yaml               # LoRA 微调配置
├── data/
│   ├── raw/                    # 原始数据
│   └── processed/              # 预处理后数据
├── src/
│   ├── data/
│   │   ├── preprocess.py       # 数据清洗与划分
│   │   └── dataset.py          # PyTorch Dataset
│   ├── models/
│   │   ├── classifier.py       # BertSentimentClassifier
│   │   └── lora_config.py      # LoRA 配置工厂
│   ├── training/
│   │   ├── trainer.py          # 统一训练器
│   │   ├── full_finetune.py    # 全量微调入口
│   │   └── lora_finetune.py    # LoRA 微调入口
│   ├── evaluation/
│   │   ├── evaluator.py        # 评估指标计算
│   │   └── compare.py          # 对比分析与可视化
│   └── utils/
│       ├── seed.py             # 随机种子固定
│       └── logger.py           # 训练日志
├── models/                     # 保存的模型（训练产物）
│   ├── full/
│   └── lora/
├── gui/
│   ├── predictor.py            # 推理封装
│   └── app.py                  # GUI 主程序
├── results/                    # 训练结果与日志
├── pyproject.toml              # uv 项目配置
├── requirements.txt            # pip 依赖
└── README.md
```

## 🚀 快速开始

### 1. 环境配置

```bash
# 使用 uv（推荐）
uv sync

# 或使用 pip
pip install -r requirements.txt
```

### 2. 数据预处理

```bash
# 自动下载 ChnSentiCorp 并预处理
python -m src.data.preprocess

# 如需使用 HuggingFace 镜像
set HF_ENDPOINT=https://hf-mirror.com
python -m src.data.preprocess
```

### 3. 训练

```bash
# 全量微调
python -m src.training.full_finetune --config configs/full_finetune.yaml

# LoRA 微调
python -m src.training.lora_finetune --config configs/lora.yaml
```

### 4. 结果对比

```bash
python -m src.evaluation.compare
```

### 5. GUI 推理

```bash
python -m gui.app
```

## ⚙️ 配置说明

### 全量微调 (`configs/full_finetune.yaml`)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| bert_lr | 2e-5 | BERT 层学习率 |
| head_lr | 1e-4 | 分类头学习率 |
| batch_size | 16 | 训练 batch（+ 梯度累积 ×2 = 等效 32） |
| epochs | 5 | 训练轮次 |
| dropout | 0.1 | 分类头 Dropout |

### LoRA 微调 (`configs/lora.yaml`)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| lr | 1e-4 | 统一学习率 |
| batch_size | 64 | 训练 batch |
| epochs | 10 | 训练轮次（参数少需更多） |
| lora_r | 8 | LoRA 秩 |
| lora_alpha | 16 | LoRA 缩放因子 |
| lora_dropout | 0.05 | LoRA Dropout |

## 📊 预期结果

| 方法 | Test Accuracy | Test F1-macro | 峰值显存 |
|------|:---:|:---:|:---:|
| Full Fine-tuning | 94~96% | 0.94~0.96 | ~8GB |
| LoRA (r=8) | 92~95% | 0.92~0.95 | ~4~5GB |

## 🖥️ GUI 使用

运行 `python -m gui.app` 启动推理界面：

1. **模型选择**：下拉框切换 `full` / `lora` 模型
2. **文本输入**：在输入框中输入待分析的中文文本
3. **点击分析**：查看预测结果（情感标签 + 置信度）

## 📚 参考文献

- Cui et al. (2020). *Pre-Training with Whole Word Masking for Chinese BERT*
- Hu et al. (2021). *LoRA: Low-Rank Adaptation of Large Language Models*
- 谭松波 (2012). *ChnSentiCorp 中文情感语料库*
- [HuggingFace PEFT 文档](https://huggingface.co/docs/peft)
- [HFL 中文预训练模型](https://github.com/ymcui/Chinese-BERT-wwm)

## 📝 许可

本项目仅供学习研究使用。
