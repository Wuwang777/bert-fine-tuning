# 基于 BERT 的中文情感分析微调对比研究

> 全量微调 / LoRA 两方法性能对比 · 含内置 Web 接口及 Claude 风格前端控制台

## 📋 项目概述

本项目以中文二分类情感分析为任务，基于 `hfl/chinese-roberta-wwm-ext` 预训练模型，深度对比**全量微调（Full Fine-tuning）**与 **LoRA** 两种微调方法的性能与资源消耗差异。

为了方便推理测试与应用展示，项目不仅集成了本地 tkinter GUI 推理应用，还额外实现了基于 **FastAPI** 的 RESTful API 服务，并搭载了一个**极具质感的 Claude 暖沙色毛玻璃风格前端 Web 控制台**，支持动态模型热切换。

### 核心特性

- 🔬 **系统对比**：全量微调 vs LoRA，统一实验框架与超参数控制下的公平对比。
- 📊 **完整评估**：Accuracy、F1-macro、混淆矩阵、训练曲线等多维度详细分析。
- 🌐 **Web API 服务**：基于 FastAPI 构建，提供单条预测、批量预测、设备/就绪查询及模型在线热切换功能。
- 🎨 **Claude 风格 Web 控制台**：内置于 API 服务中，采用高端毛玻璃（Glassmorphism）质感，融入 Anthropic / Claude 的暖沙、陶土红及鼠尾草绿品牌色系，交互动效优雅。
- 🖥️ **本地 GUI 应用**：内置轻量化 tkinter 推理界面，支持 full/lora 模型无缝热切换。
- 🔄 **高复现性**：固定随机种子，配置文件驱动，训练日志与元数据完整记录。

---

## 🏗️ 技术选型

| 组件 | 选型 | 说明 |
|------|------|------|
| **底座模型** | `hfl/chinese-roberta-wwm-ext` | 中文全词掩码 + RoBERTa，约 102M 参数 |
| **数据集** | `seamew/ChnSentiCorp` | 经典中文情感二分类数据集，约 12K 条数据 |
| **分类头** | 两层 FFN | `Linear` ➔ `LayerNorm` ➔ `GELU` ➔ `Dropout` ➔ `Linear` |
| **LoRA 库** | `peft >= 0.10` | 注入 Query/Value 权重，设定 $r=8, \alpha=16$ |
| **Web 框架** | `FastAPI` | 高性能、基于 Pydantic 数据验证的异步 API 框架 |
| **Web 渲染** | `HTML5 / Vanilla CSS & JS` | 零依赖纯前端实现，搭载磨砂玻璃面板及马卡龙色背景流光 |
| **GUI 框架** | `tkinter` | Python 零依赖跨平台轻量界面 |

---

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
│   ├── api/
│   │   ├── __init__.py
│   │   ├── app.py              # FastAPI Web 服务主程序
│   │   └── static/
│   │       └── index.html      # Claude 风格 Web 交互控制台
│   └── utils/
│       ├── seed.py             # 随机种子固定
│       └── logger.py           # 训练日志
├── models/                     # 保存的模型权重（训练产物）
│   ├── full/                   # 全量微调模型权重及配置
│   └── lora/                   # LoRA 适配器权重及配置
├── gui/
│   ├── predictor.py            # 统一推理封装 (Full / LoRA)
│   └── app.py                  # tkinter 本地 GUI 启动程序
├── results/                    # 训练过程 CSV 指标与可视化图表
├── pyproject.toml              # uv 项目依赖及清华源配置
├── requirements.txt            # 导出依赖
├── run_full.py                 # 全量微调一键训练脚本
├── run_lora.py                 # LoRA 一键微调训练脚本
├── run_api.py                  # Web 服务及前端控制台启动器
└── README.md
```

---

## 🚀 快速开始

### 1. 环境配置

推荐使用高效的包管理器 `uv`（已内置清华镜像源配置）：

```bash
# 使用 uv 同步安装依赖（推荐）
uv sync

# 或使用标准 pip 安装
pip install -r requirements.txt
```

### 2. 数据预处理

一键下载 `ChnSentiCorp` 数据集并执行分词预处理，数据集将保存在 `data/` 目录下：

```bash
python -m src.data.preprocess

# 如在国内网络遇到下载困难，可使用 Hugging Face 镜像源：
set HF_ENDPOINT=https://hf-mirror.com
python -m src.data.preprocess
```

### 3. 模型训练

项目为全量微调与 LoRA 提供了独立的一键启动程序：

```bash
# 执行全量微调训练
python run_full.py

# 执行 LoRA 微调训练
python run_lora.py
```

### 4. 结果对比与评估

运行对比分析脚本，会自动读取 `results/` 下的训练日志，生成准确率与 Loss 的收敛曲线，并输出混淆矩阵与性能对比图：

```bash
python -m src.evaluation.compare
```

---

## 🌐 Web API 与前端控制台使用

项目内置了完整的后台推理接口及可视化的前端测试面板。

### 1. 启动 Web 服务

在虚拟环境中运行项目根目录下的 API 启动器：

```bash
python run_api.py
```
> **注**：服务会自动扫描 `models/` 目录，默认优先加载 `full` 模型。如未检测到模型，服务依然会正常启动，您可在完成训练后通过接口或网页控制台动态加载。

### 2. 访问地址

* **Claude 风格 Web 控制台**：[http://localhost:8000/](http://localhost:8000/)
* **交互式 Swagger API 文档**：[http://localhost:8000/docs](http://localhost:8000/docs)

### 3. 核心 API 接口说明

* **`GET /health`**：健康状态检查。返回当前加载的模型类型（`full`/`lora`）、计算设备（`cuda`/`cpu`）以及本地磁盘已训练就绪的模型。
* **`POST /predict`**：单条文本情感分析。
  * 请求体：`{"text": "文本内容"}`
  * 返回：包含情感标签（如 `正面 😊` / `负面 😞`）、置信度以及各个类别的具体概率。
* **`POST /predict/batch`**：批量文本分析。
  * 请求体：`{"texts": ["文本1", "文本2"]}`
* **`POST /model/switch`**：**动态热切换模型**。
  * 请求体：`{"method": "lora"}` 或 `{"method": "full"}`
  * 说明：无须重启服务器，实时更换内存/显存中的推理模型，支持高并发安全访问。

---

## 🖥️ 本地 GUI 客户端

除了 Web 控制台外，项目保留了基于 `tkinter` 的本地 GUI 客户端：

```bash
python -m gui.app
```

1. **模型选择**：下拉框快速热切换 `full` / `lora` 模型。
2. **文本输入**：提供文本域输入待分析的中文文本。
3. **可视化进度条**：直观展示预测置信度，根据情感倾向（正/负）动态变色（绿/红）。

---

## ⚙️ 实验配置说明

### 全量微调 (`configs/full_finetune.yaml`)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `bert_lr` | 2e-5 | BERT 预训练层学习率 |
| `head_lr` | 1e-4 | 顶层分类头学习率 |
| `batch_size` | 16 | 训练 batch_size（配置梯度累积，等效为 32） |
| `epochs` | 5 | 训练轮次 |
| `dropout` | 0.1 | 分类层 Dropout 比例 |

### LoRA 微调 (`configs/lora.yaml`)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `lr` | 1e-4 | 统一优化学习率 |
| `batch_size` | 64 | 训练 batch_size |
| `epochs` | 10 | 训练轮次（参数量极少，需要更多收敛轮次） |
| `lora_r` | 8 | LoRA 秩 (Rank) |
| `lora_alpha` | 16 | LoRA 缩放因子 (Alpha) |
| `lora_dropout` | 0.05 | LoRA 层的 Dropout |

---

## 📊 性能基准 (实测数据)

基于本地环境微调训练得到的真实性能数据对比：

| 方法 | 最佳准确率 (Best Acc) | 训练耗时 (Minutes) | 可训练参数量 | 峰值显存 (Peak VRAM) |
|------|:---:|:---:|:---:|:---:|
| **Full Fine-tuning** | **95.29%** | 11.11 min | 102,465,538 (100.0%) | 2634 MB (~2.57 GB) |
| **LoRA (r=8)** | 92.89% | **7.78 min** | 492,802 (**~0.48%**) | **1233 MB** (~1.20 GB) |

> **实验结论**：LoRA 微调在此任务中表现出极佳的资源与效率优势。它仅需调整 **0.48%** 的可训练参数（约 49.2 万个参数），便达到了全量微调 **97.4%** 的情感分类准确度；在资源控制上，LoRA 相比全量微调**降低了 53% 的显存占用（仅需 1.20 GB）**，训练时长缩短了 **30%**。这极大地降低了模型在消费级显卡上的微调与部署门槛。

---

## 📚 参考文献

- Cui et al. (2020). *Pre-Training with Whole Word Masking for Chinese BERT*
- Hu et al. (2021). *LoRA: Low-Rank Adaptation of Large Language Models*
- 谭松波 (2012). *ChnSentiCorp 中文情感语料库*
- [HuggingFace PEFT 官方文档](https://huggingface.co/docs/peft)
- [ymcui / Chinese-BERT-wwm 预训练仓库](https://github.com/ymcui/Chinese-BERT-wwm)

---

## 📝 许可

本项目仅供学术研究和教学展示使用，遵循 MIT 开源许可协议。
