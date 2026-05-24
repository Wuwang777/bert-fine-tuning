# 基于 BERT 的中文情感分析微调对比研究
> 全量微调 / LoRA 两方法性能对比 · 含 GUI 推理应用

---

## 一、项目概述

### 1.1 研究背景

情感分析（Sentiment Analysis）是自然语言处理领域的基础任务之一，在舆情监控、用户画像、商品评价分析等场景中具有广泛应用价值。随着预训练语言模型（PLM）的兴起，以 BERT 为代表的模型通过在大规模语料上进行自监督预训练，获得了强大的语言表示能力。

然而，在实际落地场景中，全量微调（Full Fine-tuning）对显存和算力的要求较高。近年来兴起的参数高效微调方法（PEFT），尤其是 LoRA，通过在预训练权重上引入低秩适配器，大幅降低了微调成本，但在中文情感任务上与全量微调的系统性对比研究仍较为有限。

本项目以中文二分类情感分析为具体任务，构建统一的训练与评估框架，对全量微调与 LoRA 两种方法进行系统对比，并在此基础上将最优模型封装为 GUI 推理应用，实现完整的"训练→保存→部署"闭环。

### 1.2 研究目标

1. 构建完整的中文情感分析微调流水线，覆盖数据处理、模型构建、训练、评估全流程
2. 在统一实验条件下，对比全量微调与 LoRA 两种微调范式的性能差异
3. 分析超参数（学习率、LoRA rank 等）对模型性能的影响规律
4. 将微调后的最优模型保存至本地 `./models` 目录，供后续推理使用
5. 基于保存的模型开发 GUI 情感预测应用，实现对用户输入文本的实时情感分析

### 1.3 任务定义

- **任务类型**：文本二分类（正面 / 负面）
- **输入**：中文评论文本（截断至 128 tokens）
- **输出**：情感类别标签 `{0: 负面, 1: 正面}` 及对应置信度
- **核心指标**：Accuracy、F1（二分类取正类 F1 或 macro 均可）

---

## 二、技术选型

### 2.1 底座模型：chinese-roberta-wwm-ext

| 项目 | 说明 |
|------|------|
| 模型名称 | `hfl/chinese-roberta-wwm-ext` |
| 来源 | 哈工大讯飞联合实验室（HFL） |
| 预训练策略 | 全词掩码（Whole Word Masking） + RoBERTa 动态掩码 |
| 参数量 | ~102M（BERT-base 同规模） |
| 选型理由 | 全词掩码对中文分词更友好；在多项中文 NLP 基准上优于原版 bert-base-chinese；社区维护活跃，HuggingFace Hub 直接加载 |

> **为什么不选 bert-base-chinese？** 原版 bert-base-chinese 采用字级别掩码，对中文理解不如全词掩码；chinese-roberta-wwm-ext 在 ChnSentiCorp 等情感任务上公开结果普遍高出 1~2 个百分点。
>
> **为什么不选 macbert-base？** MacBERT 将 MLM 替换为近义词掩码策略，但其提升主要体现在 MRC 任务；情感分类任务上与 roberta-wwm-ext 差异不显著，而 roberta-wwm-ext 社区资源更丰富。

### 2.2 分类头：两层 FFN

在 BERT 输出的 `[CLS]` 语义向量（768 维）后接如下分类头：

```
[CLS] hidden (768) 
  → Linear(768 → 256) → LayerNorm(256) → GELU → Dropout(p=0.1)
  → Linear(256 → 2)
  → CrossEntropyLoss
```

设计考量：
- 单层 Linear 表达能力不足，三层以上易过拟合小数据
- LayerNorm 稳定训练梯度，配合 GELU 激活效果优于 ReLU
- Dropout 率 0.1 与 BERT 内部保持一致
- 二分类输出维度为 2（负面 / 正面），推理时取 softmax 概率作为置信度展示

### 2.3 数据集：ChnSentiCorp

| 属性 | 值 |
|------|----|
| 数据来源 | 谭松波中文情感语料（酒店/书籍/电子产品评论） |
| HuggingFace 标识 | `seamew/ChnSentiCorp` |
| 原始标签 | 二分类（0: 负面 / 1: 正面），直接使用，无需改造 |
| 总量 | ~12,000 条（正负基本均衡） |
| 划分 | 按 8:1:1 划分 train / dev / test（分层采样） |

> ChnSentiCorp 原始即为二分类，正负样本比例约 1:1，无需额外处理类别不平衡问题，可直接用于训练。

### 2.4 微调方法概述

| 方法 | 可训练参数 | 显存占用（估算） | 适用场景 |
|------|-----------|----------------|---------|
| 全量微调（Full） | ~102M（全部） | ~10GB（fp16） | 算力充足，追求最优性能 |
| LoRA | ~0.3M（rank=8） | ~4~5GB（fp16） | 显存受限，性能接近全量 |

### 2.5 GUI 框架：tkinter

| 项目 | 说明 |
|------|------|
| 框架 | Python 标准库 tkinter（无需额外安装） |
| 选型理由 | 零依赖、跨平台（Windows / macOS / Linux）、轻量，适合本地推理工具 |
| 交互设计 | 文本输入框 + 预测按钮 + 结果展示区（标签 + 置信度进度条） |

---

## 三、实验设计

### 3.1 数据处理流程

```
原始 ChnSentiCorp（seamew/ChnSentiCorp）
  ↓ 加载（HuggingFace datasets）
  ↓ 去重 + 长度过滤（< 5 tokens 或 > 512 tokens 的样本丢弃）
  ↓ 按 8:1:1 划分 train / dev / test（分层采样保持正负均衡）
  ↓ Tokenizer（BertTokenizer，max_length=128，padding/truncation）
  ↓ DataLoader（train: shuffle=True，dev/test: shuffle=False）
```

二分类无需类别构建或上采样，流程较三分类显著简化。

### 3.2 两种微调方案详细配置

#### 方案 A：全量微调（Full Fine-tuning）

```
优化器：AdamW（weight_decay=0.01，beta1=0.9，beta2=0.999）
学习率：2e-5（BERT 层）/ 1e-4（分类头）—— 差异化学习率
调度器：线性 warmup（前 10% steps）+ 线性衰减
Batch Size：32（T4×2 DDP）/ 16（3060 单卡，梯度累积×2）
Epochs：5
混合精度：fp16（torch.cuda.amp）
```

#### 方案 B：LoRA 微调

```
基础库：peft >= 0.10
注入位置：BERT 每层的 query / value 投影矩阵
rank（r）：搜索范围 {4, 8, 16}，默认 8
alpha：16（通常设为 2×r）
Dropout：0.05
Batch Size：64（显存更低，可放大）
Epochs：10（参数少，需更多轮次收敛）
学习率：1e-4
```

> LoRA 仅冻结 BERT 主干，分类头全量更新。

### 3.3 超参数搜索

对以下关键超参数做网格搜索，使用 dev F1 作为选择标准：

| 参数 | 方案 | 搜索范围 |
|------|------|---------|
| 学习率（lr） | Full | {5e-5, 2e-5, 1e-5} |
| 学习率（lr） | LoRA | {5e-4, 1e-4, 5e-5} |
| LoRA rank（r） | LoRA | {4, 8, 16} |
| Dropout（FFN） | 两者 | {0.1, 0.2} |

搜索策略：独立 grid search（不使用 Optuna），每组跑 3 epochs 快速筛选，最优配置再跑完整训练。

### 3.4 评估指标

**主指标**
- **Accuracy**：整体分类准确率
- **F1-macro**：正负两类 F1 的宏平均

**辅助记录**（写入日志，不作为主要对比）
- 正 / 负类各自的 Precision / Recall / F1
- 混淆矩阵（2×2）
- 训练时长（分钟）
- 峰值显存占用（`torch.cuda.max_memory_allocated()`）
- 可训练参数量

**对比维度总结**

| 维度 | Full | LoRA |
|------|------|------|
| Test Accuracy | ✓ | ✓ |
| Test F1-macro | ✓ | ✓ |
| 训练曲线（loss / F1） | ✓ | ✓ |
| 训练时长 | ✓ | ✓ |
| 峰值显存 | ✓ | ✓ |
| 可训练参数 | ✓ | ✓ |

---

## 四、模型保存规范

训练结束后，将最优 checkpoint 统一保存至项目根目录的 `./models` 下，目录结构如下：

```
./models/
├── full/
│   ├── best_model.pt          # 模型权重（state_dict）
│   ├── config.json            # 模型超参配置（num_classes、hidden、dropout 等）
│   └── tokenizer/             # BertTokenizer 本地副本（from_pretrained 保存）
│       ├── vocab.txt
│       ├── tokenizer_config.json
│       └── special_tokens_map.json
└── lora/
    ├── adapter_model.safetensors  # LoRA 适配器权重（peft 格式）
    ├── adapter_config.json        # LoRA 配置（r、alpha、target_modules 等）
    ├── config.json                # 分类头超参配置
    └── tokenizer/                 # 同上
```

**保存逻辑说明**

全量微调模型使用 `torch.save(model.state_dict(), ...)` 保存权重，加载时需先实例化相同结构的 `BertSentimentClassifier` 再 `load_state_dict`。

LoRA 模型使用 `model.save_pretrained("./models/lora")` 保存适配器，加载时先加载基底模型再通过 `PeftModel.from_pretrained` 恢复适配器。

Tokenizer 统一使用 `tokenizer.save_pretrained("./models/{method}/tokenizer")` 保存本地副本，确保推理时无需联网。

---

## 五、项目代码框架

### 5.1 目录结构

```
bert-sentiment/
├── data/
│   ├── raw/                    # 原始下载数据
│   └── processed/              # 处理后的 train / dev / test split
│
├── src/
│   ├── data/
│   │   ├── dataset.py          # SentimentDataset（torch Dataset）
│   │   └── preprocess.py       # 数据清洗、划分
│   │
│   ├── models/
│   │   ├── classifier.py       # BertSentimentClassifier（含 FFN 头）
│   │   └── lora_config.py      # LoRA PEFT 配置工厂函数
│   │
│   ├── training/
│   │   ├── trainer.py          # 统一 Trainer 类（两种方法共用）
│   │   ├── full_finetune.py    # 全量微调入口（含模型保存）
│   │   └── lora_finetune.py    # LoRA 微调入口（含适配器保存）
│   │
│   ├── evaluation/
│   │   ├── evaluator.py        # 评估函数（accuracy, f1, 混淆矩阵）
│   │   └── compare.py          # 两方法结果汇总对比表生成
│   │
│   └── utils/
│       ├── logger.py           # 训练日志（loss / metric 每 epoch 记录）
│       └── seed.py             # 随机种子固定
│
├── models/                     # 保存的模型文件（训练产物）
│   ├── full/
│   └── lora/
│
├── gui/
│   ├── app.py                  # GUI 主程序（tkinter）
│   └── predictor.py            # 推理封装（加载模型 + 前处理 + 后处理）
│
├── configs/
│   ├── full_finetune.yaml
│   └── lora.yaml
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_train_full.ipynb
│   ├── 03_train_lora.ipynb
│   └── 04_compare_results.ipynb
│
├── results/
│   ├── full/
│   └── lora/
│
├── requirements.txt
└── README.md
```

### 5.2 核心模块示意代码

#### `src/models/classifier.py` — 模型主体

```python
import torch
import torch.nn as nn
from transformers import AutoModel

class BertSentimentClassifier(nn.Module):
    """
    底座：chinese-roberta-wwm-ext
    分类头：两层 FFN（Linear → LayerNorm → GELU → Dropout → Linear）
    输入：[CLS] hidden state（768 维）
    输出：二分类 logits（负面 / 正面）
    """
    def __init__(self, model_name: str, num_classes: int = 2, dropout: float = 0.1):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        hidden_size = self.bert.config.hidden_size  # 768

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        cls_hidden = outputs.last_hidden_state[:, 0, :]  # [B, 768]
        logits = self.classifier(cls_hidden)              # [B, 2]
        return logits
```

#### `src/models/lora_config.py` — LoRA PEFT 配置工厂

```python
from peft import LoraConfig, TaskType, get_peft_model

def get_lora_config(r: int = 8, alpha: int = 16, dropout: float = 0.05) -> LoraConfig:
    return LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=["query", "value"],
        bias="none"
    )

def build_lora_model(base_model, r: int = 8, alpha: int = 16):
    lora_cfg = get_lora_config(r=r, alpha=alpha)
    model = get_peft_model(base_model, lora_cfg)
    model.print_trainable_parameters()
    return model
```

#### `src/training/trainer.py` — 统一训练器（含模型保存）

```python
import os
import json
import torch
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from src.evaluation.evaluator import compute_metrics
from src.utils.logger import TrainingLogger

class SentimentTrainer:
    def __init__(self, model, train_loader, dev_loader, config: dict, device):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.dev_loader = dev_loader
        self.config = config
        self.device = device
        self.logger = TrainingLogger(config["output_dir"])

        bert_params = [p for n, p in model.named_parameters() if "bert" in n and p.requires_grad]
        head_params = [p for n, p in model.named_parameters() if "classifier" in n and p.requires_grad]
        self.optimizer = AdamW([
            {"params": bert_params, "lr": config["bert_lr"]},
            {"params": head_params, "lr": config["head_lr"]},
        ], weight_decay=config.get("weight_decay", 0.01))

        total_steps = len(train_loader) * config["epochs"]
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=int(total_steps * 0.1),
            num_training_steps=total_steps
        )

        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.scaler = torch.cuda.amp.GradScaler()

    def train_one_epoch(self, epoch: int):
        self.model.train()
        total_loss = 0.0
        for batch in self.train_loader:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)

            self.optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                logits = self.model(input_ids, attention_mask)
                loss = self.loss_fn(logits, labels)

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(self.train_loader)
        metrics = self.evaluate()
        self.logger.log(epoch, avg_loss, metrics)
        return avg_loss, metrics

    def evaluate(self) -> dict:
        self.model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in self.dev_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                logits = self.model(input_ids, attention_mask)
                preds = logits.argmax(dim=-1).cpu().tolist()
                all_preds.extend(preds)
                all_labels.extend(batch["labels"].tolist())
        return compute_metrics(all_labels, all_preds)

    def save_model(self, save_dir: str, tokenizer, method: str = "full"):
        """将最优模型和 tokenizer 保存至 ./models/{method}/"""
        os.makedirs(save_dir, exist_ok=True)
        if method == "full":
            torch.save(self.model.state_dict(), os.path.join(save_dir, "best_model.pt"))
            with open(os.path.join(save_dir, "config.json"), "w") as f:
                json.dump({"num_classes": 2, "dropout": self.config.get("dropout", 0.1)}, f)
        elif method == "lora":
            # peft 模型调用 save_pretrained 保存适配器
            self.model.save_pretrained(save_dir)
        tokenizer.save_pretrained(os.path.join(save_dir, "tokenizer"))
        print(f"模型已保存至 {save_dir}")

    def train(self, tokenizer=None, method: str = "full"):
        best_f1 = 0.0
        save_dir = os.path.join("./models", method)
        for epoch in range(1, self.config["epochs"] + 1):
            loss, metrics = self.train_one_epoch(epoch)
            print(f"Epoch {epoch} | Loss: {loss:.4f} | "
                  f"Acc: {metrics['accuracy']:.4f} | F1: {metrics['f1_macro']:.4f}")
            if metrics["f1_macro"] > best_f1:
                best_f1 = metrics["f1_macro"]
                if tokenizer:
                    self.save_model(save_dir, tokenizer, method=method)
        print(f"训练完成，最优 Dev F1: {best_f1:.4f}，模型已保存至 {save_dir}")
```

#### `src/evaluation/evaluator.py` — 评估函数

```python
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

def compute_metrics(labels: list, preds: list) -> dict:
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro"),
        "f1_per_class": f1_score(labels, preds, average=None).tolist(),
        "confusion_matrix": confusion_matrix(labels, preds).tolist()
    }

def print_report(labels: list, preds: list, class_names=["负面", "正面"]):
    print(classification_report(labels, preds, target_names=class_names))
```

#### `gui/predictor.py` — 推理封装

```python
import os
import json
import torch
import torch.nn.functional as F
from transformers import BertTokenizer, AutoModel
from peft import PeftModel
from src.models.classifier import BertSentimentClassifier

LABEL_NAMES = {0: "负面 😞", 1: "正面 😊"}

class SentimentPredictor:
    """
    统一推理封装，支持全量微调和 LoRA 两种模型格式的加载。
    method: "full" | "lora"
    model_dir: ./models/full 或 ./models/lora
    """
    def __init__(self, model_dir: str, base_model_name: str, method: str = "full"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.method = method

        # 加载 tokenizer（使用本地副本，无需联网）
        tokenizer_dir = os.path.join(model_dir, "tokenizer")
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_dir)

        # 加载模型
        if method == "full":
            with open(os.path.join(model_dir, "config.json")) as f:
                cfg = json.load(f)
            self.model = BertSentimentClassifier(
                model_name=base_model_name,
                num_classes=cfg["num_classes"],
                dropout=cfg["dropout"]
            )
            self.model.load_state_dict(
                torch.load(os.path.join(model_dir, "best_model.pt"), map_location=self.device)
            )
        elif method == "lora":
            base = BertSentimentClassifier(model_name=base_model_name, num_classes=2)
            self.model = PeftModel.from_pretrained(base, model_dir)

        self.model.to(self.device)
        self.model.eval()

    def predict(self, text: str) -> dict:
        """
        输入：原始中文文本
        输出：{"label": "正面 😊", "confidence": 0.97, "probs": [0.03, 0.97]}
        """
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=128,
            truncation=True,
            padding="max_length"
        )
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)

        with torch.no_grad():
            logits = self.model(input_ids, attention_mask)
            probs = F.softmax(logits, dim=-1).squeeze().cpu().tolist()

        pred_id = int(torch.tensor(probs).argmax())
        return {
            "label": LABEL_NAMES[pred_id],
            "confidence": probs[pred_id],
            "probs": probs  # [负面概率, 正面概率]
        }
```

#### `gui/app.py` — GUI 主程序（tkinter）

```python
import tkinter as tk
from tkinter import ttk, messagebox
from gui.predictor import SentimentPredictor

# 启动时指定使用哪个模型（可改为下拉选择）
MODEL_DIR = "./models/full"
BASE_MODEL = "hfl/chinese-roberta-wwm-ext"
METHOD = "full"

class SentimentApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("中文情感分析 · BERT 微调")
        self.geometry("560x420")
        self.resizable(False, False)
        self._build_ui()
        self._load_model()

    def _build_ui(self):
        # 标题
        tk.Label(self, text="中文情感分析", font=("微软雅黑", 16, "bold")).pack(pady=(20, 4))
        tk.Label(self, text="输入文本，点击「分析」查看情感预测结果",
                 font=("微软雅黑", 10), fg="gray").pack()

        # 输入区
        frame_input = tk.LabelFrame(self, text="输入文本", padx=10, pady=8)
        frame_input.pack(fill="x", padx=24, pady=(16, 0))
        self.text_input = tk.Text(frame_input, height=5, font=("微软雅黑", 11), wrap="word")
        self.text_input.pack(fill="x")

        # 按钮
        tk.Button(self, text="分 析", font=("微软雅黑", 11),
                  width=12, command=self._predict).pack(pady=12)

        # 结果区
        frame_result = tk.LabelFrame(self, text="分析结果", padx=10, pady=10)
        frame_result.pack(fill="x", padx=24)

        self.label_result = tk.Label(frame_result, text="—", font=("微软雅黑", 14, "bold"),
                                     fg="#333333")
        self.label_result.pack()

        tk.Label(frame_result, text="置信度", font=("微软雅黑", 9), fg="gray").pack(anchor="w")
        self.progress = ttk.Progressbar(frame_result, length=400, maximum=100)
        self.progress.pack(fill="x", pady=(0, 4))
        self.label_conf = tk.Label(frame_result, text="", font=("微软雅黑", 9), fg="gray")
        self.label_conf.pack()

        # 状态栏
        self.label_status = tk.Label(self, text="模型加载中...", fg="gray",
                                     font=("微软雅黑", 9))
        self.label_status.pack(side="bottom", pady=6)

    def _load_model(self):
        try:
            self.predictor = SentimentPredictor(MODEL_DIR, BASE_MODEL, METHOD)
            self.label_status.config(text=f"模型已就绪（{METHOD}）", fg="green")
        except Exception as e:
            self.label_status.config(text=f"模型加载失败：{e}", fg="red")
            self.predictor = None

    def _predict(self):
        if not self.predictor:
            messagebox.showerror("错误", "模型未加载，请检查 ./models 目录")
            return
        text = self.text_input.get("1.0", "end").strip()
        if not text:
            messagebox.showwarning("提示", "请先输入文本")
            return

        result = self.predictor.predict(text)

        # 根据情感设置颜色
        color = "#27ae60" if "正面" in result["label"] else "#e74c3c"
        self.label_result.config(text=result["label"], fg=color)

        conf_pct = result["confidence"] * 100
        self.progress["value"] = conf_pct
        self.label_conf.config(
            text=f"{conf_pct:.1f}%  （负面 {result['probs'][0]*100:.1f}% / 正面 {result['probs'][1]*100:.1f}%）"
        )

if __name__ == "__main__":
    app = SentimentApp()
    app.mainloop()
```

---

## 六、实验流程

### 6.1 阶段划分

| 阶段 | 任务 | 预计时间 | 主要硬件 |
|------|------|---------|---------|
| 阶段一 | 环境搭建 + 数据加载与预处理 | 0.5~1 天 | 本机 |
| 阶段二 | 模型结构搭建 + smoke test（500 条跑通全流程） | 0.5~1 天 | 本机 3060 |
| 阶段三 | 超参数搜索（grid search，3 epochs per config） | 1~2 天 | Kaggle T4×2 |
| 阶段四 | 两种方法正式训练 + 模型保存至 `./models` | 1~2 天 | Kaggle T4×2 |
| 阶段五 | 结果分析 + 对比可视化 + 文档整理 | 1 天 | 本机 |
| 阶段六 | GUI 程序开发与本地测试 | 1 天 | 本机 |

**总预计周期**：5~8 天（视实验迭代情况）

### 6.2 本机 vs Kaggle 分工

```
本机 RTX 3060 Laptop（6GB）
├── 代码开发与调试
├── smoke test（mini 数据集，验证全流程）
├── LoRA 本地完整训练（fp16 显存 ~4~5GB，batch=8 + 梯度累积×4）
├── 模型文件本地验证（predictor.py 加载测试）
└── GUI 程序开发与测试（app.py）

Kaggle Tesla T4 × 2（15GB × 2）
├── 超参 grid search（Full / LoRA）
├── Full 微调正式训练（DDP 双卡）
├── LoRA 正式训练（单卡）
└── 导出 ./models 目录（zip 下载至本机）
```

### 6.3 可复现性保障

- 所有实验固定随机种子（`seed=42`，覆盖 Python random / numpy / torch / CUDA）
- 超参配置以 YAML 文件记录，每次实验附带配置快照
- 训练日志记录每 epoch 的 loss / accuracy / F1，保存为 CSV
- 最优 checkpoint 保存至 `./models`，附带 `config.json` 和 tokenizer 本地副本

---

## 七、预期结果与分析

### 7.1 性能预期（基于已有公开实验参考）

| 方法 | 预期 Test Accuracy | 预期 Test F1-macro | 峰值显存（估算） |
|------|-------------------|-------------------|----------------|
| Full Fine-tuning | 94~96% | 0.94~0.96 | ~8GB（fp16） |
| LoRA（r=8） | 92~95% | 0.92~0.95 | ~4~5GB |

> 二分类任务相比三分类更容易，预期指标整体偏高。以上为估算范围，以实际实验结果为准。

### 7.2 预期分析方向

1. **性能 vs 显存权衡**：LoRA 相比全量微调损失多少性能，节省多少显存？
2. **超参敏感性**：LoRA rank 对性能的影响是否显著？r=4 / 8 / 16 是否有明显拐点？
3. **训练稳定性**：从训练曲线对比两种方法的收敛速度和稳定性
4. **GUI 实测体验**：在典型正面 / 负面 / 模糊文本上的预测置信度分布

### 7.3 结果可视化计划

- 两方法训练曲线（loss / F1，x 轴为 epoch）对比折线图
- 最终指标对比柱状图（Accuracy + F1-macro）
- 混淆矩阵热力图（2×2）
- 显存占用与性能对比图

---

## 八、风险与应对

| 风险 | 可能原因 | 应对策略 |
|------|---------|---------|
| Kaggle T4 session 超时 | 训练时间过长 | 分段保存 checkpoint；超参搜索阶段减少 epochs |
| 全量微调显存不足（T4 单卡 15GB） | Batch size 过大 | 梯度累积；DDP 双卡分担 |
| 两种方法超参不统一导致对比不公平 | 设计问题 | 固定等效 batch size=64，固定总训练 steps 数 |
| LoRA 在本机 6GB 显存下 OOM | fp16 仍超限 | 降低 batch size 至 4，梯度累积×16；或用 gradient checkpointing |
| GUI 加载模型路径错误 | 模型未正确保存或路径不一致 | predictor.py 加入路径存在性检查，给出明确错误提示 |
| GUI 在无 GPU 机器上推理过慢 | CPU 推理 | predictor 自动 fallback 至 CPU；文本短（≤128 token）时 CPU 推理仍在秒级 |

---

## 九、参考资料

- Cui et al. (2020). *Pre-Training with Whole Word Masking for Chinese BERT*. arXiv:1906.08101
- Hu et al. (2021). *LoRA: Low-Rank Adaptation of Large Language Models*. arXiv:2106.09685
- 谭松波 (2012). *ChnSentiCorp 中文情感语料库*
- HuggingFace PEFT 文档：https://huggingface.co/docs/peft
- HFL 中文预训练模型：https://github.com/ymcui/Chinese-BERT-wwm

---

*本计划书版本：v1.2 | 研究性质：个人技术实践*
