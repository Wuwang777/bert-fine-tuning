"""
GUI 主程序 — 中文情感分析推理应用

基于 tkinter 构建，功能包括：
- 文本输入框
- 预测按钮
- 结果展示区（标签 + 置信度进度条）
- 模型选择下拉框（full / lora 热切换）
- 使用 threading 加载模型避免 UI 卡顿
"""

import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
import sys
import threading
import tkinter as tk
from tkinter import ttk, messagebox

# 确保项目根目录在 sys.path 中
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gui.predictor import SentimentPredictor


# ──────────────────── 配置 ────────────────────
BASE_MODEL = "hfl/chinese-roberta-wwm-ext"
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

# 颜色方案
COLORS = {
    "bg": "#f5f7fa",
    "card": "#ffffff",
    "primary": "#4a90d9",
    "primary_hover": "#357abd",
    "positive": "#27ae60",
    "negative": "#e74c3c",
    "text": "#2c3e50",
    "text_light": "#7f8c8d",
    "border": "#dfe6e9",
}


class SentimentApp(tk.Tk):
    """中文情感分析 GUI 应用"""

    def __init__(self):
        super().__init__()
        self.title("中文情感分析 · BERT 微调")
        self.geometry("580x500")
        self.resizable(False, False)
        self.configure(bg=COLORS["bg"])

        self.predictor = None
        self.current_method = tk.StringVar(value="full")

        self._build_ui()
        self._load_model_async()

    def _build_ui(self):
        """构建用户界面"""
        # ─── 标题区 ───
        title_frame = tk.Frame(self, bg=COLORS["bg"])
        title_frame.pack(fill="x", pady=(20, 0))

        tk.Label(
            title_frame, text="🔍 中文情感分析",
            font=("微软雅黑", 18, "bold"), fg=COLORS["text"], bg=COLORS["bg"],
        ).pack()
        tk.Label(
            title_frame, text="输入文本，点击「分析」查看情感预测结果",
            font=("微软雅黑", 10), fg=COLORS["text_light"], bg=COLORS["bg"],
        ).pack(pady=(2, 0))

        # ─── 模型选择区 ───
        model_frame = tk.Frame(self, bg=COLORS["bg"])
        model_frame.pack(fill="x", padx=30, pady=(12, 0))

        tk.Label(
            model_frame, text="模型选择：",
            font=("微软雅黑", 10), fg=COLORS["text"], bg=COLORS["bg"],
        ).pack(side="left")

        self.model_combo = ttk.Combobox(
            model_frame,
            textvariable=self.current_method,
            values=["full", "lora"],
            state="readonly",
            width=12,
            font=("微软雅黑", 10),
        )
        self.model_combo.pack(side="left", padx=(4, 0))
        self.model_combo.bind("<<ComboboxSelected>>", self._on_model_change)

        # ─── 输入区 ───
        input_frame = tk.LabelFrame(
            self, text="  输入文本  ", font=("微软雅黑", 10),
            bg=COLORS["card"], fg=COLORS["text"],
            bd=1, relief="solid", padx=12, pady=8,
        )
        input_frame.pack(fill="x", padx=30, pady=(12, 0))

        self.text_input = tk.Text(
            input_frame, height=5, font=("微软雅黑", 11),
            wrap="word", bd=0, bg=COLORS["card"],
            insertbackground=COLORS["primary"],
        )
        self.text_input.pack(fill="x")
        self.text_input.insert("1.0", "这家酒店环境非常好，服务也很周到，下次还会再来！")

        # ─── 按钮区 ───
        btn_frame = tk.Frame(self, bg=COLORS["bg"])
        btn_frame.pack(pady=12)

        self.btn_predict = tk.Button(
            btn_frame, text="分  析", font=("微软雅黑", 12, "bold"),
            bg=COLORS["primary"], fg="white",
            activebackground=COLORS["primary_hover"], activeforeground="white",
            bd=0, padx=30, pady=6, cursor="hand2",
            command=self._predict,
        )
        self.btn_predict.pack()

        # 按钮悬停效果
        self.btn_predict.bind("<Enter>",
            lambda e: self.btn_predict.config(bg=COLORS["primary_hover"]))
        self.btn_predict.bind("<Leave>",
            lambda e: self.btn_predict.config(bg=COLORS["primary"]))

        # ─── 结果区 ───
        result_frame = tk.LabelFrame(
            self, text="  分析结果  ", font=("微软雅黑", 10),
            bg=COLORS["card"], fg=COLORS["text"],
            bd=1, relief="solid", padx=16, pady=12,
        )
        result_frame.pack(fill="x", padx=30)

        self.label_result = tk.Label(
            result_frame, text="—",
            font=("微软雅黑", 16, "bold"), fg=COLORS["text"],
            bg=COLORS["card"],
        )
        self.label_result.pack(pady=(0, 8))

        # 置信度
        conf_label_frame = tk.Frame(result_frame, bg=COLORS["card"])
        conf_label_frame.pack(fill="x")
        tk.Label(
            conf_label_frame, text="置信度",
            font=("微软雅黑", 9), fg=COLORS["text_light"], bg=COLORS["card"],
        ).pack(side="left")

        # 进度条样式
        style = ttk.Style()
        style.theme_use("default")
        style.configure(
            "Custom.Horizontal.TProgressbar",
            troughcolor=COLORS["border"],
            background=COLORS["primary"],
            thickness=16,
        )

        self.progress = ttk.Progressbar(
            result_frame, length=440, maximum=100,
            style="Custom.Horizontal.TProgressbar",
        )
        self.progress.pack(fill="x", pady=(4, 2))

        self.label_conf = tk.Label(
            result_frame, text="",
            font=("微软雅黑", 9), fg=COLORS["text_light"], bg=COLORS["card"],
        )
        self.label_conf.pack()

        # ─── 状态栏 ───
        self.label_status = tk.Label(
            self, text="⏳ 模型加载中...",
            font=("微软雅黑", 9), fg=COLORS["text_light"], bg=COLORS["bg"],
        )
        self.label_status.pack(side="bottom", pady=8)

    def _load_model_async(self):
        """异步加载模型（避免 UI 卡顿）"""
        method = self.current_method.get()
        self.label_status.config(text=f"⏳ 正在加载 {method} 模型...", fg="orange")
        self.btn_predict.config(state="disabled")

        def _load():
            try:
                model_dir = os.path.join(MODELS_DIR, method)
                self.predictor = SentimentPredictor(model_dir, BASE_MODEL, method)
                self.after(0, lambda m=method: self._on_model_loaded(m))
            except Exception as e:
                err_msg = str(e)
                self.after(0, lambda msg=err_msg: self._on_model_error(msg))

        thread = threading.Thread(target=_load, daemon=True)
        thread.start()

    def _on_model_loaded(self, method: str):
        """模型加载成功回调"""
        self.label_status.config(
            text=f"✅ 模型已就绪（{method}）",
            fg=COLORS["positive"],
        )
        self.btn_predict.config(state="normal")

    def _on_model_error(self, error_msg: str):
        """模型加载失败回调"""
        self.label_status.config(
            text=f"❌ 模型加载失败",
            fg=COLORS["negative"],
        )
        self.btn_predict.config(state="disabled")
        self.predictor = None
        messagebox.showerror(
            "模型加载失败",
            f"无法加载模型，请检查 {MODELS_DIR} 目录：\n\n{error_msg}",
        )

    def _on_model_change(self, event=None):
        """模型切换回调（热切换）"""
        self.predictor = None
        self.label_result.config(text="—", fg=COLORS["text"])
        self.progress["value"] = 0
        self.label_conf.config(text="")
        self._load_model_async()

    def _predict(self):
        """执行预测"""
        if not self.predictor:
            messagebox.showerror("错误", "模型未加载，请检查 ./models 目录")
            return

        text = self.text_input.get("1.0", "end").strip()
        if not text:
            messagebox.showwarning("提示", "请先输入文本")
            return

        try:
            result = self.predictor.predict(text)
        except Exception as e:
            messagebox.showerror("预测错误", f"推理过程出错：\n{e}")
            return

        # 根据情感设置颜色
        is_positive = "正面" in result["label"]
        color = COLORS["positive"] if is_positive else COLORS["negative"]

        self.label_result.config(text=result["label"], fg=color)

        # 更新进度条颜色
        style = ttk.Style()
        style.configure(
            "Custom.Horizontal.TProgressbar",
            background=color,
        )

        conf_pct = result["confidence"] * 100
        self.progress["value"] = conf_pct
        self.label_conf.config(
            text=f"{conf_pct:.1f}%  （负面 {result['probs'][0]*100:.1f}% / "
                 f"正面 {result['probs'][1]*100:.1f}%）"
        )


def main():
    """启动 GUI 应用"""
    app = SentimentApp()
    app.mainloop()


if __name__ == "__main__":
    main()
