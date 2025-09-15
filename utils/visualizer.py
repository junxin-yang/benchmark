from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import ConfusionMatrixDisplay
from matplotlib import pyplot as plt
import os
import json
import numpy as np

def plot_roc_curve(self, labels, probs):
    # 实现ROC曲线绘制
    fig, ax = plt.subplots()
    RocCurveDisplay.from_predictions(labels, probs, ax=ax)
    ax.set_title('ROC Curve')
    return fig
    
def plot_confusion_matrix(self, labels, preds):
    fig, ax = plt.subplots()
    ConfusionMatrixDisplay.from_predictions(labels, preds, ax=ax)
    ax.set_title('Confusion Matrix')
    return fig

def plot_bar(models: list, datasets: list, task_name: str, metric: str, result_dir: str, fig_dir: str):
    # 绘制柱状图比较不同模型在多个数据集上的性能
    fig_dir = os.path.join(fig_dir, task_name)
    os.makedirs(fig_dir, exist_ok=True)
    metric_values = {}

    for dataset in datasets:
        metric_values[dataset] = []
        for model in models:
            task_dir = os.path.join(result_dir, task_name)
            model_dir = os.path.join(task_dir, model)
            dataset_dir = os.path.join(model_dir, dataset)
            metrics_path = os.path.join(dataset_dir, 'metrics.json')
            if os.path.exists(metrics_path):
                with open(metrics_path, 'r') as f:
                    metrics = json.load(f)
                    metric_value = metrics.get(metric, 0)
                    metric_values[dataset].append(metric_value)
            else:
                metric_values[dataset].append(0)

    # 绘制柱状图
    x = np.arange(len(models))
    width = 0.6

    fig, ax = plt.subplots()
    ax.bar(x, metric_values, width, color='skyblue')
    ax.set_xlabel('Models')
    ax.set_ylabel(metric)
    ax.set_xticks(x)
    ax.set_xticklabels(models)

    fig.tight_layout()
    fig_path = os.path.join(fig_dir, f"{dataset}_{metric}_bar.png")
    fig.savefig(fig_path)
    plt.close(fig)