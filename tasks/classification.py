import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
import matplotlib.pyplot as plt
from core.base_task import BaseTask
from core.base_model import BaseModel
from core.base_dataset import BaseDataset
from torch.utils.data import DataLoader

class ClassificationTask(BaseTask):
    def __init__(self, metrics: list, output_root="results"):
        super().__init__(task_name="classification", output_root=output_root)
        self.metrics = metrics

    def evaluate(self, model: BaseModel, dataset: BaseDataset, **kwargs):
        all_preds = []
        all_labels = []
        metric_results = {}

        # 遍历数据集
        for feature, label in dataset:
            pred = model.classify(feature, kwargs["num_classes"])
            all_preds.append(pred)
            all_labels.append(label)
        
        # 计算指标
        for metric_fn in self.metrics:
            metric_name = metric_fn.__name__
            metric_results[metric_name] = metric_fn(all_labels, all_preds)

        return metric_results, all_preds
