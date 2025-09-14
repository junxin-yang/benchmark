import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
import matplotlib.pyplot as plt
from core.base_task import BaseTask
from core.base_model import BaseModel
from core.base_dataset import BaseDataset

class WSIClassificationTask(BaseTask):
    def __init__(self, **kwargs):
        super().__init__(task_name="classification", **kwargs)
        
    def evaluate(self, model: BaseModel, dataset: BaseDataset, **kwargs):
        all_preds = []
        all_labels = []
        
        # 遍历数据集
        for wsi_path, label in dataset:
            pred = model.predict(wsi_path)
            all_preds.append(pred)
            all_labels.append(label)
        
        # 计算指标
        accuracy = accuracy_score(all_labels, np.round(all_preds))
        auc = roc_auc_score(all_labels, all_preds)
        
        metrics = {
            'accuracy': accuracy,
            'auc': auc
        }
        
        return metrics, all_preds
    
