from abc import ABC, abstractmethod
import os
import numpy as np
import json
from typing import Dict, Any
from .base_model import BaseModel
from .base_dataset import BaseDataset

class BaseTask(ABC):
    def __init__(self, task_name: str, metrics: list, output_root: str = "results"):
        self.task_name = task_name
        self.metrics = metrics
        self.output_root = output_root
        os.makedirs(output_root, exist_ok=True)
        
    @abstractmethod
    def evaluate(self, model: BaseModel, dataset: BaseDataset, **kwargs) -> Dict[str, Any]:
        """评估模型在指定数据集上的性能，返回指标字典"""
        pass
  
    def save_results(self, model_name: str, dataset_name: str, 
                    metrics: Dict[str, Any], predictions: Dict[str, Any]):
        """保存结果到结构化目录"""
        
        # 构建任务/模型/数据集/的路径
        os.makedirs(self.output_root, exist_ok=True)
        task_dir = os.path.join(self.output_root, self.task_name)
        os.makedirs(task_dir, exist_ok=True)
        model_dir = os.path.join(task_dir, model_name)
        os.makedirs(model_dir, exist_ok=True)
        dataset_dir = os.path.join(model_dir, dataset_name)
        os.makedirs(dataset_dir, exist_ok=True)
        
        metrics_path = os.path.join(dataset_dir, 'metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=4)
            
        pred_path = os.path.join(dataset_dir, 'predictions.json')
        with open(pred_path, 'w') as f:
            json.dump(predictions, f, indent=4)

        print(f"Results saved to: {dataset_dir}")