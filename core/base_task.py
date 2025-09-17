from abc import ABC, abstractmethod
import os
import numpy as np
import json
from typing import Dict, Any
from .base_model import BaseModel
from .base_dataset import BaseDataset
from utils.logger import default_logger as logger

class BaseTask(ABC):
    def __init__(self, task_name: str, metrics: list, output_root: str = "results"):
        self.task_name = task_name
        self.metrics = metrics
        self.output_root = output_root
        os.makedirs(output_root, exist_ok=True)
        
    @abstractmethod
    def evaluate(self, model: BaseModel, dataset: BaseDataset, **kwargs) -> Dict[str, Any]:
        pass
  
    def save_results(self, model_name: str, dataset_name: str, 
                    metrics: Dict[str, Any], predictions: Dict[str, Any]):
        
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

        logger.info(f"Results saved to: {dataset_dir}")