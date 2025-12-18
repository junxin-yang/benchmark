from abc import ABC, abstractmethod
import os
import sys
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)
import numpy as np
from typing import Dict, Any
from core.base_dataset import BaseDataset
from utils.logger import default_logger as logger

import yaml


######################## 加载配置文件 ###########################

with open(os.path.join(PROJECT_ROOT, 'configs/datasets.yaml'), 'r') as f:
            dataset_configs = yaml.safe_load(f)
with open(os.path.join(PROJECT_ROOT, 'configs/models.yaml'), 'r') as f:
            model_configs = yaml.safe_load(f)
################################################################

class BaseTask(ABC):
    """
    BaseTask 的 Docstring
    BaseTask 是所有任务的基类，提供了任务的基本框架和方法。
    Attributes:
        dataset (BaseDataset): 任务使用的数据集。
        total_splited_dataset (Dict[str, Any]): 分割后的数据集，包括训练、验证和测试集。
        embed_dim (int): 输入嵌入维度。
        dataset_name (str): 数据集名称。
        dataset_config (Dict[str, Any]): 数据集配置。
        model (str): 使用的模型名称。
        result_dir (str): 结果保存目录。
    """
    def __init__(self, dataset: BaseDataset):
        self.dataset = dataset
        self.total_splited_dataset = dataset.split_dataset()
        self.embed_dim = self.total_splited_dataset["train"][0]['embedding'].shape[1]
        self.dataset_name = dataset.dataset_name
        self.dataset_config = dataset_configs[self.dataset_name]
        self.device = dataset.device
        print(f"🎉  ==> Dataset {self.dataset_name} has been split into train, validate, and test sets.")
        self.model = dataset.model
        self.model_config = model_configs[self.model]
        print(f"🎄  ==> Model {self.model} is being used for the task on device {self.device}.")
        if self.model_config.get("slide_encoder_module") is not None:
            self.model_type = "slide_model"
            self.result_dir = os.path.join(self.dataset.data_root, "results", self.dataset.slide_encoder_class, self.task_name)
        else:
            self.model_type = "patch_model"
            self.result_dir = os.path.join(self.dataset.data_root, "results", self.dataset.encoder_class, self.task_name)
        os.makedirs(self.result_dir, exist_ok=True)
        
  
    def save_results(self, model_name: str, dataset_name: str, 
                    metrics: Dict[str, Any], predictions: Dict[str, Any]):
        
        pass