import os
import sys
import torch
from typing import Any, List, Optional
import pandas as pd
import yaml
import glob
from torch.utils.data import Dataset
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)
from core.base_dataset import BaseDataset


class TCGA_LUSC(BaseDataset):

    def __init__(
            self,
            model: str,
            supported_tasks = ["classification"],
            check_feature: bool = True,
            **kwargs: Any,
    ):
        self.dataset_name = "TCGA-LUSC"
        self.model = model
        self.supported_tasks = supported_tasks
        self.check_feature = check_feature
        super().__init__(**kwargs)
        self.slides = glob.glob(os.path.join(self.slide_dir, "*", "*.svs"))

        # 记载数据集有的所有下游任务的标签，假设所有标签都是两列：| slide_name | , | label |
        # 这样加载就可以动态的根据supported_tasks加载不同数量的label
        if self.check_feature:
            self._build()

    
            

if __name__ == "__main__":
    models = ["UNI_V1", "CONCH_V1", "CTransPath", "Virchow_V1", "ProvGigaPath"]
    dataset = TCGA_LUSC("CONCH_V15", label_load=False)
    # for sample in dataset:
    #     print(sample)