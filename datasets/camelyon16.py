import os
import sys
import torch
from typing import Any, List, Optional
import glob
from torch.utils.data import Dataset
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)
from core.base_dataset import BaseDataset


class Camelyon16(BaseDataset):
    """
    Camelyon16数据集类，继承自BaseDataset。
    supported_tasks: 支持的下游任务列表，例如["classification", "segmentation"]等。
    check_feature: 是否在初始化时检查预处理特征文件的存在性。
    """
    def __init__(
            self,
            model: str,
            supported_tasks = ["classification"],
            check_feature: bool = True,
            **kwargs: Any,
    ):
        self.dataset_name = "Camelyon16"
        self.model = model
        self.supported_tasks = supported_tasks
        self.check_feature = check_feature
        super().__init__(**kwargs)
        self.slides = glob.glob(os.path.join(self.slide_dir, "*", "*", "*.tif"))

        # 记载数据集有的所有下游任务的标签，假设所有标签都是两列：| slide_name | , | label |
        # 这样加载就可以动态的根据supported_tasks加载不同数量的label

        if self.check_feature:
            self._build()

    
            

if __name__ == "__main__":
    models = ["UNI_V1", "CONCH_V1", "CTransPath", "Virchow_V1", "ProvGigaPath"]
    dataset = Camelyon16("CONCH_V15", check_feature=True, label_load=False)
    for sample in dataset:
        print(sample)