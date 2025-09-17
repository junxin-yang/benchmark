import os
import torch
from typing import Any, List, Optional
import pandas as pd
import yaml

from core.base_dataset import BaseDataset


class Dataset(BaseDataset):

    def __init__(
            self,
            data_root: str,
            dataset_name: str,
            method: str,
            supported_tasks: Optional[List[str]] = None,
            **kwargs: Any,
    ):
        super().__init__(data_root, **kwargs)
        self.dataset_name = dataset_name
        self.method = method
        self.slide_dir = os.path.join(self.slide_base_dir, self.dataset_name)
        self.slides = os.listdir(self.slide_dir)
        with open('configs/model.yaml', 'r') as f:
            method_configs = yaml.safe_load(f)
        self.encoder = method_configs[self.method]['patch_encoder']
        self.supported_tasks = supported_tasks

        if supported_tasks is not None:
            self._supported_tasks = supported_tasks

        # 记载数据集有的所有下游任务的标签，假设所有标签都是两列：| slide_name | , | label |
        # 这样加载就可以动态的根据supported_tasks加载不同数量的label
        self.total_labels = {}
        for task in supported_tasks:
            self.total_labels[task] = pd.read_csv(
                os.path.join(self.label_base_dir, self.dataset_name, f"{task}.csv"))

    def __len__(self):
        return len(self.slides)

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self.slides):
            raise IndexError("Index out of range")
        slide_name = self.slides[idx]  # 包含文件后缀名-->文件格式
        slide = os.path.join(self.slide_dir, slide_name)
        feature = os.path.join(self.processed_base_dir, self.dataset_name, self.encoder,
                               slide_name + ".pt")  # 特征文件假设为pt文件
        # 校验特征文件是否存在
        if os.path.exists(feature):
            embedding = torch.load(feature, map_location="cpu")
        else:
            # 触发预处理流程生成文件(用slide)
            embedding = None

        slide_info = {
            "slide_name": slide_name,
            "slide_path": slide,
        }

        # 对单张玻片迭代每个下游任务去取对应任务csv文件中的玻片标签
        for task_name, label_csv in self.total_labels.items():
            slide_info[task_name] = label_csv.loc[label_csv["slide_name"] == slide_name, "label"].iloc[0]

        return {
            "embedding": embedding,
            "slide_info": slide_info
        }


if __name__ == "__main__":
    print("hello")
    # Camelyon16(data_root="/benchmark/", method="PRISM")