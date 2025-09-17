import os
import torch
from typing import Any, List, Optional
import pandas as pd
from core.base_dataset import BaseDataset


class Camelyon16(BaseDataset):

    def __init__(
        self,
        data_root: str,
        method: str,
        supported_tasks: Optional[List[str]] = None,
        **kwargs: Any,
    ):
        super().__init__(data_root, **kwargs)
        self.dataset_name = "Camelyon16"
        self.method = method
        self.slide_dir = os.path.join(self.slide_base_dir, self.dataset_name)
        self.slides = os.listdir(self.slide_dir)
        if supported_tasks is not None:
            self._supported_tasks = supported_tasks
        # 打开数据集有的所有标签类型，假设所有标签都是两列：slide_name label
        self.label_classification = pd.read_csv(
            os.path.join(self.label_base_dir, self.dataset_name, "classification.csv"))

        self.label_report_generation = pd.read_csv(
            os.path.join(self.label_base_dir, self.dataset_name, "report_generation.csv"))

    def __len__(self):
        return len(self.slides)

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self.slides):
            raise IndexError("Index out of range")
        slide_name = self.slides[idx]  # 包含文件后缀名-->文件格式
        slide = os.path.join(self.slide_dir, slide_name)
        feature = os.path.join(self.processed_base_dir, self.dataset_name, self.method, slide_name + ".pt")  # 特征文件假设为pt文件
        # 校验特征文件是否存在
        if os.path.exists(feature):
            embedding = torch.load(feature, map_location="cpu")
        else:
            # 触发预处理流程生成文件(用slide)
            embedding = None
        row_class = self.label_classification[self.label_classification["slide_name"] == slide_name]
        row_report = self.label_report_generation[self.label_report_generation["slide_name"] == slide_name]
        classification_label = row_class.values
        report_generation_label = row_report.values
        slide_info = {
            "slide_name": slide_name,
            "slide_path": slide,
            "classification_label": classification_label,
            "report_generation_label": report_generation_label
        }
        return {
            "embedding": embedding,
            "slide_info": slide_info
        }


if __name__ == "__main__":
    Camelyon16(data_root="/benchmark/", method="PRISM")