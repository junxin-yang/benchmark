from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple
import os
from torch.utils.data import Dataset


class BaseDataset(Dataset, ABC):
    """
    所有数据集的抽象基类。
    定义数据集的统一接口，确保不同的WSI数据集可以以一致的方式被加载和评估。
    data_root: 数据集的存储根路径，对应数据的存储路径是  root/dataset name/slide/
    processed_dir: 存储数据集的多个encoder的特征文件  root/dataset name/processed_dir/encoder name/
    label_dir: 存储数据集的多种标签（可能只有一种标签） root/dataset name/label/label name/
    """

    def __init__(self, data_root: str, **kwargs: Any):
        super().__init__()
        self.data_root = data_root
        self.slide_base_dir = os.path.join(self.data_root, "slides")
        self.processed_base_dir = os.path.join(self.data_root, "preprocessed")
        self.label_base_dir = os.path.join(self.data_root, "label")

    @abstractmethod
    def __getitem__(self, index: int) -> Dict[str, Any]:

        pass

    @abstractmethod
    def __len__(self) -> int:

        return len(self.data_list)