from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple
import os
from torch.utils.data import Dataset

class BaseDataset(Dataset, ABC):
    """
    所有数据集的抽象基类。
    定义数据集的统一接口，确保不同的WSI数据集可以以一致的方式被加载和评估。
    """
    
    def __init__(self, data_root: str, **kwargs: Any):

        super().__init__()
        self.data_root = data_root
        self.data_list = []  # 用于存储样本信息（例如，WSI路径和标签的元组）
        self._load_data_list(**kwargs)  # 调用方法来加载数据列表

    @abstractmethod
    def _load_data_list(self, **kwargs: Any) -> None:

        pass

    @abstractmethod
    def __getitem__(self, index: int) -> Dict[str, Any]:

        pass

    def __len__(self) -> int:

        return len(self.data_list)