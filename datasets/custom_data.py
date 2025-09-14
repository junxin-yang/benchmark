from core.base_dataset import BaseDataset

class CustomDataset(BaseDataset):
    def __init__(self, data_root: str, **kwargs):
        super().__init__(data_root=data_root, **kwargs)
    
    def load_data(self):
        # 实现加载自定义数据集的逻辑
        pass

    def __getitem__(self, index: int):
        # 实现获取单个样本的逻辑
        pass