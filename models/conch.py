from core.base_model import BaseModel
import torch.nn as nn
import torch

class CONCH(BaseModel): # 继承自BaseModel
    def __init__(self, model_name: str, **kwargs):
        super().__init__(model_name=model_name, **kwargs)
        self.model = nn.Sequential(
            # 模型层定义 (例如特征提取器、分类头等)
        )

    def load_weights(self, weight_path: str):
        """加载预训练权重"""
        state_dict = torch.load(weight_path)
        self.model.load_state_dict(state_dict)

    def predict(self, wsi_path: str, **kwargs):
        """对单个WSI进行预测"""
        output = self.model(...)
        return output