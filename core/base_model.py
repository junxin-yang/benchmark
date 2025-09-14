from abc import ABC, abstractmethod

class BaseModel(ABC):
    def __init__(self, model_name: str, **kwargs):
        self.model_name = model_name
        
    @abstractmethod
    def load_weights(self, weight_path: str):
        """加载预训练权重"""
        pass
        
    @abstractmethod
    def predict(self, wsi_path: str, **kwargs):
        """对单个WSI进行预测"""
        pass
        
    @abstractmethod
    def generate_report(self, wsi_path: str, **kwargs):
        """生成病理报告（如果模型支持）"""
        pass