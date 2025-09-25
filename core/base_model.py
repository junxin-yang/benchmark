from abc import ABC, abstractmethod
from transformers import AutoModel
import os
from utils.logger import default_logger as logger

class BaseModel(ABC):
    def __init__(self, model_path: str, device: str, model_name: str):
        self.model_name = model_name
        self.model_path = model_path
        self.device = device
        if self.model_path is None or not os.path.exists(self.model_path):
            logger.info(f"⚠️ Warning: Model path '{self.model_path}' is None or does not exist. Model will not be loaded.")
            self.model = None
            # raise ValueError(f"Model path {self.model_path} does not exist.")
        else:
            self.model = AutoModel.from_pretrained(
                model_path,
                trust_remote_code=True
            ).to(self.device)
        logger.info(f"🚀 Successfully loaded {self.model_name}")
        
    @abstractmethod
    def report_generate(self, feature):
        """Generate pathology report (if supported by the model)"""
        pass

    @abstractmethod
    def classify(self, feature, num_classes):
        """Pathology classification. Returns predicted class or probabilities."""
        pass

    @abstractmethod
    def survival_predict(self, feature, time_horizon=None):
        """
        Survival prediction.
        Args:
            feature: input features for prediction
            time_horizon: optional, predict survival at a specific time point
        Returns:
            Survival probability or risk score
        """
        pass