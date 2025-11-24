from abc import ABC, abstractmethod
from transformers import AutoModel
import os
import json
import socket
import yaml
import torch
from typing import Optional
from tqdm import tqdm
# from utils.logger import default_logger as logger

# 获取项目根路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

config_path = os.path.join(project_root, "configs", "models.yaml")

def get_weights_path(model_type, encoder_name):
    """
    Retrieve the path to the weights file for a given model name.
    This function looks up the path to the weights file in a local checkpoint
    registry (local_ckpts.json). If the path in the registry is absolute, it
    returns that path. If the path is relative, it joins the relative path with
    the provided weights_root directory.
    Args:
        weights_root (str): The root directory where weights files are stored.
        name (str): The name of the model whose weights path is to be retrieved.
    Returns:
        str: The absolute path to the weights file.
    """

    assert model_type in ['patch', 'slide', 'seg'], f"Encoder type must be 'patch' or 'slide' or 'seg', not '{model_type}'"

    if model_type == 'patch' or model_type == 'slide':
        root = os.path.join(os.path.dirname(__file__), f"{model_type}_models")
    else:
        root = os.path.join(os.path.dirname(__file__), "segmentation_models")

    registry_path = os.path.join(root, "local_ckpts.json")
    with open(registry_path, "r") as f:
        registry = json.load(f)

    path = registry.get(encoder_name)    
    if path:
        path = path if os.path.isabs(path) else os.path.abspath(os.path.join(root, 'model_zoo', path)) # Make path absolute
        if not os.path.exists(path):
            path = ""

    return path

def has_internet_connection(timeout=3.0) -> bool:
    endpoint = os.environ.get("HF_ENDPOINT", "huggingface.co")
    
    if endpoint.startswith(("http://", "https://")):
        from urllib.parse import urlparse
        endpoint = urlparse(endpoint).netloc
    
    try:
        # Fast socket-level check
        socket.create_connection((endpoint, 443), timeout=timeout)
        return True
    except OSError:
        pass

    try:
        # Fallback HTTP-level check (if requests is available)
        import requests
        url = f"https://{endpoint}" if not endpoint.startswith(("http://", "https://")) else endpoint
        r = requests.head(url, timeout=timeout)
        return r.status_code < 500
    except Exception:
        return False
    

class BasePatchModel(torch.nn.Module):
    _has_internet = has_internet_connection()
    
    def __init__(self, weights_path: Optional[str] = None, **build_kwargs):
        """
        Initialize BasePatchEncoder.

        Args:
            weights_path (Optional[str]): 
                Optional path to local model weights. If None, the model is loaded from the model registry or downloaded from Hugging Face Hub.
            **build_kwargs: 
                Additional keyword arguments passed to the `_build()` method to customize model creation.

        Attributes:
            enc_name (Optional[str]): Name of the encoder architecture (set during `_build()`).
            weights_path (Optional[str]): Path to local model weights (if provided).
            model (nn.Module): The instantiated encoder model.
            eval_transforms (Callable): Evaluation-time preprocessing transforms.
            precision (torch.dtype): Precision used for inference.
        """

        super().__init__()
        self.weights_path: Optional[str] = weights_path
        self.model_configs = self.load_config(config_path, self.enc_name)
        self.model, self.eval_transforms, self.precision = self._build(**build_kwargs)
        

    def ensure_valid_weights_path(self, weights_path):
        if weights_path and not os.path.isfile(weights_path):
            raise FileNotFoundError(f"Expected checkpoint at '{weights_path}', but the file was not found.")
    
    def ensure_has_internet(self, enc_name):
        if not BasePatchModel._has_internet:
            raise FileNotFoundError(
                f"Internet connection does seem not available. Auto checkpoint download is disabled."
                f"To proceed, please manually download: {enc_name},\n"
                f"and place it in the model registry in:\n`trident/patch_encoder_models/local_ckpts.json`"
            )
        
    def _get_weights_path(self):
        """
        If self.weights_path is provided, use it. 
        If not provided, check the model registry. 
            If path in model registry is empty, auto-download from huggingface
            else, use the path from the registry.
        """
        if self.weights_path:
            self.ensure_valid_weights_path(self.weights_path)
            return self.weights_path
        else:
            weights_path = get_weights_path('patch', self.enc_name)
            self.ensure_valid_weights_path(weights_path)
            return weights_path

    def forward(self, x):
        """
        Can be overwritten if model requires special forward pass.
        """
        x = x.to(self.device, dtype=self.precision)
        z = self.model(x)
        return z
    
    def load_config(self, file_path, section, defaults={"patch_model_path": "", "device": "cpu"}):
        """加载指定 section 的配置，并返回 dict，使用 defaults 补全缺失值"""
        defaults = defaults or {}
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Config file not found: {file_path}")
        
        with open(file_path, "r") as f:
            config = yaml.safe_load(f) or {}
        section_cfg = config.get(section, {})
        return section_cfg
    

    def encode_patches(self, loader, device):
        features = torch.Tensor().to(device)
        with torch.no_grad():
            for _, input in tqdm(enumerate(loader), total=len(loader)):
                input = input.to(device)
                feature = self.forward(input)
                features = torch.cat((features, feature), dim=0)
        return features.cpu()
        
    @abstractmethod
    def _build(self, **build_kwargs):
        pass


class BaseSlideModel(torch.nn.Module):
    
    def __init__(self, freeze: bool = True, **build_kwargs: dict) -> None:
        """
        Parent class for all pretrained slide encoders.
        """
        super().__init__()
        self.enc_name = None
        self.model, self.precision, self.embedding_dim = self._build(**build_kwargs)

        # Set all parameters to be non-trainable
        if freeze and self.model is not None:
            for param in self.model.parameters():
                param.requires_grad = False
            self.model.eval()
        
    def forward(self, batch):
        """
        Can be overwritten if model requires special forward pass.
        """
        batch = batch.to(next(self.model.parameters()).device)  # 保证输入与权重同设备
        z = self.model(batch)
        return z

    def load_config(self, file_path, section, defaults={"slide_model_path": "", "device": "cpu"}):
        """加载指定 section 的配置，并返回 dict，使用 defaults 补全缺失值"""
        defaults = defaults or {}
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Config file not found: {file_path}")
        
        with open(file_path, "r") as f:
            config = yaml.safe_load(f) or {}
        section_cfg = config.get(section, {})
        return section_cfg
    
    def encode_slide(self, loader, device):
        pass
      
    @abstractmethod
    def _build(self, **build_kwargs):
        """
        Initialization method, must be defined in child class.
        """
        pass