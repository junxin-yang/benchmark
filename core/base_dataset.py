from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple, Optional
import os
import importlib
from torch.utils.data import Dataset
import sys
import h5py
import torchvision.transforms as transforms
import torch
import yaml
from tqdm import tqdm

######################## 加载配置文件 ###########################
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)
with open(os.path.join(PROJECT_ROOT, 'configs/datasets.yaml'), 'r') as f:
            dataset_configs = yaml.safe_load(f)
with open(os.path.join(PROJECT_ROOT, 'configs/models.yaml'), 'r') as f:
            model_configs = yaml.safe_load(f)
################################################################

class BaseDataset(Dataset, ABC):
    """
    所有数据集的抽象基类。
    定义数据集的统一接口，确保不同的WSI数据集可以以一致的方式被加载和评估。
    data_root: 数据集的存储根路径，对应数据的存储路径是  root/dataset name/slide/
    processed_dir: 存储数据集的多个encoder的特征文件  root/dataset name/processed_dir/encoder name/
    label_dir: 存储数据集的多种标签（可能只有一种标签） root/dataset name/label/label name/
    """

    def __init__(
            self: str,
            supported_tasks: Optional[List[str]] = None,
            dataset_configs: Optional[dict] = dataset_configs,
            model_configs: Optional[dict] = model_configs,
            **kwargs: Any
        ):
        super().__init__()
        # 导入参数
        self.dataset_configs = dataset_configs
        self.model_configs = model_configs
        self.supported_tasks = supported_tasks

        # 读取dataset_configs和model_configs中的相关配置
        self.data_root = os.path.join(self.dataset_configs[self.dataset_name]['data_root'], self.dataset_name)
        self.slide_dir = self.dataset_configs[self.dataset_name]['slide_file_path']
        self.encoder_module = self.model_configs[self.model]['patch_encoder_module']
        self.encoder_class = self.model_configs[self.model]['patch_encoder_class']
        self.encode_batch_size = self.model_configs[self.model].get('encode_batch_size', 256)
        self.encode_workers_num = self.model_configs[self.model].get('encode_workers_num', 4)
        self.device = self.model_configs[self.model].get('device', 'cuda')

        # 构建数据集路径
        self.slide_base_dir = os.path.join(self.data_root, "slides")
        self.processed_base_dir = os.path.join(self.data_root, "preprocessed")
        self.label_base_dir = os.path.join(self.data_root, "label")

    @abstractmethod
    def __getitem__(self, index: int) -> Dict[str, Any]:

        pass

    @abstractmethod
    def __len__(self) -> int:

        return len(self.data_list)

    
    def _build(self):
        self.feature_extract()
    
    def _instantiate_patch_encoder(self, **kwargs) -> Tuple[object, str]:
        encoder_module = str(self.encoder_module)
        encoder_class = str(self.encoder_class)
        full_module = f"models.patch_models.{encoder_module}"
        try:
            module = importlib.import_module(full_module)
        except Exception as e:
            raise ImportError(f"无法导入模块 {full_module}: {e}")
        
        if not hasattr(module, encoder_class):
            raise ImportError(f"模块 {full_module} 中不存在类 {encoder_class}")
        
        EncoderClass = getattr(module, encoder_class)
        # 传入必要参数（如果有），例如 device / model_path 等；kwargs 可来自 model_configs 或外部传入
        return EncoderClass(**kwargs), encoder_class
    

    def feature_extract(self, overwrite: bool = False):
        """
        遍历 self.slides，对每张玻片执行特征提取并保存为 .pt 文件。
        如果已经存在目标 .pt 文件且 overwrite 为 False 则跳过。
        """
        out_dir = os.path.join(self.processed_base_dir, self.encoder_module)
        os.makedirs(out_dir, exist_ok=True)

        try:
            encoder, cls_name = self._instantiate_patch_encoder()
        except Exception as e:
            print(f"[feature_extract] 无法实例化 encoder {self.encoder_module}: {e}")
            return

        pbar = tqdm(self.slides)
        for idx, slide_path in enumerate(pbar):
            try:
                # 支持 glob 返回的完整路径或仅文件名两种情况
                slide_basename = os.path.basename(slide_path)
                pbar.set_description(f"⛏️  ==> [{idx + 1}/{len(self.slides)}] Processing slide: {slide_basename}")
                slide_type = os.path.splitext(slide_basename)[-1].lstrip('.')
                slide_id = os.path.splitext(slide_basename)[0]  # 不带后缀的文件名，用作输出名
                h5_path = os.path.join(self.processed_base_dir, "patching", slide_type, "patches", f"{slide_id}.h5")
                grid = h5py.File(h5_path, 'r')['coords'][:]
                out_path = os.path.join(out_dir, slide_basename + ".pt")
                trans = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
                ])

                if os.path.exists(out_path) and not overwrite:
                    pbar.set_description(f"⚠️  ==> Feature file {out_path} already exists. Skipping...")
                    # 已存在则跳过
                    continue
                
                from core.wsi_dataset import SingleWSIDataset
                wsi_dataset = SingleWSIDataset(grid=grid, slide_path=slide_path, transform=trans)
                loader = torch.utils.data.DataLoader(
                    wsi_dataset,
                    batch_size=self.encode_batch_size,
                    shuffle=False,
                    num_workers=self.encode_workers_num,
                    pin_memory=True
                )

                # 调用encoder的类的提特征的方法对loader进行特征提取
                embedding = encoder.encode_patches(loader, self.device if torch.cuda.is_available() else 'cpu')
                wsi_dataset.close_slide()

                # --- 关键：关闭 DataLoader worker ---
                if hasattr(loader, '_iterator') and loader._iterator is not None:
                    loader._iterator._shutdown_workers()
                del loader
                
                pbar.set_description(f'👍 ==> Feature Extract Done!')

                torch.save(embedding, out_path)
                pbar.set_description(f'🥰 ==> Saved feature to {out_path}')
            except Exception as e:
                log_path = os.path.join(out_dir, "feature_extract_errors.log")
                error_msg = f"{slide_basename}: {e}\n"
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(error_msg)
                print(f"[feature_extract] Error processing slide {slide_path}: {e}")