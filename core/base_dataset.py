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
import pandas as pd
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
            dataset_configs: Optional[dict] = dataset_configs,
            model_configs: Optional[dict] = model_configs,
            label_load: bool = True,
            **kwargs: Any
        ):
        super().__init__()
        # 导入参数
        self.label_load = label_load
        self.dataset_configs = dataset_configs
        self.model_configs = model_configs
        

        # 读取dataset_configs和model_configs中的相关配置
        self.data_root = os.path.join(self.dataset_configs[self.dataset_name]['data_root'], self.dataset_name)
        self.slide_dir = self.dataset_configs[self.dataset_name]['slide_file_path']
        self.encoder_module = self.model_configs[self.model]['patch_encoder_module']
        self.encoder_class = self.model_configs[self.model]['patch_encoder_class']
        self.patch_size = self.model_configs[self.model].get("patch_size", 512)

        self.slide_encoder_module = self.model_configs[self.model].get("slide_encoder_module", None)
        self.slide_encoder_class = self.model_configs[self.model].get("slide_encoder_class", None)


        self.encode_batch_size = self.model_configs[self.model].get('encode_batch_size', 256)
        self.encode_workers_num = self.model_configs[self.model].get('encode_workers_num', 4)
        self.device = self.model_configs[self.model].get('device', 'cuda')

        # 构建数据集路径
        self.slide_base_dir = os.path.join(self.data_root, "slides")
        self.processed_base_dir = os.path.join(self.data_root, "preprocessed")
        self.label_base_dir = os.path.join(self.data_root, "label")

        # 加载标签文件
        if self.label_load:
            self.total_labels = {}
            self.get_label()

        print("\n")
        print("="*50)    
        print(f"🚩  ==> Loading features for model: {self.model} ")



    def __len__(self):
        return len(self.slides)



    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self.slides):
            raise IndexError("Index out of range")
        slide_name = os.path.basename(self.slides[idx])  # 包含文件后缀名-->文件格式
        slide_path = os.path.join(self.slide_dir, slide_name)
        if self.slide_encoder_module is not None:
            feature_path = os.path.join(self.processed_base_dir, self.slide_encoder_module,
                                        slide_name + ".pt")  # 特征文件假设为pt文件
        else:
            feature_path = os.path.join(self.processed_base_dir, self.encoder_module,
                                        slide_name + ".pt")  # 特征文件假设为pt文件
        # 校验特征文件是否存在
        if os.path.exists(feature_path):
            embedding = torch.load(feature_path, map_location="cpu")
        else:
            # 触发预处理流程生成文件(用slide)
            print(f"⚠️  ==> Feature file {feature_path} not found. Triggering feature extraction...")
            embedding = None

        slide_info = {
            "slide_name": slide_name,
            "slide_path": slide_path,
        }

        # 对单张玻片迭代每个下游任务去取对应任务csv文件中的玻片标签
        for task_name, label_csv in self.total_labels.items():
            slide_info[task_name] = label_csv.loc[label_csv["slide_name"] == slide_name, "label"].iloc[0]

        return {
            "embeddings": embedding,
            "slide_info": slide_info
        }

    
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
        agg_out_dir = os.path.join(self.processed_base_dir, self.slide_encoder_module or "")

        os.makedirs(out_dir, exist_ok=True)
        os.makedirs(agg_out_dir, exist_ok=True)


        try:
            encoder, cls_name = self._instantiate_patch_encoder()
            
        except Exception as e:
            print(f"[feature_extract] 无法实例化 encoder {self.encoder_module}: {e}")
            return
        if self.slide_encoder_module is not None:
            slide_encoder_module = __import__(f"models.slide_models.{self.slide_encoder_module}", fromlist=[self.slide_encoder_class])
            slide_encoder_class = getattr(slide_encoder_module, self.slide_encoder_class)
            aggregator = slide_encoder_class()
        

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
                agg_out_path = os.path.join(agg_out_dir, slide_basename + ".pt")

                trans = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
                ])

                if os.path.exists(out_path) and not overwrite and os.path.exists(agg_out_path):
                    pbar.set_description(f"⚠️  ==> Feature file {agg_out_path} already exists. Skipping...")
                    # 已存在则跳过
                    continue
                if os.path.exists(out_path) and not overwrite and self.slide_encoder_module is None:
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
                    pin_memory=False
                )

                # 调用encoder的类的提特征的方法对loader进行特征提取
                if not os.path.exists(out_path) and not overwrite:
                    embeddings = encoder.encode_patches(loader, self.device if torch.cuda.is_available() else 'cpu')
                else:
                    embeddings = torch.load(out_path, map_location=self.device if torch.cuda.is_available() else 'cpu')

                wsi_dataset.close_slide()

                # --- 关键：关闭 DataLoader worker ---
                if hasattr(loader, '_iterator') and loader._iterator is not None:
                    loader._iterator._shutdown_workers()
                del loader
                
                pbar.set_description(f'👍 ==> Feature Extract Done!')

                torch.save(embeddings, out_path)

                # 如果是slide-level模型，则进行 slide-level 特征聚合
                if self.model_configs[self.model].get("slide_encoder_module") is not None:
                    pbar.set_description(f"⚙️  ==> 使用 slide encoder {self.slide_encoder_module} 进行特征提取")
                    embeddings = embeddings.unsqueeze(0)  # 添加 batch 维度
                    aggregat_embedding = {
                        "embeddings": embeddings,
                        "coords": torch.tensor(grid, dtype=torch.long).unsqueeze(0),  # 添加 batch 维度
                        "attributes": {
                            "patch_size_level0": self.patch_size
                        }
                    }
                    try:
                        aggregat_embedding = aggregator(aggregat_embedding, self.device if torch.cuda.is_available() else 'cpu')
                    except torch.cuda.OutOfMemoryError as e:
                        print(f"[feature_extract] GPU 内存不足，尝试使用 CPU 进行 slide-level 特征聚合")
                        aggregat_embedding = aggregator(aggregat_embedding, 'cpu')
                    torch.save(aggregat_embedding, agg_out_path)

                pbar.set_description(f'🥰 ==> Saved feature to {out_path}')

            except Exception as e:
                if agg_out_dir == "":
                    log_path = os.path.join(out_dir, "feature_extract_errors.log")
                else:
                    log_path = os.path.join(agg_out_dir, "feature_extract_errors.log")
                error_msg = f"{slide_basename}: {e}\n"
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(error_msg)
                print(f"[feature_extract] Error processing slide {slide_path}: {e}")

    def get_label(self):
        for task in self.supported_tasks:
            self.total_labels[task] = pd.read_csv(
                os.path.join(self.label_base_dir, f"{task}.csv"))
            

    def split_dataset(self, split_ratio: List[float] = [0.8, 0.1, 0.1], random_seed: int = 42):
        """
        如果没有设置划分规则，则默认按照入参的比例划分
        根据给定的比例划分数据集为训练集、验证集和测试集。
        返回划分后的索引列表。
        """
        print("⚠️   ==> No specific split ratio, use default split ratio")
        print(f"🎲  ==> Splitting dataset into train, validate, and test sets with ratios: {split_ratio}")
        from torch.utils.data import DataLoader, random_split
        total_size = len(self)
        train_size = int(split_ratio[0] * total_size)
        val_size = int(split_ratio[1] * total_size)
        test_size = total_size - train_size - val_size
        train_set, val_set, test_set = random_split(
            self,
            [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(random_seed)
        )

        total_dataset = {
            "train": train_set,
            "validate": val_set,
            "test": test_set
        }
        

        return total_dataset