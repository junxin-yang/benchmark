import json
import pickle
import csv
import yaml
import os
import shutil
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from datetime import datetime

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_directory(dir_path: Union[str, Path], parents: bool = True, exist_ok: bool = True) -> Path:
    """
    安全地创建目录（包括父目录）。
    
    Args:
        dir_path: 要创建的目录路径。
        parents: 是否创建父目录，类似于 `mkdir -p`。
        exist_ok: 如果目录已存在，是否将其视为成功（而不是抛出异常）。

    Returns:
        Path: 创建的目录的 Path 对象。

    Raises:
        OSError: 如果目录创建失败且 exist_ok 为 False。
    """
    dir_path = Path(dir_path)
    try:
        dir_path.mkdir(parents=parents, exist_ok=exist_ok)
        logger.debug(f"Directory created or already exists: {dir_path}")
        return dir_path
    except OSError as e:
        logger.error(f"Failed to create directory {dir_path}: {e}")
        raise

def safe_file_write(file_path: Union[str, Path], content: Any, mode: str = 'w', encoding: str = 'utf-8') -> bool:
    """
    安全地将内容写入文件。会自动创建不存在的父目录。

    Args:
        file_path: 要写入的文件路径。
        content: 要写入的内容（字符串或字节）。
        mode: 写入模式，'w' 表示文本写入，'wb' 表示二进制写入。
        encoding: 文本编码方式。

    Returns:
        bool: 成功则返回 True，失败则返回 False。
    """
    file_path = Path(file_path)
    try:
        # 确保父目录存在
        create_directory(file_path.parent, parents=True, exist_ok=True)
        
        if 'b' in mode:
            # 二进制写入
            with open(file_path, mode) as f:
                f.write(content)
        else:
            # 文本写入
            with open(file_path, mode, encoding=encoding) as f:
                f.write(content)
                
        logger.debug(f"Content successfully written to: {file_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to write to file {file_path}: {e}")
        return False

def read_json(file_path: Union[str, Path]) -> Optional[Dict[str, Any]]:
    """
    从JSON文件中读取数据。

    Args:
        file_path: JSON文件路径。

    Returns:
        Optional[Dict[str, Any]]: 解析后的字典数据，如果读取失败则返回None。
    """
    file_path = Path(file_path)
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.debug(f"JSON data read from: {file_path}")
        return data
    except Exception as e:
        logger.error(f"Failed to read JSON from {file_path}: {e}")
        return None

def write_json(data: Dict[str, Any], file_path: Union[str, Path], indent: int = 4, ensure_ascii: bool = False) -> bool:
    """
    将数据写入JSON文件。

    Args:
        data: 要写入的字典数据。
        file_path: 输出的JSON文件路径。
        indent: JSON格式化缩进。
        ensure_ascii: 是否确保ASCII编码（通常设为False以支持中文）。

    Returns:
        bool: 成功则返回True，失败则返回False。
    """
    try:
        # 使用 safe_file_write 确保目录存在并写入
        json_str = json.dumps(data, indent=indent, ensure_ascii=ensure_ascii)
        return safe_file_write(file_path, json_str, 'w', encoding='utf-8')
    except Exception as e:
        logger.error(f"Failed to write JSON to {file_path}: {e}")
        return False

def read_csv(file_path: Union[str, Path]) -> Optional[List[List[str]]]:
    """
    从CSV文件中读取数据。

    Args:
        file_path: CSV文件路径。

    Returns:
        Optional[List[List[str]]]: CSV行列表，每行是一个字符串列表。失败则返回None。
    """
    file_path = Path(file_path)
    try:
        with open(file_path, 'r', encoding='utf-8', newline='') as f:
            reader = csv.reader(f)
            data = list(reader)
        logger.debug(f"CSV data read from: {file_path}")
        return data
    except Exception as e:
        logger.error(f"Failed to read CSV from {file_path}: {e}")
        return None

def write_csv(data: List[List[Any]], file_path: Union[str, Path]) -> bool:
    """
    将数据写入CSV文件。

    Args:
        data: 二维列表数据。
        file_path: 输出的CSV文件路径。

    Returns:
        bool: 成功则返回True，失败则返回False。
    """
    try:
        # 使用 safe_file_write 的逻辑，但CSV写入需要特殊处理
        create_directory(Path(file_path).parent, parents=True, exist_ok=True)
        with open(file_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(data)
        logger.debug(f"CSV data written to: {file_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to write CSV to {file_path}: {e}")
        return False

def read_yaml(file_path: Union[str, Path]) -> Optional[Dict[str, Any]]:
    """
    从YAML文件中读取配置。

    Args:
        file_path: YAML文件路径。

    Returns:
        Optional[Dict[str, Any]]: 解析后的字典数据，失败则返回None。
    """
    file_path = Path(file_path)
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        logger.debug(f"YAML data read from: {file_path}")
        return data
    except Exception as e:
        logger.error(f"Failed to read YAML from {file_path}: {e}")
        return None

def write_yaml(data: Dict[str, Any], file_path: Union[str, Path]) -> bool:
    """
    将数据写入YAML文件。

    Args:
        data: 要写入的字典数据。
        file_path: 输出的YAML文件路径。

    Returns:
        bool: 成功则返回True，失败则返回False。
    """
    try:
        # PyYAML 的 dump 方法返回字符串，然后安全写入
        yaml_str = yaml.dump(data, default_flow_style=False, allow_unicode=True, encoding='utf-8', sort_keys=False)
        # 注意：yaml.dump 可能返回 bytes，需要解码
        if isinstance(yaml_str, bytes):
            yaml_str = yaml_str.decode('utf-8')
        return safe_file_write(file_path, yaml_str, 'w', encoding='utf-8')
    except Exception as e:
        logger.error(f"Failed to write YAML to {file_path}: {e}")
        return False

def save_pickle(obj: Any, file_path: Union[str, Path]) -> bool:
    """
    使用 pickle 序列化并保存 Python 对象到文件。

    Args:
        obj: 要序列化的Python对象。
        file_path: 输出文件路径。

    Returns:
        bool: 成功则返回True，失败则返回False。
    """
    file_path = Path(file_path)
    try:
        create_directory(file_path.parent, parents=True, exist_ok=True)
        with open(file_path, 'wb') as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.debug(f"Object pickled to: {file_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to pickle object to {file_path}: {e}")
        return False

def load_pickle(file_path: Union[str, Path]) -> Optional[Any]:
    """
    从文件中加载并反序列化 Python 对象。

    Args:
        file_path: pickle文件路径。

    Returns:
        Optional[Any]: 反序列化的Python对象，失败则返回None。
    """
    file_path = Path(file_path)
    try:
        with open(file_path, 'rb') as f:
            obj = pickle.load(f)
        logger.debug(f"Object unpickled from: {file_path}")
        return obj
    except Exception as e:
        logger.error(f"Failed to unpickle object from {file_path}: {e}")
        return None

def list_files(dir_path: Union[str, Path], pattern: str = "*", recursive: bool = False) -> List[Path]:
    """
    列出目录中符合模式的文件。

    Args:
        dir_path: 要搜索的目录路径。
        pattern: 文件匹配模式（例如 "*.npy", "*/embeddings/*.csv"）。
        recursive: 是否递归搜索子目录。

    Returns:
        List[Path]: 匹配到的文件路径列表。
    """
    dir_path = Path(dir_path)
    if not dir_path.exists() or not dir_path.is_dir():
        logger.warning(f"Directory does not exist or is not a directory: {dir_path}")
        return []
    
    if recursive:
        files = list(dir_path.rglob(pattern))
    else:
        files = list(dir_path.glob(pattern))
    
    # 过滤，只保留文件
    files = [f for f in files if f.is_file()]
    return files

def get_file_size(file_path: Union[str, Path]) -> Optional[int]:
    """
    获取文件大小（字节）。

    Args:
        file_path: 文件路径。

    Returns:
        Optional[int]: 文件大小（字节），如果文件不存在或出错则返回None。
    """
    file_path = Path(file_path)
    try:
        return file_path.stat().st_size
    except OSError as e:
        logger.error(f"Failed to get file size for {file_path}: {e}")
        return None

def file_exists(file_path: Union[str, Path]) -> bool:
    """
    检查文件是否存在且为文件。

    Args:
        file_path: 文件路径。

    Returns:
        bool: 存在且为文件则返回True，否则返回False。
    """
    file_path = Path(file_path)
    return file_path.is_file()

def copy_file(src_path: Union[str, Path], dst_path: Union[str, Path], overwrite: bool = False) -> bool:
    """
    复制文件。

    Args:
        src_path: 源文件路径。
        dst_path: 目标文件路径。
        overwrite: 如果目标文件已存在，是否覆盖。

    Returns:
        bool: 成功则返回True，失败则返回False。
    """
    src_path, dst_path = Path(src_path), Path(dst_path)
    
    if not src_path.is_file():
        logger.error(f"Source file does not exist: {src_path}")
        return False
        
    if dst_path.exists():
        if not overwrite:
            logger.error(f"Destination file already exists and overwrite is False: {dst_path}")
            return False
        else:
            # 尝试删除已存在的目标文件
            try:
                dst_path.unlink()
            except OSError as e:
                logger.error(f"Failed to remove existing destination file {dst_path}: {e}")
                return False
                
    try:
        # 确保目标目录存在
        create_directory(dst_path.parent, parents=True, exist_ok=True)
        shutil.copy2(src_path, dst_path) # 使用 copy2 尽可能保留元数据
        logger.debug(f"File copied from {src_path} to {dst_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to copy file from {src_path} to {dst_path}: {e}")
        return False

def generate_timestamped_dir(base_dir: Union[str, Path], prefix: Optional[str] = None) -> Path:
    """
    在基础目录下生成一个带时间戳的子目录（格式：prefix_YYYYMMDD_HHMMSS）。
    常用于创建实验运行的结果目录。

    Args:
        base_dir: 基础目录。
        prefix: 目录名前缀（例如 'exp', 'run'）。

    Returns:
        Path: 生成的新目录的完整路径。
    """
    base_dir = Path(base_dir)
    create_directory(base_dir, exist_ok=True) # 确保基础目录存在
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if prefix:
        dir_name = f"{prefix}_{timestamp}"
    else:
        dir_name = timestamp
        
    new_dir = base_dir / dir_name
    create_directory(new_dir) # 创建新目录
    logger.info(f"Generated timestamped directory: {new_dir}")
    return new_dir