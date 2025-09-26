import logging
import sys
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime

# 定义一个全局的日志记录器字典，用于保存不同名称的日志器实例，避免重复创建
_logger_instances = {}

def setup_logger(
    name: str = "WSIBench",
    log_dir: Optional[str] = "logs",
    level: int = logging.INFO,
    console_output: bool = True,
    file_output: bool = True,
    mode: str = 'a'
) -> logging.Logger:
    """
    配置并获取一个命名的日志记录器。
    如果同名日志器已存在，则直接返回现有的实例。

    Args:
        name (str): 日志记录器的名称，通常使用模块名 __name__。默认为 "WSIBench"。
        log_dir (Optional[str]): 日志文件存储的目录路径。默认为 "logs"。
                                如果为 None 或空字符串，则仅输出到控制台（如果启用）。
        level (int): 日志级别。默认为 logging.INFO。
        console_output (bool): 是否将日志输出到控制台。默认为 True。
        file_output (bool): 是否将日志输出到文件。默认为 True。
        mode (str): 日志文件的写入模式。'a' 为追加，'w' 为覆盖。默认为 'a'。

    Returns:
        logging.Logger: 配置好的日志记录器实例。

    Example:
        >>> logger = setup_logger(__name__)
        >>> logger.info("This is an info message.")
    """
    global _logger_instances
    
    # 如果该名称的日志器已经配置过，直接返回
    if name in _logger_instances:
        return _logger_instances[name]
        
    # 创建一个日志记录器
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # 避免重复添加handler（防止Jupyter等环境中重复打印）
    if logger.handlers:
        return logger
        
    # 定义日志格式
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 输出到控制台
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        console_handler.setLevel(level)
        logger.addHandler(console_handler)
    
    # 输出到文件
    if file_output and log_dir:
        # 确保日志目录存在
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        
        # 生成带时间戳的日志文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_path / f"log_{timestamp}.log"
        
        file_handler = logging.FileHandler(log_file, mode=mode, encoding='utf-8')
        file_handler.setFormatter(formatter)
        file_handler.setLevel(level)
        logger.addHandler(file_handler)
        
    # 保存实例并返回
    _logger_instances[name] = logger
    return logger

def get_logger(name: str = "WSIBench") -> logging.Logger:
    """
    快速获取一个已配置的日志记录器。如果尚未配置，则使用默认设置进行配置。

    Args:
        name (str): 日志记录器的名称。默认为 "WSIBench"。

    Returns:
        logging.Logger: 日志记录器实例。
    """
    if name in _logger_instances:
        return _logger_instances[name]
    else:
        # 默认配置：日志目录为 "logs"，同时输出到控制台和文件
        return setup_logger(name)


default_logger = setup_logger()