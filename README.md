```
    benchmark_demo/
    │
    ├── configs/               # 配置文件目录
    │   ├── models.yaml       # 模型配置（路径、参数）
    │   ├── datasets.yaml     # 数据集配置（路径、预处理）
    │   └── config.json        # 任务配置
    │
    ├── core/                 # 核心模块
    │   ├── __init__.py
    │   ├── base_model.py    # 模型基类（统一接口）
    │   ├── base_dataset.py  # 数据集基类（统一接口）
    │   └── base_task.py     # 任务基类（核心）
    │
    ├── models/               # 模型实现（继承 base_model）
    │   ├── __init__.py
    │   ├── conch.py
    │   ├── uni.py
    │   ├── prism.py
    │   └── titan.py
    │
    ├── datasets/             # 数据集处理（继承 base_dataset）
    │   ├── __init__.py
    │   ├── tcga.py
    │   ├── camelyon16.py
    │   └── custom_data.py
    │
    ├── tasks/                # 具体任务（继承 base_task）
    │   ├── __init__.py
    │   ├── classification.py   # WSI 分类任务
    │   ├── report_generation.py # 报告生成任务
    │   └── survival_prediction.py # 生存预测任务
    │
    ├── utils/                # 工具函数
    │   ├── __init__.py
    │   ├── logger.py         # 日志记录
    │   ├── visualizer.py     # 结果可视化（绘图）
    │   ├── file_utils.py     # 文件操作辅助
    |   └── metrics.py
    │
    ├── results/              # 结果根目录（自动生成）
    │   ├── classification/
    |   |   ├──CONCH
    |   |   |   ├──CAMELYON16
    |   |   |   |   ├──metrics.json
    |   |   |   |   └──predictions.json
    │   ├── report_generation/
    │   └── survival_prediction/
    |
    |── figures/               # 图片生成根目录
    |   ├── classification/
    |   |   ├──ACC
    |   |   |  ├──CAMELYON16.png
    |   |   |  ├──CUSTOM_DATASET.png
    |   |   |  └──TCGA_BRCA.png
    |   |   ├──AUC
    |   |   |  ├──CAMELYON16.png
    |   |   |  ├──CUSTOM_DATASET.png
    |   |   |  └──TCGA_BRCA.png
    │   ├── report_generation/
    │   └── survival_prediction/
    │
    ├── main.py               # 主程序（一键运行入口）
    └── requirements.txt       # 依赖库
```