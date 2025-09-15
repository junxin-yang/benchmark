import yaml
import json
from models import CONCH, UNI, PRISM, TITAN
from datasets import Camelyon16, TCGA_BRCA, CustomDataset
from tasks import WSIClassificationTask, ReportGenerationTask, SurvivalPredictionTask

def main():
    # 加载配置文件
    with open('configs/config.json', 'r') as f:
        config = json.load(f)
    
    with open('configs/models.yaml', 'r') as f:
        model_configs = yaml.safe_load(f)
    
    with open('configs/datasets.yaml', 'r') as f:
        dataset_configs = yaml.safe_load(f)

    # 任务映射字典，将配置中的任务名映射到对应的任务类
    task_mapping = {
        'Classification': WSIClassificationTask,
        'ReportGeneration': ReportGenerationTask,
        'SurvivalPrediction': SurvivalPredictionTask
    }

    # 数据集映射字典
    dataset_mapping = {
        'TCGA_BRCA': TCGA_BRCA,
        'CAMELYON16': Camelyon16,
        'CUSTOM_DATASET': CustomDataset
    }

    # 模型映射字典
    model_mapping = {
        'UNI': UNI,
        'PRISM': PRISM,
        'TITAN': TITAN,
        'CONCH': CONCH
    }

    # 第一层循环：遍历所有任务
    for task_name, task_config in config.items():
        print(f"\n=== 开始处理任务: {task_name} ===")
        
        # 获取当前任务的模型列表
        task_models = task_config['models']
        # 获取当前任务的数据集列表
        task_datasets = task_config['dataset']
        
        # 第二层循环：遍历当前任务的所有模型
        for model_name in task_models:
            print(f"\n--- 使用模型: {model_name} ---")
            
            # 第三层循环：遍历当前任务的所有数据集
            for dataset_info in task_datasets:
                dataset_name = dataset_info['name']
                print(f"\n处理数据集: {dataset_name}")
                
                try:
                    # 初始化模型
                    model_class = model_mapping[model_name]
                    model = model_class(model_configs[model_name])
                    
                    # 初始化数据集
                    dataset_class = dataset_mapping[dataset_name]
                    dataset = dataset_class(dataset_info, dataset_configs[dataset_name])
                    
                    # 初始化任务
                    task_class = task_mapping[task_name]
                    task = task_class(model, dataset, task_config)
                    
                    # 执行任务
                    results = task.execute()
                    
                    # 输出结果
                    print(f"任务 {task_name} - 模型 {model_name} - 数据集 {dataset_name} 完成")
                    print(f"结果: {results}")
                    
                except Exception as e:
                    print(f"错误: 任务 {task_name} - 模型 {model_name} - 数据集 {dataset_name} 执行失败: {str(e)}")
                    continue

    print("\n=== 所有任务处理完成 ===")

if __name__ == "__main__":
    main()