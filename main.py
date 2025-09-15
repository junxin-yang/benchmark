import yaml
import json
from models import CONCH, UNI, PRISM, TITAN
from datasets import Camelyon16, TCGA_BRCA, CustomDataset
from tasks import ClassificationTask, ReportGenerationTask, SurvivalPredictionTask
from utils.visualizer import plot_bar

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
        'Classification': ClassificationTask,
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
        task_models = task_config.get('models')
        # 获取当前任务的数据集列表
        task_datasets = task_config.get('datasets')
        # 获取当前任务的评分函数
        task_metrics = task_config.get('metrics')
        # 获取当前任务的结果保存路径
        result_dir = task_config.get('result_dir')
        # 获取当前任务的图像结果保存路径
        fig_dir = task_config.get('fig_dir')
        
        # 第二层循环：遍历当前任务的所有模型
        for model_name in task_models:
            print(f"\n--- 使用模型: {model_name} ---")
            
            # 第三层循环：遍历当前任务的所有数据集
            for dataset_info in task_datasets:
                dataset_name = dataset_info.get('name')
                test_configs = dataset_info.get("configs")
                print(f"\n处理数据集: {dataset_name}")
                
                try:
                    # 初始化模型
                    model_class = model_mapping[model_name]
                    model = model_class(model_path = model_configs.get("model_path"), 
                                        model_name = model_configs.get(model_name), 
                                        device = model_configs.get("device"))
                    
                    # 初始化数据集
                    dataset_class = dataset_mapping[dataset_name]
                    dataset = dataset_class(dataset_info, dataset_configs[dataset_name])
                    
                    # 初始化任务
                    task_class = task_mapping[task_name]
                    task = task_class(task_metrics, output_root=result_dir)
                    
                    # 执行任务
                    metric_results, all_preds = task.evaluate(model, dataset, **test_configs)
                    # 保存结果
                    task.save_results(model_name=model_name, dataset_name=dataset_name, metrics=metric_results, predictions=all_preds)
                    
                    # 输出结果
                    print(f"任务 {task_name} - 模型 {model_name} - 数据集 {dataset_name} 完成")
                    print(f"结果: {metric_results}")

                except Exception as e:
                    print(f"错误: 任务 {task_name} - 模型 {model_name} - 数据集 {dataset_name} 执行失败: {str(e)}")
                    continue
        # plot_bar(models: list, datasets: list, task_name: str, metric: str, result_dir: str, fig_dir: str)
        for metric in task_metrics:
            plot_bar(models=task_models, datasets=[d.get('name') for d in task_datasets], 
                     task_name=task_name, metric=metric, 
                     result_dir=result_dir, fig_dir=fig_dir)

    print("\n=== 所有任务处理完成 ===")


if __name__ == "__main__":
    main()