import yaml
from models import CONCH, UNI, PRISM, TITAN
from datasets import Camelyon16, TCGA_BRCA, CustomDataset
from tasks import WSIClassificationTask, ReportGenerationTask, SurvivalPredictionTask

def main():
    # 加载配置
    with open('configs/models.yaml', 'r') as f:
        model_configs = yaml.safe_load(f)
    with open('configs/datasets.yaml', 'r') as f:
        dataset_configs = yaml.safe_load(f)
    with open('configs/tasks.yaml', 'r') as f:
        task_configs = yaml.safe_load(f)
    
    # 初始化任务
    tasks = {
        'classification': WSIClassificationTask(),
        'report_generation': ReportGenerationTask(),
        'survival_prediction': SurvivalPredictionTask()
    }
    
    # 初始化模型
    models = {
        'CONCH': CONCH(**model_configs['CONCH']),
        'UNI': UNI(**model_configs['UNI']),
        'Prism': PRISM(**model_configs['Prism']),
        'TITAN': TITAN(**model_configs['TITAN'])
    }
    
    # 初始化数据集
    datasets = {
        'CAMELYON16': Camelyon16(**dataset_configs['Camelyon16']),
        'TCGA_BRCA': TCGA_BRCA(**dataset_configs['TCGA_BRCA']),
        'CustomDataset': CustomDataset(**dataset_configs['CustomDataset'])
    }
    
    # 一键式循环测试
    for task_name, task in tasks.items():
        for model_name, model in models.items():
            for dataset_name, dataset in datasets.items():
                print(f"Evaluating {model_name} on {dataset_name} for {task_name}...")
                
                # 执行评估
                metrics, predictions, figures = task.evaluate(model, dataset)
                
                # 保存结果
                task.save_results(model_name, dataset_name, metrics, predictions, figures)
                

if __name__ == "__main__":
    main()