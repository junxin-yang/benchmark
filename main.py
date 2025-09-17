import yaml
import json
from models import CONCH, UNI, PRISM, TITAN
from datasets import Camelyon16, TCGA_BRCA, CustomDataset
from tasks import ClassificationTask, ReportGenerationTask, SurvivalPredictionTask
from utils.visualizer import plot_bar
from utils.metrics import acc, precision, recall, f1, auc, bleu, c_index, auc_survival
from utils.logger import default_logger as logger


from core.base_dataset import BaseDataset
import os
class SimpleDataset(BaseDataset):
    def __init__(self, data_root: str, **kwargs):
        super().__init__(data_root, **kwargs)

        self.slides = ["slide1.svs", "slide2.svs", "slide3.svs"]
        self.dataset_name = "SimpleDataset"
        self.method = "dummy_method"

        self.label_classification = {
            "slide1.svs": 0,
            "slide2.svs": 1,
            "slide3.svs": 0
        }
        self.label_report_generation = {
            "slide1.svs": "Benign tissue identified.",
            "slide2.svs": "Inflammatory changes present.",
            "slide3.svs": "Benign tissue identified."
        }
        self.label_survival_prediction = {
            "slide1.svs": (10, 1),
            "slide2.svs": (20, 0),
            "slide3.svs": (30, 1)
        }

    def __len__(self):
        return len(self.slides)

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self.slides):
            raise IndexError("Index out of range")
        slide_name = self.slides[idx]
        slide = os.path.join(self.slide_base_dir, slide_name)
        feature = [float(idx+1), float(idx+2), float(idx+3)]
        embedding = feature

        classification_label = self.label_classification[slide_name]
        report_generation_label = self.label_report_generation[slide_name]
        survival_prediction_label = self.label_survival_prediction[slide_name]

        slide_info = {
            "slide_name": slide_name,
            "slide_path": slide,
            "classification_label": classification_label,
            "report_generation_label": report_generation_label,
            "survival_prediction_label": survival_prediction_label
        }
        return {
            "embedding": embedding,
            "slide_info": slide_info
        }
    

def main():
    # load configurations
    with open('configs/config.json', 'r') as f:
        config = json.load(f)
    
    with open('configs/models.yaml', 'r') as f:
        model_configs = yaml.safe_load(f)
    
    with open('configs/datasets.yaml', 'r') as f:
        dataset_configs = yaml.safe_load(f)

    # task mapping dictionary
    task_mapping = {
        'Classification': ClassificationTask,
        'ReportGeneration': ReportGenerationTask,
        'SurvivalPrediction': SurvivalPredictionTask
    }

    # dataset mapping dictionary
    dataset_mapping = {
        'TCGA_BRCA': TCGA_BRCA,
        'CAMELYON16': Camelyon16,
        'CUSTOM_DATASET': CustomDataset
    }

    # model mapping dictionary
    model_mapping = {
        'UNI': UNI,
        'PRISM': PRISM,
        'TITAN': TITAN,
        'CONCH': CONCH
    }

    # metrics mapping dictionary
    metrics_mapping = {
        'ACC': acc,
        'Precision': precision,
        'Recall': recall,
        'F1': f1,
        'AUC': auc,
        'BLEU': bleu,
        'c_index': c_index,
        'AUC_Survival': auc_survival
    }

    # first layer loop: iterate over tasks
    for task_name, task_config in config.items():
        logger.info(f"=== Start processing tasks: {task_name} ===")
        
        # get current task's model list
        task_models = task_config.get('models')
        # get current task's dataset list
        task_datasets = task_config.get('datasets')
        # get current task's metric list
        task_metrics = task_config.get('metrics')
        # get current task's result directory
        result_dir = task_config.get('result_dir')
        # get current task's figure directory
        fig_dir = task_config.get('fig_dir')

        # second layer loop: iterate over current task's models
        for model_name in task_models:
            logger.info(f"--- Use the model: {model_name} ---")
            model_config = model_configs.get(model_name)

            # third layer loop: iterate over current task's datasets
            for dataset_info in task_datasets:
                dataset_name = dataset_info.get('name')
                test_configs = dataset_info.get("configs", {})
                logger.info(f"\nProcessing dataset: {dataset_name}")
                
                try:
                    # initialize model
                    model_class = model_mapping[model_name]
                    model = model_class(model_path = model_config.get("model_path"), 
                                        model_name = model_name, 
                                        device = model_config.get("device"))
                    
                    # [todo]
                    # # initialize dataset
                    # dataset_class = dataset_mapping[dataset_name]
                    # dataset = dataset_class(dataset_configs[dataset_name].get("root_dir"))
                    
                    # [todo]
                    dataset = SimpleDataset(data_root="dummy_path")

                    # initialize task
                    task_class = task_mapping[task_name]
                    metric_fns = [metrics_mapping[m] for m in task_metrics]
                    task = task_class(task_name=task_name, metrics=metric_fns, output_root=result_dir)

                    # execute evaluation
                    metric_results, all_preds = task.evaluate(model, dataset, **test_configs)
                    
                    # save results
                    task.save_results(model_name=model_name, dataset_name=dataset_name, metrics=metric_results, predictions=all_preds)

                    # output results
                    logger.info(f"task {task_name} - model {model_name} - dataset {dataset_name} finished.")
                    logger.info(f"result: {metric_results}")

                except Exception as e:
                    logger.error(f"Erro: task {task_name} - model {model_name} - dataset {dataset_name} Execution failed: {str(e)}")
                    continue

        for metric in task_metrics:
            plot_bar(models=task_models, datasets=[d.get('name') for d in task_datasets], 
                    task_name=task_name, metric=metric, 
                    result_dir=result_dir, fig_dir=fig_dir)
        

    logger.info("\n=== All tasks have been completed ===")


if __name__ == "__main__":
    main()
