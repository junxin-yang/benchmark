from core.base_task import BaseTask
from core.base_model import BaseModel
from core.base_dataset import BaseDataset

class ClassificationTask(BaseTask):
    def __init__(self, task_name: str, metrics: list, output_root="results"):
        super().__init__(task_name=task_name, metrics=metrics, output_root=output_root)

    def evaluate(self, model: BaseModel, dataset: BaseDataset, **kwargs):
        all_preds = []
        all_labels = []
        metric_results = {}

        for item in dataset:
            feature = item.get("embedding")
            slide_info = item.get("slide_info")
            label = slide_info.get("classification_label")
            pred = model.classify(feature, kwargs.get("num_classes"))
            all_preds.append(pred)
            all_labels.append(label)
        
        for metric_fn in self.metrics:
            metric_name = metric_fn.__name__
            metric_results[metric_name] = metric_fn(all_labels, all_preds)

        return metric_results, all_preds
