from core.base_task import BaseTask
from core.base_model import BaseModel
from core.base_dataset import BaseDataset

class ReportGenerationTask(BaseTask):
    def __init__(self, **kwargs):
        super().__init__(task_name="report_generation", **kwargs)
        
    def evaluate(self, model: BaseModel, dataset: BaseDataset, **kwargs):
        pass