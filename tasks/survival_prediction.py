from core.base_task import BaseTask
from core.base_model import BaseModel
from core.base_dataset import BaseDataset

class SurvivalPredictionTask(BaseTask):
    def __init__(self, **kwargs):
        super().__init__(task_name="survival_prediction", **kwargs)

    def evaluate(self, model: BaseModel, dataset: BaseDataset, **kwargs):
        pass