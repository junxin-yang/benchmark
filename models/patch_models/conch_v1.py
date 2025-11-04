import os, sys
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(PROJECT_ROOT)
import torch
import traceback
from core.base_model import BasePatchModel
import yaml


# 获取项目根路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

config_path = os.path.join(project_root, "configs", "models.yaml")

class CONCH(BasePatchModel):
    def __init__(self, **build_kwargs):
        """
        CONCH initialization.
        """
        super().__init__(**build_kwargs)

    def _build(self, with_proj=False, normalize=False):
        self.enc_name = 'conch_v1'
        self.with_proj = with_proj
        self.normalize = normalize

        try:
            from conch.open_clip_custom import create_model_from_pretrained
        except:
            traceback.print_exc()
            raise Exception("Please install CONCH `pip install git+https://github.com/Mahmoodlab/CONCH.git`")
        
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        weights_path = config['CONCH'].get('model_path', None)

        if weights_path:
            try:
                model, eval_transform = create_model_from_pretrained('conch_ViT-B-16', checkpoint_path=weights_path)
                print(f"🚁Loaded CONCH v1 model weights from {weights_path}")
            except:
                traceback.print_exc()
                raise Exception(
                    f"Failed to create CONCH v1 model from local checkpoint at '{weights_path}'. "
                    "You can download the required `pytorch_model.bin` from: https://huggingface.co/MahmoodLab/CONCH."
                )
        else:
            self.ensure_has_internet(self.enc_name)
            try:
                model, eval_transform = create_model_from_pretrained('conch_ViT-B-16', checkpoint_path="hf_hub:MahmoodLab/conch")
            except:
                traceback.print_exc()
                raise Exception("Failed to download CONCH v1 model, make sure that you were granted access and that you correctly registered your token")
    
        precision = torch.float32
        
        return model, eval_transform, precision
    
    def forward(self, x):
        return self.model.encode_image(x, proj_contrast=self.with_proj, normalize=self.normalize)

    def classify(self, feature, num_classes):
        import random
        pred_class = random.randint(0, num_classes - 1)
        probs = [random.random() for _ in range(num_classes)]
        total = sum(probs)
        probs = [p / total for p in probs]
        return {"pred_class": pred_class, "probabilities": probs}
        
        # embedding_data = torch.load(feature)
        # tile_embeddings = embedding_data['embeddings'].unsqueeze(0).to(self.device)

        # with torch.autocast(self.device, torch.float16), torch.inference_mode():
        #     logits = self.model.classify(tile_embeddings)
        #     probs = torch.softmax(logits, dim=-1)
        #     pred_class = torch.argmax(probs, dim=-1).item()
        # return {"pred_class": pred_class, "probabilities": probs.squeeze().tolist()}

    def survival_predict(self, feature, time_horizon=None):
        """
        Survival prediction.
        Args:
            feature: input features for prediction
            time_horizon: optional, predict survival at a specific time point
        Returns:
            Survival probability or risk score
        """
        import random
        risk_score = random.random()
        return {"risk_score": risk_score}

        # embedding_data = torch.load(feature)
        # tile_embeddings = embedding_data['embeddings'].unsqueeze(0).to(self.device)

        # with torch.autocast(self.device, torch.float16), torch.inference_mode():
        #     if hasattr(self.model, "survival_predict"):
        #         result = self.model.survival_predict(tile_embeddings, time_horizon)
        #     else:
        #         logits = self.model.classify(tile_embeddings)
        #         probs = torch.softmax(logits, dim=-1)
        #         result = 1 - probs.max().item() 

        # return {"risk_score": result}

    def report_generate(self, feature):
        raise NotImplementedError("CONCH does not support report generation.")
    
if __name__ == "__main__":
    model = CONCH()