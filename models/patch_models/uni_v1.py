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

class UNI_V1(BasePatchModel):
    def __init__(self, **build_kwargs):
        """
        UNI initialization.
        """
        self.enc_name = 'UNI_V1'
        super().__init__(**build_kwargs)

    def _build(
        self, 
        timm_kwargs={"dynamic_img_size": True, "num_classes": 0, "init_values": 1e-5}
    ):
        import timm
        from torchvision import transforms

        
        self.weights_path = self.model_configs.get("patch_model_path")
        self.device = self.model_configs.get("device")

        if self.weights_path:
            try:
                timm_kwargs = {
                    'img_size': 224,
                    'patch_size': 16,
                    'init_values': 1e-5,
                    'num_classes': 0,
                    'dynamic_img_size': True,
                }
                model = timm.create_model("vit_large_patch16_224", **timm_kwargs)
                model.load_state_dict(torch.load(self.weights_path, map_location="cpu"), strict=True)
                print(f"🚁  ==> Loaded {self.enc_name} model weights from {self.weights_path}")
            except:
                traceback.print_exc()
                raise Exception(
                    f"Failed to create UNI model from local checkpoint at '{self.weights_path}'. "
                    "You can download the required `pytorch_model.bin` from: https://huggingface.co/MahmoodLab/UNI."
                )
        else:
            self.ensure_has_internet(self.enc_name)
            try:
                model = timm.create_model("hf-hub:MahmoodLab/uni", pretrained=True, **timm_kwargs)
            except:
                traceback.print_exc()
                raise Exception("Failed to download UNI model, make sure that you were granted access and that you correctly registered your token")

        eval_transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

        precision = torch.float32
        model = model.to(self.device, dtype=precision)
        return model, eval_transform, precision

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
        raise NotImplementedError("UNI does not support report generation.")

if __name__ == "__main__":
    model = UNI_V1()
    dummy_input = torch.randn(2, 3, 224, 224)
    output = model.forward(dummy_input)
    print(output.shape)