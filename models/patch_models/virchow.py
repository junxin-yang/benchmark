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


class Virchow(BasePatchModel):
    import timm
    
    def __init__(self, **build_kwargs):
        """
        Virchow initialization.
        """
        super().__init__(**build_kwargs)

    def _build(
        self,
        return_cls=False,
        timm_kwargs={'mlp_layer': timm.layers.SwiGLUPacked, 'act_layer': torch.nn.SiLU}
    ):
        import timm
        import torchvision
        from torchvision import transforms

        self.enc_name = 'virchow'
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        
        weights_path = config['Virchow'].get('model_path', None)

        if weights_path:
            try:
                timm_kwargs = {
                    "img_size": 224,
                    "init_values": 1e-5,
                    "num_classes": 0,
                    "mlp_ratio": 5.3375,
                    "global_pool": "",
                    "dynamic_img_size": True,
                    'mlp_layer': timm.layers.SwiGLUPacked,
                    'act_layer': torch.nn.SiLU,
                }
                model = timm.create_model("vit_huge_patch14_224", **timm_kwargs)
                model.load_state_dict(state_dict=torch.load(weights_path, map_location="cpu"), strict=True)
                print(f"🚁Loaded Virchow model weights from {weights_path}")
            except:
                traceback.print_exc()
                raise Exception(
                    f"Failed to create Virchow model from local checkpoint at '{weights_path}'. "
                    "You can download the required `pytorch_model.bin` from: https://huggingface.co/paige-ai/Virchow."
                )
        else:
            self.ensure_has_internet(self.enc_name)
            try:
                model = timm.create_model("hf-hub:paige-ai/Virchow", pretrained=True, **timm_kwargs)
            except:
                traceback.print_exc()
                raise Exception("Failed to download Virchow model, make sure that you were granted access and that you correctly registered your token")

        eval_transform = transforms.Compose(
            [
                transforms.Resize(224, interpolation=torchvision.transforms.InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )
        precision = torch.float16
        self.return_cls = return_cls
        
        return model, eval_transform, precision

    def forward(self, x):
        output = self.model(x)
        class_token = output[:, 0]

        if self.return_cls:
            return class_token
        else:
            patch_tokens = output[:, 1:]
            embeddings = torch.cat([class_token, patch_tokens.mean(1)], dim=-1)
            return embeddings
        
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
    model = Virchow()