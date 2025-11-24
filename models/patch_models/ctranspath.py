import os, sys
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(PROJECT_ROOT)
import torch
import traceback
from core.base_model import BasePatchModel
from models.patch_models.utils.constants import get_constants
from models.patch_models.utils.transform_utils import get_eval_transforms
import yaml


class CTransPath(BasePatchModel):

    def __init__(self, **build_kwargs):
        """
        CTransPath initialization.
        """
        self.enc_name = 'CTransPath'
        super().__init__(**build_kwargs)

    def _build(self):
        from torchvision.transforms import InterpolationMode
        from torch import nn

        try:
            from models.patch_models.model_zoo.ctranspath.ctran import ctranspath
        except:
            traceback.print_exc()
            raise Exception("Failed to import CTransPath model, make sure timm_ctp is installed. `pip install timm_ctp`")
        
        self.weights_path = self.model_configs.get("patch_model_path")
        self.device = self.model_configs.get("device")

        model = ctranspath(img_size=224)
        model.head = nn.Identity()

        if not self.weights_path:
            self.ensure_has_internet(self.enc_name)
            try:
                from huggingface_hub import hf_hub_download   
                self.weights_path = hf_hub_download(
                    repo_id="MahmoodLab/hest-bench",
                    repo_type="dataset",
                    filename="CHIEF_CTransPath.pth",
                    subfolder="fm_v1/ctranspath",
                )
            except:
                traceback.print_exc()
                raise Exception("Failed to download CTransPath model, make sure that you were granted access and that you correctly registered your token")

        try:
            state_dict = torch.load(self.weights_path, weights_only=True)['model']
            print(f"🚁  ==> Loaded {self.enc_name} model weights from {self.weights_path}")

        except:
                traceback.print_exc()
                raise Exception(
                    f"Failed to create CTransPath model from local checkpoint at '{self.weights_path}'. "
                    "You can download the required `CHIEF_CTransPath.pth` from: https://huggingface.co/datasets/MahmoodLab/hest-bench/tree/main/fm_v1/ctranspath."
                )
        state_dict = {key: val for key, val in state_dict.items() if 'attn_mask' not in key}
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        assert len(unexpected) == 0, f"Unexpected keys found in state dict: {unexpected}"
        assert missing == ['layers.0.blocks.1.attn_mask', 'layers.1.blocks.1.attn_mask', 'layers.2.blocks.1.attn_mask', 'layers.2.blocks.3.attn_mask', 'layers.2.blocks.5.attn_mask'], f"Unexpected missing keys: {missing}"

        mean, std = get_constants('imagenet')
        eval_transform = get_eval_transforms(mean, std, target_img_size=224, interpolation=InterpolationMode.BILINEAR, max_size=None, antialias=True)

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
        raise NotImplementedError("CONCH does not support report generation.")
    

if __name__ == "__main__":
    model = CTransPath()
    dummy_input = torch.randn(2, 3, 224, 224)  # batch_size=2, 3 channels, 224x224 image
    output = model.forward(dummy_input)
    print(output.shape)  # Expected: (2, embedding_dim)