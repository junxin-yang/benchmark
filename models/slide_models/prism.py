import sys
import os
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(PROJECT_ROOT)
import torch
import traceback
from abc import abstractmethod
from einops import rearrange
from typing import Optional, Tuple
from core.base_model import BaseSlideModel

import yaml

# 获取项目根路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

config_path = os.path.join(project_root, "configs", "models.yaml")

class PRISM(BaseSlideModel):
    def __init__(self, **build_kwargs):
        """
        PRISM initialization.
        """
        super().__init__(**build_kwargs)
    
    def _build(self, pretrained=True):
        
        self.enc_name = 'PRISM'

        if sys.version_info < (3, 10):
            raise RuntimeError("PRISM requires Python 3.10 or above. Please update your Python interpreter.")

        try:
            import environs  # weird dependencies required by PRISM
            import sacremoses
            from transformers import AutoModel, AutoConfig
        except:
            traceback.print_exc()
            raise Exception(
                "Please run `pip install environs==11.0.0 transformers==4.42.4 sacremoses==0.1.1` "
                "and ensure Python version is 3.10 or above."
            )
        
        cfg = self.load_config(config_path, self.enc_name)
        self.weights_path = cfg.get("slide_model_path")
        self.device = cfg.get("device")


        if self.weights_path:
            try:
                from model_zoo.prism.perceiver import PerceiverResampler
                from model_zoo.prism.biogpt import BioGPT
                from model_zoo.prism.configuring_prism import PrismConfig
                from model_zoo.prism.modeling_prism import Prism
                self.prism_config = PrismConfig.from_pretrained(self.weights_path)
                model = self.prism_model = Prism.from_pretrained(self.weights_path, config=self.prism_config)
                print(f"🚁Loaded PRISM model weights from {self.weights_path}")
            except:
                traceback.print_exc()
                raise Exception(
                    f"Failed to create PRISM model from local checkpoint at '{self.weights_path}'. "
                    "You can download the required `pytorch_model.bin` from: https://huggingface.co/paige-ai/Prism."
                )
        else:
            if pretrained:
                model = AutoModel.from_pretrained('paige-ai/Prism', trust_remote_code=True)
            else:
                model = AutoModel.from_config(AutoConfig.from_pretrained('paige-ai/Prism'))
        model.text_decoder = None
        precision = torch.float16
        embedding_dim = 1280
        return model, precision, embedding_dim
    
    def forward(self, batch, device='cuda'):
        # input should be of shape (batch_size, tile_seq_len, tile_embed_dim)
        x = batch['features'].to(device)
        z = self.model.slide_representations(x)
        z = z['image_embedding'] 
        return z

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

        import random
        # 随机生成一个假报告字符串
        fake_reports = [
            "No significant abnormality detected.",
            "Possible malignancy observed in the sample.",
            "Inflammatory changes present.",
            "Sample insufficient for diagnosis.",
            "Benign tissue identified."
        ]
        return random.choice(fake_reports)
        # embedding_data = torch.load(feature)
        # tile_embeddings = embedding_data['embeddings'].unsqueeze(0).to(self.device)

        # with torch.autocast(self.device, torch.float16), torch.inference_mode():
        #     reprs = self.model.slide_representations(tile_embeddings)

        # with torch.autocast('cuda', torch.float16), torch.inference_mode():
        #     genned_ids = self.model.generate(
        #         key_value_states=reprs['image_latents'],
        #         do_sample=False,
        #         num_beams=5,
        #         num_beam_groups=1,
        #     )
        #     genned_caption = self.model.untokenize(genned_ids)
        # return genned_caption

if __name__ == "__main__":
    model = PRISM(pretrained=False)
    dummy_input = {
        'features': torch.randn(1, 50, 2560)  # batch_size=2, tile_seq_len=50, tile_embed_dim=1280
    }
    output = model.forward(dummy_input, device='cpu')
    print("Output shape:", output.shape)  # Expected: (2, 1280)