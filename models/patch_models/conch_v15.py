import os, sys
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(PROJECT_ROOT)
import torch
import traceback
from core.base_model import BasePatchModel
from tqdm import tqdm
import yaml

class CONCH_V15(BasePatchModel):

    def __init__(self, **build_kwargs):
        """
        CONCHv1.5 initialization.
        """
        self.enc_name = 'CONCH_V15'
        super().__init__(**build_kwargs)

    def _build(self, img_size=448):
        from trident.patch_encoder_models.model_zoo.conchv1_5.conchv1_5 import create_model_from_pretrained

        
        self.weights_path = self.model_configs.get("patch_model_path")
        self.device = self.model_configs.get("device")

        if self.weights_path:
            try:
                model, eval_transform = create_model_from_pretrained(checkpoint_path=self.weights_path, img_size=img_size)
                print(f"🚁  ==> Loaded {self.enc_name} model weights from {self.weights_path}")
            except:
                traceback.print_exc()
                raise Exception(
                    f"Failed to create CONCH v1.5 model from local checkpoint at '{self.weights_path}'. "
                    "You can download the required `pytorch_model_vision.bin` and `config.json` from: https://huggingface.co/MahmoodLab/conchv1_5."
                )
        else:
            self.ensure_has_internet(self.enc_name)
            try:
                model, eval_transform = create_model_from_pretrained(checkpoint_path="hf_hub:MahmoodLab/conchv1_5", img_size=img_size)
            except:
                traceback.print_exc()
                raise Exception("Failed to download CONCH v1.5 model, make sure that you were granted access and that you correctly registered your token")

        precision = torch.float32
        model = model.to(self.device, dtype=precision)
        return model, eval_transform, precision


if __name__ == "__main__":
    model = CONCH_V15()
    duummy_input = torch.randn(2, 3, 448, 448)
    output = model.forward(duummy_input)
    print("Output shape:", output.shape)
