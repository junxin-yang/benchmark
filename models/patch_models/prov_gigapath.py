import os, sys
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(PROJECT_ROOT)
import torch
import traceback
from core.base_model import BasePatchModel
from models.patch_models.utils.constants import get_constants
from models.patch_models.utils.transform_utils import get_eval_transforms
import yaml


class ProvGigaPath(BasePatchModel):

    def __init__(self, **build_kwargs):
        """
        GigaPath initialization.
        """
        self.enc_name = 'ProvGigaPath'
        super().__init__(**build_kwargs)

    def _build(
        self, 
    ):
        import timm
        assert timm.__version__ == '0.9.16', f"Gigapath requires timm version 0.9.16, but found {timm.__version__}. Please install the correct version using `pip install timm==0.9.16`"
        from torchvision import transforms

        self.weights_path = self.model_configs.get("patch_model_path")
        self.device = self.model_configs.get("device")

        if self.weights_path:
            try:
                timm_kwargs = {
                    "img_size": 224,
                    "in_chans": 3,
                    "patch_size": 16,
                    "embed_dim": 1536,
                    "depth": 40,
                    "num_heads": 24,
                    "mlp_ratio": 5.33334,
                    "num_classes": 0
                }
                model = timm.create_model("vit_giant_patch14_dinov2", pretrained=False, **timm_kwargs)
                model.load_state_dict(torch.load(self.weights_path, map_location="cpu"), strict=True)
                print(f"🚁  ==> Loaded {self.enc_name} model weights from {self.weights_path}")
            except:
                traceback.print_exc()
                raise Exception(
                    f"Failed to create GigaPath model from local checkpoint at '{self.weights_path}'. "
                    "You can download the required `pytorch_model.bin` from: https://huggingface.co/prov-gigapath/prov-gigapath."
                )
        else:
            self.ensure_has_internet(self.enc_name)
            try:
                model = timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=True)
            except:
                traceback.print_exc()
                raise Exception("Failed to download GigaPath model, make sure that you were granted access and that you correctly registered your token")

        mean, std = get_constants('imagenet')
        eval_transform = transforms.Compose(
            [
                transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
        precision = torch.float32
        model = model.to(self.device, dtype=precision)
        return model, eval_transform, precision
    
if __name__ == "__main__":
    model = ProvGigaPath()
    dummy_input = torch.randn(2, 3, 224, 224)
    output = model.forward(dummy_input)
    print(output.shape)