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

class GigaPathSlideEncoder(BaseSlideModel):

    def __init__(self, **build_kwargs):
        """
        GigaPath initialization.
        """
        super().__init__(**build_kwargs)

    def _build(self, pretrained=True):

        self.enc_name = 'ProvGigaPath'

        cfg = self.load_config(config_path, self.enc_name)
        self.weights_path = cfg.get("slide_model_path")
        self.device = cfg.get("device")

        try:
            from gigapath.slide_encoder import create_model
        except:
            traceback.print_exc()
            raise Exception("Please install fairscale and gigapath using `pip install fairscale git+https://github.com/prov-gigapath/prov-gigapath.git`.")
        
        # Make sure flash_attn is correct version
        try:
            import flash_attn; assert flash_attn.__version__ == '2.5.8'
        except:
            traceback.print_exc()
            raise Exception("Please install flash_attn version 2.5.8 using `pip install flash_attn==2.5.8`.")
        
        if pretrained:
            
            try:
                model = create_model(self.weights_path, "gigapath_slide_enc12l768d", 1536, global_pool=True, device=self.device)
                print(f"🚁Loaded GigaPath Slide Encoder model weights from {self.weights_path}")
            except:
                model = create_model("hf_hub:prov-gigapath/prov-gigapath", "gigapath_slide_enc12l768d", 1536, global_pool=True)
                print("🚁Downloaded GigaPath Slide Encoder model weights from HuggingFace Hub")
        else:
            model = create_model("", "gigapath_slide_enc12l768d", 1536, global_pool=True)
        
        
        precision = torch.float16
        embedding_dim = 768
        return model, precision, embedding_dim

    def forward(self, batch, device='cuda'):
        device = self.device if self.device else device
        self.model.tile_size = batch['attributes']['patch_size_level0']
        z = self.model(batch['features'].to(device), batch['coords'].to(device), all_layer_embed=True)[11]
        return z
    

if __name__ == "__main__":
    model = GigaPathSlideEncoder(pretrained=True)
    dummy_input = {
        'features': torch.randn(1, 50, 1536),  # batch_size=1, tile_seq_len=50, tile_embed_dim=1536
        'coords': torch.randint(0, 20000, (1, 50, 2), dtype=torch.long),      # batch_size=1, tile_seq_len=50, coord_dim=2
        'attributes': {'patch_size_level0': 512}
    }
    with torch.cuda.amp.autocast(dtype=torch.float16):
        output = model.forward(dummy_input)
    print("Output shape:", output.shape)  # Expected: (1, embedding_dim)