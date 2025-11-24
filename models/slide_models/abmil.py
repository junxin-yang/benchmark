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

class ABMILSlideEncoder(BaseSlideModel):

    def __init__(self, **build_kwargs):
        """
        ABMIL initialization.
        """
        super().__init__(**build_kwargs)
    
    def _build(
        self,
        input_feature_dim: int,
        n_heads: int,
        head_dim: int,
        dropout: float,
        gated: bool,
        pretrained: bool = False
    ) -> Tuple[torch.nn.ModuleDict, torch.dtype, int]:
        
        from trident.slide_encoder_models.model_zoo.reusable_blocks.ABMIL import ABMIL
        import torch.nn as nn

        self.enc_name = 'abmil'
        
        assert pretrained is False, "ABMILSlideEncoder has no corresponding pretrained models. Please load with pretrained=False."
                                
        pre_attention_layers = nn.Sequential(
            nn.Linear(input_feature_dim, input_feature_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        image_pooler = ABMIL(
            n_heads=n_heads,
            feature_dim=input_feature_dim,
            head_dim=head_dim,
            dropout=dropout,
            n_branches=1,
            gated=gated
        )
        
        post_attention_layers = nn.Sequential(
            nn.Linear(input_feature_dim, input_feature_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        model = nn.ModuleDict({
            'pre_attention_layers': pre_attention_layers,
            'image_pooler': image_pooler,
            'post_attention_layers': post_attention_layers
        })
        
        precision = torch.float32
        embedding_dim = input_feature_dim
        return model, precision, embedding_dim

    def forward(self, batch, device='cuda', return_raw_attention=False):
        image_features = self.model['pre_attention_layers'](batch['features'].to(device))
        image_features, attn = self.model['image_pooler'](image_features) # Features shape: (b n_branches f), where n_branches = 1. Branching is not used in this implementation.
        image_features = rearrange(image_features, 'b 1 f -> b f')
        image_features = self.model['post_attention_layers'](image_features)# Attention scores shape: (b r h n), where h is number of attention heads 
        if return_raw_attention:
            return image_features, attn
        return image_features
    

if __name__ == "__main__":
    model = ABMILSlideEncoder(
        input_feature_dim=1024,
        n_heads=8,
        head_dim=64,
        dropout=0.1,
        gated=True,
        pretrained=False
    )
    dummy_input = {
        'features': torch.randn(2, 50, 1024)  # batch_size=2, tile_seq_len=50, tile_embed_dim=1024
    }
    output = model.forward(dummy_input, device='cpu')
    print("Output shape:", output.shape)  # Expected: (2, 1024)