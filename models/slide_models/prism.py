import sys
import os
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(PROJECT_ROOT)
import torch
from torch import einsum
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
                from models.slide_models.model_zoo.prism.perceiver import PerceiverResampler
                from models.slide_models.model_zoo.prism.biogpt import BioGPT
                from models.slide_models.model_zoo.prism.configuring_prism import PrismConfig
                from models.slide_models.model_zoo.prism.modeling_prism import Prism
                self.prism_config = PrismConfig.from_pretrained(self.weights_path)
                model = self.prism_model = Prism.from_pretrained(self.weights_path, config=self.prism_config)
                model = model.to(self.device)
                print(f"🚁  ==>Loaded PRISM model weights from {self.weights_path}")
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
        precision = torch.float16
        embedding_dim = 1280
        return model, precision, embedding_dim
    
    def forward(self, batch, device='cuda'):
        # input should be of shape (batch_size, tile_seq_len, tile_embed_dim)
        x = batch["embeddings"].to(device)
        z = self.model.slide_representations(x)
        z = z['image_embedding'] 
        return z

    def generate_report(self, batch, device='cuda', prompt: Optional[str]=None):
        """
        generate_report 的 Docstring
        
        :param self: 
        :param batch: 加载的virchow_v1的一个batch数据
        :param device: 放在哪个设备上运行
        :param prompt: 生成报告时的提示语
        :return: 生成的报告
        """
        x = batch["embeddings"].unsqueeze(0).to(device)
        z = self.model.slide_representations(x)
        if prompt is not None:
            prompt_ids = self.model.tokenize([prompt]).to(device)
        else:
            prompt_ids = None
        generated_embedding = self.model.generate(
            inputs = prompt_ids,
            key_value_states=z['image_latents'],
            do_sample=False,
            num_beams=5,
            num_beam_groups=1,
        )
        generated_caption = self.model.untokenize(generated_embedding)
        return generated_caption
    
    def zero_shot(self, batch, class_prompt: dict[str, list[str]], device='cuda'):
        """
        zero_shot 的 Docstring
        
        :param self: 
        :param batch: 加载的virchow_v1的一个batch数据
        :param class_prompt: zero-shot分类的提示语
        :type class_prompt: dict[str, list[str]]
        :param device: 使用的设备
        :return: 分类得分
        """
        x = batch["embeddings"].unsqueeze(0).to(device)
        image_embedding = self.model.slide_representations(x)['image_embedding']
        
        # zero-shot prompts
        zero_shot_prompts = [p for lst in class_prompt.values() for p in lst]
        zero_shot_token_ids = self.model.tokenize(zero_shot_prompts)[:, :-1].to(device)
        dummy_image_latents = torch.empty(
            (len(zero_shot_prompts), 1, self.model.text_decoder.context_dim), device=device
        )
        decoder_out = self.model.text_decoder(zero_shot_token_ids, dummy_image_latents)
    
        # zero-shot probabilities
        text_proj = self.model.text_to_latents(decoder_out['text_embedding'])
        image_proj = self.model.img_to_latents(image_embedding)

        sim = einsum('i d, j d -> i j', image_proj, text_proj)  # (image, text)
        sim = sim * self.model.temperature.exp()

        assert sim.shape[0] == len(image_embedding)
        assert sim.shape[1] == len(zero_shot_prompts)

        zero_shot_probs = torch.softmax(sim.to(torch.float), dim=-1)
        
        # 对概率进行分割
        lengths = [len(lst) for lst in class_prompt.values()]
        split_probs = torch.split(zero_shot_probs, lengths, dim=1)
        class_probs = torch.stack([p.sum(dim=1) for p in split_probs], dim=1)

        result = {}
        for i, key in enumerate(class_prompt.keys()):
            # 取出第 i 个类别的概率（假设 batch_size=1，取第 0 个元素）
            result[key] = class_probs[0, i].item()
        return result

        
if __name__ == "__main__":
    model = PRISM(pretrained=False)
    dummy_input = torch.randn(1, 50, 2560)  # batch_size=2, tile_seq_len=50, tile_embed_dim=1280
    output = model.forward(dummy_input, device='cpu')
    print("Output shape:", output.shape)  # Expected: (2, 1280)