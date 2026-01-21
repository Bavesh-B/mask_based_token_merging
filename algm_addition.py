import torch
from typing import Tuple

from models.vit import ViT
from timm.models.vision_transformer import Block, Attention, VisionTransformer

from models.algm_layers.local_merge import conditional_pooling, merge_source, merge_wavg
from models.algm_layers.global_merge_bavesh import turbo_matching

from models.algm_layers.utils import parse_r



class TurboBlock(Block):
    """
    Modifications:
     - Apply ALGM between the attention and mlp blocks
    """

    def _drop_path1(self, x):
        return self.drop_path1(x) if hasattr(self, "drop_path1") else self.drop_path(x)

    def _drop_path2(self, x):
        return self.drop_path2(x) if hasattr(self, "drop_path2") else self.drop_path(x)

    def forward(self, x: torch.Tensor, num_protected_tokens: int = 1) -> torch.Tensor:
        attn_size = self._turbo_info["size"] if self._turbo_info["prop_attn"] else None
        x_attn, metric = self.attn(self.norm1(x), attn_size)
        if hasattr(self, "ls1"):
            x_attn = self.ls1(x_attn)
        x = x + self._drop_path1(x_attn)
        layer_idx = self._turbo_layer_idx
        
        if self._turbo_info["source"] is None: # if layer_idx == 1:
                
                merge  = conditional_pooling(
                    x,
                    self._turbo_info["threshold"],
                    self._turbo_info["window_size"],
                    num_prefix_tokens=num_protected_tokens,
                )
                if self._turbo_info["trace_source"]:
                        self._turbo_info["source"] = merge_source(
                            merge, x, self._turbo_info["source"]
                        )
                x, self._turbo_info["size"] = merge_wavg(merge, x, self._turbo_info["size"])
                
              
        else:
              
                merge = turbo_matching(
                    x,
                    layer_idx,
                    self._turbo_info["source"],
                    self._turbo_info["class_token"],
                    self._turbo_info["distill_token"],
                    num_prefix_tokens=num_protected_tokens,
                )
                if self._turbo_info["trace_source"]:
                    self._turbo_info["source"] = merge_source(
                        merge, x, self._turbo_info["source"]
                    )
                x, self._turbo_info["size"] = merge_wavg(merge, x, self._turbo_info["size"])
        x_mlp = self.mlp(self.norm2(x))
        if hasattr(self, "ls2"):
            x_mlp = self.ls2(x_mlp)
        x = x + self._drop_path2(x_mlp)
        
        return x 


class TurboAttention(Attention):

    def forward(
        self, x: torch.Tensor,size: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Note: this is copied from timm.models.vision_transformer.Attention with modifications.
        # we do not change anything here, and do not use  q.mean(1)
        B, N, C = x.shape
        
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        
        q, k, v = qkv[0], qkv[1], qkv[2]

        if hasattr(self, "q_norm") and hasattr(self, "k_norm"):
            q, k = self.q_norm(q), self.k_norm(k)

        dropout_p = self.attn_drop.p if self.training else 0.0
        if getattr(self, "fused_attn", False):
            x = torch.nn.functional.scaled_dot_product_attention(q, k, v, None, dropout_p)
        else:
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, q.mean(1)

class TurboVisionTransformer(VisionTransformer):
   

    def forward(self, *args, **kwdargs) -> torch.Tensor:
    
        self._turbo_info["size"] = None
        self._turbo_info["source"] = None
        self._turbo_info["rel_pos"] = None
        self._turbo_info["selected_layers"] = list(self.selected_layers)
        self._turbo_info["window_size"] = self.window_size
        self._turbo_info["threshold"] = self.threshold
       


        return super().forward(*args, **kwdargs)


def apply_patch(
    model: ViT,
    selected_layers: list = [1, 7],
    trace_source: bool = False,
    prop_attn: bool = True,
):

    model = model.backbone
    model.__class__ = TurboVisionTransformer
    
    model.selected_layers = list(selected_layers)
    model.window_size = (2,2)
    model.threshold = 0.88
    model._turbo_info = {
        "size": None,
        "source": None,
        "trace_source": trace_source,
        "prop_attn": prop_attn,
        "class_token": model.cls_token is not None,
        "distill_token": False,
        "rel_pos": None,
        "selected_layers": list(model.selected_layers),
        "window_size":model.window_size,
        "threshold":model.threshold,
    }

    if hasattr(model, "dist_token") and model.dist_token is not None:
        model._turbo_info["distill_token"] = True


    print(model.selected_layers)
    len_att = 1
    len_block = 1
    for module in model.modules():
        if isinstance(module, Block):
            if len_att in model.selected_layers:
                module.__class__ = TurboBlock
                module._turbo_info = model._turbo_info
                module._turbo_layer_idx = len_att
            len_att +=1 
        elif isinstance(module, Attention):
            if len_block in model.selected_layers: 
                module.__class__ = TurboAttention
                module._turbo_info = model._turbo_info
            len_block +=1 

