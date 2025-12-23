import torch
import torch.nn as nn

class LearnableUnmergeLayered(nn.Module):
    def __init__(self, dim):
        super().__init__()
        
        # time embedding（类似 diffusion）
        self.time_proj = nn.Sequential(
            nn.Linear(1, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )

    def forward(self, x_merged, mask, layer_ratio):
        """
        x_merged:   [B, K, C]
        mask:       [B, K, N]
        layer_ratio: float in [0,1], e.g. layer_idx / total_layers
        """
        B, K, C = x_merged.shape
        
        # 1) embed t
        t = torch.tensor([[layer_ratio]], device=x_merged.device)  # [1,1]
        t_emb = self.time_proj(t)                                # [1, dim]
        t_emb = t_emb.view(1, 1, C)                              # broadcast
        
        # 2) conditioning
        c = x_merged + t_emb                                     # [B,K,C]

        # 3) learnable transform
        c_proj = self.decoder(c)                                 # [B,K,C]

        # 4) mask-guided re-projection（保持拓扑）
        x_unmerged = torch.einsum('bkn,bkc->bnc', mask, c_proj)

        return x_unmerged
