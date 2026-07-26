import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.embeds.embodiment_encoder import SinusoidalPositionalEncoding, swish



class CategorySpecificLinear(nn.Module):
    def __init__(
        self,
        num_categories: int,
        input_dim: int,
        hidden_dim: int,
    ):
        super(CategorySpecificLinear, self).__init__()
        self.num_categories = num_categories
        self.W = nn.Parameter(0.02 * torch.randn(num_categories, input_dim, hidden_dim))
        self.b = nn.Parameter(torch.zeros(num_categories, hidden_dim))
    
    def forward(self, x: torch.Tensor, cat_ids: torch.Tensor) -> torch.Tensor:
        target_device = self.W.data.device
        
        selected_W = self.W[cat_ids.to(target_device)]
        selected_b = self.b[cat_ids.to(target_device)]
        
        selected_W = selected_W.to(device=x.device, dtype=x.dtype)
        selected_b = selected_b.to(device=x.device, dtype=x.dtype)

        return torch.bmm(x, selected_W) + selected_b.unsqueeze(1)
        
# state encoder, # action decoder
class CategorySpecificMLP(nn.Module):
    def __init__(
        self,
        num_categories: int,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
    ):
        super(CategorySpecificMLP, self).__init__()
        self.num_categories = num_categories
        self.layer1 = CategorySpecificLinear(num_categories=num_categories, input_dim=input_dim, hidden_dim=hidden_dim)
        self.layer2 = CategorySpecificLinear(num_categories=num_categories, input_dim=hidden_dim, hidden_dim=output_dim)
    
    def forward(self, x: torch.Tensor, cat_ids: torch.Tensor) -> torch.Tensor:
        hidden = F.relu(self.layer1(x, cat_ids))
        return self.layer2(hidden, cat_ids)

# action encoder
class MultiEmbodimentActionEncoder(nn.Module):
    def __init__(self, action_dim, hidden_size, num_embodiments):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_embodiments = num_embodiments

        # W1: R^{w x d}, W2: R^{w x 2w}, W3: R^{w x w}
        self.W1 = CategorySpecificLinear(num_embodiments, action_dim, hidden_size)  # (d -> w)
        self.W2 = CategorySpecificLinear(num_embodiments, 2 * hidden_size, hidden_size)  # (2w -> w)
        self.W3 = CategorySpecificLinear(num_embodiments, hidden_size, hidden_size)  # (w -> w)
        self.pos_encoding = SinusoidalPositionalEncoding(hidden_size)

    def forward(self, actions, timesteps, cat_ids):
        """
        actions:   shape (B, T, action_dim)
        timesteps: shape (B,)  -- a single scalar per batch item
        cat_ids:   shape (B,)
        returns:   shape (B, T, hidden_size)
        """
        B, T, _ = actions.shape

        # Standard action MLP step for shape => (B, T, w)
        timesteps = timesteps.to(device=actions.device)
        a_emb = self.W1(actions, cat_ids)

        # 3) Get the sinusoidal encoding (B, T, w)
        tau_emb = self.pos_encoding(timesteps).to(dtype=a_emb.dtype)

        # 4) Concat along last dim => (B, T, 2w), then W2 => (B, T, w), swish
        x = torch.cat([a_emb, tau_emb], dim=-1)
        x = swish(self.W2(x, cat_ids))

        # 5) Finally W3 => (B, T, w)
        x = self.W3(x, cat_ids)
        return x
