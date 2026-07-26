import torch
import torch.nn as nn
import torch.nn.functional as F


def swish(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(
        self,
        dimension: int,
    ):
        super(SinusoidalPositionalEncoding, self).__init__()
        self.dimensions = dimension

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        timesteps = timesteps.float()

        B, T = timesteps.shape
        device = timesteps.device

        half_dim = self.dimensions // 2
        exponent = -torch.arange(half_dim, dtype=torch.float, device=device) * (
            torch.log(torch.tensor(10000.0)) / half_dim
        )
        freqs = timesteps.unsqueeze(-1) * exponent.exp()

        sin = torch.sin(freqs)
        cos = torch.cos(freqs)
        enc = torch.cat([sin, cos], dim=-1)
        return enc


class EmbodimentSpecificLinear(nn.Module):
    """Linear layer with category specific weights and biases for multi-embodiment support"""

    def __init__(
        self, num_categories: int, input_dimensions: int, hidden_dimensions: int
    ):
        super(EmbodimentSpecificLinear, self).__init__()
        self.num_categories = num_categories
        self.W = nn.Parameter(
            0.02 * torch.randn(num_categories, input_dimensions, hidden_dimensions)
        )
        self.b = nn.Parameter(torch.zeros(num_categories, hidden_dimensions))

    def forward(self, x: torch.Tensor, category_ids: torch.Tensor) -> torch.Tensor:
        """
        Input:
            x: [B, T, input_dimension]
            category_ids: [B]

        Returns:
            [B, T, hidden_dimension]
        """
        selected_W = self.W[category_ids]
        selected_b = self.b[category_ids]
        return torch.bmm(x, selected_W) + selected_b.unsqueeze(1)


class EmbodimentSpecificMLP(nn.Module):
    def __init__(
        self,
        num_categories: int,
        input_dimensions: int,
        hidden_dimension: int = 1536,
        output_dimensions: int = 1536,
        dropout: float = 0.1,
    ):
        super(EmbodimentSpecificMLP, self).__init__()
        self.num_categories = num_categories
        self.input_dimensions = input_dimensions
        self.hidden_dimensions = hidden_dimension
        self.output_dimensions = output_dimensions
        self.layer1 = EmbodimentSpecificLinear(
            num_categories=num_categories,
            input_dimensions=input_dimensions,
            hidden_dimensions=hidden_dimension,
        )
        #self.norm = nn.LayerNorm(hidden_dimension)
        #self.drop = nn.Dropout(dropout)
        self.layer2 = EmbodimentSpecificLinear(
            num_categories=num_categories,
            input_dimensions=hidden_dimension,
            hidden_dimensions=output_dimensions,
        )

    def pad_to_universal_dimension(self, state_vector: torch.Tensor) -> torch.Tensor:
        current_dimension = state_vector.shape[-1]
        if current_dimension > self.input_dimensions:
            raise ValueError(
                f"State dimensions ({current_dimension}) exceeds hidden dimensions ({self.input_dimensions})"
            )
        elif current_dimension < self.input_dimensions:
            pad_size = self.input_dimensions - current_dimension
            pad_shape = list(state_vector.shape)
            pad_shape[-1] = pad_size
            padding = torch.zeros(
                pad_shape, dtype=state_vector.dtype, device=state_vector.device
            )
            state_vector = torch.cat([state_vector, padding], dim=-1)
        return state_vector

    def forward(self, x: torch.Tensor, category_ids: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.layer1(x, category_ids))
        x = self.layer2(x, category_ids)
        return x


class MultiEmbodimentActionEncoder(nn.Module):
    def __init__(
        self, action_dimenstions: int, hidden_dimenstions: int, num_embodiments: int
    ):
        super(MultiEmbodimentActionEncoder, self).__init__()
        self.hidden_dimensions = hidden_dimenstions
        self.num_embodiments = num_embodiments
        self.W1 = EmbodimentSpecificLinear(
            num_categories=num_embodiments,
            input_dimensions=action_dimenstions,
            hidden_dimensions=hidden_dimenstions,
        )
        self.W2 = EmbodimentSpecificLinear(
            num_categories=num_embodiments,
            input_dimensions=2 * hidden_dimenstions,
            hidden_dimensions=hidden_dimenstions,
        )
        self.W3 = EmbodimentSpecificLinear(
            num_categories=num_embodiments,
            input_dimensions=hidden_dimenstions,
            hidden_dimensions=hidden_dimenstions,
        )
        self.position_encoding = SinusoidalPositionalEncoding(
            dimension=hidden_dimenstions
        )

    def forward(
        self, actions: torch.Tensor, timesteps: torch.Tensor, category_ids: torch.Tensor
    ):
        B, T, _ = actions.shape
        if timesteps.dim() == 1 and timesteps.shape[0] == B:
            timesteps = timesteps.unsqueeze(1).expand(-1, T)
        else:
            raise ValueError(f"Expected timesteps to have shape (B,)")
        action_embed = self.W1(actions, category_ids)
        tau_embed = self.position_encoding(timesteps).to(dtype=action_embed.dtype)
        x = torch.cat([action_embed, tau_embed], dim=-1)
        x = swish(self.W2(x, category_ids))
        x = self.W3(x, category_ids)
        return x
