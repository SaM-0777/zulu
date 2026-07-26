from typing import List, Optional, Union
import torch
import torch.nn as nn
from transformers import AutoImageProcessor, AutoModel  # type: ignore
from PIL import Image
from pathlib import Path

DEFAULT_MODEL_BASE_PATH = Path("models/dinov2breg/").resolve()


"""
DinoV2 
"attention_probs_dropout_prob": 0.0,
  "drop_path_rate": 0.0,
  "dtype": "float32",
  "hidden_act": "gelu",
  "hidden_dropout_prob": 0.0,
  "hidden_size": 768,
  "image_size": 518,
  "initializer_range": 0.02,
  "interpolate_antialias": true,
  "interpolate_offset": 0.0,
  "layer_norm_eps": 1e-06,
  "layerscale_value": 1.0,
  "mlp_ratio": 4,
  "model_type": "dinov2_with_registers",
  "num_attention_heads": 12,
  "num_channels": 3,
  "num_hidden_layers": 12,
  "num_register_tokens": 4,
  "out_features": [
    "stage12"
  ],
  "out_indices": [
    12
  ],
  "patch_size": 14,
  "qkv_bias": true,
  "reshape_hidden_states": true,
  "stage_names": [
    "stem",
    "stage1",
    "stage2",
    "stage3",
    "stage4",
    "stage5",
    "stage6",
    "stage7",
    "stage8",
    "stage9",
    "stage10",
    "stage11",
    "stage12"
  ],
  "transformers_version": "5.12.1",
  "use_swiglu_ffn": false
"""


class DinoV2Embedding(nn.Module):
    """
    Uses Meta's DINOv2 (with Registers) to compress raw camera frames into
    dense, physically accurate latent vectors.
    """

    def __init__(
        self,
        dtype: torch.dtype,
        frame_size: int = 224,  # default 224
        dimensions: int = 768,
        model_base_path: Optional[Union[str, Path]] = None,
        return_patches: bool = False,
    ) -> None:
        super(DinoV2Embedding, self).__init__()
        self.model_path = model_base_path or DEFAULT_MODEL_BASE_PATH
        self.frame_size = frame_size
        self.dimensions = dimensions
        self.return_patches = return_patches

        self.processor = AutoImageProcessor.from_pretrained(
            self.model_path,
            backend="torchvision",
            do_normalize=True,
            do_scale=True,
            do_resize=True,
            size={"height": 154, "width": 308},
        )
        self.model = AutoModel.from_pretrained(
            self.model_path,
            torch_dtype=dtype,
        )

        expected_dimensions = self.model.config.hidden_size
        if self.dimensions != expected_dimensions:
            raise ValueError(
                f"Requested dimesntion {self.dimensions} differs from model native {expected_dimensions}"
            )

        for param in self.model.parameters():
            param.requires_grad = False  # Freeze the all models parameters
        self.model.eval()

    def train(self, mode: bool = True):
        """Safety: Prevent main training loop from accidentaly unfreezing DINOv2's Dropout / LayerNorm layers when `model.train()` is called"""
        super().train(mode=mode)
        self.model.eval()
        return self

    def preprocess_images(
        self, frames: Union[Image.Image, List[Image.Image]]
    ) -> torch.Tensor:
        """
        Helper method to convert raw PIL Images into the pre-processed PyTorch tensor
        expected by the forward pass.
        """
        inputs = self.processor(images=frames, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.model.device)
        return pixel_values

    @torch.no_grad()
    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        frames = frames.to(self.model.device)
        outputs = self.model(pixel_values=frames)
        last_hidden_state = outputs.last_hidden_state

        if self.return_patches:
            """
            Extract only the spatial patches (Ignore the 0th CLS token)
            Shape: [Batch, num_patches, hidden_dim]
            Model is loaded with registers
            """
            num_registers = getattr(self.model.config, "num_register_tokens", 0)
            start_idx = 1 + num_registers
            return last_hidden_state[:, start_idx:, :]
        else:
            """
            Extract only the CLS token, preserving the sequence dimension
            Shape: [Batch, 1, hidden_dim]
            """
            return last_hidden_state[:, 0:1, :]
