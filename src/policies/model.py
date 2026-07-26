from pathlib import Path
import time
from typing import List, Literal, Optional, Tuple
from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as Fn
from torch.distributions import Beta
import torchvision.io as tv_io
from torchvision.transforms import v2
from transformers.feature_extraction_utils import BatchFeature

from src.embeds.dinov2 import DinoV2Embedding
from src.embeds.text_encoder import TextEncoder

from src.policies.flow_unipc_multistep_scheduler import FlowUniPCMultistepScheduler
from src.policies.scheduler import FlowMatchScheduler
from src.policies.dit import DiTBackbone


class Model(nn.Module):
    _keys_to_ignore_on_save = None

    def __init__(
        self,
        # device: torch.device,
        dtype: torch.dtype,
        # vision
        patch_size: int = 14,
        frame_size: int = 224,
        dino_dim: int = 768,
        # text
        text_len: int = 200,
        text_dim: int = 768,
        # state
        max_state_dim: int = 64,
        # action
        action_horizon: int = 24,
        action_dim: int = 32,
        freq_dim: int = 256,
        num_embodiments: int = 1,
        # model
        eps: float = 1e-06,
        in_dim: int = 768,
        out_dim: int = 768,
        ffn_dim: int = 4096,
        dim: int = 2048,
        hidden_size: int = 1024,
        num_heads: int = 16,
        num_layers: int = 32,
        num_action_per_block: int = 32,
        num_state_per_block: int = 1,
        num_frame_per_block: int = 1,
        max_chunk_size: int = 4,
        # noise
        frame_noise_beta_alpha: float = 1.5,
        frame_noise_beta_beta: float = 1.0,
        high_noise_beta_alpha: float = 3.0,
        decouple_frame_action_noise: bool = True,
        use_high_noise_emphasis: bool = False,
        dynamics_loss_weight: float = 1.0,
        action_loss_weight: float = 1.0,
        use_residual_frame_prediction: bool = True,
    ):
        super(Model, self).__init__()
        # self.device = device
        self.dtype = dtype
        self.frame_noise_beta_alpha = frame_noise_beta_alpha
        self.frame_noise_beta_beta = frame_noise_beta_beta
        self.high_noise_beta_alpha = high_noise_beta_alpha
        self.decouple_frame_action_noise = decouple_frame_action_noise
        self.use_high_noise_emphasis = use_high_noise_emphasis
        self.dynamics_loss_weight = dynamics_loss_weight
        self.action_loss_weight = action_loss_weight
        self.use_residual_frame_prediction = use_residual_frame_prediction
        self.num_train_timesteps = 100
        self.num_frame_per_block = num_frame_per_block  # 8
        self.noise_not_logged = False
        self.num_inference_steps = 50 # was 16
        self.max_chunk_size = max_chunk_size
        self.action_horizon = action_horizon

        self.frame_size = frame_size
        self.frame_h = 154
        self.frame_w = 308
        self.patch_size = patch_size
        self.dino_dim = dino_dim

        self.text_len = text_len
        self.text_dim = text_dim

        self.max_state_dim = max_state_dim

        self.action_dim = action_dim

        self.dim = dim
        self.hidden_size = hidden_size

        self.freq_dim = freq_dim

        self.num_layers = num_layers

        self.cfg_scale = 1.0
        self.current_start_frame = 0
        self.kv_caches = None
        self.crossattn_caches = None

        self.scheduler = FlowMatchScheduler(
            shift=5,
            #sigma_min=0.0,
            extra_one_step=True,
        )
        if self.training:  # keep it unconditional
            self.scheduler.set_timesteps(1000, training=True)

        self.vision_encoder = DinoV2Embedding(
            dtype=dtype,
            frame_size=self.frame_size,
            dimensions=self.dino_dim,
            return_patches=True,
        )
        self.vision_encoder.model = self.vision_encoder.model
        self.grid_h = self.frame_h // self.patch_size  # 11
        self.grid_w = self.frame_w // self.patch_size  # 22
        self.tokens_per_frame = self.grid_h * self.grid_w # 242

        self.text_encoder = TextEncoder(
            dtype=dtype,
            dim=text_dim,
            max_length=text_len,
        )

        self.normalize_frames = v2.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

        self.dit_backbone = DiTBackbone(
            patch_size=(1, 1, 1),
            frame_seq_len=self.tokens_per_frame,
            text_len=text_len,
            text_dim=text_dim,
            in_dim=in_dim,
            out_dim=out_dim,
            dim=dim,
            ffn_dim=ffn_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            action_dim=action_dim,
            max_state_dim=max_state_dim,
            max_num_embodiment=num_embodiments,
            num_frame_per_block=num_frame_per_block,
            num_action_per_block=num_action_per_block,
            num_state_per_block=num_state_per_block,
            max_chunk_size=max_chunk_size,
            eps=eps,
            hidden_size=hidden_size,
        )

        self.frame_beta_dist = Beta(frame_noise_beta_alpha, frame_noise_beta_beta)
        self.high_noise_beta_dist = Beta(high_noise_beta_alpha, 1.0)

    def load_state_dict(
        self, state_dict, strict: bool = True, assign: bool = False, **kwargs
    ):
        # 1. Create a mapping of cleaned keys to actual model keys
        # This handles both "_orig_mod." (torch.compile) and "module." (DDP) prefixes
        model_state_dict = self.state_dict()
        clean_to_actual_key = {}
        for k in model_state_dict.keys():
            clean_k = k
            if clean_k.startswith("module."):
                clean_k = clean_k[7:]
            if clean_k.startswith("_orig_mod."):
                clean_k = clean_k[10:]
            # Also handle if it's nested like dit_backbone._orig_mod.
            clean_k = clean_k.replace("._orig_mod.", ".")
            clean_to_actual_key[clean_k] = k

        print("clean_to_actual_key ", list(clean_to_actual_key.keys())[:5], " ... ", list(clean_to_actual_key.keys())[-5:])

        # 2. Process the checkpoint state_dict
        new_state_dict = {}
        i = 0
        for k, v in state_dict.items():
            if not isinstance(v, torch.Tensor):
                i = i + 1
                continue
                
            clean_k = k
            if clean_k.startswith("module."):
                clean_k = clean_k[7:]
            if clean_k.startswith("_orig_mod."):
                clean_k = clean_k[10:]
            clean_k = clean_k.replace("._orig_mod.", ".")
            
            # If the cleaned key exists in the model, use the ACTUAL model key
            if clean_k in clean_to_actual_key:
                actual_k = clean_to_actual_key[clean_k]
                
                # Check for shape mismatches
                if v.shape == model_state_dict[actual_k].shape:
                    new_state_dict[actual_k] = v
                else:
                    print(f"⚠️ Shape mismatch for {actual_k}: checkpoint {v.shape} vs model {model_state_dict[actual_k].shape}. Skipping.")

        print("new_state_dict keys ", list(new_state_dict.keys())[:5], " ... ", list(new_state_dict.keys())[-5:])
        
        # 3. Load the filtered state dict
        load_result = super().load_state_dict(new_state_dict, strict=False, **kwargs)

        if len(load_result.missing_keys) > 0:
            print(
                f"✅ Model successfully resumed. Safely ignored {len(load_result.missing_keys)} "
                "missing keys belonging to stripped frozen backbones."
            )

        from torch.nn.modules.module import _IncompatibleKeys
        
        # Calculate truly unexpected keys (keys in checkpoint that didn't match anything)
        # unexpected = [k for k in state_dict.keys() if k not in new_state_dict and isinstance(state_dict[k], torch.Tensor)]
        return _IncompatibleKeys(
            missing_keys=load_result.missing_keys, unexpected_keys=load_result.unexpected_keys
        )

    def forward(
        self,
        feature_input: dict,
    ):
        frames = feature_input["images"]  # [B, T, H, W, C]
        actions = feature_input["action"]
        states = feature_input["state"]
        text = feature_input["text"]
        text_attention_mask = feature_input["text_attention_mask"]
        embodiment_id = feature_input["embodiment_id"]
        action_mask = feature_input["action_mask"]
        has_real_action = feature_input["has_real_action"]

        embodiment_id = torch.zeros_like(embodiment_id)

        B, F, H, W, C = frames.shape
        assert (F - 1) % self.num_frame_per_block == 0, (
            f"(num_frames - 1) = {F-1} must be divisible by "
            f"num_frame_per_block = {self.num_frame_per_block}. "
            f"Got remainder {(F-1) % self.num_frame_per_block}."
        )

        if actions is not None:
            actions = actions.to(self.dtype)
        if states is not None:
            states = states.to(dtype=self.dtype)

        text_embedding = self.text_encoder(text, text_attention_mask)
        context_emb = text_embedding["text_embeddings"]  # [B, text_len, text_dim]

        frames = rearrange(frames, "b t h w c -> b t c h w")

        # create dino tokens
        latents = self._prepare_dino_latents(frames).to(
            self.dtype
        )  # [B, 768, F, 16, 16]
        _, dino_dim, F_d, grid_h, grid_w = latents.shape
        dino_features = latents.permute(0, 2, 3, 4, 1).reshape(
            B, F_d * grid_h * grid_w, dino_dim
        )

        seq_len = F * self.tokens_per_frame
        timestep_dino = torch.zeros(B, F, dtype=torch.long, device=frames.device)

        noise_action: torch.Tensor | None = None
        timestep_action_id: torch.Tensor | None = None
        timestep_action: torch.Tensor | None = None
        noisy_actions: torch.Tensor | None = None
        training_target_action: torch.Tensor | None = None

        if actions is not None and actions.numel() > 0:
            noise_action = torch.randn_like(actions) # B A a_dim 1 96 32
            timestep_action_id = torch.randint(
                0,
                self.scheduler.num_train_timesteps,
                (actions.shape[0], actions.shape[1]),
            )
            timestep_action = self.scheduler.timesteps[timestep_action_id]
            noisy_actions = self.scheduler.add_noise(
                actions.flatten(0, 1),
                noise_action.flatten(0, 1),
                timestep_action.flatten(0, 1),
            ).unflatten(0, (actions.shape[0], actions.shape[1]))
            training_target_action = self.scheduler.training_target(
                actions, noise_action, timestep_action
            )

        if not self.noise_not_logged:
            action_mean = (
                timestep_action_id.float().mean().item()
                if timestep_action_id is not None
                else 0.0
            )
            print(
                f"[NOISE] DINO tokens: CLEAN (t=0), "
                f"Action: INDEPENDENT Uniform mean_t = {action_mean:.0f}"
            )
            self.noise_not_logged = True

        device_type = next(self.parameters()).device.type
        with torch.amp.autocast(dtype=self.dtype, device_type=device_type):
            if actions is not None and actions.numel() > 0:
                frame_pred, action_noise_pred = self.dit_backbone(
                    x=dino_features,
                    timestep=timestep_dino,
                    timestep_action=timestep_action.long(),
                    context=context_emb,
                    seq_len=seq_len,
                    grid_size=(F, grid_h, grid_w),  #
                    action=(
                        noisy_actions.to(self.dtype)
                        if noisy_actions is not None
                        else None
                    ),
                    state=states,
                    embodiment_id=embodiment_id,
                )
            else:
                frame_pred, action_noise_pred = self.dit_backbone(
                    x=dino_features,
                    timestep=timestep_dino,
                    timestep_action=timestep_action.long(),
                    context=context_emb,
                    seq_len=seq_len,
                    grid_size=(F, grid_h, grid_w),  #
                    action=None,
                    state=states,
                    embodiment_id=embodiment_id,
                )

        tpf = self.tokens_per_frame # 242
        pred_frames = frame_pred[:, : (F - 1) * tpf, :]  # [B, (F-1)*tpf, dino_dim]
        raw_target_frames = dino_features[:, tpf:, :]  # [B, (F-1)*tpf, dino_dim]
        target_frames = self.dit_backbone.dino_input_norm(raw_target_frames)

        dynamics_loss_per_token = torch.nn.functional.mse_loss(
            pred_frames.float(),
            target_frames.float(),
            reduction="none",
        ).mean(
            dim=-1
        )  # [B, (F-1)*tpf] — mean over channel dim
        dynamics_loss = dynamics_loss_per_token.mean()

        loss: torch.Tensor | None = None
        action_loss = torch.tensor(0.0, device=frames.device)

        if (
            actions is not None
            and actions.numel() > 0
            and action_noise_pred is not None
            and training_target_action is not None
        ):    
            action_loss_per_sample = torch.nn.functional.mse_loss(
                action_noise_pred.float(),
                training_target_action.float(),
                reduction="none",
            )
            device = action_loss_per_sample.device

            if action_mask is not None:
                mask = action_mask.to(device)
                mask = (
                    action_mask.unsqueeze(-1)
                    if action_mask.dim() == 2
                    else action_mask
                )
                action_loss_per_sample = action_loss_per_sample * mask
                mask_sum = mask.sum(dim=-1).clamp(min=1)  # Shape: [B, H], values = 8
                action_loss_per_sample = action_loss_per_sample.sum(dim=-1) / mask_sum  # Shape: [B, H]
                #action_loss_per_sample = action_loss_per_sample.mean(dim=-1)

                if has_real_action is not None:
                    real_action_indicator = has_real_action.to(device)
                    action_loss_per_sample = (
                        real_action_indicator[:, None].float()
                        * action_loss_per_sample
                    )

                weight_action_vec = self.scheduler.training_weight(
                    timestep_action.flatten(0, 1)  # type: ignore
                ).to(device)
                
                weight_action = action_loss_per_sample * weight_action_vec.unflatten(
                    0, (noise_action.shape[0], noise_action.shape[1])  # type: ignore
                )

                action_loss = weight_action.mean()

        loss = (
            self.dit_backbone.dynamics_loss_weight * dynamics_loss
            + self.dit_backbone.action_loss_weight * action_loss
        )
        
        return BatchFeature(
            data={
                "loss": loss,
                "dynamic_loss": dynamics_loss,
                "action_loss": action_loss,
            }
        )

    def _prepare_dino_latents(self, frames: torch.Tensor) -> torch.Tensor:
        B, F, C, H, W = frames.shape
        P = self.patch_size  # 14

        if frames.dtype == torch.uint8:
            frames = frames.float() / 255.0

        self.grid_h = self.frame_h // P
        self.grid_w = self.frame_w // P

        if H != self.frame_h or W != self.frame_w:
            frames = Fn.interpolate(
                frames.view(B * F, C, H, W),
                size=(self.frame_h, self.frame_w),
                mode="bilinear",
                align_corners=False,
            ).view(B, F, C, self.frame_h, self.frame_w)

        flat_frames = frames.view(B * F, C, self.frame_h, self.frame_w)

        # process in chunks to prevent CUDA OOM
        chunk_size = 8
        patch_features_list = []
        for i in range(0, B * F, chunk_size):
            chunk = flat_frames[i : i + chunk_size]
            # chunk = chunk.to("cpu")
            chunk_features = self.vision_encoder(chunk)
            # chunk_features = chunk_features.to(frames.device)  # back to GPU
            patch_features_list.append(chunk_features)

        patch_features = torch.concat(patch_features_list, dim=0)
        # patch_features = self.vision_encoder(flat_frames)
        bf, num_patches, d = patch_features.shape
        patch_features = patch_features.transpose(1, 2).view(
            bf, d, self.grid_h, self.grid_w
        )
        return patch_features.view(B, F, d, self.grid_h, self.grid_w).transpose(1, 2)

    #def _run_diffusion_steps(
    #    self,
    #    x: torch.Tensor,
    #    timestep: torch.Tensor,
    #    timestep_action: Optional[torch.Tensor],
    #    state: torch.Tensor,
    #    embodiment_id: torch.Tensor,
    #    contexts: list[torch.Tensor],
    #    seq_len: int,
    #    grid_size: Tuple[int, int, int],
    #    kv_caches: list[list[torch.Tensor]],
    #    crossattn_caches: list[list[dict]],
    #    kv_cache_metadata: dict[str, bool | int],
    #    action: Optional[torch.Tensor],
    #):
    #    predictions = []
    #    for index, prompt_emb in enumerate(contexts):
    #        kv_cache = kv_caches[index]
    #        crossattn_cache = crossattn_caches[index]
    #        with torch.no_grad():
    #            frame_pred, action_noise_pred, updated_kv_caches = self.dit_backbone(
    #                x=x,
    #                timestep=timestep,
    #                timestep_action=timestep_action,
    #                context=prompt_emb,
    #                seq_len=seq_len,
    #                grid_size=grid_size,
    #                kv_cache=kv_cache,
    #                crossattn_cache=crossattn_cache,
    #                current_start_frame=kv_cache_metadata["start_frame"],
    #                action=action,
    #                state=state,
    #                embodiment_id=embodiment_id,
    #            )

    #        if kv_cache_metadata["update_kv_cache"]:
    #            for block_index, updated_kv_cache in enumerate(updated_kv_caches):
    #                kv_cache[block_index] = updated_kv_cache.clone()

    #        frame_pred = frame_pred.clone()
    #        if action_noise_pred is not None:
    #            action_noise_pred = action_noise_pred.clone()
    #        else:
    #            action_noise_pred = torch.tensor(0.0, device=frame_pred.device)

    #        predictions.append((frame_pred, action_noise_pred))

    #    return predictions

    # def generate_noise(self, shape, seed=None, device="cpu", dtype=torch.float16):
    #    generator = None if seed is None else torch.Generator(device).manual_seed(seed)
    #    noise = torch.randn(shape, generator=generator, device=device, dtype=dtype)
    #    return noise

    def lazy_joint_frame_action(self, feature_input: dict):
        self.vision_encoder.eval()
        self.text_encoder.eval()

        frames = feature_input["images"]  # [B, T, H, W, C]
        text = feature_input["text"]
        text_attention_mask = feature_input["text_attention_mask"]
        text_negative = feature_input["text_negative"] # can be None
        text_attention_mask_negative = feature_input["text_attention_mask_negative"] # can be None
        embodiment_id = feature_input["embodiment_id"]
        states = feature_input["state"]

        embodiment_id = torch.zeros_like(embodiment_id)
        frames = rearrange(frames, "b t h w c -> b t c h w")
        states = states.to(dtype=self.dtype)

        B = frames.shape[0]
        device = frames.device
        tpf = self.tokens_per_frame
        grid_h = self.grid_h
        grid_w = self.grid_w

        # self.max_context_frames = 33  # from config

        text_inputs = [(text, text_attention_mask)]
        if self.cfg_scale > 1.0 and text_negative is not None:
            text_inputs.append((text_negative, text_attention_mask_negative))
        prompt_embs = [
            self.text_encoder(t, m)["text_embeddings"] for t, m in text_inputs
        ]

        with torch.no_grad():
            latents = self._prepare_dino_latents(frames=frames).to(self.dtype)
            _, dino_dim, F_d, gh, gw = latents.shape
            dino_tokens = latents.permute(0, 2, 3, 4, 1).reshape(
                B, F_d * gh * gw, dino_dim
            )
        F_ctx = frames.shape[1]  # 33

        num_frame_per_block = self.dit_backbone.num_frame_per_block  # 8
        num_action_per_block = self.dit_backbone.num_action_per_block  # 24
        num_state_per_block = self.dit_backbone.num_state_per_block  # 1
        num_image_blocks = (F_ctx - 1) // num_frame_per_block  # 4
        action_horizon = num_image_blocks * num_action_per_block  # 96
        state_horizon = num_image_blocks * num_state_per_block  # 4

        noisy_input_action = torch.randn(
            (B, action_horizon, self.action_dim), device=device, dtype=self.dtype
        )
        
        # 🎯 CRITICAL FIX 1: Silence the 24 padded dimensions
        physical_action_dim = 8 # 7 joints + 1 gripper
        noisy_input_action[..., physical_action_dim:] = 0.0
        
        sample_scheduler = FlowUniPCMultistepScheduler(
            num_train_timesteps=self.scheduler.num_train_timesteps,  # 1000
            shift=1,
            use_dynamic_shifting=False,
            solver_order=2,
            prediction_type="flow_prediction",
        )
        sample_scheduler.set_timesteps(
            self.num_inference_steps, device=device, shift=5.0
        )

        for index, action_timestep in enumerate(sample_scheduler.timesteps):
            timestep_dino = torch.zeros(B, F_ctx, dtype=torch.int64, device=device)
            timestep_action = torch.ones(
                [B, action_horizon],
                device=device,
                dtype=torch.int64,
            ) * int(action_timestep)

            predictions = []
            for prompt_emb in prompt_embs:
                with torch.no_grad():
                    frame_pred, action_noise_pred = self.dit_backbone(
                        x=dino_tokens,
                        timestep=timestep_dino,
                        timestep_action=timestep_action,
                        context=prompt_emb,
                        seq_len=F_ctx * tpf,
                        grid_size=(F_ctx, grid_h, grid_w),
                        action=noisy_input_action,
                        state=states,
                        embodiment_id=embodiment_id,
                    )
                predictions.append((frame_pred, action_noise_pred))

            if self.cfg_scale > 1.0 and len(predictions) > 1:
                _, action_cond = predictions[0]
                _, action_uncond = predictions[1]
                action_noise_pred = action_uncond + self.cfg_scale * (
                    action_cond - action_uncond
                )
            else:
                _, action_noise_pred = predictions[0]

            noisy_input_action = sample_scheduler.step(
                model_output=action_noise_pred,
                timestep=action_timestep,
                sample=noisy_input_action,
                step_index=index,
                return_dict=False,
            )[0]

            noisy_input_action[..., physical_action_dim:] = 0.0

        latents_action = noisy_input_action

        timestep_dino_clean = torch.zeros(B, F_ctx, dtype=torch.int64, device=device)
        timestep_action_clean = torch.zeros(
            B, action_horizon, dtype=torch.int64, device=device
        )

        predictions = []
        for prompt_emb in prompt_embs:
            with torch.no_grad():
                frame_pred, _ = self.dit_backbone(
                    x=dino_tokens,  # ALL 33 frames
                    timestep=timestep_dino_clean,
                    timestep_action=timestep_action_clean,
                    context=prompt_emb,
                    seq_len=F_ctx * tpf,
                    grid_size=(F_ctx, grid_h, grid_w),
                    action=latents_action,  # Clean sampled action
                    state=states,
                    embodiment_id=embodiment_id,
                )
            predictions.append(frame_pred)

        frame_pred = predictions[0]
        next_frame_tokens = frame_pred[:, -tpf:, :]

        return BatchFeature(
            data={
                "action_pred": latents_action,
                "frame_pred": next_frame_tokens,
            }
        )

    # deprecated and irrelevant
    #def ssss(self, feature_input: dict):
    #    self.vision_encoder.eval()
    #    self.text_encoder.eval()

    #    print(f"feature_input {feature_input.keys()}")

    #    frames = feature_input["images"]  # [B, T, H, W, C]
    #    text = feature_input["text"]
    #    text_attention_mask = feature_input["text_attention_mask"]
    #    text_negative = feature_input["text_negative"]
    #    text_attention_mask_negative = feature_input["text_attention_mask_negative"]
    #    embodiment_id = feature_input["embodiment_id"]
    #    states = feature_input["state"]

    #    embodiment_id = torch.zeros_like(embodiment_id)

    #    frames = rearrange(frames, "b t h w c -> b t c h w")

    #    states = states.to(dtype=self.dtype)

    #    B = frames.shape[0]
    #    device = frames.device
    #    tpf = self.tokens_per_frame
    #    grid_h = self.grid_h
    #    grid_w = self.grid_w

    #    if getattr(self, "language", None) is None or self.language is None:
    #        print("language is None, reset current_start_frame to 0")
    #        self.language = text
    #        self.current_start_frame = 0
    #    elif not torch.equal(self.language, text):
    #        print("language changed, reset current_start_frame to 0")
    #        self.current_start_frame = 0
    #        self.language = text
    #    elif frames.shape[1] == 1:
    #        print("frames.shape[1] == 1, reset current_start_frame to 0")
    #        self.current_start_frame = 0
    #    elif self.current_start_frame >= self.dit_backbone.local_attn_size:
    #        print(
    #            "current_start_frame >= local_attn_size, reset current_start_frame to 0"
    #        )
    #        self.current_start_frame = 0

    #    text_inputs = [(text, text_attention_mask)]
    #    if self.cfg_scale != 1.0:
    #        text_inputs.append((text_negative, text_attention_mask_negative))
    #    prompt_embs = [
    #        self.text_encoder(t, m)["text_embeddings"] for t, m in text_inputs
    #    ]

    #    num_frame_per_block = self.dit_backbone.num_frame_per_block  # 8
    #    num_action_per_block = self.dit_backbone.num_action_per_block  # 24
    #    num_state_per_block = self.dit_backbone.num_state_per_block  # 1

    #    self.max_context_frames = 25  # from config
    #    num_image_blocks = (self.max_context_frames - 1) // num_frame_per_block
    #    action_horizon = num_image_blocks * num_action_per_block
    #    state_horizon = num_image_blocks * num_state_per_block

    #    if self.current_start_frame == 0:
    #        F_available = frames.shape[1]

    #        if F_available < self.max_context_frames:
    #            deficit = self.max_context_frames - F_available
    #            last_frame = frames[:, -1:]  # [B, 1, C, H, W]
    #            padding = last_frame.expand(-1, deficit, -1, -1, -1)
    #            frames_context = torch.cat([frames, padding], dim=1)
    #            print(
    #                f"[INIT] Padded {deficit} frames (had {F_available}, need {self.max_context_frames})"
    #            )
    #        elif F_available > self.max_context_frames:
    #            frames_context = frames[:, -self.max_context_frames :]
    #        else:
    #            frames_context = frames

    #        with torch.no_grad():
    #            latents = self._prepare_dino_latents(frames_context).to(self.dtype)
    #            _, dino_dim, F_d, gh, gw = latents.shape
    #            self.dino_tokens = latents.permute(0, 2, 3, 4, 1).reshape(
    #                B, F_d * gh * gw, dino_dim
    #            )

    #        F_ctx = frames_context.shape[1]

    #        self.kv_caches = list(
    #            self.dit_backbone._create_kv_caches(
    #                batch_size=B,
    #                dtype=self.dtype,
    #                device=device,
    #                frame_seqlen=tpf,
    #            )
    #        )
    #        # → [kv_cache1, kv_cache_neg]

    #        self.crossattn_caches = list(
    #            self.dit_backbone._create_crossattn_caches(
    #                batch_size=B,
    #                dtype=self.dtype,
    #                device=device,
    #            )
    #        )
    #        # → [crossattn_cache, crossattn_cache_neg]

    #        kv_caches_for_init = (
    #            self.kv_caches if self.cfg_scale > 1.0 else self.kv_caches[:1]
    #        )
    #        crossattn_for_init = (
    #            self.crossattn_caches
    #            if self.cfg_scale > 1.0
    #            else self.crossattn_caches[:1]
    #        )

    #        timestep_dino = torch.zeros(B, F_ctx, dtype=torch.int64, device=device)
    #        seq_len_init = F_ctx * tpf

    #        self._run_diffusion_steps(
    #            x=self.dino_tokens,
    #            timestep=timestep_dino,
    #            timestep_action=None,
    #            state=states,
    #            embodiment_id=embodiment_id,
    #            contexts=prompt_embs,
    #            seq_len=seq_len_init,
    #            grid_size=(F_ctx, grid_h, grid_w),
    #            kv_caches=kv_caches_for_init,
    #            crossattn_caches=crossattn_for_init,
    #            kv_cache_metadata=dict(start_frame=0, update_kv_cache=True),
    #            action=None,
    #        )

    #        self.current_start_frame = F_ctx

    #    assert self.kv_caches is not None
    #    assert self.crossattn_caches is not None

    #    noisy_input_action = torch.randn(
    #        (B, action_horizon, self.action_dim), device=device, dtype=self.dtype
    #    )

    #    sample_scheduler = FlowUniPCMultistepScheduler(
    #        num_train_timesteps=self.scheduler.num_train_timesteps,
    #        shift=1,
    #        use_dynamic_shifting=False,
    #        solver_order=2,
    #        prediction_type="flow_prediction",
    #    )
    #    sample_scheduler.set_timesteps(
    #        self.num_inference_steps, device=device, shift=5.0
    #    )

    #    last_frame_tokens = self.dino_tokens[:, -tpf:]
    #    F_new = 1

    #    kv_caches_for_diff = (
    #        self.kv_caches if self.cfg_scale > 1.0 else self.kv_caches[:1]
    #    )
    #    crossattn_for_diff = (
    #        self.crossattn_caches if self.cfg_scale > 1.0 else self.crossattn_caches[:1]
    #    )

    #    for index, action_timestep in enumerate(sample_scheduler.timesteps):
    #        timestep_dino_new = torch.zeros(B, F_new, dtype=torch.int64, device=device)
    #        timestep_action = torch.ones(
    #            [B, action_horizon],
    #            device=device,
    #            dtype=torch.int64,
    #        ) * int(action_timestep)

    #        predictions = self._run_diffusion_steps(
    #            x=last_frame_tokens,
    #            timestep=timestep_dino_new,
    #            timestep_action=timestep_action,
    #            state=states,
    #            embodiment_id=embodiment_id,
    #            contexts=prompt_embs,
    #            seq_len=tpf,
    #            grid_size=(F_new, grid_h, grid_w),
    #            kv_caches=kv_caches_for_diff,
    #            crossattn_caches=crossattn_for_diff,
    #            kv_cache_metadata=dict(
    #                start_frame=self.current_start_frame,
    #                update_kv_cache=False,  # READ-ONLY during diffusion
    #            ),
    #            action=noisy_input_action,
    #        )

    #        if self.cfg_scale > 1.0:
    #            _, action_cond = predictions[0]
    #            _, action_uncond = predictions[1]
    #            action_noise_pred = action_uncond + self.cfg_scale * (
    #                action_cond - action_uncond
    #            )
    #        else:
    #            _, action_noise_pred = predictions[0]

    #        noisy_input_action = sample_scheduler.step(
    #            model_output=action_noise_pred,
    #            timestep=action_timestep,
    #            sample=noisy_input_action,
    #            step_index=index,
    #            return_dict=False,
    #        )[0]

    #    latents_action = noisy_input_action

    #    # 7. FRAME PREDICTION (Regression — Update Caches)
    #    timestep_dino_clean = torch.zeros(B, F_new, dtype=torch.int64, device=device)
    #    timestep_action_clean = torch.zeros(
    #        B, action_horizon, dtype=torch.int64, device=device
    #    )

    #    kv_caches_for_frame = (
    #        self.kv_caches if self.cfg_scale > 1.0 else self.kv_caches[:1]
    #    )
    #    crossattn_for_frame = (
    #        self.crossattn_caches if self.cfg_scale > 1.0 else self.crossattn_caches[:1]
    #    )

    #    predictions = self._run_diffusion_steps(
    #        x=last_frame_tokens,
    #        timestep=timestep_dino_clean,
    #        timestep_action=timestep_action_clean,
    #        state=states,
    #        embodiment_id=embodiment_id,
    #        contexts=prompt_embs,
    #        seq_len=tpf,
    #        grid_size=(F_new, grid_h, grid_w),
    #        kv_caches=kv_caches_for_frame,
    #        crossattn_caches=crossattn_for_frame,
    #        kv_cache_metadata=dict(
    #            start_frame=self.current_start_frame,
    #            update_kv_cache=True,  # WRITE
    #        ),
    #        action=latents_action,
    #    )

    #    frame_pred = predictions[0][0]

    #    next_frame_tokens = frame_pred
    #    self.dino_tokens = torch.cat([self.dino_tokens, next_frame_tokens], dim=1)

    #    F_current = self.dino_tokens.shape[1] // tpf
    #    if F_current > self.max_context_frames:
    #        self.dino_tokens = self.dino_tokens[:, -self.max_context_frames * tpf :]

    #    self.current_start_frame += 1
    #    torch.cuda.synchronize()

    #    return BatchFeature(
    #        data={
    #            "action_pred": latents_action,
    #            "frame_pred": next_frame_tokens,
    #        }
    #    )
