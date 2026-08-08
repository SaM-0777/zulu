import json
import os
from pathlib import Path
from typing import Any, Dict
from peft import PeftModel
from src.configs.train import (
    video_keys,
    state_keys,
    action_keys,
    all_modality_configs,
    experiment_conf,
)

from src.data.lerobot import ModalityConfig, build_transform_pipeline
from src.data.schema import DatasetMetadata, EmbodimentTag
from src.data.transforms_base import ComposedModalityTransform
import matplotlib
import matplotlib.markers
import numpy as np
import torch
from tianshou.data import Batch
from src.policies.model import Model
import matplotlib.pyplot as plt
from torchvision.io import write_file
from datetime import datetime

# from tianshou.policy import BasePolicy as BaseTianshouPolicy


def unsqueeze_dict_values(data: dict[str, Any]) -> dict[str, Any]:
    """
    Unsqueeze the values of a dictionary.
    This converts the data to be batched of size 1.
    """
    unsqueezed_data = {}
    for k, v in data.items():
        if isinstance(v, np.ndarray):
            unsqueezed_data[k] = np.expand_dims(v, axis=0)
        elif isinstance(v, list):
            unsqueezed_data[k] = np.array(v)
        elif isinstance(v, torch.Tensor):
            unsqueezed_data[k] = v.unsqueeze(0)
        elif isinstance(v, str):
            unsqueezed_data[k] = np.array([v])
        else:
            unsqueezed_data[k] = v
    return unsqueezed_data


class WAMPolicy:

    def __init__(
        self,
        checkpoint_path: str | Path,
        metadata_json_path: str | Path,
        # stats_json_path: str,
        finetuned_checkpoint_path: str | Path | None = None,
        action_horizon: int = 24,
        embodiment: EmbodimentTag = EmbodimentTag.OXE_DROID,
    ) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.action_horizon = action_horizon
        self.embodiment_tag = embodiment

        # self.experiment = WAMExperiment(experiment_conf)
        # self.model = self.experiment.model
        self.model = self.create_model(experiment_conf)
        self.dtype = self.model.dtype
        self.eval_bf16 = False

        weights_path = os.path.join(checkpoint_path, "pytorch_model.bin")
        print(f"Loading weights from {weights_path}...")
        state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
        self.model.load_state_dict(state_dict, strict=False)
        
        if finetuned_checkpoint_path is not None:
            finetuned_checkpoint_path = Path(finetuned_checkpoint_path)
            print(f"Loading LoRA adapter from {finetuned_checkpoint_path}...")

            self.model = PeftModel.from_pretrained(
                self.model,
                str(finetuned_checkpoint_path),
                is_trainable=False,
            )

            print("Merging LoRA weights into base model for fast inference...")
            self.model = self.model.merge_and_unload()
            #if merge_lora:

        self.model.eval()
        self.model.requires_grad_(False)
        self.model.to(self.device)
        print("Model loaded in EVAL mode.")
        
        self.model.cfg_scale = 1.0

        torch.cuda.empty_cache()

        with open(metadata_json_path, "r") as f:
            metadatas = json.load(f)
        metadata = DatasetMetadata.model_validate(metadatas[embodiment.value])

        assert self.embodiment_tag.value in experiment_conf.get(
            "transforms", {}
        ), f"{self.embodiment_tag.value=}, {experiment_conf.get(
            "transforms", {}
        ).keys()=}"
        eval_transform_cfg = experiment_conf.get("transforms", {})[
            self.embodiment_tag.value
        ]

        eval_transform = build_transform_pipeline(
            video_keys=video_keys,
            state_keys=state_keys,
            action_keys=action_keys,
        )
        assert isinstance(
            eval_transform, ComposedModalityTransform
        ), f"{eval_transform=}"
        eval_transform.set_metadata(metadata)

        # relative_action_per_horizon = experiment_conf.get(
        #    "relative_action_per_horizon", False
        # )
        # if relative_action_per_horizon:
        #    pass

        #eval_transform.eval()
        self.eval_transform = eval_transform

        self.modality_configs: dict[str, ModalityConfig]

        if self.embodiment_tag.value in all_modality_configs:
            self.modality_configs = all_modality_configs[self.embodiment_tag.value]

        self._video_delta_indices = np.array(
            self.modality_configs["video"].delta_indices
        )
        self._video_horizon = len(self._video_delta_indices)

        if "state" in self.modality_configs:
            self._state_delta_indices = np.array(
                self.modality_configs["state"].delta_indices
            )
            self.assert_delta_indices(self._state_delta_indices)
            self._state_horizon = len(self._state_delta_indices)
        else:
            self._state_horizon = None
            self._state_delta_indices = None
        self._raw_data_image_transform = None

        print(f"video_h {self._video_horizon}")
        print(f"state_h {self._state_horizon}")

    def create_model(self, cfg: Dict):
        action_head_cfg = cfg["action_head_cfg"]
        model_dtype = action_head_cfg["model_dtype"]
        num_frames = action_head_cfg["num_frames"]
        num_frame_per_block = action_head_cfg["num_frame_per_block"]
        use_gradient_checkpointing = action_head_cfg["use_gradient_checkpointing"]
        max_state_dim = action_head_cfg["max_state_dim"]
        max_action_dim = action_head_cfg["max_action_dim"]
        hidden_size = action_head_cfg["hidden_size"]
        input_embedding_dim = action_head_cfg["input_embedding_dim"]

        dit_config = action_head_cfg["dit_cfg"]
        dim = dit_config["dim"]
        in_dim = dit_config["in_dim"]
        out_dim = dit_config["out_dim"]
        ffn_dim = dit_config["ffn_dim"]
        eps = dit_config["eps"]
        num_heads = dit_config["num_heads"]
        num_layers = dit_config["num_layers"]
        max_chunk_size = dit_config["max_chunk_size"]
        num_frame_per_block = dit_config["num_frame_per_block"]
        num_action_per_block = dit_config["num_action_per_block"]
        num_state_per_block = dit_config["num_state_per_block"]
        eps = dit_config["eps"]
        noise_beta_alpha = action_head_cfg["noise_beta_alpha"]
        noise_beta_beta = action_head_cfg["noise_beta_beta"]
        noise_s = action_head_cfg["noise_s"]
        decouple_video_action_noise = action_head_cfg["decouple_video_action_noise"]
        video_noise_beta_alpha = action_head_cfg["video_noise_beta_alpha"]
        video_noise_beta_beta = action_head_cfg["video_noise_beta_beta"]
        # concat_first_frame_latent = dit_config["concat_first_frame_latent"]

        dtype = torch.float32

        # if model_dtype == "bfloat16":
        #    dtype = torch.bfloat16
        # elif model_dtype == "float16":
        #    dtype = torch.float16

        model = Model(
            dtype=dtype,
            max_state_dim=max_state_dim,
            action_dim=max_action_dim,
            text_dim=input_embedding_dim,
            num_frame_per_block=num_frame_per_block,
            dim=dim,
            in_dim=in_dim,
            out_dim=out_dim,
            ffn_dim=ffn_dim,
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_layers=num_layers,
            max_chunk_size=max_chunk_size,
            num_action_per_block=num_action_per_block,
            num_state_per_block=num_state_per_block,
            eps=eps,
            frame_noise_beta_alpha=video_noise_beta_alpha,
            frame_noise_beta_beta=video_noise_beta_beta,
            decouple_frame_action_noise=decouple_video_action_noise,
            num_embodiments=1,
        )

        print("Model dtype, device ", model.dtype)
        self.print_model_parameters(model)

        return model

    def offload_to_cpu(self):
        """Offload the model to CPU to free GPU memory."""
        # if hasattr(self.model, "action_head") and hasattr(self.model.action_head, "image_encoder"):
        ## For models with vram management, disable it and move to CPU
        # if hasattr(self.model.action_head, 'disable_vram_management'):
        #    self.model.action_head.disable_vram_management()

        self.model.to(device="cpu")
        torch.cuda.empty_cache()
        print(f"Model offloaded to CPU")

    def load_to_gpu(self):
        """Load the model to GPU for inference."""
        print(f"Loading model to GPU...")

        # Move model to GPU
        # if hasattr(self.model, "action_head") and hasattr(self.model.action_head, "image_encoder"):
        # self.model.action_head.enable_vram_management()
        # else:
        self.model.to(device=self.device)

        # Apply bf16 if needed
        if self.eval_bf16:
            self.model = self.model.to(dtype=torch.bfloat16)

        torch.cuda.empty_cache()

    def ensure_model_on_gpu(self):
        """Ensure the model is loaded on GPU before inference."""
        # Check if model is on CPU
        model_device = next(self.model.parameters()).device
        if model_device.type == "cpu":
            self.load_to_gpu()

    def to_device(self, data, device):
        """Recursively moves all tensors in a dict, list, or custom object to the target device."""
        if isinstance(data, torch.Tensor):
            return data.to(device)
        elif isinstance(data, dict):
            return {k: self.to_device(v, device) for k, v in data.items()}
        elif isinstance(data, list):
            return [self.to_device(v, device) for v in data]
        elif isinstance(data, tuple):
            return tuple(self.to_device(v, device) for v in data)
        elif hasattr(data, "__dict__"):
            # For custom objects like BatchFeature
            for k, v in vars(data).items():
                setattr(data, k, self.to_device(v, device))
            return data
        return data

    def assert_delta_indices(self, delta_indices: np.ndarray):
        # All delta indices should be non-positive because there's no way to get the future observations
        assert np.all(delta_indices <= 0), f"{delta_indices=}"
        # The last delta index should be 0 because it doesn't make sense to not use the latest observation
        assert delta_indices[-1] == 0, f"{delta_indices=}"
        if len(delta_indices) > 1:
            # The step is consistent
            assert np.all(
                np.diff(delta_indices) == delta_indices[1] - delta_indices[0]
            ), f"{delta_indices=}"
            # And the step is positive
            assert (delta_indices[1] - delta_indices[0]) > 0, f"{delta_indices=}"

    def apply(self, batch: Batch, **kwargs) -> Batch:
        """Normalize inputs"""
        obs = batch.obs

        normalized_input = self.eval_transform(obs)

        batch.normalized_obs = normalized_input
        return batch

    def unapply(self, batch: Batch, obs: dict | None = None, **kwargs):
        """Unnormalize actions and convert relative actions to absolute if needed"""
        unnormalized_action = self.eval_transform.unapply(
            dict(action=batch.normalized_action.cpu())
        )

        # Check if relative_action is enabled and convert relative to absolute
        relative_action = experiment_conf.get("relative_action", False)
        relative_action_per_horizon = experiment_conf.get(
            "relative_action_per_horizon", False
        )
        relative_action_keys = experiment_conf.get("relative_action_keys", [])
        #print("relative_action", relative_action)
        #print("relative_action_per_horizon", relative_action_per_horizon)
        #print("relative_action_keys", relative_action_keys)
        if (
            (relative_action or relative_action_per_horizon)
            and relative_action_keys
            and obs is not None
        ):
            for key in relative_action_keys:
                action_key = f"action.{key}"
                state_key = f"state.{key}"

                if action_key not in unnormalized_action:
                    continue

                # Try to find the state data - check multiple possible key formats
                last_state = None

                if last_state is None and state_key in obs:
                    # Format 1: Direct key like "state.joint_position"
                    last_state = obs[state_key]
                elif last_state is None:
                    # Format 2: Search for keys containing both "state" and the key name
                    for obs_key in obs.keys():
                        if "state" in obs_key and key in obs_key:
                            last_state = obs[obs_key]
                            break

                    # Format 3: If key is "joint_position" and obs has "state" key directly
                    # This handles cases where the observation uses modality-level keys
                    if last_state is None and "state" in obs:
                        state_data = obs["state"]
                        # Check if the state data shape matches the action shape
                        action_dim = unnormalized_action[action_key].shape[-1]
                        if torch.is_tensor(state_data):
                            state_dim = state_data.shape[-1]
                        elif isinstance(state_data, np.ndarray):
                            state_dim = state_data.shape[-1]
                        else:
                            state_dim = None

                        if state_dim == action_dim:
                            last_state = state_data

                if last_state is None:
                    continue

                if torch.is_tensor(last_state):
                    last_state = last_state.cpu().numpy()

                # Shape is (B, T, D) or (T, D), we want the last timestep
                # After indexing: (B, D) or (D,)
                if len(last_state.shape) >= 2:
                    last_state = last_state[..., -1, :]  # Get the last timestep

                # Action shape is (horizon, D) or (B, horizon, D)
                # Expand dims to broadcast: (D,) -> (1, D) or (B, D) -> (B, 1, D)
                if len(unnormalized_action[action_key].shape) > len(last_state.shape):
                    last_state = np.expand_dims(
                        last_state, axis=-2
                    )  # Add horizon dimension

                # Add state to relative action to get absolute action
                #print(
                #    "last_state",
                #    last_state.shape,
                #    "unnormalized_action[action_key]",
                #    unnormalized_action[action_key].shape,
                #)
                unnormalized_action[action_key] = (
                    unnormalized_action[action_key] + last_state
                )

        batch.act = unnormalized_action
        return batch

    def evaluate_action_quality(
        self, act_dict: dict, obs_dict: dict, plot: bool = False
    ):
        """
        Quick diagnostic check for the predicted action trajectory.
        Evaluates teleportation (anchor error), jitter (velocity spikes), and gripper bounds.
        """
        action_key = "action.joint_position"
        gripper_key = "action.gripper_position"

        if action_key not in act_dict:
            return

        action_traj = act_dict[action_key]
        state_key = "state.joint_position"
        curr_state = obs_dict.get(state_key)

        gripper_traj = act_dict.get(gripper_key)
        curr_gripper = obs_dict.get("state.gripper_position")

        # Convert to pure numpy for math
        if torch.is_tensor(action_traj):
            action_traj = action_traj.detach().cpu().numpy()
        if curr_state is not None and torch.is_tensor(curr_state):
            curr_state = curr_state.detach().cpu().numpy()

        if gripper_traj is not None and torch.is_tensor(gripper_traj):
            gripper_traj = gripper_traj.detach().cpu().numpy()
        if curr_gripper is not None and torch.is_tensor(curr_gripper):
            curr_gripper = curr_gripper.detach().cpu().numpy()

        # Isolate the first batch (B=0) if it's batched
        if len(action_traj.shape) == 3:  # [B, Horizon, Dim] -> [1, 96, 7]
            action_traj = action_traj[0]
        if curr_state is not None:
            if len(curr_state.shape) == 3:  # [B, Horizon, Dim] -> [1, 4, 7]
                curr_state = curr_state[0, -1]  # Last state of horizon
            elif len(curr_state.shape) == 2:  # [Horizon, Dim]
                curr_state = curr_state[-1]

        if gripper_traj is not None:
            if len(gripper_traj.shape) == 3:  # [1, 96, 1]
                gripper_traj = gripper_traj[0]
        if curr_gripper is not None:
            if len(curr_gripper.shape) == 3:  # [1, 4, 1]
                curr_gripper = curr_gripper[0, -1]
            elif len(curr_gripper.shape) == 2:
                curr_gripper = curr_gripper[-1]

        # 1. Anchor Check (All 7 Arm Joints) - Is the start position teleporting?
        start_error = 0.0
        if curr_state is not None:
            start_error = float(np.linalg.norm(action_traj[0, :] - curr_state[:]))

        # 2. Smoothness Check (All 7 Arm Joints) - Max step-to-step velocity jump
        step_deltas = np.diff(action_traj, axis=0)
        max_velocity = float(np.max(np.abs(step_deltas)))

        print("\n" + "=" * 60)
        print(f"🎯 ACTION QUALITY DIAGNOSTICS: {action_key}")
        print("=" * 60)
        print(f"Anchor Error (Start vs Real State): {start_error:.4f} (Ideal: ~0.0)")
        print(f"Max Step Velocity (Jitter Check):   {max_velocity:.4f} (Ideal: < 0.1)")

        if gripper_traj is not None:
            g_min, g_max = float(gripper_traj.min()), float(gripper_traj.max())
            print(f"Gripper Range [Min, Max]:           [{g_min:.2f}, {g_max:.2f}]")

        if start_error > 0.1:
            print("⚠️  WARNING: High anchor error! Robot might jerk to the start.")
        if max_velocity > 0.2:
            print("⚠️  WARNING: High velocity! Trajectory might be jittery/noisy.")
        print("=" * 60 + "\n")

        if plot:
            self._plot_trajectory(
                action_traj, curr_state, gripper_traj, curr_gripper, action_key
            )

    def _plot_trajectory(
        self,
        action_traj: np.ndarray,
        curr_state: np.ndarray | None,
        gripper_traj: np.ndarray | None,
        curr_gripper: np.ndarray | None,
        title: str,
    ):
        """Plots the predicted 96 steps vs the current anchor state"""
        import matplotlib.pyplot as plt

        fig, ax1 = plt.subplots(figsize=(10, 5))

        # Plot the arm dimensions (7 DOF)
        num_joints = action_traj.shape[-1]
        for i in range(num_joints):
            ax1.plot(action_traj[:, i], label=f"Joint {i}")
            if curr_state is not None:
                ax1.scatter(0, curr_state[i], color="black", zorder=5)

        ax1.set_title(f"Trajectory Quality Check: {title}")
        ax1.set_xlabel("Future Timesteps")
        ax1.set_ylabel("Joint Position")
        ax1.grid(True, alpha=0.3)

        # Plot gripper on secondary Y axis
        if gripper_traj is not None:
            gripper_plot = np.asarray(gripper_traj).squeeze()

            # Handle shapes like (), (T,), (T, 1), or (1, T, 1)
            if gripper_plot.ndim == 0:
                gripper_plot = gripper_plot.reshape(1)
            elif gripper_plot.ndim > 1:
                gripper_plot = gripper_plot.reshape(-1)

            ax2 = ax1.twinx()
            ax2.plot(
                gripper_plot,
                label="Gripper",
                linestyle="--",
                color="red",
                linewidth=2,
            )

            if curr_gripper is not None:
                curr_gripper_value = float(np.asarray(curr_gripper).squeeze().reshape(-1)[-1])
                ax2.scatter(
                    0,
                    curr_gripper_value,
                    color="red",
                    zorder=5,
                    marker=matplotlib.markers.MarkerStyle("x"),
                    s=100,
                )
        else:
            ax1.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

        plt.tight_layout()
        plt.show()

    def plot_actions(self, action: torch.Tensor, pred_action: torch.Tensor | None = None):
        squeezed = action.squeeze(0)
        is_all_zero = torch.all(squeezed == 0, dim=0)
        
        active_indices = torch.where(~is_all_zero)[0]
        
        if len(active_indices) == 0:
            print("Error: The entire action tensor is empty/zero.")
            return
        
        num_joints = active_indices[-1].item() + 1
        action_np = squeezed[:, :num_joints].cpu().numpy()
        
        if pred_action is not None:
            pred_squeezed = pred_action.squeeze(0)
            pred_action_np = pred_squeezed[:, :num_joints].cpu().numpy()
        
        print(f"Automatically detected {num_joints} active action dimensions (dropping {32 - num_joints} padding dimensions).")
        
        import math
        num_cols = 2
        num_rows = math.ceil(num_joints / num_cols)
        
        fig, axs = plt.subplots(num_rows, num_cols, figsize=(14, 2.5 * num_rows), sharex=True)
        axs = axs.flatten()
        
        for i in range(num_joints):
            axs[i].plot(action_np[:, i], color="C"+str(i), linewidth=2, label="Ground Truth")
            
            if pred_action is not None:
                axs[i].plot(pred_action_np[:, i], color="C"+str(i), linewidth=2, linestyle="--", alpha=0.8, label="Prediction")
            
            # Label the last active dimension as Gripper, others as Joints
            if i == num_joints - 1:
                title = f"Gripper)"
            else:
                title = f"Joint {i+1}"
                
            axs[i].set_ylabel("Position", fontsize=9)
            axs[i].set_title(title, loc="left", fontsize=10, fontweight="bold")
            axs[i].grid(True, linestyle="--", alpha=0.5)
            
            #if pred_action is not None:
            #    axs[i].legend(loc="upper right", fontsize=8)
            
            if i >= num_joints - num_cols:
                axs[i].set_xlabel("Time Step (Prediction Horizon)")
        
        for j in range(num_joints, len(axs)):
            axs[j].set_visible(False)
        
        plt.suptitle("Model Action Trajectory Prediction", fontsize=14, y=0.99)
        plt.tight_layout()
        plt.savefig("Figure_apf-1500.png", dpi=300, bbox_inches="tight")
        plt.close()
        #plt.show()

    def lazy_joint_forward(
        self,
        batch: Batch,
        state=None,
        check_quality: bool = True,
        plot_trajectory: bool = True,
        **kwargs,
    ):
        #print(batch.obs.keys())
        
        is_batched = self._check_state_is_batched(batch.obs)
        if not is_batched:
            batch.obs = unsqueeze_dict_values(batch.obs)

        batch = self.apply(batch)
        
        normalized_input = batch.normalized_obs
        normalized_input: Any = self.to_device(normalized_input, torch.device("cuda"))
        action = normalized_input["action"]
        frames = normalized_input["images"].squeeze(0)
        frames = frames.cpu()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"outputs/predicted_trajectory_{timestamp}.mp4"
        #write_file(output_path, frames)
        #print(f"Video saved successfully to {output_path}")
        
        #print(f"images shape {type(normalized_input["images"])} {normalized_input["images"].shape} {normalized_input["images"].dtype}") # 1, 33, 180, 320, 3
        #print(f"state shape {normalized_input["state"].shape}")
        #print(f"state_mask shape {normalized_input["state_mask"].shape}")
        #print(f"has_real_action shape {normalized_input["has_real_action"].shape}")
        #print(f"action shape {normalized_input["action"].shape}")
        #print(f"action_mask shape {normalized_input["action_mask"].shape}")
        #print(f"text shape {normalized_input["text"].shape}")
        #print(f"text_attention_mask shape {normalized_input["text_attention_mask"].shape}")
        #print(f"embodiment_id shape {normalized_input["embodiment_id"].shape}")

        with torch.inference_mode():
            model_pred = self.model.lazy_joint_frame_action(normalized_input)
        normalized_action = model_pred["action_pred"].float()
        frame_pred = model_pred["frame_pred"]
        
        print(f"normalized_action shape {normalized_action.shape}")
        
        #self.plot_actions(action, normalized_action)
        #self.calculate_error(action, normalized_action)

        original_obs = batch.obs

        pred_relative = self.eval_transform.unapply(
            dict(action=normalized_action.cpu())
        )

        batch = self.unapply(
            Batch(normalized_action=normalized_action), obs=original_obs
        )
        batch.act_relative = pred_relative
        
        #pred_joints = batch.act["action.joint_position"]  # [96, 7]
        #pred_gripper = batch.act["action.gripper_position"]  # [96, 1]
        ## These should be unnormalized (real joint angles in radians)
        #print(f"Pred joints range: [{pred_joints.min():.3f}, {pred_joints.max():.3f}]")
        #print(f"Pred gripper range: [{pred_gripper.min():.3f}, {pred_gripper.max():.3f}]")

        ##print(f"original_obs shapes: {[(k, v.shape) for k, v in original_obs.items()]}")
        ##print(f"batch shapes: {[(k, v.shape) for k, v in batch.items()]}")

        #if check_quality:
        #    self.evaluate_action_quality(batch.act, original_obs, plot=plot_trajectory)

        if not is_batched:
            batch.act = squeeze_dict_values(batch.act)

        return batch, frame_pred

    def _check_state_is_batched(self, obs: dict[str, Any]) -> bool:
        for k, v in obs.items():
            if "state" in k and len(v.shape) < 3:  # (B, Time, Dim)
                return False
        return True

    def calculate_error(self, action, pred_action):
        physical_dim = 8
        true_active = action.squeeze(0)[:, :physical_dim]
        pred_active = pred_action.squeeze(0)[:, :physical_dim]
        
        mse_loss = torch.nn.functional.mse_loss(pred_active, true_active)
        l1_loss = torch.nn.functional.l1_loss(pred_active, true_active)
        
        print(f"True Model MSE: {mse_loss.item()}")
        print(f"True Model L1: {l1_loss.item()}")
        
    def print_model_parameters(self, model):
        # 1. Count all parameters (including frozen ones like a pre-trained Vision Encoder)
        total_params = sum(p.numel() for p in model.parameters())
        
        # 2. Count ONLY the parameters that are currently being trained
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"Total Parameters: {total_params / 1e6:.2f} M")
        print(f"Trainable Parameters: {trainable_params / 1e6:.2f} M")
        print(f"Percentage Trainable: {(trainable_params / total_params) * 100:.2f}%")


def squeeze_dict_values(data: dict[str, Any]) -> dict[str, Any]:
    """
    Squeeze the values of a dictionary. This removes the batch dimension.
    """
    squeezed_data = {}
    for k, v in data.items():
        if isinstance(v, np.ndarray):
            squeezed_data[k] = np.squeeze(v)
        elif isinstance(v, torch.Tensor):
            squeezed_data[k] = v.squeeze()
        else:
            squeezed_data[k] = v
    return squeezed_data


if __name__ == "__main__":
    cwd = Path(os.getcwd())
    checkpoint_path = cwd / "checkpoint-11300"
    metadata_json_path = checkpoint_path / "experiment_cfg/metadata.json"
    inference = WAMPolicy(
        checkpoint_path=checkpoint_path, metadata_json_path=metadata_json_path
    )
