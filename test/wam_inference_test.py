import os
from pathlib import Path

from src.data.lerobot import ShardedLeRobotSubLangSingleActionChunkDatasetDROID
from src.data.schema import EmbodimentTag
from src.experiment.wam_inference import WAMPolicy
import numpy as np
import torch
import torch.distributed as dist
from tianshou.data import Batch
import torchvision


def _make_dataset_forward_loop(policy: WAMPolicy, dataset_path: str| Path, num_caliberation_trajs: int = 1, episode_index: int = 0):
    dino_proj_weight = policy.model.dit_backbone.dino_proj.weight
    #print(f"dino_proj weight norm: {dino_proj_weight.norm().item():.4f}")
    #print(f"dino_proj weight max: {dino_proj_weight.max().item():.4f}")
    
    def forward_loop(model):
        #print(f"Caliberation: loading dataset from {dataset_path} ({num_caliberation_trajs} trajs)")

        dataset = ShardedLeRobotSubLangSingleActionChunkDatasetDROID(
            dataset_path=dataset_path,
            modality_configs=policy.modality_configs,
            embodiment_tag=policy.embodiment_tag,
            video_backend="decord",
            transforms=policy.eval_transform, # policy.lazy_joint_forward applies transforms
            #use_global_metadata=False,
            max_chunk_size=4,
            relative_action=True,
            relative_action_per_horizon=False,
            relative_action_keys=["joint_position"]
        )
        
        dataset.start_cache_shard(0)
        dataset.finish_cache_shard()
        
        action_horizon = policy.model.action_horizon
        num_frame_per_block = policy.model.num_frame_per_block
        
        total_episodes = len(dataset.trajectory_lengths)
        if not (0 <= episode_index < total_episodes):
            raise ValueError(f"Episode index {episode_index} is out of bounds. Dataset contains {total_episodes} episodes (0 to {total_episodes - 1}).")
        
        traj_len = int(dataset.trajectory_lengths[episode_index])
        print(f"Targeting Episode Index: {episode_index} | Length: {traj_len} frames | Action horizon: {action_horizon}")
        
        for step in range(0, traj_len, action_horizon):
            indices = {
                k: np.clip(v + step, 0, traj_len - 1)
                for k, v in dataset.delta_indices.items()
            }
            
            # Fetch data point exclusively for this specific episode index
            data_point = dataset.get_step_data(episode_index, indices)
            batch = Batch(obs=data_point)
            
            with torch.no_grad():
                result_batch, video_pred = policy.lazy_joint_forward(batch=batch)
                
            print(f"Processed chunk starting at step {step} for episode {episode_index}")
        
        policy.model.current_start_frame = 0
        policy.model.kv_caches = None
        policy.model.crossattn_caches = None
        
        #traj_id = target_episode_index
        #if traj_id >= len(dataset.trajectory_lengths):
        #    raise ValueError(f"Episode index {traj_id} out of range for dataset with {len(dataset.trajectory_lengths)} trajectories.")
        
        #traj_len = int(dataset.trajectory_lengths[traj_id])
        #latent_video = None
        
        #max_steps = min(traj_len, 1 * action_horizon)
        #for step in range(0, max_steps, action_horizon):
        #    indices = {
        #        k: np.clip(v + step, 0, traj_len - 1)
        #        for k, v in dataset.delta_indices.items()
        #    }
        
        #    data_point = dataset.get_step_data(traj_id, indices)
        #    batch = Batch(obs=data_point)
            
        #    print(f"Instruction {batch["obs"]["annotation.language.language_instruction"]}")
            
        #    with torch.no_grad():
        #        result_batch, video_pred = policy.lazy_joint_forward(batch=batch)
            
        #    if video_pred is not None:
        #        latent_video = video_pred[:, :, -num_frame_per_block:]
        
        #policy.model.current_start_frame = 0
        #policy.model.kv_caches = None
        #policy.model.crossattn_caches = None
    
    return forward_loop

                
if __name__ == "__main__":
    cwd = Path(os.getcwd())
    checkpoint_path = cwd / "checkpoint-0"
    finetuned_checkpoint_path = cwd / "checkpoint-sim-finetune-1500"
    metadata_json_path = finetuned_checkpoint_path / "experiment_cfg" / "metadata.json"
    policy = WAMPolicy(
        checkpoint_path=checkpoint_path, 
        metadata_json_path=metadata_json_path, 
        finetuned_checkpoint_path=finetuned_checkpoint_path
    )
    
    dataset_path = cwd / "data" / "panda_pickplace_droid_v3_fix_state_action_gripper_inverted_2i"
    #dataset_path = cwd / "data" / "dreamzero_droid_first3"
    target_episode_index = 0

    dataset_forward_loop = _make_dataset_forward_loop(policy=policy, dataset_path=dataset_path, episode_index=target_episode_index)

    dataset_forward_loop(model=None)
