import os
from pathlib import Path

from src.data.lerobot import ShardedLeRobotSubLangSingleActionChunkDatasetDROID
from src.data.schema import EmbodimentTag
from src.experiment.wam_inference import WAMPolicy
import numpy as np
import torch
import torch.distributed as dist
from tianshou.data import Batch
#import torch._dynamo.config.re
#from wam.quality import ActionQualityTester
import torchvision


def _make_dataset_forward_loop(policy: WAMPolicy, dataset_path: str| Path, num_caliberation_trajs: int = 1):
    dino_proj_weight = policy.model.dit_backbone.dino_proj.weight
    #print(f"dino_proj weight norm: {dino_proj_weight.norm().item():.4f}")
    #print(f"dino_proj weight max: {dino_proj_weight.max().item():.4f}")
    
    def forward_loop(model):
        #print(f"Caliberation: loading dataset from {dataset_path} ({num_caliberation_trajs} trajs)")

        dataset = ShardedLeRobotSubLangSingleActionChunkDatasetDROID(
            dataset_path=dataset_path,
            modality_configs=policy.modality_configs,
            embodiment_tag=policy.embodiment_tag,
            #embodiment_tag=EmbodimentTag.OXE_DROID,
            video_backend="decord",
            #video_backend_kwargs=None,
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
        
        #print(f"action_horizon {action_horizon}, num_frame_per_block {num_frame_per_block}") # action_horizon 24, num_frame_per_block 8
        
        
        #torch._dynamo.config.recompile_limit = 500
        #print(f"num_caliberation_trajs {num_caliberation_trajs}, trajectory_lengths {len(dataset.trajectory_lengths)}") # num_caliberation_trajs 2, trajectory_lengths 2000
        
        for traj_id in range(min(num_caliberation_trajs, len(dataset.trajectory_lengths))): # 2
            traj_len = int(dataset.trajectory_lengths[traj_id])
            latent_video = None
            
            max_steps = min(traj_len, 1 * action_horizon) # simulate for 5 times
            for step in range(0, max_steps, action_horizon):
                indices = {
                    k: np.clip(v + step, 0, traj_len - 1)
                    for k, v in dataset.delta_indices.items()
                }
                
                #print(f"indices {indices}")
                
                data_point = dataset.get_step_data(traj_id, indices)
                batch = Batch(obs=data_point)
                
            #    #dist.barrier()
                with torch.no_grad():
                    result_batch, video_pred = policy.lazy_joint_forward(batch=batch)
                #dist.barrier()
                
                #pred_relative = policy.eval_transform.unapply(
                #    dict(action=model_pred["action_pred"].cpu())
                #)
                #print(f"Relative actions (before adding state): mean={pred_relative['action.joint_position'].mean():.4f}")
                #print(f"Absolute actions (after adding state): mean={result_batch.act['action.joint_position'].mean():.4f}")
                #print(f"Last state: {data_point['state.joint_position'][0, -1]}")
                #print(f"GT actions: mean={data_point['action.joint_position'][0].mean():.4f}")
                
                #ActionQualityTester.run_all_tests(
                #    act_dict=result_batch.act,
                #    obs_dict=data_point,
                #    save_path="action_comparison_9350.png"
                #)
                
                #ActionQualityTester.run_all_tests(
                #    act_dict=result_batch.act_relative,
                #    obs_dict=data_point,
                #    save_path="action_comparison_relative_9350.png",
                #)
                
                #policy.evaluate_action_quality(
                #    act_dict=result_batch.act,
                #    obs_dict=data_point,
                #    plot=True,
                #)
                
                #if video_pred is not None:
                #    latent_video = video_pred[:, :, -num_frame_per_block:]
                
            policy.model.current_start_frame = 0
            policy.model.kv_caches = None
            policy.model.crossattn_caches = None
    
    return forward_loop

                
if __name__ == "__main__":
    cwd = Path(os.getcwd())
    checkpoint_path = cwd / "checkpoint-0"
    finetuned_checkpoint_path = cwd / "checkpoint-finetune-2800"
    metadata_json_path = checkpoint_path / "experiment_cfg" / "metadata.json"
    policy = WAMPolicy(
        checkpoint_path=checkpoint_path, metadata_json_path=metadata_json_path, finetuned_checkpoint_path=finetuned_checkpoint_path
    )
    
    dataset_path = cwd / "data" / "dreamzero_droid_pick_g"

    dataset_forward_loop = _make_dataset_forward_loop(policy=policy, dataset_path=dataset_path)

    dataset_forward_loop(model=None)
                