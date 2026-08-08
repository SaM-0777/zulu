import os
import time
from pathlib import Path
from collections import deque
import numpy as np
import torch
import imageio
import h5py
import robosuite as suite
from robosuite.controllers import load_part_controller_config
from robosuite.controllers.composite.composite_controller_factory import (
    refactor_composite_controller_config,
)
from tianshou.data import Batch

from src.experiment.wam_inference import WAMPolicy
from src.data.lerobot import ShardedLeRobotSubLangSingleActionChunkDatasetDROID


# ──────────────────────────────────────────────────────────────────────
class ActionReplay:
    """Replay recorded actions in a Robosuite environment with physics."""

    # ── Defaults ────────────────────────────────────────────────────
    WRIST_CAM = "robot0_eye_in_hand"
    FRONT_CAM = "frontview"
    LEFT_CAM = "left_cam"
    RIGHT_CAM = "right_cam"
    CAM_H, CAM_W = 180, 320
    GRIPPER_MAX = 0.04
    HIGH_FRICTION = [10.0, 0.1, 0.1]
    FRICTION_KEYWORDS = ["can", "finger", "table", "tray", "milk", "bread", "cereal"]
    HARD_CONTACT_KEYWORDS = ["can", "finger"]
    IMAGE_HORIZON = 33
    STATE_HORIZON = 4

    def __init__(
        self,
        checkpoint_path: str,
        finetuned_path: str,
        dataset_path: str,
        hdf5_path: str,
        episode_index: int = 0,
        control_freq: int = 15,
    ):
        self.episode_index = episode_index
        self.control_freq = control_freq

        # ── Policy & Dataset ─────────────────────────────────────
        metadata_json = Path(finetuned_path) / "experiment_cfg" / "metadata.json"
        print("Loading policy …")
        self.policy = WAMPolicy(
            checkpoint_path=checkpoint_path,
            metadata_json_path=metadata_json,
            finetuned_checkpoint_path=finetuned_path,
        )

        print(f"Loading dataset from {dataset_path} …")
        self.dataset = ShardedLeRobotSubLangSingleActionChunkDatasetDROID(
            dataset_path=dataset_path,
            modality_configs=self.policy.modality_configs,
            embodiment_tag=self.policy.embodiment_tag,
            video_backend="decord",
            transforms=None,
            max_chunk_size=4,
            relative_action=True,
            relative_action_per_horizon=False,
            relative_action_keys=["joint_position"],
        )
        self.dataset.start_cache_shard(0)
        self.dataset.finish_cache_shard()
        
        self.max_chunk = 4
        self.nfpb = 8

        self.traj_len = int(self.dataset.trajectory_lengths[episode_index])
        print(f"Episode {episode_index} | {self.traj_len} steps")

        # 8─ HDF5 reference ───────────────────────────────────────
        self.hdf5_path = hdf5_path

        # ── Environment ──────────────────────────────────────────
        self._create_env()
        self._set_initial_state_from_hdf5()
        self._compute_joint_addresses()
        self._configure_physics()
        self._reset_to_dataset_initial_state()

    # ──────────────────────────────────────────────────────────────
    # Setup helpers
    # ──────────────────────────────────────────────────────────────
    def _create_env(self):
        ctrl_config = refactor_composite_controller_config(
            load_part_controller_config(default_controller="JOINT_POSITION"),
            robot_type="Panda",
            arms=["right"],
        )
        self.env = suite.make(
            env_name="PickPlaceCan",
            robots="Panda",
            controller_configs=ctrl_config,
            has_renderer=True,
            has_offscreen_renderer=True,
            use_camera_obs=True,
            camera_names=[self.WRIST_CAM, self.LEFT_CAM, self.RIGHT_CAM],
            camera_heights=self.CAM_H,
            camera_widths=self.CAM_W,
            control_freq=self.control_freq,
            horizon=self.traj_len + 50,
            ignore_done=True,
            reward_shaping=False,
        )
        self.env._visualize = lambda: None
        self.sim = self.env.sim
        self.obs = self.env.reset()

    def _set_initial_state_from_hdf5(self):
        with h5py.File(self.hdf5_path, "r") as f:
            demo = f[f"data/demo_{self.episode_index + 1}"]
            model_xml = demo.attrs["model_file"]
            initial_state = demo["mj_states"][0]

        xml = self.env.edit_model_xml(model_xml) if hasattr(self.env, "edit_model_xml") else model_xml
        self.env.reset_from_xml_string(xml)
        self.sim.set_state_from_flattened(initial_state)
        self.sim.forward()

    def _compute_joint_addresses(self):
        self.arm_qpos_addrs = []
        self.arm_qvel_addrs = []
        for i in range(7):
            jid = self.sim.model.joint_name2id(f"robot0_joint{i+1}")
            self.arm_qpos_addrs.append(self.sim.model.jnt_qposadr[jid])
            self.arm_qvel_addrs.append(self.sim.model.jnt_dofadr[jid])

        self.gripper_qpos_addrs = []
        for name in ["gripper0_right_finger_joint1", "gripper0_right_finger_joint2"]:
            jid = self.sim.model.joint_name2id(name)
            self.gripper_qpos_addrs.append(self.sim.model.jnt_qposadr[jid])

        self.timestep = self.sim.model.opt.timestep
        self.n_substeps = int(round(1.0 / (self.control_freq * self.timestep)))
        self.control_dt = self.n_substeps * self.timestep

    def _geom_name(self, idx: int) -> str:
        if hasattr(self.sim.model, "geom_names") and idx < len(self.sim.model.geom_names):
            return self.sim.model.geom_names[idx] or f"geom_{idx}"
        return f"geom_{idx}"

    def _configure_physics(self):
        sim = self.sim

        # Disable robot damping
        for addr in self.arm_qvel_addrs:
            sim.model.dof_damping[addr] = 0.0
            sim.model.dof_frictionloss[addr] = 0.0
            sim.model.dof_armature[addr] = 0.0

        # Full friction contact model
        sim.model.geom_condim[:] = 6

        # High friction on relevant geoms
        for i in range(sim.model.ngeom):
            name = self._geom_name(i).lower()
            if any(kw in name for kw in self.FRICTION_KEYWORDS):
                sim.model.geom_friction[i] = self.HIGH_FRICTION

        # Harder contacts for can + fingers
        for i in range(sim.model.ngeom):
            name = self._geom_name(i).lower()
            if any(kw in name for kw in self.HARD_CONTACT_KEYWORDS):
                sim.model.geom_solref[i] = [0.005, 1.0]
                sim.model.geom_solimp[i][:3] = [0.99, 0.999, 0.001]

        # Lighter can (easier to grip)
        for i in range(sim.model.nbody):
            name = (sim.model.body_names[i] or "").lower()
            if "can" in name:
                sim.model.body_mass[i] *= 0.1
                print(f"  ↓ {sim.model.body_names[i]}: mass → {sim.model.body_mass[i]:.4f}")

        # More solver iterations
        sim.model.opt.iterations = 100
        sim.model.opt.ls_iterations = 50
        sim.forward()

    def _reset_to_dataset_initial_state(self):
        dp = self._get_step_data(0)
        init_joints = dp["state.joint_position"][0][:7]
        for i, addr in enumerate(self.arm_qpos_addrs):
            self.sim.data.qpos[addr] = init_joints[i]
        self.sim.data.qvel[:] = 0
        self.sim.forward()

    # ─────────────────────────────────────────────────7─────────────
    # Data access
    # ──────────────────────────────────────────────────────────────
    def _get_step_data(self, step: int):
        indices = {
            k: np.clip(v + step, 0, self.traj_len - 1).astype(int)
            for k, v in self.dataset.delta_indices.items()
        }
        return self.dataset.get_step_data(self.episode_index, indices)

    # ──────────────────────────────────────────────────────────────
    # Core action execution
    # ──────────────────────────────────────────────────────────────
    def _pin_and_step(self, desired_q: np.ndarray, desired_vel: np.ndarray, gripper_qpos: float):
        """Pin robot at desired position/velocity, then step physics."""
        for _ in range(self.n_substeps):
            for i, addr in enumerate(self.arm_qpos_addrs):
                self.sim.data.qpos[addr] = desired_q[i]
            for i, addr in enumerate(self.arm_qvel_addrs):
                self.sim.data.qvel[addr] = desired_vel[i]
            self.sim.data.qpos[self.gripper_qpos_addrs[0]] = gripper_qpos
            self.sim.data.qpos[self.gripper_qpos_addrs[1]] = -gripper_qpos
            self.sim.step()

    # ──────────────────────────────────────────────────────────────
    # Rendering
    # ──────────────────────────────────────────────────────────────
    def _capture_frames(self):
        self.env.render()
        wrist = np.flipud(self.sim.render(height=self.CAM_H, width=self.CAM_W, camera_name=self.WRIST_CAM)).copy()
        front = np.flipud(self.sim.render(height=self.CAM_H, width=self.CAM_W, camera_name=self.FRONT_CAM)).copy()
        return wrist, front

    def init_history(self):
        self.buf_left    = deque(maxlen=self.IMAGE_HORIZON)
        self.buf_right   = deque(maxlen=self.IMAGE_HORIZON)
        self.buf_wrist   = deque(maxlen=self.IMAGE_HORIZON)
        self.buf_joint   = deque(maxlen=self.IMAGE_HORIZON)
        self.buf_gripper = deque(maxlen=self.IMAGE_HORIZON)
        
    def get_obs_images(self):
        left  = np.flipud(self.sim.render(height=self.CAM_H, width=self.CAM_W, camera_name=self.LEFT_CAM)).copy()
        right = np.flipud(self.sim.render(height=self.CAM_H, width=self.CAM_W, camera_name=self.RIGHT_CAM)).copy()
        wrist = np.flipud(self.sim.render(height=self.CAM_H, width=self.CAM_W, camera_name=self.WRIST_CAM)).copy()
        return left, right, wrist
    
    def update_history(self):
        left, right, wrist = self.get_obs_images()
        self.buf_left.append(left)
        self.buf_right.append(right)
        self.buf_wrist.append(wrist)

        joint = np.array(
            [self.sim.data.qpos[addr] for addr in self.arm_qpos_addrs],
            dtype=np.float32,
        )
        grip_qpos = np.array(
            [self.sim.data.qpos[addr] for addr in self.gripper_qpos_addrs],
            dtype=np.float32,
        )
        grip = np.array([np.mean(grip_qpos)], dtype=np.float32)

        self.buf_joint.append(joint)
        self.buf_gripper.append(grip)
    
    def pack_for_policy(self, language="pick up the can and place it in the bin"):
        def pad_video(buf):
            frames = list(buf)
            while len(frames) < self.IMAGE_HORIZON:
                frames.insert(0, frames[0].copy())
            return np.stack(frames, axis=0)

        def pad_state(buf):
            states = list(buf)
            while len(states) < self.IMAGE_HORIZON:
                states.insert(0, states[0].copy())
            selected = []
            for block_idx in range(self.max_chunk):
                offset = (self.max_chunk - 1 - block_idx) * self.nfpb
                idx = len(states) - 1 - offset
                selected.append(states[idx])
            return np.stack(selected, axis=0)

        packed = {
            "video.exterior_image_1_left": pad_video(self.buf_left),
            "video.exterior_image_2_left": pad_video(self.buf_right),
            "video.wrist_image_left":      pad_video(self.buf_wrist),
            "state.joint_position":        pad_state(self.buf_joint),
            "state.gripper_position":      pad_state(self.buf_gripper),
            "annotation.language.language_instruction": language,
        }
        return packed

    @staticmethod
    def save_video(frames, path: str, fps: int = 15):
        if not frames:
            print("No frames to save!")
            return
        print(f"Saving {len(frames)} frames → {path}")
        imageio.mimsave(path, frames, fps=fps)
        
    # ──────────────────────────────────────────────────────────────
    # Main evalu loop
    # ──────────────────────────────────────────────────────────────
    def run_eval(self, max_steps=20, action_horizon=48, language = "pick up the can and place it in the bin", output_prefix: str = "eval"):
        wrist_frames = []
        front_frames = []

        self.env.reset()                           # init env internals
        self._set_initial_state_from_hdf5()
        self._configure_physics()
        self._reset_to_dataset_initial_state()     # set robot to dataset state

        self.init_history()
        self.update_history()

        for step in range(max_steps):
            print(f"\n===== step {step} =====")
            
            packed = self.pack_for_policy(language)
            batch = Batch(obs=packed)
            
            print("state.joint_position:\n", batch.obs["state.joint_position"])
            print("state.gripper_position:\n", batch.obs["state.gripper_position"])
            
            out_batch, _ = self.policy.lazy_joint_forward(
                batch, check_quality=False, plot_trajectory=False
            )
            
            # absolute actions
            pred_joints = out_batch.act["action.joint_position"] # np shape (96, 7)
            pred_gripper = out_batch.act["action.gripper_position"] # np shape (96,)
            
            for t in range(action_horizon):
                target_joints = pred_joints[t]
                target_gripper = pred_gripper[t]
                
                current_q = np.array([self.sim.data.qpos[a] for a in self.arm_qpos_addrs])
                desired_vel = (target_joints - current_q) / self.control_dt
                
                gripper_qpos = (1.0 - target_gripper) * self.GRIPPER_MAX
                
                print(f"target_joints {target_joints}")
                print(f"target_gripper {target_gripper}")
                print(f"current_q {current_q}")
                print(f"desired_vel {desired_vel}")
                print(f"gripper_qpos {gripper_qpos}")
                print("-" * 50)
                
                self._pin_and_step(target_joints, desired_vel, gripper_qpos)
                
                self.update_history()
                
                wrist, front = self._capture_frames()
                wrist_frames.append(wrist)
                front_frames.append(front)
                
                time.sleep(0.02)
        
        self.save_video(wrist_frames, f"{output_prefix}_wrist.mp4", fps=15)
        self.save_video(front_frames, f"{output_prefix}_front.mp4", fps=15)
        print("Eval done.")
                

    # ──────────────────────────────────────────────────────────────
    # Main replay loop
    # ──────────────────────────────────────────────────────────────
    def run(self, output_prefix: str = "action_replay"):
        wrist_frames = []
        front_frames = []
        
        self.env.reset()                           # init env internals
        self._set_initial_state_from_hdf5()
        self._configure_physics()
        self._reset_to_dataset_initial_state()     # set robot to dataset state

        for step in range(self.traj_len - 24):
            dp = self._get_step_data(step)

            # Action delta (index [1] = one-step-ahead; [0] is always 0 for relative)
            delta = dp["action.joint_position"][1]
            gripper_action = dp["action.gripper_position"][0][0]

            current_q = np.array([self.sim.data.qpos[a] for a in self.arm_qpos_addrs])
            desired_q = current_q + delta
            desired_vel = delta / self.control_dt
            gripper_qpos = (1.0 - gripper_action) * self.GRIPPER_MAX

            self._pin_and_step(desired_q, desired_vel, gripper_qpos)

            wrist, front = self._capture_frames()
            wrist_frames.append(wrist)
            front_frames.append(front)

            if step % 100 == 0:
                print(f"  step {step:4d}/{self.traj_len - 24}")

            time.sleep(0.02)

        self.save_video(wrist_frames, f"{output_prefix}_wrist.mp4", fps=15)
        self.save_video(front_frames, f"{output_prefix}_front.mp4", fps=15)
        self.env.close()
        print("Done.")


# ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    CWD = Path(os.getcwd())

    replay = ActionReplay(
        checkpoint_path=str(CWD / "checkpoint-0"),
        finetuned_path=str(CWD / "checkpoint-sim-finetune-1500"),
        dataset_path=str(CWD / "data" / "panda_pickplace_droid_v3_fix_state_action_gripper_inverted_2i"),
        hdf5_path=str(CWD / "robosuite_data" / "pickplace_teleop_droid" / "20260801_151327.hdf5"),
        episode_index=0,
        control_freq=15,
    )

    #replay.run(output_prefix="action_replay")
    replay.run_eval()
