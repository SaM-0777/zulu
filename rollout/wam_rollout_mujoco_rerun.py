import numpy as np
import torch
import time
import pathlib
import mujoco
import mujoco.viewer
import rerun as rr
from collections import deque
from tqdm import tqdm
from tianshou.data import Batch
from src.experiment.wam_inference import WAMPolicy

class WAMRolloutMuJoCo:
    def __init__(
        self, 
        policy: WAMPolicy, 
        xml_path: str,
        text_prompt: str, 
        text_prompt_2: str | None, 
        text_prompt_3: str | None, 
        video_history_len: int = 33,
        state_history_len: int = 4,
        execution_horizon: int = 10,
        max_steps: int = 500
    ):
        self.policy = policy
        self.text_prompt = text_prompt
        self.text_prompt_2 = text_prompt_2
        self.text_prompt_3 = text_prompt_3
        self.video_history_len = video_history_len
        self.state_history_len = state_history_len
        self.execution_horizon = execution_horizon
        self.max_steps = max_steps
        
        self.video_keys = [
            "video.exterior_image_1_left",
            "video.exterior_image_2_left",
            "video.wrist_image_left"
        ]
        self.state_keys = [
            "state.joint_position",
            "state.gripper_position"
        ]
        
        # --- MuJoCo Setup ---
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        
        # Setup Offscreen Renderers (320x180)
        self.renderer_left = mujoco.Renderer(self.model, height=180, width=320)
        self.renderer_right = mujoco.Renderer(self.model, height=180, width=320)
        self.renderer_wrist = mujoco.Renderer(self.model, height=180, width=320)
        
        # Get Camera IDs
        self.cam_left_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, "left_cam")
        self.cam_right_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, "right_cam")
        self.cam_wrist_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, "wrist_cam")
        
        # Joint indices and qpos addresses
        self.arm_joint_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"joint{i+1}") for i in range(7)]
        self.finger_joint_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "finger_joint1"), 
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "finger_joint2")
        ]
        self.arm_qpos_adr = [self.model.jnt_qposadr[i] for i in self.arm_joint_ids]
        self.finger_qpos_adr = [self.model.jnt_qposadr[i] for i in self.finger_joint_ids]
        
        self.wrist_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "hand")

    def _get_camera_image(self, renderer, cam_id):
        """Renders an image from a specific MuJoCo camera ID."""
        renderer.update_scene(self.data, camera=cam_id)
        return renderer.render()

    def _map_sim_obs_to_model(self) -> dict:
        """Get current sim state and format it for the model."""
        arm_pos = np.array([self.data.qpos[adr] for adr in self.arm_qpos_adr])
        finger_pos = np.array([self.data.qpos[adr] for adr in self.finger_qpos_adr])
        gripper_pos = np.array([finger_pos.mean()])
        
        img1 = self._get_camera_image(self.renderer_left, self.cam_left_id)
        img2 = self._get_camera_image(self.renderer_right, self.cam_right_id)
        img3 = self._get_camera_image(self.renderer_wrist, self.cam_wrist_id)
        
        return {
            "video.exterior_image_1_left": img1,
            "video.exterior_image_2_left": img2,
            "video.wrist_image_left": img3,
            "state.joint_position": arm_pos,
            "state.gripper_position": gripper_pos
        }

    def _init_history(self, initial_obs: dict):
        """Pre-fill the history buffers."""
        self.video_history = {k: deque(maxlen=self.video_history_len) for k in self.video_keys}
        self.state_history = {k: deque(maxlen=self.state_history_len) for k in self.state_keys}
        for _ in range(self.video_history_len):
            for k in self.video_keys:
                self.video_history[k].append(initial_obs[k])
        for _ in range(self.state_history_len):
            for k in self.state_keys:
                self.state_history[k].append(initial_obs[k])

    def _construct_model_input(self) -> dict:
        """Construct the observation dictionary expected by WAMPolicy."""
        obs_dict = {}
        for k in self.video_keys:
            obs_dict[k] = np.stack(list(self.video_history[k]))
        for k in self.state_keys:
            obs_dict[k] = np.stack(list(self.state_history[k]))
        obs_dict["annotation.language.language_instruction"] = self.text_prompt
        #obs_dict["annotation.language.language_instruction_2"] = self.text_prompt_2
        #obs_dict["annotation.language.language_instruction_3"] = self.text_prompt_3
        return obs_dict

    def _calculate_future_trajectory_xyz(self, action_chunk):
        """Forward kinematics: Convert 96 joint angles into 3D XYZ points for Rerun."""
        original_qpos = self.data.qpos.copy()
        future_points = []
        for action in action_chunk:
            for i, adr in enumerate(self.arm_qpos_adr):
                self.data.qpos[adr] = action[i]
            for adr in self.finger_qpos_adr:
                self.data.qpos[adr] = action[7]
            mujoco.mj_forward(self.model, self.data)
            future_points.append(self.data.xpos[self.wrist_body_id].copy())
        # Restore original state
        self.data.qpos[:] = original_qpos
        mujoco.mj_forward(self.model, self.data)
        return np.array(future_points)

    def run_episode(self, rerun_save_path="rollout.rrd"):
        """Runs one full episode in MuJoCo and logs to Rerun."""
        # --- Rerun Setup ---
        rr.init("WAM_MuJoCo_Rollout")
        rr.save(rerun_save_path)
        rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP)
        
        mujoco.mj_resetData(self.model, self.data)
        
        # --- MuJoCo Non-Blocking Viewer ---
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            
            print("Warming up environment...")
            for _ in range(10):
                mujoco.mj_step(self.model, self.data)
                viewer.sync()
                _ = self._get_camera_image(self.renderer_left, self.cam_left_id)
                _ = self._get_camera_image(self.renderer_right, self.cam_right_id)
                _ = self._get_camera_image(self.renderer_wrist, self.cam_wrist_id)
                time.sleep(0.05)
            
            initial_obs = self._map_sim_obs_to_model()
            self._init_history(initial_obs)
            
            step = 0
            done = False
            
            with tqdm(total=self.max_steps, desc="MuJoCo WAM Rollout") as pbar:
                while not done and step < self.max_steps:
                    # 1. Construct input and predict
                    obs_dict = self._construct_model_input()
                    batch = Batch(obs=obs_dict)
                    
                    # [INFERENCE HAPPENS HERE - Viewer remains interactive!]
                    act_dict = self.policy.lazy_joint_forward(batch=batch)
                    act = act_dict.act
                    
                    joint_actions = act["action.joint_position"]
                    gripper_actions = act["action.gripper_position"]
                    if gripper_actions.ndim == 1:
                        gripper_actions = np.expand_dims(gripper_actions, axis=-1)
                    action_chunk = np.concatenate([joint_actions, gripper_actions], axis=-1)
                    
                    # 2. Log predicted 3D trajectory to Rerun
                    future_xyz = self._calculate_future_trajectory_xyz(action_chunk)
                    rr.log("world/predicted_trajectory", rr.LineStrips3D([future_xyz]))
                    
                    # 3. Execute the first `execution_horizon` actions
                    for i in range(self.execution_horizon):
                        if step >= self.max_steps:
                            break
                        action = action_chunk[i]
                        
                        # Set controls (Directly setting qpos for stable visualization)
                        for j, adr in enumerate(self.arm_qpos_adr):
                            self.data.qpos[adr] = action[j]
                        gripper_target = float(np.clip(action[7], 0.0, 0.04))
                        for adr in self.finger_qpos_adr:
                            self.data.qpos[adr] = gripper_target
                            
                        self.data.qvel[:] = 0
                        mujoco.mj_forward(self.model, self.data)
                        viewer.sync()
                        
                        # 4. Update history and log to Rerun
                        mapped_next = self._map_sim_obs_to_model()
                        for k in self.video_keys:
                            self.video_history[k].append(mapped_next[k])
                        for k in self.state_keys:
                            self.state_history[k].append(mapped_next[k])
                        
                        step += 1
                        pbar.update(1)
                        
                        # --- RERUN LOGGING ---
                        img1 = mapped_next["video.exterior_image_1_left"]
                        img2 = mapped_next["video.exterior_image_2_left"]
                        img3 = mapped_next["video.wrist_image_left"]
                        
                        # Stitch exactly as model sees it
                        wrist_wide = np.repeat(img3, 2, axis=1)
                        bottom_row = np.hstack((img1, img2))
                        stitched_view = np.vstack((wrist_wide, bottom_row))
                        
                        rr.set_time("step", sequence=step)
                        rr.log("observation/stitched_view", rr.Image(stitched_view))
                        
                        # Log actual joint states
                        rr.log("state/joint_1", rr.Scalars(mapped_next["state.joint_position"][0]))
                        
        print(f"\nRerun recording saved to: {rerun_save_path}")

if __name__ == "__main__":
    XML_PATH = "./env/franka_emika_panda/panda_pick_n_place.xml" 
    
    checkpoint_path = pathlib.Path("checkpoint-0")
    finetuned_checkpoint_path = pathlib.Path("checkpoint-finetune-2800")
    metadata_json_path = checkpoint_path / "experiment_cfg/metadata.json"
    
    policy = WAMPolicy(
        checkpoint_path=checkpoint_path,
        metadata_json_path=metadata_json_path,
        #finetuned_checkpoint_path=finetuned_checkpoint_path
    )
    
    evaluator = WAMRolloutMuJoCo(
        policy=policy,
        xml_path=XML_PATH,
        text_prompt="Pick up the blue ring from the table and put it in the wooden tray",
        #text_prompt_2="Put the blue ring on the tray",
        #text_prompt_3="Put the blue ring in the box",
        text_prompt_2=None,
        text_prompt_3=None,
        video_history_len=33,
        state_history_len=4,
        execution_horizon=10,
        max_steps=500
    )
    
    evaluator.run_episode(rerun_save_path="wam_rollout_mujoco.rrd")
