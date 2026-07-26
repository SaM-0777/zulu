from __future__ import annotations
from collections import defaultdict
import copy
import json
import logging
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
import time
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

from src.data.transforms.concat import ConcatTransform
from src.data.transforms.state_action import StateActionToTensor, StateActionTransform
from src.data.transforms.video import (
    VideoColorJitter,
    VideoCrop,
    VideoResize,
    VideoToNumpy,
    VideoToTensor,
)
from src.data.transforms_base import ComposedModalityTransform, ModalityTransform
from src.data.transforms.language import LanguageTransform
from src.data.video_utils import get_frames_by_timestamps
from src.data.zulu_transform import ZuluTransform
from tqdm import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizer  # type: ignore

from src.data.schema import (
    DatasetMetadata,
    DatasetStatisticalValues,
    EmbodimentTag,
    LeRobotModalityMetadata,
    LeRobotStateActionMetadata,
    StateActionMetadata,
)
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, IterableDataset, get_worker_info
from pydantic import BaseModel, Field, ValidationError, model_validator

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


LE_ROBOT_MODALITY_FILENAME = "meta/modality.json"
LE_ROBOT_EPISODE_FILENAME = "meta/episodes.jsonl"
LE_ROBOT_TASKS_FILENAME = "meta/tasks.jsonl"
LE_ROBOT_INFO_FILENAME = "meta/info.json"
LE_ROBOT_STATS_FILENAME = "meta/stats.json"
LE_ROBOT_DATA_FILENAME = "data/*/*.parquet"
METADATA_DIR = Path("")  # dummy
LE_ROBOT_TASK_EMBEDDINGS_FILENAME = "meta/task_embeddings.pt"
LEROBOT_RELATIVE_STATS_FILE_NAME = "meta/relative_stats_zuluz.json"
LEROBOT_RELATIVE_HORIZON_STATS_FILE_NAME = (
    "meta/relative_horizon_stats_zuluz.json"
)
STEP_FILTER_FILENAME = "meta/step_filter.jsonl"
LE_ROBOT_DETAILED_GLOBAL_INSTRUCTION_FILENAME = (
    "meta/episodes_detail_global_instruction.jsonl"
)

METADATA_LANG_KEYS = [
    "detailed_global_instruction_medium",
    "detailed_global_instruction_concise",
]


def calculate_dataset_statistics(
    parquet_paths: list[Path], features: list[str] | None = None
) -> dict[str, DatasetStatisticalValues]:
    """Calculate the dataset statistics of all columns for a list of parquet files.

    Args:
        parquet_paths (list[Path]): List of paths to parquet files to process.
        features (list[str] | None): List of feature names to compute statistics for.
            If None, computes statistics for all columns in the data.

    Returns:
        dict[str, DatasetStatisticalValues]: Dictionary mapping feature names to their
            statistical values (mean, std, min, max, q01, q99).
    """
    # Dataset statistics
    all_low_dim_data_list = []
    # Collect all the data
    for parquet_path in tqdm(
        sorted(list(parquet_paths)),
        desc="Collecting all parquet files...",
    ):
        # Load the parquet file
        parquet_data = pd.read_parquet(parquet_path)
        parquet_data = parquet_data
        all_low_dim_data_list.append(parquet_data)
    all_low_dim_data = pd.concat(all_low_dim_data_list, axis=0)
    # Compute dataset statistics
    dataset_statistics = {}
    if features is None:
        features = list(all_low_dim_data.columns)
    for le_modality in features:
        print(f"Computing statistics for {le_modality}...")
        np_data = np.vstack(
            [np.asarray(x, dtype=np.float32) for x in all_low_dim_data[le_modality]]
        )
        dataset_statistics[le_modality] = DatasetStatisticalValues(
            mean=np.mean(np_data, axis=0).tolist(),
            std=np.std(np_data, axis=0).tolist(),
            min=np.min(np_data, axis=0).tolist(),
            max=np.max(np_data, axis=0).tolist(),
            q01=np.quantile(np_data, 0.01, axis=0).tolist(),
            q99=np.quantile(np_data, 0.99, axis=0).tolist(),
        )
    return dataset_statistics


class MemorySafeCopyTransform(ModalityTransform):
    """
    Forces Python to physically copy the PyAV C++ memory block into native RAM.
    This prevents Segmentation Faults when the underlying video cache is destroyed.
    """

    apply_to: List[str] = []
    training: bool = True

    def apply(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply the transformation to the data dictionary and return the processed copy.
        """
        return copy.deepcopy(data)


class ModalityConfig(BaseModel):
    delta_indices: List[int]
    modality_keys: List[str]
    eval_delta_indices: List[int] | None = None

    def model_post_init(self, *args, **kwargs):
        super().model_post_init(*args, **kwargs)
        if self.eval_delta_indices is None:
            self.eval_delta_indices = self.delta_indices


class LeRobotSingleDataset(Dataset):
    def __init__(
        self,
        dataset_path: Path | str,
        modality_configs: Dict[str, ModalityConfig],
        embodiment_tag: EmbodimentTag,
        use_global_metadata: bool = False,
        metadata_version: str | None = None,
        max_chunk_size: int = 5,
        transforms: ComposedModalityTransform | None = None,
        video_backend: str = "pyav",
        video_backend_kwargs: dict | None = None,
        fps: float | None = None,
        relative_action: bool = True,
        relative_action_per_horizon: bool = True,
        relative_action_keys: list[str] | None = None,
        discard_bad_trajectories: bool = True,
    ):
        if not Path(dataset_path).exists():
            raise FileNotFoundError(f"Dataset path {dataset_path} does not exist")

        self.modality_configs = modality_configs
        self.use_global_metadata = use_global_metadata
        self.metadata_version = metadata_version
        self.video_backend = video_backend
        self.video_backend_kwargs = (
            video_backend_kwargs if video_backend_kwargs is not None else {}
        )
        self.fps = fps
        self.transforms = (
            transforms
            if transforms is not None
            else ComposedModalityTransform(transforms=[])
        )
        self.max_chunk_size = max_chunk_size
        self.discard_bad_trajectories = discard_bad_trajectories

        self.relative_action = relative_action
        self.relative_action_per_horizon = relative_action_per_horizon
        self.relative_action_keys = relative_action_keys
        self.relative_action_keys_input = relative_action_keys  # Store original input

        self._dataset_path = Path(dataset_path)
        self._dataset_name = self._dataset_path.name

        self._lerobot_modality_meta = self._get_lerobot_modality_meta()
        self._lerobot_info_meta = self._get_lerobot_info_meta()
        self._lerobot_stats_meta = self._get_lerobot_stats_meta()
        self._trajectory_ids, self._trajectory_lengths = self._get_trajectories()
        self._data_path_pattern = self._get_data_path_pattern()
        self._chunk_size = self._get_chunk_size()
        self.tag = embodiment_tag

        if self.relative_action and self.relative_action_keys_input is None:
            # Default: apply to all action keys except those containing 'gripper'
            action_keys = self.modality_configs.get(
                "action", ModalityConfig(delta_indices=[0], modality_keys=[])
            ).modality_keys
            self.relative_action_keys = [
                k.replace("action.", "")
                for k in action_keys
                if "gripper" not in k.lower()
            ]
            print(
                f"Relative action will be applied to keys: {self.relative_action_keys}"
            )

        self._lerobot_relative_stats_meta = (
            self._get_lerobot_relative_stats_meta() if self.relative_action else {}
        )
        self._lerobot_relative_horizon_stats_meta = (
            self._get_lerobot_relative_horizon_stats_meta()
            if self.relative_action_per_horizon
            else {}
        )
        self._metadata = self._get_metadata()
        self._step_filter = self._get_step_filter()
        self._all_steps = self._get_all_steps()
        self._modality_keys = self._get_modality_keys()
        self._delta_indices = self._get_delta_indices()
        self._max_delta_index = self._get_max_delta_index()

        # NOTE(YL): method to predict the task progress
        if "action.task_progress" in self._modality_keys["action"]:
            # from groot.vla.data.schema import StateActionMetadata

            print("we will add task progress to the action modality")
            self._modality_keys["action"].append("action.task_progress")
            self._metadata.modalities.action["task_progress"] = StateActionMetadata(
                absolute=True, rotation_type=None, shape=(1,), continuous=True
            )
            # assume the task progress is uniformly distributed between 0 and 1
            self._metadata.statistics.action["task_progress"] = (
                DatasetStatisticalValues(
                    max=np.array([1.0]),
                    min=np.array([0.0]),
                    mean=np.array([0.5]),
                    std=np.array([0.2887]),
                    q01=np.array([0.01]),
                    q99=np.array([0.99]),
                )
            )

        self.set_transforms_metadata(self._metadata)
        self.set_epoch(0)

        print(f"Initialized dataset {self._dataset_name} with {embodiment_tag}")

        # LeRobot-specific config (some already initialized above for relative stats)
        self._video_path_pattern = self._get_video_path_pattern()
        self._tasks = self._get_tasks()
        self._detailed_global_instructions = self._get_detailed_global_instructions()
        self.curr_traj_data = None
        self.curr_traj_id = None

        self._check_integrity()

    @property
    def dataset_path(self) -> Path:
        """The path to the dataset that contains the METADATA_FILENAME file."""
        return self._dataset_path

    @property
    def metadata(self) -> DatasetMetadata:
        """The metadata for the dataset, loaded from metadata.json in the dataset directory"""
        return self._metadata

    @property
    def trajectory_ids(self) -> np.ndarray:
        """The trajectory IDs in the dataset, stored as a 1D numpy array of strings."""
        return self._trajectory_ids

    @property
    def trajectory_lengths(self) -> np.ndarray:
        """The trajectory lengths in the dataset, stored as a 1D numpy array of integers.
        The order of the lengths is the same as the order of the trajectory IDs.
        """
        return self._trajectory_lengths

    @property
    def all_steps(self) -> list[tuple[int, int]]:
        """The trajectory IDs and base indices for all steps in the dataset.
        Example:
            self.trajectory_ids: [0, 1, 2]
            self.trajectory_lengths: [3, 2, 4]
            return: [
                ("traj_0", 0), ("traj_0", 1), ("traj_0", 2),
                ("traj_1", 0), ("traj_1", 1),
                ("traj_2", 0), ("traj_2", 1), ("traj_2", 2), ("traj_2", 3)
            ]
        """
        return self._all_steps

    @property
    def modality_keys(self) -> dict:
        """The modality keys for the dataset. The keys are the modality names, and the values are the keys for each modality.

        Example: {
            "video": ["video.image_side_0", "video.image_side_1"],
            "state": ["state.eef_position", "state.eef_rotation"],
            "action": ["action.eef_position", "action.eef_rotation"],
            "language": ["language.human.task"],
            "timestamp": ["timestamp"],
            "reward": ["reward"],
        }
        """
        return self._modality_keys

    @property
    def delta_indices(self) -> dict[str, np.ndarray]:
        """The delta indices for the dataset. The keys are the modality.key, and the values are the delta indices for each modality.key."""
        return self._delta_indices

    def _get_max_delta_index(self) -> int:
        """Calculate the maximum delta index across all modalities.

        Returns:
            int: The maximum delta index value.
        """
        max_delta_index = 0
        for delta_index in self.delta_indices.values():
            max_delta_index = max(max_delta_index, delta_index.max())
        return max_delta_index

    @property
    def max_delta_index(self) -> int:
        """The maximum delta index across all modalities."""
        return self._max_delta_index

    @property
    def dataset_name(self) -> str:
        """The name of the dataset."""
        return self._dataset_name

    @property
    def lerobot_modality_meta(self) -> LeRobotModalityMetadata:
        """The metadata for the LeRobot dataset."""
        return self._lerobot_modality_meta

    @property
    def lerobot_info_meta(self) -> dict:
        """The metadata for the LeRobot dataset."""
        return self._lerobot_info_meta

    @property
    def lerobot_stats_meta(self) -> dict[str, DatasetStatisticalValues]:
        """The metadata for the LeRobot dataset."""
        return self._lerobot_stats_meta

    @property
    def lerobot_relative_stats_meta(self) -> dict[str, DatasetStatisticalValues]:
        """The relative action stats metadata for the LeRobot dataset."""
        return self._lerobot_relative_stats_meta

    @property
    def lerobot_relative_horizon_stats_meta(self) -> dict[str, dict[str, list]]:
        """The per-horizon relative action stats metadata for the LeRobot dataset.

        Format: {action_key: {stat_name: [[h0_vals], [h1_vals], ...]}}
        """
        return self._lerobot_relative_horizon_stats_meta

    @property
    def step_filter(self) -> dict[int, np.ndarray]:
        """The step filter for the dataset."""
        return self._step_filter

    @property
    def data_path_pattern(self) -> str:
        """The path pattern for the LeRobot dataset."""
        return self._data_path_pattern

    @property
    def video_path_pattern(self) -> str:
        """The path pattern for the LeRobot dataset."""
        return self._video_path_pattern

    @property
    def chunk_size(self) -> int:
        """The chunk size for the LeRobot dataset."""
        return self._chunk_size

    @property
    def tasks(self) -> pd.DataFrame:
        """The tasks for the dataset."""
        return self._tasks

    def _get_lerobot_modality_meta(self) -> LeRobotModalityMetadata:
        """Get the metadata for the LeRobot dataset."""
        if self.use_global_metadata:
            assert (
                self.metadata_version is not None
            ), "metadata_version must be provided if use_global_metadata is True"
            modality_meta_path = (
                METADATA_DIR  # dummy
                / self.tag.value
                / self.metadata_version
                / Path(LE_ROBOT_MODALITY_FILENAME).name
            )
            assert (
                modality_meta_path.exists()
            ), f"Please provide a {Path(LE_ROBOT_MODALITY_FILENAME).name} file in {METADATA_DIR / self.tag.value / self.metadata_version}"
            with open(modality_meta_path, "r") as f:
                modality_meta = LeRobotModalityMetadata.model_validate(json.load(f))
            return modality_meta
        else:
            modality_meta_path = self.dataset_path / LE_ROBOT_MODALITY_FILENAME
            assert (
                modality_meta_path.exists()
            ), f"Please provide a {LE_ROBOT_MODALITY_FILENAME} file in {self.dataset_path}"
            with open(modality_meta_path, "r") as f:
                modality_meta = LeRobotModalityMetadata.model_validate(json.load(f))
            return modality_meta

    def _get_lerobot_info_meta(self) -> dict:
        """Get the metadata for the LeRobot dataset."""
        info_meta_path = self.dataset_path / LE_ROBOT_INFO_FILENAME
        with open(info_meta_path, "r") as f:
            info_meta = json.load(f)
        return info_meta

    def _get_lerobot_stats_meta(self) -> dict[str, DatasetStatisticalValues]:
        """Get the metadata for the LeRobot dataset."""
        if self.use_global_metadata:
            assert (
                self.metadata_version is not None
            ), "metadata_version must be provided if use_global_metadata is True"
            stats_path = (
                METADATA_DIR
                / self.tag.value
                / self.metadata_version
                / Path(LE_ROBOT_STATS_FILENAME).name
            )
        else:
            stats_path = self.dataset_path / LE_ROBOT_STATS_FILENAME
        try:
            with open(stats_path, "r") as f:
                stats: dict = json.load(f)
            for name in ["num_trajectories", "total_trajectory_length"]:
                stats.pop(name, None)
            for name, stat in stats.items():
                stats[name] = DatasetStatisticalValues.model_validate(stat)
            return stats
        except (FileNotFoundError, ValidationError) as e:
            if self.use_global_metadata:
                raise ValueError(
                    f"{e}: Please provide a {Path(LE_ROBOT_STATS_FILENAME).name} file in {stats_path}"
                    " and ensure the metadata format is correct."
                )
            print(f"Failed to load dataset statistics: {e}")
            print(f"Calculating dataset statistics for {self.dataset_name}")
            # Get all parquet files in the dataset paths
            parquet_files = list((self.dataset_path).glob(LE_ROBOT_DATA_FILENAME))
            lowdim_features = []
            le_features = self.lerobot_info_meta["features"]
            for feature in le_features:
                if "float" in le_features[feature]["dtype"]:
                    lowdim_features.append(feature)

            stats = calculate_dataset_statistics(parquet_files, lowdim_features)
            stats_serialized = {k: v.model_dump(mode="json") for k, v in stats.items()}
            with open(stats_path, "w") as f:
                json.dump(stats_serialized, f, indent=4)
            return stats

    def _get_lerobot_relative_stats_meta(self) -> dict[str, DatasetStatisticalValues]:
        """Get the relative action stats metadata for the LeRobot dataset.

        Returns:
            dict[str, DatasetStatisticalValues]: Dictionary mapping action keys to their relative stats.
        """
        # Determine the path for relative stats file
        if self.use_global_metadata:
            assert (
                self.metadata_version is not None
            ), "metadata_version must be provided if use_global_metadata is True"
            stats_path = (
                METADATA_DIR
                / self.tag.value
                / self.metadata_version
                / Path(LEROBOT_RELATIVE_STATS_FILE_NAME).name
            )
            assert (
                stats_path.exists()
            ), f"Please provide a {Path(LEROBOT_RELATIVE_STATS_FILE_NAME).name} file in {METADATA_DIR / self.tag.value / self.metadata_version}"
        else:
            stats_path = self.dataset_path / LEROBOT_RELATIVE_STATS_FILE_NAME

        # Try to load existing relative stats
        if stats_path.exists():
            print(f"Loading relative action stats from {stats_path}")
            with open(stats_path, "r") as f:
                stats: dict = json.load(f)
            for name, stat in stats.items():
                stats[name] = DatasetStatisticalValues.model_validate(stat)
            return stats

        # Calculate relative stats if file doesn't exist
        print(f"Relative stats file not found at {stats_path}")
        print(f"Calculating relative action stats for {self.dataset_name}")

        # Get action keys from modality configs, filtered by relative_action_keys
        all_action_keys = self.modality_configs.get(
            "action", ModalityConfig(delta_indices=[0], modality_keys=[])
        ).modality_keys
        if not all_action_keys:
            print(
                "No action keys found in modality configs, skipping relative stats calculation"
            )
            return {}

        # Filter to only the keys that should use relative action
        action_keys_to_process = []
        for key in all_action_keys:
            subkey = key.replace("action.", "")
            if self.relative_action_keys is None or subkey in self.relative_action_keys:
                action_keys_to_process.append(subkey)

        if not action_keys_to_process:
            print("No action keys to process for relative stats")
            return {}

        print(f"Will calculate relative stats for: {action_keys_to_process}")

        stats = {}
        for action_key in action_keys_to_process:
            print(f"Calculating relative stats for action key: {action_key}")
            try:
                relative_stats = self._calculate_relative_stats_for_key(action_key)
                stats[action_key] = relative_stats
            except Exception as e:
                print(f"Failed to calculate relative stats for {action_key}: {e}")
                continue

        if stats:
            # Save the calculated stats
            stats_serialized = {k: v.model_dump(mode="json") for k, v in stats.items()}
            # Only save to dataset path (not global metadata path)
            save_path = self.dataset_path / LEROBOT_RELATIVE_STATS_FILE_NAME
            print(f"Saving relative action stats to {save_path}")
            with open(save_path, "w") as f:
                json.dump(stats_serialized, f, indent=4)

        return stats

    def _get_lerobot_relative_horizon_stats_meta(self) -> dict[str, dict[str, list]]:
        """Get the per-horizon relative action stats metadata for the LeRobot dataset.

        Similar to _get_lerobot_relative_stats_meta but calculates separate stats for each
        action horizon index. Will load from file if exists, otherwise calculate and save.

        Returns:
            dict[str, dict[str, list]]: Nested dictionary where:
                - Outer key is the action key (e.g., 'joint_position')
                - Inner key is the stat name (e.g., 'max', 'min', 'mean', 'std', 'q01', 'q99')
                - Value is a list of stat values per horizon index
        """
        # Determine the path for per-horizon relative stats file
        if self.use_global_metadata:
            assert (
                self.metadata_version is not None
            ), "metadata_version must be provided if use_global_metadata is True"
            stats_path = (
                METADATA_DIR
                / self.tag.value
                / self.metadata_version
                / Path(LEROBOT_RELATIVE_HORIZON_STATS_FILE_NAME).name
            )
            assert (
                stats_path.exists()
            ), f"Please provide a {Path(LEROBOT_RELATIVE_HORIZON_STATS_FILE_NAME).name} file in {METADATA_DIR / self.tag.value / self.metadata_version}"
        else:
            stats_path = self.dataset_path / LEROBOT_RELATIVE_HORIZON_STATS_FILE_NAME

        # Try to load existing per-horizon relative stats
        if stats_path.exists():
            print(f"Loading per-horizon relative action stats from {stats_path}")
            with open(stats_path, "r") as f:
                stats: dict = json.load(f)
            return stats

        # Calculate per-horizon relative stats if file doesn't exist
        print(f"Per-horizon relative stats file not found at {stats_path}")
        print(f"Calculating per-horizon relative action stats for {self.dataset_name}")

        # Get action keys from modality configs, filtered by relative_action_keys
        all_action_keys = self.modality_configs.get(
            "action", ModalityConfig(delta_indices=[0], modality_keys=[])
        ).modality_keys
        if not all_action_keys:
            print(
                "No action keys found in modality configs, skipping per-horizon relative stats calculation"
            )
            return {}

        # Filter to only the keys that should use relative action
        action_keys_to_process = []
        for key in all_action_keys:
            subkey = key.replace("action.", "")
            if self.relative_action_keys is None or subkey in self.relative_action_keys:
                action_keys_to_process.append(subkey)

        if not action_keys_to_process:
            print("No action keys to process for per-horizon relative stats")
            return {}

        print(
            f"Will calculate per-horizon relative stats for: {action_keys_to_process}"
        )

        stats = {}
        for action_key in action_keys_to_process:
            print(
                f"Calculating per-horizon relative stats for action key: {action_key}"
            )
            try:
                relative_stats = self._calculate_relative_stats_for_key_per_horizon(
                    action_key
                )
                stats[action_key] = relative_stats
            except Exception as e:
                print(
                    f"Failed to calculate per-horizon relative stats for {action_key}: {e}"
                )
                continue

        if stats:
            # Only save to dataset path (not global metadata path)
            save_path = self.dataset_path / LEROBOT_RELATIVE_HORIZON_STATS_FILE_NAME
            print(f"Saving per-horizon relative action stats to {save_path}")
            with open(save_path, "w") as f:
                json.dump(stats, f, indent=4)

        return stats

    def _calculate_relative_stats_for_key(
        self, action_key: str
    ) -> DatasetStatisticalValues:
        """Calculate relative action statistics for a specific action key.

        Args:
            action_key: The action key to calculate stats for (e.g., 'joint_position')

        Returns:
            DatasetStatisticalValues: The calculated statistics for the relative action.
        """
        # Get state and action metadata from lerobot modality config
        state_key = action_key  # Assume state key matches action key

        # Get the modality metadata to find original column names and indices
        state_meta = self.lerobot_modality_meta.state.get(state_key)
        action_meta = self.lerobot_modality_meta.action.get(action_key)

        if state_meta is None:
            raise ValueError(f"State key '{state_key}' not found in modality metadata")
        if action_meta is None:
            raise ValueError(
                f"Action key '{action_key}' not found in modality metadata"
            )

        # Get the original column names (e.g., 'observation.state', 'action')
        state_original_key = state_meta.original_key
        action_original_key = action_meta.original_key

        # Get the indices to slice from the concatenated vectors
        state_start, state_end = state_meta.start, state_meta.end
        action_start, action_end = action_meta.start, action_meta.end

        state_delta_indices = self.modality_configs.get(
            "state", ModalityConfig(delta_indices=[0], modality_keys=[])
        ).delta_indices
        action_delta_indices = self.modality_configs["action"].delta_indices

        print(f"Calculating relative stats for {action_key}:")
        print(
            f"  State: column='{state_original_key}', indices=[{state_start}:{state_end}]"
        )
        print(
            f"  Action: column='{action_original_key}', indices=[{action_start}:{action_end}]"
        )

        # # Calculate relative actions for all trajectories
        all_relative_actions = []

        # for traj_id in tqdm(self.trajectory_ids, desc=f"Calculating relative stats for {action_key}"):
        max_trajs_for_stats = 10000
        traj_ids_to_process = self.trajectory_ids
        if len(traj_ids_to_process) > max_trajs_for_stats:
            # Randomly sample 500 trajectories
            rng = np.random.default_rng(seed=42)
            sampled_indices = rng.choice(
                len(traj_ids_to_process), size=max_trajs_for_stats, replace=False
            )
            traj_ids_to_process = traj_ids_to_process[sampled_indices]
            print(
                f"Sampling {max_trajs_for_stats} trajectories out of {len(self.trajectory_ids)} for stats calculation"
            )

        # Calculate relative actions for sampled trajectories
        all_relative_actions = []

        for traj_id in tqdm(
            traj_ids_to_process, desc=f"Calculating relative stats for {action_key}"
        ):
            try:
                # Load trajectory data
                traj_data = self._load_trajectory_data(traj_id)
                if traj_data is None:
                    continue

                # Check if columns exist
                if (
                    state_original_key not in traj_data.columns
                    or action_original_key not in traj_data.columns
                ):
                    print(
                        f"Missing columns: state='{state_original_key}' exists={state_original_key in traj_data.columns}, "
                        f"action='{action_original_key}' exists={action_original_key in traj_data.columns}"
                    )
                    continue

                # Load full state and action arrays, then slice to get the specific component
                full_state_data = np.stack(traj_data[state_original_key].tolist())
                full_action_data = np.stack(traj_data[action_original_key].tolist())

                # Slice to get just the component we care about (e.g., joint_position)
                state_data = full_state_data[:, state_start:state_end]
                action_data = full_action_data[:, action_start:action_end]

                # Calculate usable length based on action delta indices
                usable_length = len(traj_data) - max(action_delta_indices)

                for i in range(usable_length):
                    # Get reference state (last state before action chunk)
                    ref_state_idx = state_delta_indices[-1] + i
                    if ref_state_idx >= len(state_data):
                        continue
                    ref_state = state_data[ref_state_idx]

                    # Get action chunk
                    action_indices = [idx + i for idx in action_delta_indices]
                    if max(action_indices) >= len(action_data):
                        continue
                    actions = action_data[action_indices]

                    # print("actions shape", actions.shape, "ref_state shape", ref_state.shape)

                    # Calculate relative actions (action - reference state)
                    relative_actions = actions - ref_state
                    all_relative_actions.extend(relative_actions)

            except Exception as e:
                print(f"Error processing trajectory {traj_id}: {e}")
                continue

        if not all_relative_actions:
            raise ValueError(f"No relative actions calculated for {action_key}")

        all_relative_actions = np.array(all_relative_actions)
        print(
            f"Collected {len(all_relative_actions)} relative action samples for {action_key}"
        )

        return DatasetStatisticalValues(
            max=np.max(all_relative_actions, axis=0).tolist(),
            min=np.min(all_relative_actions, axis=0).tolist(),
            mean=np.mean(all_relative_actions, axis=0).tolist(),
            std=np.std(all_relative_actions, axis=0).tolist(),
            q01=np.quantile(all_relative_actions, 0.01, axis=0).tolist(),
            q99=np.quantile(all_relative_actions, 0.99, axis=0).tolist(),
        )

    def _calculate_relative_stats_for_key_per_horizon(
        self, action_key: str
    ) -> dict[str, list]:
        """Calculate relative action statistics for each delta index (horizon step) separately.

        Unlike `_calculate_relative_stats_for_key` which pools all horizon steps together,
        this method calculates separate statistics for each action horizon index.

        Args:
            action_key: The action key to calculate stats for (e.g., 'joint_position')

        Returns:
            dict[str, list]: Dictionary where keys are stat names (max, min, mean, std, q01, q99)
                and values are lists of stat values per horizon index.
                Format: {"max": [[h0_vals], [h1_vals], ...], "min": [...], ...}
        """
        # Get state and action metadata from lerobot modality config
        state_key = action_key  # Assume state key matches action key

        # Get the modality metadata to find original column names and indices
        state_meta = self.lerobot_modality_meta.state.get(state_key)
        action_meta = self.lerobot_modality_meta.action.get(action_key)

        if state_meta is None:
            raise ValueError(f"State key '{state_key}' not found in modality metadata")
        if action_meta is None:
            raise ValueError(
                f"Action key '{action_key}' not found in modality metadata"
            )

        # Get the original column names (e.g., 'observation.state', 'action')
        state_original_key = state_meta.original_key
        action_original_key = action_meta.original_key

        # Get the indices to slice from the concatenated vectors
        state_start, state_end = state_meta.start, state_meta.end
        action_start, action_end = action_meta.start, action_meta.end

        state_delta_indices = self.modality_configs.get(
            "state", ModalityConfig(delta_indices=[0], modality_keys=[])
        ).delta_indices
        action_delta_indices = self.modality_configs["action"].delta_indices

        print(f"Calculating per-horizon relative stats for {action_key}:")
        print(
            f"  State: column='{state_original_key}', indices=[{state_start}:{state_end}]"
        )
        print(
            f"  Action: column='{action_original_key}', indices=[{action_start}:{action_end}]"
        )
        print(f"  Action delta indices: {action_delta_indices}")

        # Initialize separate lists for each horizon index
        all_relative_actions_per_horizon: dict[int, list] = {
            delta_idx: [] for delta_idx in action_delta_indices
        }

        max_trajs_for_stats = 10000
        traj_ids_to_process = self.trajectory_ids
        if len(traj_ids_to_process) > max_trajs_for_stats:
            # Randomly sample trajectories
            rng = np.random.default_rng(seed=42)
            sampled_indices = rng.choice(
                len(traj_ids_to_process), size=max_trajs_for_stats, replace=False
            )
            traj_ids_to_process = traj_ids_to_process[sampled_indices]
            print(
                f"Sampling {max_trajs_for_stats} trajectories out of {len(self.trajectory_ids)} for stats calculation"
            )

        for traj_id in tqdm(
            traj_ids_to_process,
            desc=f"Calculating per-horizon relative stats for {action_key}",
        ):
            try:
                # Load trajectory data
                traj_data = self._load_trajectory_data(traj_id)
                if traj_data is None:
                    continue

                # Check if columns exist
                if (
                    state_original_key not in traj_data.columns
                    or action_original_key not in traj_data.columns
                ):
                    continue

                # Load full state and action arrays, then slice to get the specific component
                full_state_data = np.stack(traj_data[state_original_key].tolist())
                full_action_data = np.stack(traj_data[action_original_key].tolist())

                # Slice to get just the component we care about (e.g., joint_position)
                state_data = full_state_data[:, state_start:state_end]
                action_data = full_action_data[:, action_start:action_end]

                # Calculate usable length based on action delta indices
                usable_length = len(traj_data) - max(action_delta_indices)

                for i in range(usable_length):
                    # Get reference state (last state before action chunk)
                    ref_state_idx = state_delta_indices[-1] + i
                    if ref_state_idx >= len(state_data):
                        continue
                    ref_state = state_data[ref_state_idx]

                    # Get action for each horizon index separately
                    for delta_idx in action_delta_indices:
                        action_idx = delta_idx + i
                        if action_idx >= len(action_data):
                            continue
                        action = action_data[action_idx]

                        # Calculate relative action (action - reference state)
                        relative_action = action - ref_state
                        all_relative_actions_per_horizon[delta_idx].append(
                            relative_action
                        )

            except Exception as e:
                print(f"Error processing trajectory {traj_id}: {e}")
                continue

        # Calculate stats for each horizon index and organize by stat name
        stat_names = ["max", "min", "mean", "std", "q01", "q99"]
        stats_by_name: dict[str, list] = {name: [] for name in stat_names}

        for delta_idx in action_delta_indices:
            relative_actions = all_relative_actions_per_horizon[delta_idx]
            if not relative_actions:
                print(
                    f"Warning: No relative actions calculated for {action_key} at horizon index {delta_idx}"
                )
                # Add empty/placeholder values
                for name in stat_names:
                    stats_by_name[name].append([])
                continue

            relative_actions_array = np.array(relative_actions)
            print(
                f"Collected {len(relative_actions_array)} relative action samples for {action_key} at horizon {delta_idx}"
            )

            stats_by_name["max"].append(np.max(relative_actions_array, axis=0).tolist())
            stats_by_name["min"].append(np.min(relative_actions_array, axis=0).tolist())
            stats_by_name["mean"].append(
                np.mean(relative_actions_array, axis=0).tolist()
            )
            stats_by_name["std"].append(np.std(relative_actions_array, axis=0).tolist())
            stats_by_name["q01"].append(
                np.quantile(relative_actions_array, 0.01, axis=0).tolist()
            )
            stats_by_name["q99"].append(
                np.quantile(relative_actions_array, 0.99, axis=0).tolist()
            )

        return stats_by_name

    def get_relative_stats_per_horizon(
        self,
        action_keys: list[str] | None = None,
        save_to_file: bool = True,
    ) -> dict[str, dict[str, list]]:
        """Get relative action stats calculated separately for each horizon index.

        This is useful when you want different normalization for different action horizon
        steps, e.g., near-future actions vs far-future actions might have different distributions.

        Args:
            action_keys: List of action keys to calculate stats for. If None, uses
                relative_action_keys (all action keys except gripper by default).
            save_to_file: Whether to save the calculated stats to a file.

        Returns:
            dict[str, dict[str, list]]: Nested dictionary where:
                - Outer key is the action key (e.g., 'joint_position')
                - Inner key is the stat name (e.g., 'max', 'min', 'mean', 'std', 'q01', 'q99')
                - Value is a list of stat values per horizon index

        Example output format:
            {
                "joint_position": {
                    "max": [[h0_vals], [h1_vals], ...],
                    "min": [[h0_vals], [h1_vals], ...],
                    ...
                }
            }
        """
        # Determine which action keys to process
        if action_keys is None:
            all_action_keys = self.modality_configs.get(
                "action", ModalityConfig(delta_indices=[0], modality_keys=[])
            ).modality_keys
            if not all_action_keys:
                print("No action keys found in modality configs")
                return {}
            # Default: apply to all action keys except those containing 'gripper'
            action_keys = [
                k.replace("action.", "")
                for k in all_action_keys
                if "gripper" not in k.lower()
            ]

        if not action_keys:
            print("No action keys to process for per-horizon relative stats")
            return {}

        print(f"Calculating per-horizon relative stats for: {action_keys}")

        all_stats: dict[str, dict[str, list]] = {}

        for action_key in action_keys:
            print(f"Processing action key: {action_key}")
            try:
                stats_per_horizon = self._calculate_relative_stats_for_key_per_horizon(
                    action_key
                )
                all_stats[action_key] = stats_per_horizon
            except Exception as e:
                print(
                    f"Failed to calculate per-horizon relative stats for {action_key}: {e}"
                )
                continue

        if save_to_file and all_stats:
            # Save to the designated file
            save_path = self.dataset_path / LEROBOT_RELATIVE_HORIZON_STATS_FILE_NAME
            print(f"Saving per-horizon relative action stats to {save_path}")

            with open(save_path, "w") as f:
                json.dump(all_stats, f, indent=4)

        return all_stats

    def load_relative_stats_per_horizon(
        self,
        stats_path: Path | str | None = None,
    ) -> dict[str, dict[str, list]]:
        """Load pre-computed per-horizon relative stats from a file.

        Args:
            stats_path: Path to the stats file. If None, uses the default path in the dataset.

        Returns:
            dict[str, dict[str, list]]: Nested dictionary of stats per horizon.
        """
        if stats_path is None:
            stats_path = self.dataset_path / LEROBOT_RELATIVE_HORIZON_STATS_FILE_NAME
        else:
            stats_path = Path(stats_path)

        if not stats_path.exists():
            print(f"Per-horizon relative stats file not found at {stats_path}")
            return {}

        print(f"Loading per-horizon relative stats from {stats_path}")
        with open(stats_path, "r") as f:
            all_stats = json.load(f)

        return all_stats

    def _load_trajectory_data(self, traj_id: int) -> pd.DataFrame | None:
        """Load trajectory data from parquet file.

        Args:
            traj_id: The trajectory ID to load.

        Returns:
            pd.DataFrame or None if loading fails.
        """
        try:
            chunk_index = traj_id // self.chunk_size
            parquet_path = (
                self.dataset_path
                / f"data/chunk-{chunk_index:03d}/episode_{traj_id:06d}.parquet"
            )
            if not parquet_path.exists():
                # Try alternative pattern
                parquet_files = list(
                    self.dataset_path.glob(f"data/*/episode_{traj_id:06d}.parquet")
                )
                if parquet_files:
                    parquet_path = parquet_files[0]
                else:
                    return None
            return pd.read_parquet(parquet_path)
        except Exception:
            return None

    def _get_step_filter(self) -> dict[int, np.ndarray]:
        """Get the step filter for the dataset."""
        step_filter_path = self.dataset_path / STEP_FILTER_FILENAME
        step_filter = {}
        if step_filter_path.exists():
            with open(step_filter_path, "r") as f:
                for line in f:
                    episode_step_filter = json.loads(line)
                    trajectory_id = episode_step_filter["episode_index"]
                    all_indices = np.arange(
                        self.trajectory_lengths[trajectory_id].item()
                    )
                    indices_to_filter = np.array(episode_step_filter["step_indices"])
                    step_filter[trajectory_id] = np.setdiff1d(
                        all_indices, indices_to_filter
                    )
        else:
            for trajectory_id in self.trajectory_ids:
                step_filter[trajectory_id] = np.arange(
                    self.trajectory_lengths[trajectory_id].item()
                )
        return step_filter

    def _get_metadata(self) -> DatasetMetadata:
        """Get the metadata for the dataset.

        Returns:
            dict: The metadata for the dataset.
        """

        # 1. Modality metadata
        # 1.1. State and action modalities
        simplified_modality_meta: dict[str, dict] = {}
        for modality in ["state", "action"]:
            simplified_modality_meta[modality] = {}
            le_state_action_meta: dict[str, LeRobotStateActionMetadata] = getattr(
                self.lerobot_modality_meta, modality
            )
            for subkey in le_state_action_meta:
                state_action_dtype = np.dtype(le_state_action_meta[subkey].dtype)
                if np.issubdtype(state_action_dtype, np.floating):
                    continuous = True
                else:
                    continuous = False
                simplified_modality_meta[modality][subkey] = {
                    "absolute": le_state_action_meta[subkey].absolute,
                    "rotation_type": le_state_action_meta[subkey].rotation_type,
                    "shape": [
                        le_state_action_meta[subkey].end
                        - le_state_action_meta[subkey].start
                    ],
                    "continuous": continuous,
                }

        # 1.2. Video modalities
        le_info_path = self.dataset_path / LE_ROBOT_INFO_FILENAME
        assert (
            le_info_path.exists()
        ), f"Please provide a {LE_ROBOT_INFO_FILENAME} file in {self.dataset_path}"
        with open(le_info_path, "r") as f:
            le_info = json.load(f)
        simplified_modality_meta["video"] = {}
        for new_key in self.lerobot_modality_meta.video:
            original_key = self.lerobot_modality_meta.video[new_key].original_key
            if original_key is None:
                original_key = new_key
            le_video_meta = le_info["features"][original_key]
            height = le_video_meta["shape"][le_video_meta["names"].index("height")]
            width = le_video_meta["shape"][le_video_meta["names"].index("width")]
            # NOTE(FH): different lerobot dataset versions have different keys for the number of channels and fps
            try:
                channels = le_video_meta["shape"][
                    le_video_meta["names"].index("channel")
                ]
                fps = le_video_meta["video_info"]["video.fps"]
            except (ValueError, KeyError):
                # channels = le_video_meta["shape"][le_video_meta["names"].index("channels")]
                channels = le_video_meta["info"]["video.channels"]
                fps = le_video_meta["info"]["video.fps"]
            simplified_modality_meta["video"][new_key] = {
                "resolution": [width, height],
                "channels": channels,
                "fps": fps,
            }

        # 2. Dataset statistics
        dataset_statistics = {}
        le_statistics = {k: v.model_dump() for k, v in self.lerobot_stats_meta.items()}
        # Prepare relative stats if available
        relative_stats = {}
        if self.relative_action and hasattr(self, "_lerobot_relative_stats_meta"):
            relative_stats = {
                k: v.model_dump() for k, v in self._lerobot_relative_stats_meta.items()
            }

        # Prepare per-horizon relative stats if available
        per_horizon_stats = {}
        if self.relative_action_per_horizon and hasattr(
            self, "_lerobot_relative_horizon_stats_meta"
        ):
            per_horizon_stats = self._lerobot_relative_horizon_stats_meta

        for our_modality in ["state", "action"]:
            dataset_statistics[our_modality] = {}
            for subkey in simplified_modality_meta[our_modality]:
                dataset_statistics[our_modality][subkey] = {}
                state_action_meta = self.lerobot_modality_meta.get_key_meta(
                    f"{our_modality}.{subkey}"
                )
                assert isinstance(state_action_meta, LeRobotStateActionMetadata)

                # Check if we should use per-horizon relative stats for this action key
                should_use_per_horizon = (
                    our_modality == "action"
                    and self.relative_action_per_horizon
                    and subkey in per_horizon_stats
                    and (
                        self.relative_action_keys is None
                        or subkey in self.relative_action_keys
                    )
                )

                # Use relative stats for action modality if relative_action is enabled and stats are available
                # Also check if this subkey is in the list of keys that should use relative action
                should_use_relative = (
                    our_modality == "action"
                    and self.relative_action
                    and subkey in relative_stats
                    and (
                        self.relative_action_keys is None
                        or subkey in self.relative_action_keys
                    )
                )

                if should_use_per_horizon:
                    # Use per-horizon relative action stats (format: {stat_name: [[h0_vals], [h1_vals], ...]})
                    for stat_name in per_horizon_stats[subkey]:
                        dataset_statistics[our_modality][subkey][stat_name] = (
                            per_horizon_stats[subkey][stat_name]
                        )
                    print(f"Using per-horizon relative stats for {subkey}")
                elif should_use_relative:
                    # Use relative action stats directly
                    for stat_name in relative_stats[subkey]:
                        dataset_statistics[our_modality][subkey][stat_name] = (
                            relative_stats[subkey][stat_name]
                        )
                    print(
                        f"Using relative stats for {subkey}: {dataset_statistics[our_modality][subkey]}"
                    )
                else:
                    # Use original absolute stats
                    le_modality = state_action_meta.original_key
                    for stat_name in le_statistics[le_modality]:
                        indices = np.arange(
                            state_action_meta.start,
                            state_action_meta.end,
                        )
                        stat = np.array(le_statistics[le_modality][stat_name])
                        dataset_statistics[our_modality][subkey][stat_name] = stat[
                            indices
                        ].tolist()

        # 3. Full dataset metadata
        metadata = DatasetMetadata(
            statistics=dataset_statistics,  # type: ignore
            modalities=simplified_modality_meta,  # type: ignore
            embodiment_tag=self.tag,
        )

        return metadata

    def _get_trajectories(self) -> tuple[np.ndarray, np.ndarray]:
        """Get the trajectories in the dataset."""
        # Get trajectory lengths, IDs, and whitelist from dataset metadata
        episode_path = self.dataset_path / LE_ROBOT_EPISODE_FILENAME
        with open(episode_path, "r") as f:
            episode_metadata = [json.loads(line) for line in f]
        trajectory_ids = []
        trajectory_lengths = []
        for episode in episode_metadata:
            trajectory_ids.append(episode["episode_index"])
            trajectory_lengths.append(episode["length"])
        return np.array(trajectory_ids), np.array(trajectory_lengths)

    def _get_all_steps(self) -> list[tuple[int, int]]:
        """Get the trajectory IDs and base indices for all steps in the dataset.

        Returns:
            list[tuple[int, int]]: A list of (trajectory_id, base_index) tuples.

        Example:
            self.trajectory_ids: [0, 1, 2]
            self.step_filter: {
                0: [0, 1, 2],
                1: [0, 1],
                2: [0, 2, 3]
            }
            return: [
                (0, 0), (0, 1), (0, 2),
                (1, 0), (1, 1),
                (2, 0), (2, 2), (2, 3)
            ]
        """
        all_steps: list[tuple[int, int]] = []
        # All steps is used in single dataset, so we need to discard bad trajectories
        # Mixture dataset directly use trajectory_ids, so we handle it by changing the sampling weights
        discarded_episode_indices = []
        if self.discard_bad_trajectories:
            discarded_episode_indices = self._lerobot_info_meta.get(
                "discarded_episode_indices", []
            )

        for trajectory_id in self.trajectory_ids:
            if trajectory_id in discarded_episode_indices:
                continue
            for base_index in self.step_filter[trajectory_id]:
                all_steps.append((trajectory_id, base_index))
        return all_steps

    def _get_modality_keys(self) -> dict:
        """Get the modality keys for the dataset.

        Returns:
            dict: Dictionary mapping modality names to their keys.
        """
        modality_keys = defaultdict(list)
        for modality, config in self.modality_configs.items():
            modality_keys[modality] = config.modality_keys
        return modality_keys

    def _get_delta_indices(self) -> dict[str, np.ndarray]:
        """Restructure the delta indices to use modality.key as keys instead of just the modalities."""
        delta_indices: dict[str, np.ndarray] = {}
        for config in self.modality_configs.values():
            for key in config.modality_keys:
                delta_indices[key] = np.array(config.delta_indices)
        return delta_indices

    def _get_data_path_pattern(self) -> str:
        """Get the data path pattern for the LeRobot dataset."""
        return self.lerobot_info_meta["data_path"]

    def _get_video_path_pattern(self) -> str:
        """Get the video path pattern for the LeRobot dataset."""
        return self.lerobot_info_meta["video_path"]

    def _get_chunk_size(self) -> int:
        """Get the chunk size for the LeRobot dataset."""
        return self.lerobot_info_meta["chunks_size"]

    def _get_tasks(self) -> pd.DataFrame:
        """Get the tasks for the dataset."""
        tasks_path = self.dataset_path / LE_ROBOT_TASKS_FILENAME
        with open(tasks_path, "r") as f:
            tasks = [json.loads(line) for line in f]
        df = pd.DataFrame(tasks)
        return df.set_index("task_index")

    def _get_task_embeddings(self) -> dict:
        """Get the task embeddings for the dataset."""
        task_embeddings_path = self.dataset_path / LE_ROBOT_TASK_EMBEDDINGS_FILENAME
        return torch.load(task_embeddings_path)

    def _get_detailed_global_instructions(self) -> dict[int, dict]:
        """Get the detailed global instructions for the dataset.

        Loads from episodes_detail_global_instruction.jsonl if it exists.

        Returns:
            dict[int, dict]: Mapping from episode_index to detailed instruction dict.
        """
        detailed_instruction_path = (
            self.dataset_path / LE_ROBOT_DETAILED_GLOBAL_INSTRUCTION_FILENAME
        )
        if not detailed_instruction_path.exists():
            return {}
        with open(detailed_instruction_path, "r") as f:
            instructions_list = [json.loads(line) for line in f]
        return {entry["episode_index"]: entry for entry in instructions_list}

    def _check_integrity(self):
        """Use the config to check if the keys are valid and detect silent data corruption."""
        ERROR_MSG_HEADER = (
            f"Error occurred in initializing dataset {self.dataset_name}:\n"
        )

        for modality, modality_config in self.modality_configs.items():
            if modality in [
                "lapa_action",
                "dream_actions",
                "rl_info",
                "task_embedding",
            ]:
                continue
            for key in modality_config.modality_keys:

                if key == "action.task_progress":
                    continue
                # Skip metadata-based language keys (they don't need modality metadata)
                if modality == "language" and key.startswith("annotation."):
                    lang_subkey = key.replace("annotation.", "")
                    if lang_subkey in METADATA_LANG_KEYS:
                        continue
                # Check if the key is valid
                try:
                    self.lerobot_modality_meta.get_key_meta(key)
                except Exception as e:
                    raise ValueError(
                        ERROR_MSG_HEADER
                        + f"Unable to find key {key} in modality metadata:\n{e}"
                    )

    def set_transforms_metadata(self, metadata: DatasetMetadata):
        """Set the metadata for the transforms. This is useful for transforms that need to know the metadata, such as the normalization values."""
        self.transforms.set_metadata(metadata)
        # Also set per-horizon statistics if available
        if self.relative_action_per_horizon and hasattr(
            self, "_lerobot_relative_horizon_stats_meta"
        ):
            if hasattr(self.transforms, "set_per_horizon_statistics"):
                self.transforms.set_per_horizon_statistics(self._lerobot_relative_horizon_stats_meta)  # type: ignore

    def set_epoch(self, epoch: int):
        """Set the epoch for the dataset.

        Args:
            epoch (int): The epoch to set.
        """
        self.epoch = epoch

    def __len__(self) -> int:
        """Get the total number of data points in the dataset.

        Returns:
            int: the total number of data points in the dataset.
        """
        return len(self.all_steps)

    def __str__(self) -> str:
        """Get the description of the dataset."""
        return f"{self.dataset_name} ({len(self)} steps)"

    def __getitem__(self, index: int) -> dict:
        """Get the data for a single step in a trajectory.

        Args:
            index (int): The index of the step to get.

        Returns:
            dict: The data for the step.
        """
        trajectory_id, base_index = self.all_steps[index]
        indices = {
            key: delta_indices + base_index
            for key, delta_indices in self.delta_indices.items()
        }
        return self.transforms(self.get_step_data(trajectory_id, indices))  # type: ignore

    def get_step_data(self, trajectory_id: int, indices: dict[str, np.ndarray]) -> dict:
        """Get the RAW data for a single step in a trajectory. No transforms are applied.

        Args:
            trajectory_id (int): The name of the trajectory.
            indices (dict[str, np.ndarray]): The indices for each modality.

        Returns:
            dict: The RAW data for the step.

        Example return:
            {
                "video": {
                    "video.image_side_0": [B, T, H, W, C],
                    "video.image_side_1": [B, T, H, W, C],
                },
                "state": {
                    "state.eef_position": [B, T, state_dim],
                    "state.eef_rotation": [B, T, state_dim],
                },
                "action": {
                    "action.eef_position": [B, T, action_dim],
                    "action.eef_rotation": [B, T, action_dim],
                },
            }
        """
        data = {}
        # Get the data for all modalities
        self.curr_traj_data = self.get_trajectory_data(trajectory_id)
        for modality in self.modality_keys:
            # Get the data corresponding to each key in the modality
            for key in self.modality_keys[modality]:
                # Only load the data if the key is in the indices
                if key in indices:
                    data[key] = self.get_data_by_modality(
                        trajectory_id, modality, key, indices[key]
                    )
        return data

    def get_parquet_path(self, trajectory_id: int) -> Path:
        """Get the parquet path for a trajectory."""
        chunk_index = self.get_episode_chunk(trajectory_id)
        return self.dataset_path / self.data_path_pattern.format(
            episode_chunk=chunk_index, episode_index=trajectory_id
        )

    def get_trajectory_data(self, trajectory_id: int) -> pd.DataFrame:
        """Get the data for a trajectory."""
        if self.curr_traj_id == trajectory_id and self.curr_traj_data is not None:
            return self.curr_traj_data
        else:
            parquet_path = self.get_parquet_path(trajectory_id)
            assert parquet_path.exists(), f"Parquet file not found at {parquet_path}"
            return pd.read_parquet(parquet_path)

    def get_trajectory_index(self, trajectory_id: int) -> int:
        """Get the index of the trajectory in the dataset by the trajectory ID.
        This is useful when you need to get the trajectory length or sampling weight corresponding to the trajectory ID.

        Args:
            trajectory_id (str): The ID of the trajectory.

        Returns:
            int: The index of the trajectory in the dataset.
        """
        trajectory_indices = np.where(self.trajectory_ids == trajectory_id)[0]
        if len(trajectory_indices) != 1:
            raise ValueError(
                f"Error finding trajectory index for {trajectory_id}, found {trajectory_indices=}"
            )
        return trajectory_indices[0]

    def get_episode_chunk(self, ep_index: int) -> int:
        """Get the chunk index for an episode index."""
        return ep_index // self.chunk_size

    def retrieve_data_and_pad(
        self,
        array: np.ndarray,
        step_indices: np.ndarray,
        max_length: int,
        padding_strategy: str = "first_last",
    ) -> np.ndarray:
        """Retrieve the data from the dataset and pad it if necessary.

        Args:
            array (np.ndarray): The array to retrieve the data from.
            step_indices (np.ndarray): The step indices to retrieve the data for.
            max_length (int): The maximum length of the trajectory.
            padding_strategy (str): The padding strategy, either "first_last" or "zero".
                "first_last" uses first/last step data for padding, "zero" uses zero padding.

        Returns:
            np.ndarray: The retrieved and padded data.
        """
        # Get the padding indices
        front_padding_indices = step_indices < 0
        end_padding_indices = step_indices >= max_length
        padding_positions = np.logical_or(front_padding_indices, end_padding_indices)
        # Retrieve the data with the non-padding indices
        # If there exists some padding, Given T step_indices, the shape of the retrieved data will be (T', ...) where T' < T
        raw_data = array[step_indices[~padding_positions]]
        assert isinstance(raw_data, np.ndarray), f"{type(raw_data)=}"
        # This is the shape of the output, (T, ...)
        if raw_data.ndim == 1:
            expected_shape = (len(step_indices),)
        else:
            expected_shape = (len(step_indices), *array.shape[1:])

        # Pad the data
        output = np.zeros(expected_shape)
        # Assign the non-padded data
        output[~padding_positions] = raw_data
        # If there exists some padding, pad the data
        if padding_positions.any():
            if padding_strategy == "first_last":
                # Use first / last step data to pad
                front_padding_data = array[0]
                end_padding_data = array[-1]
                output[front_padding_indices] = front_padding_data
                output[end_padding_indices] = end_padding_data
            elif padding_strategy == "zero":
                # Use zero padding
                output[padding_positions] = 0
            else:
                raise ValueError(f"Invalid padding strategy: {padding_strategy}")
        return output

    def get_video_path(self, trajectory_id: int, key: str) -> Path:
        """Get the video file path for a specific trajectory and video key.

        Args:
            trajectory_id (int): The ID of the trajectory.
            key (str): The video key (without 'video.' prefix).

        Returns:
            Path: Path to the video file.
        """
        chunk_index = self.get_episode_chunk(trajectory_id)
        original_key = self.lerobot_modality_meta.video[key].original_key
        if original_key is None:
            original_key = key
        video_filename = self.video_path_pattern.format(
            episode_chunk=chunk_index,
            episode_index=trajectory_id,
            video_key=original_key,
        )
        return self.dataset_path / video_filename

    def get_video(
        self,
        trajectory_id: int,
        key: str,
        step_indices: np.ndarray,
    ) -> np.ndarray:
        """Get the video frames for a trajectory by a base index.

        Args:
            dataset (BaseSingleDataset): The dataset to retrieve the data from.
            trajectory_id (str): The ID of the trajectory.
            key (str): The key of the video.
            base_index (int): The base index of the trajectory.

        Returns:
            np.ndarray: The video frames for the trajectory and frame indices. Shape: (T, H, W, C)
        """
        # print(f"{step_indices=}")
        # Get the trajectory index
        trajectory_index = self.get_trajectory_index(trajectory_id)
        # Ensure the indices are within the valid range
        # This is equivalent to padding the video with extra frames at the beginning and end
        step_indices = np.maximum(step_indices, 0)
        step_indices = np.minimum(
            step_indices, self.trajectory_lengths[trajectory_index] - 1
        )
        assert key.startswith(
            "video."
        ), f"Video key must start with 'video.', got {key}"
        # Get the sub-key
        key = key.replace("video.", "")
        video_path = self.get_video_path(trajectory_id, key)
        # Get the action/state timestamps for each frame in the video
        assert self.curr_traj_data is not None, f"No data found for {trajectory_id=}"
        assert (
            "timestamp" in self.curr_traj_data.columns
        ), f"No timestamp found in {trajectory_id=}"
        timestamp: np.ndarray = self.curr_traj_data["timestamp"].to_numpy()
        # Get the corresponding video timestamps from the step indices
        video_timestamp = timestamp[step_indices]

        # try:
        return get_frames_by_timestamps(
            video_path.as_posix(),
            video_timestamp,
            video_backend=self.video_backend,
            video_backend_kwargs=self.video_backend_kwargs,
        )
        # except:
        # self.video_backend = "torchvision_av"
        # return get_frames_by_timestamps(
        #     video_path.as_posix(),
        #     video_timestamp,
        #     video_backend=self.video_backend,
        #     video_backend_kwargs=self.video_backend_kwargs,
        # )

    def get_state_or_action(
        self,
        trajectory_id: int,
        modality: str,
        key: str,
        step_indices: np.ndarray,
    ) -> np.ndarray:
        """Get the state or action data for a trajectory by a base index.
        If the step indices are out of range, pad with the data:
            if the data is stored in absolute format, pad with the first or last step data;
            otherwise, pad with zero.

        Args:
            dataset (BaseSingleDataset): The dataset to retrieve the data from.
            trajectory_id (int): The ID of the trajectory.
            modality (str): The modality of the data.
            key (str): The key of the data.
            base_index (int): The base index of the trajectory.

        Returns:
            np.ndarray: The data for the trajectory and step indices.
        """
        # Get the trajectory index
        trajectory_index = self.get_trajectory_index(trajectory_id)
        # Get the maximum length of the trajectory
        max_length = self.trajectory_lengths[trajectory_index]

        # Note [YL]: this handles action.task_progress if specified
        if key == "action.task_progress":
            # Get frame_index array and apply proper bounds checking and padding
            frame_index_array = self.curr_traj_data["frame_index"].to_numpy()  # type: ignore
            # Use retrieve_data_and_pad to handle out-of-bounds indices
            frame_index = self.retrieve_data_and_pad(
                array=frame_index_array,
                step_indices=step_indices,
                max_length=max_length,
                padding_strategy="first_last",  # Use first/last for task progress
            )
            # get the task progress by using "frame index / trajectory length"
            progress = frame_index / max_length
            progress = progress.reshape(-1, 1)
            return progress

        assert key.startswith(
            modality + "."
        ), f"{key} must start with {modality + '.'}, got {key}"
        # Get the sub-key, e.g. state.joint_angles -> joint_angles
        subkey = key.replace(modality + ".", "")
        # Get the lerobot key
        le_state_or_action_cfg = getattr(self.lerobot_modality_meta, modality)
        le_key = le_state_or_action_cfg[subkey].original_key
        if le_key is None:
            le_key = subkey
        # Get the data array, shape: (T, D)
        assert self.curr_traj_data is not None, f"No data found for {trajectory_id=}"
        assert (
            le_key in self.curr_traj_data.columns
        ), f"No {le_key} found in {trajectory_id=}"
        data_array: np.ndarray = np.stack(self.curr_traj_data[le_key])  # type: ignore
        if data_array.ndim == 1:
            assert (
                data_array.shape[0] == max_length
            ), f"Expected 1D array with length {max_length}, got {data_array.shape} array"
            data_array = data_array.reshape(-1, 1)
        assert data_array.ndim == 2, f"Expected 2D array, got {data_array.shape} array"
        le_indices = np.arange(
            le_state_or_action_cfg[subkey].start,
            le_state_or_action_cfg[subkey].end,
        )
        data_array = data_array[:, le_indices]
        # Get the state or action configuration
        state_or_action_cfg = getattr(self.metadata.modalities, modality)[subkey]

        # Pad the data
        return self.retrieve_data_and_pad(
            array=data_array,
            step_indices=step_indices,
            max_length=max_length,
            padding_strategy="first_last" if state_or_action_cfg.absolute else "zero",
        )

    def get_lapa_action(
        self,
        trajectory_id: int,
        key: str,
        step_indices: np.ndarray,
    ) -> np.ndarray | None:
        """Get LAPA action data for a trajectory by step indices.

        Args:
            trajectory_id (int): The ID of the trajectory.
            key (str): The key of the LAPA action data.
            step_indices (np.ndarray): The step indices to retrieve data for.

        Returns:
            np.ndarray | None: The LAPA action data, or None if the key is not found.
        """
        # Get the trajectory index
        trajectory_index = self.get_trajectory_index(trajectory_id)
        # Get the maximum length of the trajectory
        max_length = self.trajectory_lengths[trajectory_index]
        # Check key in the current trajectory data
        assert self.curr_traj_data is not None, f"No data found for {trajectory_id=}"
        if (
            key not in self.curr_traj_data.columns
        ):  # this ensures that we can still load data w/o lapa actions. will store values that are None.
            return None
        # assert key in self.curr_traj_data.columns, f"{key} not found in {trajectory_id=}"
        # Get the data array, shape: (T, D)
        data_array: np.ndarray = np.stack(self.curr_traj_data[key])  # type: ignore
        assert data_array.ndim == 2, f"Expected 2D array, got {data_array.shape} array"
        # Pad the data
        return self.retrieve_data_and_pad(
            array=data_array,
            step_indices=step_indices,
            max_length=max_length,
            padding_strategy="first_last",
        )

    def get_dream_actions(
        self,
        trajectory_id: int,
        key: str,
        step_indices: np.ndarray,
    ) -> np.ndarray | None:
        """Get DREAM action data for a trajectory by step indices.

        Args:
            trajectory_id (int): The ID of the trajectory.
            key (str): The key of the DREAM action data.
            step_indices (np.ndarray): The step indices to retrieve data for.

        Returns:
            np.ndarray | None: The DREAM action data, or None if the key is not found.
        """
        # Get the trajectory index
        trajectory_index = self.get_trajectory_index(trajectory_id)
        # Get the maximum length of the trajectory
        max_length = self.trajectory_lengths[trajectory_index]
        # Check key in the current trajectory data
        assert self.curr_traj_data is not None, f"No data found for {trajectory_id=}"
        if (
            key not in self.curr_traj_data.columns
        ):  # this ensures that we can still load data w/o lapa actions. will store values that are None.
            return None
        # assert key in self.curr_traj_data.columns, f"{key} not found in {trajectory_id=}"
        # Get the data array, shape: (T, D)
        data_array: np.ndarray = np.stack(self.curr_traj_data[key])  # type: ignore
        assert data_array.ndim == 2, f"Expected 2D array, got {data_array.shape} array"
        # Pad the data
        return self.retrieve_data_and_pad(
            array=data_array,
            step_indices=step_indices,
            max_length=max_length,
            padding_strategy="first_last",
        )

    def get_language(
        self,
        trajectory_id: int,
        key: str,
        step_indices: np.ndarray,
    ) -> list[str]:
        """Get the language annotation data for a trajectory by step indices.

        Args:
            trajectory_id (int): The ID of the trajectory.
            key (str): The key of the annotation.
            step_indices (np.ndarray): The step indices to retrieve data for.

        Returns:
            list[str]: The annotation data for the trajectory and step indices.
                If no matching data is found, return empty strings.
        """
        assert self.curr_traj_data is not None, f"No data found for {trajectory_id=}"
        # Get the trajectory index
        trajectory_index = self.get_trajectory_index(trajectory_id)
        # Get the maximum length of the trajectory
        max_length = self.trajectory_lengths[trajectory_index]
        # Get the end times corresponding to the closest indices
        step_indices = np.maximum(step_indices, 0)
        step_indices = np.minimum(step_indices, max_length - 1)
        # Get the annotations
        assert key.startswith(
            "annotation."
        ), f"Language key must start with 'annotation.', got {key}"
        subkey = key.replace("annotation.", "")
        # print("subkey", subkey)

        # Check if this is a metadata-based language key (detailed_global_instruction_medium/concise)
        if subkey in METADATA_LANG_KEYS:
            # print("return metadata language")
            return self._get_language_from_metadata(
                trajectory_id, subkey, len(step_indices)
            )

        # Otherwise, load from parquet columns (original behavior)
        annotation_meta = self.lerobot_modality_meta.annotation
        assert annotation_meta is not None, f"Annotation metadata is None for {subkey}"
        assert (
            subkey in annotation_meta
        ), f"Annotation key {subkey} not found in metadata, available annotation keys: {annotation_meta.keys()}"
        subkey_meta = annotation_meta[subkey]
        original_key = subkey_meta.original_key
        if original_key is None:
            original_key = key
        if pd.api.types.is_numeric_dtype(self.curr_traj_data[original_key]):
            # Stored as list of integers
            task_indices: list[int] = (
                self.curr_traj_data[original_key].iloc[step_indices].tolist()
            )
            return self.tasks.loc[task_indices]["task"].tolist()
        else:
            # Stored as list of strings
            return (
                self.curr_traj_data[original_key]
                .iloc[step_indices]
                .astype(str)
                .tolist()
            )

    def _get_language_from_metadata(
        self,
        trajectory_id: int,
        lang_key: str,
        nframes: int,
    ) -> list[str]:
        """Get language instruction from metadata files for special language keys.

        Supports:
        - detailed_global_instruction_medium: Longer, detailed description
        - detailed_global_instruction_concise: Short summary

        Args:
            trajectory_id (int): The ID of the trajectory (episode_index).
            lang_key (str): The language key (e.g., "detailed_global_instruction_medium").
            nframes (int): Number of frames to return the instruction for.

        Returns:
            list[str]: The instruction repeated for each frame (empty string if not found).
        """
        if trajectory_id in self._detailed_global_instructions:
            instruction = self._detailed_global_instructions[trajectory_id].get(
                lang_key, ""
            )
            # print("instruction", instruction)
        else:
            instruction = ""
        return [instruction] * nframes

    def get_rl_info(
        self,
        trajectory_id: int,
        key: str,
        step_indices: np.ndarray,
    ) -> np.ndarray:
        """Get the reward data for a trajectory by step indices.

        If the step indices are out of range, pad with first/last step data.

        Args:
            trajectory_id (int): The ID of the trajectory.
            key (str): The key of the reward data.
            step_indices (np.ndarray): The step indices to retrieve data for.

        Returns:
            np.ndarray: The reward data for the trajectory and step indices.
        """
        # Get the trajectory index
        trajectory_index = self.get_trajectory_index(trajectory_id)
        # Get the maximum length of the trajectory
        max_length = self.trajectory_lengths[trajectory_index]
        data_array: np.ndarray = np.stack(self.curr_traj_data[key])  # type: ignore

        if key == "rl_info.next.reward":
            padding_strategy = "zero"
        else:
            padding_strategy = "first_last"

        # Pad the data
        return self.retrieve_data_and_pad(
            array=data_array,
            step_indices=step_indices,
            max_length=max_length,
            padding_strategy=padding_strategy,
        )

    def get_data_by_modality(
        self,
        trajectory_id: int,
        modality: str,
        key: str,
        step_indices: np.ndarray,
    ) -> np.ndarray | list[str] | None:
        """Get the data corresponding to the modality for a trajectory by step indices.

        This method dispatches to the appropriate specialized method based on the modality.
        For the language modality, empty strings are returned if no matching data is found.

        Args:
            trajectory_id (int): The ID of the trajectory.
            modality (str): The modality of the data (video, state, action, language, etc.).
            key (str): The key of the data.
            step_indices (np.ndarray): The step indices of the trajectory.

        Returns:
            np.ndarray | list[str] | None: The data for the specified modality.
        """
        if modality == "video":
            return self.get_video(trajectory_id, key, step_indices)
        elif modality == "state" or modality == "action":
            return self.get_state_or_action(trajectory_id, modality, key, step_indices)
        elif modality == "language":
            return self.get_language(trajectory_id, key, step_indices)
        elif modality == "lapa_action":
            return self.get_lapa_action(trajectory_id, key, step_indices)
        elif modality == "dream_actions":
            return self.get_dream_actions(trajectory_id, key, step_indices)
        elif modality == "rl_info":
            return self.get_rl_info(trajectory_id, key, step_indices)
        else:
            raise ValueError(f"Invalid modality: {modality}")

    def get_initial_actions(self):
        """Load initial actions from the dataset if available.

        Returns:
            list: List containing initial actions if the file exists, empty list otherwise.
        """
        # initial_actions_path = self.dataset_path / INITIAL_ACTIONS_FILENAME
        # if initial_actions_path.exists():
        #    initial_actions = load_initial_actions(initial_actions_path)
        #    return initial_actions  # a single-element list of dict[str, dict[str, np.ndarray]]
        # else:
        #    return []
        return []


class ShardedLeRobotSubLangSingleActionChunkDatasetDROID(LeRobotSingleDataset):
    def __init__(
        self,
        num_steps_per_shard: int = int(1e4),
        *args,
        **kwargs,
    ):
        self.args = args
        self.kwargs = kwargs
        super().__init__(*args, **kwargs)

        self.num_steps_per_shard = num_steps_per_shard
        self.all_video_paths = self.get_all_video_paths()
        self.all_parquet_paths = self.get_all_parquet_paths()
        self.sharded_trajectories, self.shard_lengths = self.generate_shards()

        self.shard_start_indices: dict[int, int] | None = None
        self.cached_shard: dict[str, np.ndarray] | None = None
        self.cached_df: pd.DataFrame | None = None
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._cache_job: Future | None = None

    @property
    def num_shards(self) -> int:
        """The number of shards."""
        return len(self.sharded_trajectories)

    def get_all_video_paths(self) -> dict[int, dict[str, Path]]:
        """Get the video paths for all trajectories and all views.

        Returns:
            dict[int, dict[str, Path]]: The video paths for all trajectories.
        """
        video_paths = {}
        for trajectory_id in self._trajectory_ids:
            if isinstance(trajectory_id, np.integer):
                trajectory_id = trajectory_id.item()
            assert isinstance(
                trajectory_id, int
            ), f"trajectory_id must be an integer, got {type(trajectory_id)}"
            video_paths[trajectory_id] = {}
            for key in self._modality_keys["video"]:
                assert key.startswith(
                    "video."
                ), f"Video key must start with 'video.', got {key}"
                video_paths[trajectory_id][key] = self.get_video_path(
                    trajectory_id, key.replace("video.", "")
                )
        return video_paths

    def get_all_parquet_paths(self) -> dict[int, Path]:
        """Get the parquet paths for all trajectories.

        Returns:
            dict[int, Path]: The parquet paths for all trajectories.
        """
        return {
            trajectory_id: self.get_parquet_path(trajectory_id)
            for trajectory_id in self._trajectory_ids
        }

    def generate_shards(self) -> tuple[list[list[int]], np.ndarray]:
        sharded_trajectories = [[]]
        curr_num_steps = 0
        curr_shard_index = 0
        discarded_episode_indices = []
        trajectory_ids = self._trajectory_ids
        if self.discard_bad_trajectories:
            discarded_episode_indices = self._lerobot_info_meta.get(
                "discarded_episode_indices", []
            )
            trajectory_ids = [
                trajectory_id
                for trajectory_id in trajectory_ids
                if trajectory_id not in discarded_episode_indices
            ]

        assert len(trajectory_ids) > 0, "No valid trajectories found for dataset"
        total_steps = np.sum(
            [len(self.step_filter[trajectory_id]) for trajectory_id in trajectory_ids]
        ).astype(int)
        num_shards = np.ceil(total_steps / self.num_steps_per_shard).astype(int)
        cutoffs = np.linspace(0, total_steps, num_shards + 1)[
            1:
        ]  # Exclude the first cutoff (0)
        shard_lengths = []
        last_num_steps = 0
        for trajectory_id in trajectory_ids:
            sharded_trajectories[-1].append(trajectory_id)
            curr_num_steps += len(self.step_filter[trajectory_id])
            if curr_num_steps > cutoffs[curr_shard_index]:
                sharded_trajectories.append([])
                curr_shard_index += 1
                shard_lengths.append(curr_num_steps - last_num_steps)
                last_num_steps = curr_num_steps
        shard_lengths.append(curr_num_steps - last_num_steps)
        assert (
            curr_num_steps == total_steps
        ), "Total steps not equal to the sum of trajectory lengths"
        assert (
            len(shard_lengths) == num_shards
        ), "Number of shards not equal to the number of cutoffs"
        assert (
            len(sharded_trajectories) == num_shards
        ), "Number of shards not equal to the number of cutoffs"
        print(
            f"Generated {len(sharded_trajectories)} shards for dataset {self._dataset_path}"
        )
        return sharded_trajectories, np.array(shard_lengths)

    @staticmethod
    def get_shard(
        trajectory_ids: list[int] | np.ndarray,
        modality_keys: dict,
        video_paths: dict[int, dict[str, Path]],
        parquet_paths: dict[int, Path],
        video_backend: str = "pyav",
        video_backend_kwargs: dict | None = None,
        fps: float | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[int, int], pd.DataFrame]:
        # Optional logging to avoid stdout overhead during tight loops
        # (controlled by instance-level verbose flag)
        # Using a staticmethod, we cannot read self.verbose; defer to caller to control prints
        print("Caching shard")
        start_time = time.time()
        assert (
            "video" in modality_keys
        ), "No video modality found. No need to use caching."
        cached_frames = {}
        trajectory_start_indices = {}
        curr_step_index = 0
        cached_df = None
        for trajectory_id in trajectory_ids:
            trajectory_start_indices[trajectory_id] = curr_step_index
            parquet_path = parquet_paths[trajectory_id]
            parquet_df = pd.read_parquet(parquet_path)
            # Check timestamps are in sync
            parquet_timestamps = parquet_df["timestamp"].to_numpy()
            trajectory_length = len(parquet_timestamps)
            if isinstance(trajectory_id, np.integer):
                trajectory_id = trajectory_id.item()
            assert isinstance(
                trajectory_id, int
            ), f"trajectory_id must be an integer, got {type(trajectory_id)}"
            for key in modality_keys["video"]:
                assert key.startswith(
                    "video."
                ), f"Video key must start with 'video.', got {key}"
                if key not in cached_frames:
                    cached_frames[key] = []
                frames = get_frames_by_timestamps(
                    video_paths[trajectory_id][key].as_posix(),
                    timestamps=parquet_timestamps,
                    video_backend=video_backend,
                    video_backend_kwargs=video_backend_kwargs,
                    fps=fps,
                )
                cached_frames[key].append(frames)
            if cached_df is None:
                cached_df = parquet_df
            else:
                cached_df = pd.concat([cached_df, parquet_df])
            curr_step_index += trajectory_length

        # Concatenate the frames
        for key in cached_frames:
            cached_frames[key] = np.concatenate(cached_frames[key], axis=0)
        end_time = time.time()
        print(f"Cached shard in {end_time - start_time:.2f} seconds")
        assert cached_df is not None, "Cached dataframe is None"
        # Add global "index" column if missing (some dataset formats omit it)
        if "index" not in cached_df.columns:
            cached_df = cached_df.reset_index(drop=True)
            cached_df["index"] = cached_df.index
        return cached_frames, trajectory_start_indices, cached_df

    def start_cache_shard(self, shard_index: int) -> None:
        """Start caching a shard in a background thread."""
        self._cache_job = self._executor.submit(
            self.get_shard,
            self.sharded_trajectories[shard_index],
            self._modality_keys,
            self.all_video_paths,
            self.all_parquet_paths,
            self.video_backend,
            self.video_backend_kwargs,
            self.fps,
        )

    def finish_cache_shard(self):
        """Get the cached shard."""
        assert self._cache_job is not None
        self.cached_shard, self.shard_start_indices, self.cached_df = (
            self._cache_job.result()
        )
        self._cache_job = None  # Clear the future to allow memory to be freed

    def delete_cached_shard(self):
        """Delete the cached shard."""
        del self.cached_shard
        del self.shard_start_indices
        del self.cached_df
        # self._traj_cache.clear()

    def get_trajectories_in_shard(self) -> list[int]:
        """Get the trajectories in a shard."""
        assert self.shard_start_indices is not None
        return list(self.shard_start_indices.keys())

    def get_step_data(  # type: ignore
        self,
        trajectory_id: int,
        indices: dict[str, np.ndarray],
    ) -> dict | None:
        """Get the RAW data for a single step in a trajectory. No transforms are applied.

        Args:
            trajectory_id (int): The name of the trajectory.
            indices (dict[str, np.ndarray]): The indices for each modality.

        Returns:
            dict: The RAW data for the step.

        Example return:
            {
                "video": {
                    "video.image_side_0": [B, T, H, W, C],
                    "video.image_side_1": [B, T, H, W, C],
                },
                "state": {
                    "state.eef_position": [B, T, state_dim],
                    "state.eef_rotation": [B, T, state_dim],
                },
                "action": {
                    "action.eef_position": [B, T, action_dim],
                    "action.eef_rotation": [B, T, action_dim],
                },
            }
        """
        data = {}
        self.curr_traj_data = self.get_trajectory_data(trajectory_id)
        for modality in self._modality_keys:
            # Get the data corresponding to each key in the modality
            for key in self._modality_keys[modality]:
                # Only load the data if the key is in the indices
                if key in indices:
                    data[key] = self.get_data_by_modality(
                        trajectory_id, modality, key, indices[key]
                    )
                    # Skip this sample if state or action data is empty
                    if (
                        data[key] is not None
                        and hasattr(data[key], "__len__")
                        and len(data[key]) == 0
                    ):
                        return None
        return data

    def get_data_by_modality(
        self,
        trajectory_id: int,
        modality: str,
        key: str,
        step_indices: np.ndarray,
    ) -> np.ndarray | list[str] | None:
        """Get the data corresponding to the modality for a trajectory by step indices.

        This method dispatches to the appropriate specialized method based on the modality.
        For the language modality, empty strings are returned if no matching data is found.

        Args:
            trajectory_id (int): The ID of the trajectory.
            modality (str): The modality of the data (video, state, action, language, etc.).
            key (str): The key of the data.
            step_indices (np.ndarray): The step indices of the trajectory.

        Returns:
            np.ndarray | list[str] | None: The data for the specified modality.
        """
        if modality == "video":
            return self.get_video(trajectory_id, key, step_indices)
        elif modality == "state":
            return self.get_state(trajectory_id, modality, key, step_indices)
        elif modality == "action":
            return self.get_action(trajectory_id, modality, key, step_indices)
        elif modality == "language":
            return self.get_language(trajectory_id, key, step_indices)
        # elif modality == "lapa_action":
        #    return self.get_lapa_action(trajectory_id, key, step_indices)
        # elif modality == "dream_actions":
        #    return self.get_dream_actions(trajectory_id, key, step_indices)
        # elif modality == "rl_info":
        #    return self.get_rl_info(trajectory_id, key, step_indices)
        else:
            raise ValueError(f"Invalid modality: {modality}")

    def get_video(
        self, trajectory_id: int, key: str, step_indices: np.ndarray
    ) -> np.ndarray:
        """Get the video frames from cached shards for a trajectory by uniformly sampling from language-consistent ranges.

        Args:
            trajectory_id (int): The ID of the trajectory.
            key (str): The key of the video.
            step_indices (np.ndarray): The step indices to retrieve data for.

        Returns:
            np.ndarray: The video frames for the trajectory and frame indices. Shape: (T, H, W, C)
        """
        # Get the trajectory index
        trajectory_index = self.get_trajectory_index(trajectory_id)
        trajectory_length = self._trajectory_lengths[trajectory_index]

        # Get trajectory data to access language annotations (reuse if already loaded)
        # traj_data = (
        #     self.curr_traj_data
        #     if getattr(self, "curr_traj_data", None) is not None
        #     else self.get_trajectory_data(trajectory_id)
        # )
        traj_data = self.get_trajectory_data(trajectory_id)
        # print("trajectory id", trajectory_id, step_indices, trajectory_index)

        # Get language annotations for all steps in the trajectory
        # language_key = self.language_key

        language_key: str | None = None

        for modality in self._modality_keys:
            for modality_key in self._modality_keys[modality]:
                if modality_key.startswith("annotation."):
                    subkey = modality_key.replace("annotation.", "")
                    annotation_meta = self._lerobot_modality_meta.annotation
                    if annotation_meta is not None:
                        subkey_meta = annotation_meta[subkey]
                        language_key = subkey_meta.original_key
                        break
        assert language_key is not None, "Language key not found"
        if language_key in traj_data.columns:
            language_annotations = traj_data[language_key].values
        else:
            # Fallback to original behavior if language annotations are not available
            step_indices = np.maximum(step_indices, 0)
            step_indices = np.minimum(step_indices, trajectory_length - 1)
            assert (
                self.shard_start_indices is not None
                and self.cached_shard is not None
                and trajectory_id in self.shard_start_indices
            ), "Shard not cached. Please call `cache_next_shard` and `use_next_shard` first."
            indices_in_shard = self.shard_start_indices[trajectory_id] + step_indices
            return self.cached_shard[key][indices_in_shard]

        # Find language-consistent ranges and uniformly sample from them
        sampled_indices = self._uniform_sample_from_language_ranges(
            step_indices, np.array(language_annotations), trajectory_length
        )

        # Ensure the sampled indices are within the valid range
        sampled_indices = np.maximum(sampled_indices, 0)
        sampled_indices = np.minimum(sampled_indices, trajectory_length - 1)
        # print("sampled indices", sampled_indices)

        # Calculate the absolute indices
        assert (
            self.shard_start_indices is not None
            and self.cached_shard is not None
            and trajectory_id in self.shard_start_indices
        ), "Shard not cached. Please call `cache_next_shard` and `use_next_shard` first."
        indices_in_shard = self.shard_start_indices[trajectory_id] + sampled_indices
        return self.cached_shard[key][indices_in_shard]

    def get_state(
        self,
        trajectory_id: int,
        modality: str,
        key: str,
        step_indices: np.ndarray,
    ) -> np.ndarray:
        """Get the state data for a trajectory by a base index.
        If the step indices are out of range, pad with the data:
            if the data is stored in absolute format, pad with the first or last step data;
            otherwise, pad with zero.

        Args:
            dataset (BaseSingleDataset): The dataset to retrieve the data from.
            trajectory_id (int): The ID of the trajectory.
            modality (str): The modality of the data.
            key (str): The key of the data.
            base_index (int): The base index of the trajectory.

        Returns:
            np.ndarray: The data for the trajectory and step indices.
        """
        # Get the trajectory index
        trajectory_index = self.get_trajectory_index(trajectory_id)
        # Get the maximum length of the trajectory
        max_length = self._trajectory_lengths[trajectory_index]

        # Note [YL]: this handles action.task_progress if specified
        if key == "action.task_progress":
            # Get frame_index array and apply proper bounds checking and padding
            frame_index_array = self.curr_traj_data["frame_index"].to_numpy()
            # Use retrieve_data_and_pad to handle out-of-bounds indices
            frame_index = self.retrieve_data_and_pad(
                array=frame_index_array,
                step_indices=step_indices,
                max_length=max_length,
                padding_strategy="first_last",  # Use first/last for task progress
            )
            # get the task progress by using "frame index / trajectory length"
            progress = frame_index / max_length
            progress = progress.reshape(-1, 1)
            return progress

        assert key.startswith(
            modality + "."
        ), f"{key} must start with {modality + '.'}, got {key}"
        # Get the sub-key, e.g. state.joint_angles -> joint_angles
        subkey = key.replace(modality + ".", "")
        # Get the lerobot key
        le_state_or_action_cfg = getattr(self._lerobot_modality_meta, modality)
        le_key = le_state_or_action_cfg[subkey].original_key
        if le_key is None:
            le_key = subkey

        # Get the data array, shape: (T, D)
        assert self.curr_traj_data is not None, f"No data found for {trajectory_id=}"
        assert (
            le_key in self.curr_traj_data.columns
        ), f"No {le_key} found in {trajectory_id=}"
        data_array: np.ndarray = np.stack(self.curr_traj_data[le_key])  # type: ignore
        if data_array.ndim == 1:
            assert (
                data_array.shape[0] == max_length
            ), f"Expected 1D array with length {max_length}, got {data_array.shape} array"
            data_array = data_array.reshape(-1, 1)
        assert data_array.ndim == 2, f"Expected 2D array, got {data_array.shape} array"
        le_indices = np.arange(
            le_state_or_action_cfg[subkey].start,
            le_state_or_action_cfg[subkey].end,
        )
        data_array = data_array[:, le_indices]
        # Get the state or action configuration
        state_or_action_cfg = getattr(self._metadata.modalities, modality)[subkey]

        # Build sampled indices for state aligned with language and video sampling
        # For state, select only the anchor index per 30-frame chunk (stride 30):
        # [..., first_idx-30, first_idx, first_idx+30, ...]
        # Stop on language change at the step anchor, bounds, or when reaching 16 anchors (to match 16 chunks).
        trajectory_index = self.get_trajectory_index(trajectory_id)
        trajectory_length = self._trajectory_lengths[trajectory_index]
        # traj_data = (
        #     self.curr_traj_data
        #     if getattr(self, "curr_traj_data", None) is not None
        #     else self.get_trajectory_data(trajectory_id)
        # )
        # language_key = self.language_key

        traj_data = self.get_trajectory_data(trajectory_id)
        language_key = None
        for modality_name in self._modality_keys:
            for modality_key in self._modality_keys[modality_name]:
                if modality_key.startswith("annotation."):
                    subkey = modality_key.replace("annotation.", "")
                    annotation_meta = self._lerobot_modality_meta.annotation
                    if annotation_meta is not None:
                        subkey_meta = annotation_meta[subkey]
                        language_key = subkey_meta.original_key
                        break

        if (
            language_key is not None
            and language_key in traj_data.columns
            and len(step_indices) > 0
        ):
            language_annotations = traj_data[language_key].values
            first_idx = max(0, min(int(step_indices[0]), trajectory_length - 1))
            target_language = language_annotations[first_idx]

            # Get the number of chunks from video sampling to ensure alignment
            target_num_chunks = None
            # if first_idx in self._current_num_chunks:
            if (
                hasattr(self, "_current_num_chunks")
                and first_idx in self._current_num_chunks
            ):
                target_num_chunks = self._current_num_chunks[first_idx]
                # print(f"State: Using target_num_chunks from video: {target_num_chunks}")

            max_frames = (
                self.max_chunk_size
            )  # 16 anchors to align with 16 chunks as video/action
            sampled_list: list[int] = []

            def add_anchor(anchor_index: int) -> None:
                nonlocal sampled_list
                if len(sampled_list) >= max_frames:
                    return
                # If we have a target number of chunks, stop when we reach it
                if (
                    target_num_chunks is not None
                    and len(sampled_list) >= target_num_chunks
                ):
                    return
                # Require full 32-length window to exist for alignment with action/video
                if 0 <= anchor_index and anchor_index + 24 < trajectory_length:
                    sampled_list.append(int(anchor_index))

            # Always include first_idx anchor
            add_anchor(first_idx)

            # Expand outward in 32-frame steps
            step = 1
            back_done = False
            fwd_done = False
            while len(sampled_list) < max_frames and (not back_done or not fwd_done):
                # Stop if we've reached the target number of chunks
                if (
                    target_num_chunks is not None
                    and len(sampled_list) >= target_num_chunks
                ):
                    break

                if not back_done:
                    back_anchor = first_idx - 24 * step
                    if back_anchor < 0:
                        back_done = True
                    elif language_annotations[back_anchor] != target_language:
                        back_done = True
                    else:
                        add_anchor(back_anchor)
                if len(sampled_list) >= max_frames:
                    break
                if not fwd_done:
                    fwd_anchor = first_idx + 24 * step
                    if fwd_anchor >= trajectory_length:
                        fwd_done = True
                    elif language_annotations[fwd_anchor] != target_language:
                        fwd_done = True
                    else:
                        add_anchor(fwd_anchor)
                step += 1

            if len(sampled_list) > 0:
                sampled_indices = np.array(sorted(set(sampled_list)), dtype=int)
            else:
                sampled_indices = np.array([], dtype=int)

        else:
            # Fallback: use provided indices with bounds
            sampled_indices = np.maximum(step_indices, 0)
            sampled_indices = np.minimum(sampled_indices, trajectory_length - 1)

        # print("sampled indices for state", sampled_indices)

        # Pad the data using the computed sampled indices
        return self.retrieve_data_and_pad(
            array=data_array,
            step_indices=sampled_indices,
            max_length=max_length,
            padding_strategy="first_last" if state_or_action_cfg.absolute else "zero",
        )

    def get_action(
        self,
        trajectory_id: int,
        modality: str,
        key: str,
        step_indices: np.ndarray,
    ) -> np.ndarray:
        """Get the action data for a trajectory by a base index.
        If the step indices are out of range, pad with the data:
            if the data is stored in absolute format, pad with the first or last step data;
            otherwise, pad with zero.

        Args:
            dataset (BaseSingleDataset): The dataset to retrieve the data from.
            trajectory_id (int): The ID of the trajectory.
            modality (str): The modality of the data.
            key (str): The key of the data.
            base_index (int): The base index of the trajectory.

        Returns:
            np.ndarray: The data for the trajectory and step indices.
        """
        # Get the trajectory index
        trajectory_index = self.get_trajectory_index(trajectory_id)
        # Get the maximum length of the trajectory
        max_length = self._trajectory_lengths[trajectory_index]

        # Note [YL]: this handles action.task_progress if specified
        if key == "action.task_progress":
            # Get frame_index array and apply proper bounds checking and padding
            frame_index_array = self.curr_traj_data["frame_index"].to_numpy()
            # Use retrieve_data_and_pad to handle out-of-bounds indices
            frame_index = self.retrieve_data_and_pad(
                array=frame_index_array,
                step_indices=step_indices,
                max_length=max_length,
                padding_strategy="first_last",  # Use first/last for task progress
            )
            # get the task progress by using "frame index / trajectory length"
            progress = frame_index / max_length
            progress = progress.reshape(-1, 1)
            return progress

        assert key.startswith(
            modality + "."
        ), f"{key} must start with {modality + '.'}, got {key}"
        # Get the sub-key, e.g. state.joint_angles -> joint_angles
        subkey = key.replace(modality + ".", "")
        # Get the lerobot key
        le_state_or_action_cfg = getattr(self._lerobot_modality_meta, modality)
        le_key = le_state_or_action_cfg[subkey].original_key
        if le_key is None:
            le_key = subkey

        # Get the data array, shape: (T, D)
        assert self.curr_traj_data is not None, f"No data found for {trajectory_id=}"
        assert (
            le_key in self.curr_traj_data.columns
        ), f"No {le_key} found in {trajectory_id=}"
        data_array: np.ndarray = np.stack(self.curr_traj_data[le_key])  # type: ignore
        if data_array.ndim == 1:
            assert (
                data_array.shape[0] == max_length
            ), f"Expected 1D array with length {max_length}, got {data_array.shape} array"
            data_array = data_array.reshape(-1, 1)
        assert data_array.ndim == 2, f"Expected 2D array, got {data_array.shape} array"
        le_indices = np.arange(
            le_state_or_action_cfg[subkey].start,
            le_state_or_action_cfg[subkey].end,
        )
        data_array = data_array[:, le_indices]
        # Get the state or action configuration
        state_or_action_cfg = getattr(self._metadata.modalities, modality)[subkey]

        # Build sampled indices for action aligned with language and video sampling
        # Action runs at 30fps, so for each ±30-frame step around first_idx,
        # collect a 30-length chunk with stride 1: [anchor ... anchor+29].
        # Stop on language change at the step anchor, bounds, or when reaching 480 frames (16 chunks * 30).
        trajectory_index = self.get_trajectory_index(trajectory_id)
        trajectory_length = self._trajectory_lengths[trajectory_index]
        # traj_data = (
        #     self.curr_traj_data
        #     if getattr(self, "curr_traj_data", None) is not None
        #     else self.get_trajectory_data(trajectory_id)
        # )

        # language_key = self.language_key
        traj_data = self.get_trajectory_data(trajectory_id)
        language_key = None
        for modality_name in self._modality_keys:
            for modality_key in self._modality_keys[modality_name]:
                if modality_key.startswith("annotation."):
                    subkey = modality_key.replace("annotation.", "")
                    annotation_meta = self._lerobot_modality_meta.annotation
                    if annotation_meta is not None:
                        subkey_meta = annotation_meta[subkey]
                        language_key = subkey_meta.original_key
                        break

        if (
            language_key is not None
            and language_key in traj_data.columns
            and len(step_indices) > 0
        ):
            language_annotations = traj_data[language_key].values
            first_idx = max(0, min(int(step_indices[0]), trajectory_length - 1))
            target_language = language_annotations[first_idx]

            # Get the number of chunks from video sampling to ensure alignment
            target_num_chunks = None
            # if first_idx in self._current_num_chunks:
            if (
                hasattr(self, "_current_num_chunks")
                and first_idx in self._current_num_chunks
            ):
                target_num_chunks = self._current_num_chunks[first_idx]
                # print(f"Using target_num_chunks from video: {target_num_chunks}")

            max_frames = 24 * self.max_chunk_size
            per_step_offsets = list(range(24))  # 0..23
            sampled_list: list[int] = []

            def add_step_set(anchor_index: int) -> None:
                nonlocal sampled_list
                # Ensure the whole 32-length chunk fits within bounds
                if anchor_index < 0 or anchor_index + 24 >= trajectory_length:
                    return
                # Ensure we don't overrun the max_frames cap with a partial chunk
                if len(sampled_list) + 24 > max_frames:
                    return
                # If we have a target number of chunks, stop when we reach it
                if (
                    target_num_chunks is not None
                    and len(sampled_list) // 24 >= target_num_chunks
                ):
                    return
                for offset in per_step_offsets:
                    idx = anchor_index + offset
                    sampled_list.append(int(idx))

            # Always include first_idx chunk
            add_step_set(first_idx)

            step = 1
            back_done = False
            fwd_done = False
            while len(sampled_list) < max_frames and (not back_done or not fwd_done):
                # Stop if we've reached the target number of chunks
                if (
                    target_num_chunks is not None
                    and len(sampled_list) // 24 >= target_num_chunks
                ):
                    break

                if not back_done:
                    back_anchor = first_idx - 24 * step
                    if back_anchor < 0:
                        back_done = True
                    elif language_annotations[back_anchor] != target_language:
                        back_done = True
                    else:
                        add_step_set(back_anchor)
                if len(sampled_list) >= max_frames:
                    break
                if not fwd_done:
                    fwd_anchor = first_idx + 24 * step
                    if fwd_anchor >= trajectory_length:
                        fwd_done = True
                    elif language_annotations[fwd_anchor] != target_language:
                        fwd_done = True
                    else:
                        add_step_set(fwd_anchor)
                step += 1

            if len(sampled_list) > 0:
                unique_sorted = np.array(sorted(set(sampled_list)), dtype=int)
                # Enforce divisibility by 30 and the 480 cap
                capped_size = min(unique_sorted.size, max_frames)
                divisible_size = (capped_size // 24) * 24
                sampled_indices = unique_sorted[:divisible_size]
            else:
                sampled_indices = np.array([], dtype=int)

        else:
            # Fallback: use provided indices with bounds
            sampled_indices = np.maximum(step_indices, 0)
            sampled_indices = np.minimum(sampled_indices, trajectory_length - 1)

        # Pad the data using the computed sampled indices
        action_data = self.retrieve_data_and_pad(
            array=data_array,
            step_indices=sampled_indices,
            max_length=max_length,
            padding_strategy="first_last" if state_or_action_cfg.absolute else "zero",
        )
        # print("action data before convert", key)
        # Calculate relative action on the fly if relative_action is enabled
        # Only apply to keys that are in relative_action_keys
        subkey = key.replace("action.", "")
        should_convert_to_relative = (
            (self.relative_action or self.relative_action_per_horizon)
            and len(sampled_indices) > 0
            and (
                self.relative_action_keys is None or subkey in self.relative_action_keys
            )
        )
        if should_convert_to_relative:
            # print("action data before convert", action_data[0], action_data[-1], key)
            action_data = self._convert_to_relative_action(
                action_data=action_data,
                action_key=key,
                sampled_indices=sampled_indices,
                trajectory_id=trajectory_id,
                chunk_size=24,
            )
            # print("action data after convert", action_data[0], action_data[-1], key)

        return action_data

    def _convert_to_relative_action(
        self,
        action_data: np.ndarray,
        action_key: str,
        sampled_indices: np.ndarray,
        trajectory_id: int,
        chunk_size: int = 24,
    ) -> np.ndarray:
        """Convert absolute action to relative action by subtracting reference state.

        Args:
            action_data: Absolute action data, shape (T, D)
            action_key: The action key (e.g., 'action.left_arm_joints')
            sampled_indices: The sampled indices for the action
            trajectory_id: The trajectory ID
            chunk_size: Size of each action chunk (default 24)

        Returns:
            np.ndarray: Relative action data, shape (T, D)
        """
        # Get corresponding state key (assume state key matches action key)
        state_key = action_key.replace("action.", "state.")
        subkey = action_key.replace("action.", "")

        # Get state data from trajectory
        traj_data = self.get_trajectory_data(trajectory_id)
        le_state_cfg = getattr(self._lerobot_modality_meta, "state", None)

        if le_state_cfg is None or subkey not in le_state_cfg:
            # If no corresponding state key, return original action data
            return action_data

        le_state_key = le_state_cfg[subkey].original_key
        if le_state_key is None:
            le_state_key = subkey

        if le_state_key not in traj_data.columns:
            # If state column doesn't exist, return original action data
            return action_data

        # Get state data array
        state_array: np.ndarray = np.stack(traj_data[le_state_key].tolist())
        if state_array.ndim == 1:
            state_array = state_array.reshape(-1, 1)

        # Apply same indices as action
        le_indices = np.arange(
            le_state_cfg[subkey].start,
            le_state_cfg[subkey].end,
        )
        state_array = state_array[:, le_indices]

        # Calculate relative action for each chunk
        relative_action_data = action_data.copy()
        num_chunks = len(sampled_indices) // chunk_size

        for chunk_idx in range(num_chunks):
            chunk_start = chunk_idx * chunk_size
            chunk_end = chunk_start + chunk_size

            # Get anchor index (first index of the chunk)
            anchor_idx = sampled_indices[chunk_start]

            # Get reference state at anchor index
            if anchor_idx < len(state_array):
                reference_state = state_array[anchor_idx]

                # Subtract reference state from all actions in this chunk
                relative_action_data[chunk_start:chunk_end] = (
                    action_data[chunk_start:chunk_end] - reference_state
                )

        return relative_action_data

    def _uniform_sample_from_language_ranges(
        self,
        step_indices: np.ndarray,
        language_annotations: np.ndarray,
        trajectory_length: int,
    ) -> np.ndarray:
        """Uniformly sample from language-consistent ranges based on the first index's language.

        Args:
            step_indices (np.ndarray): Original step indices to sample.
            language_annotations (np.ndarray): Language annotations for each step in the trajectory.
            trajectory_length (int): Total length of the trajectory.

        Returns:
            np.ndarray: New indices sampled uniformly from the language-consistent range of the first index.
        """
        if len(step_indices) == 0:
            return np.array([])

        # Use only the first index to determine the target language
        first_idx = max(0, min(step_indices[0], trajectory_length - 1))
        target_language = language_annotations[first_idx]

        # Build sampled indices by moving in ±32-frame steps from first_idx
        # and adding 4 frames at 8-frame strides for each step, while:
        # - staying within trajectory bounds,
        # - keeping language consistent with target_language at the anchor step,
        # - and limiting the total collected frames to 81.
        max_frames = 8 * self.max_chunk_size + 1
        per_step_offsets = [0, 3, 6, 9, 12, 15, 18, 21]
        sampled_list: list[int] = []

        def add_step_set(anchor_index: int) -> None:
            # Only add a complete 4-frame set if it fully fits and capacity allows
            # Require full 32-frame window to exist for alignment with action/state
            nonlocal sampled_list
            if anchor_index < 0 or anchor_index + 23 >= trajectory_length:
                return
            if len(sampled_list) + len(per_step_offsets) > max_frames:
                return
            for offset in per_step_offsets:
                idx = anchor_index + offset
                sampled_list.append(int(idx))

        # Always include the set at the first_idx
        add_step_set(first_idx)

        step = 1
        back_done = False
        fwd_done = False
        while len(sampled_list) < max_frames and (not back_done or not fwd_done):
            # Backward step
            if not back_done:
                back_anchor = first_idx - 24 * step
                if back_anchor < 0:
                    back_done = True
                elif language_annotations[back_anchor] != target_language:
                    back_done = True
                else:
                    add_step_set(back_anchor)
            # Forward step
            if len(sampled_list) >= max_frames:
                break
            if not fwd_done:
                fwd_anchor = first_idx + 24 * step
                if fwd_anchor >= trajectory_length:
                    fwd_done = True
                elif language_annotations[fwd_anchor] != target_language:
                    fwd_done = True
                else:
                    add_step_set(fwd_anchor)
            step += 1

        if len(sampled_list) == 0:
            return np.array([])
        unique_sorted = np.array(sorted(set(sampled_list)), dtype=int)
        # Ensure we return at most 81 frames
        if unique_sorted.size > max_frames:
            unique_sorted = unique_sorted[:max_frames]

        # Convert to 4n+1 format by adding one more frame at the end with 8-frame stride
        if unique_sorted.size > 0:
            # Get the last index and add one more frame with 8-frame stride
            last_idx = unique_sorted[-1]
            additional_idx = last_idx + 3

            # Only add if it doesn't exceed trajectory bounds and max_frames
            if additional_idx < trajectory_length and unique_sorted.size < max_frames:
                unique_sorted = np.append(unique_sorted, additional_idx)
            else:
                # Trim to 8n+1 format. Require at least 9 frames so (noisy_frames-1)//num_frame_per_block >= 1
                # for action/state model invariant (CausalWanModel); otherwise return empty so sample is skipped.
                if unique_sorted.size <= 8:
                    return np.array([])
                unique_sorted = unique_sorted[:-7]

        # ensure that unique_sorted has 4n+1 frames
        assert (
            unique_sorted.size % 8 == 1
        ), f"unique_sorted size {unique_sorted.size} is not 4n+1"

        # Store the number of chunks for alignment with action/state
        num_video_chunks = (unique_sorted.size - 1) // 8
        if not hasattr(self, "_current_num_chunks"):
            self._current_num_chunks = {}
        # Use first_idx as a key to track the current sample's chunk count
        self._current_num_chunks[first_idx] = num_video_chunks

        # print("unique_sorted size", unique_sorted.size, "num_video_chunks", num_video_chunks)
        return unique_sorted


def build_transform_pipeline(
    video_keys: List[str], state_keys: List[str], action_keys: List[str]
) -> ComposedModalityTransform:
    transform_list = [
        MemorySafeCopyTransform(apply_to=[]),
        # --- 1. Video Pipeline ---
        VideoToTensor(apply_to=video_keys),
        VideoCrop(
            apply_to=video_keys,
            scale=0.95,
        ),
        VideoResize(apply_to=video_keys, height=160, width=320, interpolation="linear"),
        VideoColorJitter(
            apply_to=video_keys, brightness=0.3, contrast=0.4, saturation=0.5, hue=0.08
        ),
        VideoToNumpy(apply_to=video_keys),
        # --- 2. State Pipeline ---
        StateActionToTensor(apply_to=state_keys),
        StateActionTransform(
            apply_to=state_keys, normalization_modes={k: "q99" for k in state_keys}
        ),
        # --- 3. Action Pipeline ---
        StateActionToTensor(apply_to=action_keys),
        StateActionTransform(
            apply_to=action_keys, normalization_modes={k: "q99" for k in action_keys}
        ),
        # --- 4. Concatenation Fusion ---
        ConcatTransform(
            # apply_to=[],
            video_concat_order=video_keys,
            state_concat_order=state_keys,
            action_concat_order=action_keys,
        ),
        ZuluTransform(
            default_instruction="Perform the default behavior.",
            language_dropout_prob=0.0,
            always_use_default_instruction=False,
            max_state_dim=64,
            max_action_dim=32,
            max_length=512,
            state_horizon=1,
            action_horizon=24,
            tokenizer_path="models/flan-t5-base",
            embodiment_tag_mapping={
                "real_gr1_arms_only": 0,
                "real_gr1_arms_only_annotated": 1,
                "real_gr1_arms_waist": 2,
                "real_gr1_arms_waist_annotated": 3,
                "dexmg_gr1_arms_only_inspire": 4,
                "dexmg_gr1_arms_only_fourier": 5,
                "dexmg_gr1_arms_waist_fourier": 6,
                "robocasa_single_arm": 7,
                "onex_eve_gripper": 8,
                "robocasa_gr1_arms_only_inspire_hands": 9,
                "robocasa_gr1_arms_only_fourier_hands": 10,
                "robocasa_gr1_fixed_lower_body_inspire_hands": 11,
                "robocasa_gr1_fixed_lower_body_fourier_hands": 12,
                "robocasa_panda_omron": 13,
                "robocasa_bimanual_panda_parallel_gripper": 15,
                "robocasa_bimanual_panda_inspire_hand": 16,
                "oxe_droid": 17,
                "oxe_fractal": 18,
                "oxe_language_table": 19,
                "oxe_bridge": 20,
                "real_panda_single_arm": 21,
                "hot3d_hands_only": 23,
                "gr1_unified": 24,
                "robocasa_gr1_arms_waist_fourier_hands": 25,
                "agibot": 26,
                "lapa": 27,
                "oxe_mutex": 28,
                "oxe_roboset": 29,
                "oxe_plex": 30,
                "dream": 31,
                "yam": 32,
                "xdof": 22,
                "gr1_unified_segmentation": 14,
                "language_table_sim": 7,
                "gr1_isaac": 0,
                "sim_behavior_r1_pro": 31,
                "mecka_hands": 27,
                "real_r1_pro_sharpa": 28,
            },
        ),
    ]

    return ComposedModalityTransform(transforms=transform_list)

