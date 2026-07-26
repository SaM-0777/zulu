from collections import defaultdict
from concurrent.futures import Future, ThreadPoolExecutor
import copy
import glob
import hashlib
import importlib
import json
from pathlib import Path
import threading
import time
from typing import Sequence, TypeVar

import pandas as pd

from src.configs.train import (
    mixture_spec,
    all_modality_configs,
    fps,
    dataset_kwargs,
    mixture_kwargs,
    video_keys,
    action_keys,
    state_keys,
    metadata_versions,
)
from src.data.video_utils import get_frames_by_timestamps
import yaml

from tqdm import tqdm
from src.data.lerobot import (
    LeRobotSingleDataset,
    ModalityConfig,
    ShardedLeRobotSubLangSingleActionChunkDatasetDROID,
    build_transform_pipeline,
)
from src.data.schema import DatasetMetadata, EmbodimentTag
from src.data.transforms_base import ComposedModalityTransform
import numpy as np
from pydantic import BaseModel, Field, field_validator
import torch.distributed as dist
from torch.utils.data import Dataset, IterableDataset, get_worker_info

T_LeRobotMixtureDataset = TypeVar(
    "T_LeRobotMixtureDataset", bound="LeRobotMixtureDataset"
)

_VIDEO_DECODE_LOCK= threading.Lock()


class MixtureSpecElement(BaseModel):
    """Specification element for a dataset mixture defining paths and weights.

    This class validates dataset paths by embodiment tag and handles weight distribution
    across multiple dataset paths if requested.
    """

    dataset_path: dict[str, list[Path] | Path] = Field(
        ..., description="The path to the dataset."
    )
    dataset_weight: float = Field(
        ..., description="The weight of the dataset in the mixture."
    )
    distribute_weights: bool = Field(
        default=False,
        description="Whether to distribute the weights of the dataset across all the paths. If True, the weights will be evenly distributed across all the paths.",
    )

    @field_validator("dataset_path", mode="after")
    def validate_dataset_path_keys(
        cls, v: dict[str, list[Path] | Path]
    ) -> dict[str, list[Path]]:
        """Validate dataset paths and expand glob patterns.

        Args:
            v (dict[str, list[Path] | Path]): Dictionary mapping embodiment tags to paths.

        Returns:
            dict[str, list[Path]]: Validated and expanded paths.

        Raises:
            ValueError: If an invalid embodiment tag is provided.
        """
        all_globbed_paths: dict[str, list[Path]] = {}
        for embodiment_tag, paths in v.items():
            try:
                _ = EmbodimentTag(embodiment_tag)
            except ValueError:
                raise ValueError(f"Invalid embodiment tag: {embodiment_tag}")
            if isinstance(paths, Path):
                paths = [paths]
            globbed_paths = []
            for path in paths:
                globbed_paths.extend(glob.glob(str(path)))
            all_globbed_paths[embodiment_tag] = globbed_paths
        return all_globbed_paths


def safe_hash(input_tuple):
    """Generate a safe hash from an input tuple.

    Creates a deterministic hash using SHA256 and returns the lower 128 bits.
    This is used for deterministic random seed generation.

    Args:
        input_tuple: The tuple to hash.

    Returns:
        int: A 128-bit hash value.
    """
    # keep 128 bits of the hash
    tuple_string = repr(input_tuple).encode("utf-8")
    sha256 = hashlib.sha256()
    sha256.update(tuple_string)

    seed = int(sha256.hexdigest(), 16)

    return seed & 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF


class LeRobotMixtureDataset(Dataset):
    """
    A mixture of multiple datasets. This class samples a single dataset based on the dataset weights and then calls the `__getitem__` method of the sampled dataset.
    It is recommended to modify the single dataset class instead of this class.
    """

    def __init__(
        self,
        data_mixture: Sequence[tuple[LeRobotSingleDataset, float]],
        training: bool,
        balance_dataset_weights: bool = True,
        balance_trajectory_weights: bool = True,
        seed: int = 42,
        allow_padding_at_end: bool = False,
        metadata_config: dict = {
            "percentile_mixing_method": "min_max",
        },
    ):
        """
        Initialize the mixture dataset.

        Args:
            data_mixture (list[tuple[LeRobotSingleDataset, float]]): Datasets and their corresponding weights.
            training (bool): If True, __getitem__ will return different samples every epoch; if False, __getitem__ will return the same sample every epoch.
            balance_dataset_weights (bool): If True, the weight of dataset will be multiplied by the total trajectory length of each dataset.
            balance_trajectory_weights (bool): If True, sample trajectories within a dataset weighted by their length; otherwise, use equal weighting.
            seed (int): Random seed for sampling.
            allow_padding_at_end (bool): If True, allow padding at the end of the dataset.
        """
        datasets: list[LeRobotSingleDataset] = []
        dataset_sampling_weights: list[float] = []
        for dataset, weight in data_mixture:
            datasets.append(dataset)
            dataset_sampling_weights.append(weight)
        self.datasets = datasets
        self.balance_dataset_weights = balance_dataset_weights
        self.balance_trajectory_weights = balance_trajectory_weights
        self.seed = seed
        self.training = training
        self.allow_padding_at_end = allow_padding_at_end

        # Set properties for sampling

        # 1. Dataset lengths
        self._dataset_lengths = np.array([len(dataset) for dataset in self.datasets])

        # 2. Dataset sampling weights
        self._dataset_sampling_weights = np.array(dataset_sampling_weights)
        if self.balance_dataset_weights:
            self._dataset_sampling_weights *= self._dataset_lengths
        self._dataset_sampling_weights /= self._dataset_sampling_weights.sum()

        # 3. Trajectory sampling weights
        self._trajectory_sampling_weights: list[np.ndarray] = []
        for dataset in self.datasets:
            trajectory_sampling_weights = np.ones(len(dataset.trajectory_ids))
            if self.balance_trajectory_weights:
                trajectory_sampling_weights *= np.array(
                    [
                        len(dataset.step_filter[trajectory_id])
                        for trajectory_id in dataset.trajectory_ids
                    ]
                )

            if dataset.discard_bad_trajectories:
                bad_trajectory_indices = dataset.lerobot_info_meta.get(
                    "discarded_episode_indices", []
                )
                trajectory_sampling_weights[bad_trajectory_indices] = 0.0

            if trajectory_sampling_weights.sum() == 0:
                raise ValueError(f"No valid trajectories found for dataset {dataset}")

            trajectory_sampling_weights /= trajectory_sampling_weights.sum()
            self._trajectory_sampling_weights.append(trajectory_sampling_weights)

        # 4. Primary dataset indices
        self._primary_dataset_indices = np.array(dataset_sampling_weights) == 1.0

        # Set the epoch and sample the first epoch
        self.set_epoch(0)

        # Create a merged metadata for the mixture dataset (we don't need this in the future as eval will directly use `get_metadata`)
        self.update_metadata(metadata_config)

        # Set the transforms to training or evaluation mode
        if self.training:
            for dataset in self.datasets:
                dataset.transforms.train()  # type: ignore
        else:
            for dataset in self.datasets:
                dataset.transforms.eval()  # type: ignore

    @property
    def dataset_lengths(self) -> np.ndarray:
        """The lengths of each dataset."""
        return self._dataset_lengths

    @property
    def dataset_sampling_weights(self) -> np.ndarray:
        """The sampling weights for each dataset."""
        return self._dataset_sampling_weights

    @property
    def trajectory_sampling_weights(self) -> list[np.ndarray]:
        """The sampling weights for each trajectory in each dataset."""
        return self._trajectory_sampling_weights

    @property
    def primary_dataset_indices(self) -> np.ndarray:
        """The indices of the primary datasets."""
        return self._primary_dataset_indices

    def __str__(self) -> str:
        """Return a string representation of the mixture dataset with weights."""
        dataset_descriptions = []
        for dataset, weight in zip(self.datasets, self.dataset_sampling_weights):
            dataset_description = {
                "Dataset": str(dataset),
                "Sampling weight": float(weight),
            }
            dataset_descriptions.append(dataset_description)
        return yaml.dump({"Mixture dataset": dataset_descriptions})  # type: ignore

    @classmethod
    def from_mixture_spec(
        cls: type[T_LeRobotMixtureDataset],
        mixture_spec: Sequence[MixtureSpecElement | dict],
        dataset_class: type[LeRobotSingleDataset] | str,
        all_modality_configs: dict[str, dict[str, ModalityConfig]],
        all_transforms: dict[str, ComposedModalityTransform],
        metadata_versions: dict[str, str],
        fps: dict | None = None,
        dataset_kwargs: dict | None = None,
        mixture_kwargs: dict | None = None,
    ) -> T_LeRobotMixtureDataset:
        """Initialize the mixture dataset from a specification.

        Args:
            mixture_spec (Sequence[MixtureSpecElement | dict]): The specification for the mixture dataset.
            dataset_class (type[LeRobotSingleDataset] | str): The dataset class or its string path.
            all_modality_configs (dict[str, dict[str, ModalityConfig]]): The modality configs for each embodiment.
            all_transforms (dict[str, ComposedModalityTransform]): The transforms for each embodiment.
            metadata_versions (dict[str, str]): The metadata versions for each embodiment.
            dataset_kwargs (dict | None): Additional keyword arguments for the dataset classes.
            mixture_kwargs (dict | None): Additional keyword arguments for the mixture dataset.

        Returns:
            LeRobotMixtureDataset: The initialized mixture dataset.
        """
        if isinstance(dataset_class, str):
            module_name, class_name = dataset_class.rsplit(".", 1)
            module = importlib.import_module(module_name)
            dataset_class = getattr(module, class_name)
        assert not isinstance(dataset_class, str), f"{dataset_class} is a string"
        assert issubclass(
            dataset_class, LeRobotSingleDataset
        ), f"{dataset_class} is not a subclass of LeRobotSingleDataset"
        data_mixture = []

        for dataset_spec in tqdm(
            mixture_spec,
            total=len(mixture_spec),
            desc="Initializing datasets",
        ):
            start_time = time.time()
            if isinstance(dataset_spec, dict):
                dataset_spec = MixtureSpecElement.model_validate(dataset_spec)
            datasets = []
            for embodiment_tag, paths in dataset_spec.dataset_path.items():
                if isinstance(paths, Path):
                    paths = [paths]
                for dataset_path in paths:
                    if ".sh" in dataset_path or ".json" in dataset_path:  # type: ignore
                        continue
                    assert (
                        embodiment_tag in all_modality_configs
                    ), f"{embodiment_tag} not in modality_configs: {all_modality_configs.keys()}"
                    assert (
                        embodiment_tag in all_transforms
                    ), f"{embodiment_tag} not in transforms: {all_transforms.keys()}"
                    dataset = dataset_class(
                        dataset_path=dataset_path,
                        embodiment_tag=EmbodimentTag(embodiment_tag),
                        modality_configs=copy.copy(
                            all_modality_configs[embodiment_tag]
                        ),
                        transforms=copy.copy(all_transforms[embodiment_tag]),
                        metadata_version=metadata_versions[embodiment_tag],
                        fps=fps[embodiment_tag] if embodiment_tag in fps else None,  # type: ignore
                        **(dataset_kwargs if dataset_kwargs is not None else {}),
                    )
                    datasets.append(dataset)
            # dataset_lengths = np.array([len(dataset) for dataset in datasets])
            # dataset_relative_lengths = dataset_lengths / dataset_lengths.sum()

            if dataset_spec.distribute_weights:
                dataset_lengths = np.array([len(dataset) for dataset in datasets])
                dataset_relative_lengths = dataset_lengths / dataset_lengths.sum()
            else:
                dataset_relative_lengths = [1.0] * len(datasets)

            for dataset, relative_length in zip(datasets, dataset_relative_lengths):
                if dataset_spec.distribute_weights:
                    weight = relative_length * dataset_spec.dataset_weight
                else:
                    weight = dataset_spec.dataset_weight
                data_mixture.append((dataset, weight))

            print(
                f"Time taken to initialize {len(datasets)} datasets: {time.time() - start_time:.2f} seconds"
            )

        return cls(
            data_mixture=data_mixture,
            **(mixture_kwargs if mixture_kwargs is not None else {}),
        )

    def set_epoch(self, epoch: int):
        """Set the epoch for the dataset.

        Args:
            epoch (int): The epoch to set.
        """
        self.epoch = epoch
        # self.sampled_steps = self.sample_epoch()

    def sample_step(self, index: int) -> tuple[LeRobotSingleDataset, int, int]:
        """Sample a single step from the mixture dataset.

        Args:
            index (int): The index to sample (used for deterministic sampling).

        Returns:
            tuple[LeRobotSingleDataset, int, int]: A tuple of (dataset, trajectory_id, step_index).
        """
        # return self.sampled_steps[index]

        # Set seed
        if self.training:
            seed = safe_hash((self.epoch, index, self.seed))
            rng = np.random.default_rng(seed)

            # Sample dataset
            dataset_index = rng.choice(
                len(self.datasets), p=self.dataset_sampling_weights
            )
            dataset = self.datasets[dataset_index]

            if self.allow_padding_at_end:
                # Sample trajectory
                trajectory_index = rng.choice(
                    len(dataset.trajectory_ids),
                    p=self.trajectory_sampling_weights[dataset_index],
                )
                trajectory_id = dataset.trajectory_ids[trajectory_index]

                allowed_length = dataset.trajectory_lengths[trajectory_index]
            else:
                # Avoid padding at the end of the trajectory
                max_delta_index = dataset.max_delta_index
                trajectory_length = 0
                trajectory_id = None
                while trajectory_length < max_delta_index + 1:
                    # Sample trajectory
                    trajectory_index = rng.choice(
                        len(dataset.trajectory_ids),
                        p=self.trajectory_sampling_weights[dataset_index],
                    )
                    trajectory_id = dataset.trajectory_ids[trajectory_index]
                    trajectory_length = dataset.trajectory_lengths[trajectory_index]
                assert trajectory_id is not None

                # Sample step
                assert (
                    trajectory_length >= max_delta_index + 1
                ), f"{trajectory_length=}, {max_delta_index=}"
                allowed_length = trajectory_length - max_delta_index
            # Get the allowed indices from the step filter
            allowed_indices = dataset.step_filter[trajectory_id]
            # Remove indices that are too large
            allowed_indices = allowed_indices[allowed_indices <= allowed_length]
            step_index = rng.choice(allowed_indices)
            return dataset, trajectory_id, step_index
        else:
            length_cumsum = np.cumsum(self.dataset_lengths)
            dataset_index = np.searchsorted(length_cumsum, index)
            dataset = self.datasets[dataset_index]
            assert (
                len(dataset.lerobot_info_meta.get("discarded_episode_indices", [])) == 0
            ), f"Find discarded episode indices in evaluation dataset {dataset.dataset_path}"
            trajectory_id, step_index = dataset.all_steps[
                index - length_cumsum[dataset_index]
            ]
            return dataset, trajectory_id, step_index

    def __getitem__(self, index: int) -> dict:
        """Get the data for a single trajectory and start index.

        Args:
            index (int): The index of the trajectory to get.

        Returns:
            dict: The data for the trajectory and start index.
        """
        dataset, trajectory_id, step_index = self.sample_step(index)
        indices = {
            key: delta_indices + step_index
            for key, delta_indices in dataset.delta_indices.items()
        }
        return dataset.transforms(dataset.get_step_data(trajectory_id, indices))  # type: ignore

    def __len__(self) -> int:
        """Get the length of a single epoch in the mixture.

        Returns:
            int: The length of a single epoch in the mixture.
        """
        if self.training:
            return int((self.dataset_lengths * self.dataset_sampling_weights).sum())
        else:
            return int(self.dataset_lengths.sum())

    @staticmethod
    def compute_overall_statistics(
        per_task_stats: list[dict[str, dict[str, list[float] | np.ndarray]]],
        dataset_sampling_weights: list[float] | np.ndarray,
        percentile_mixing_method: str = "weighted_average",
    ) -> dict[str, dict[str, list[float]]]:
        """
        Computes overall statistics from per-task statistics using dataset sample weights.

        Args:
            per_task_stats: List of per-task statistics.
            Example format of one element in the per-task statistics list:
                {
                    "state.gripper": {
                        "min": [...],
                        "max": [...],
                        "mean": [...],
                        "std": [...],
                        "q01": [...],
                        "q99": [...],
                    },
                    ...
                }
            dataset_sampling_weights: List of sample weights for each task.
            percentile_mixing_method: The method to mix the percentiles, either "weighted_average" or "weighted_std".

        Returns:
            A dict of overall statistics per modality.
        """
        # Normalize the sample weights to sum to 1
        dataset_sampling_weights = np.array(dataset_sampling_weights)
        normalized_weights = dataset_sampling_weights / dataset_sampling_weights.sum()

        # Initialize overall statistics dict
        overall_stats: dict[str, dict[str, list[float]]] = {}

        # Get the list of modality keys
        modality_keys = per_task_stats[0].keys()

        for modality in modality_keys:
            # Check if stats are per-horizon (2D) by examining the first task's mean
            first_mean = np.array(per_task_stats[0][modality]["mean"])
            is_per_horizon = first_mean.ndim == 2  # Shape (horizon_len, action_dim)

            if is_per_horizon:
                # Handle per-horizon stats (2D arrays)
                stats_shape = first_mean.shape  # (horizon_len, action_dim)

                # Initialize accumulators for means and variances
                weighted_means = np.zeros(stats_shape)
                weighted_squares = np.zeros(stats_shape)

                # Collect min, max, q01, q99 from all tasks
                min_list = []
                max_list = []
                q01_list = []
                q99_list = []

                for task_idx, task_stats in enumerate(per_task_stats):
                    w_i = normalized_weights[task_idx]
                    stats = task_stats[modality]
                    means = np.array(stats["mean"])
                    stds = np.array(stats["std"])

                    # Update weighted sums for mean and variance
                    weighted_means += w_i * means
                    weighted_squares += w_i * (stds**2 + means**2)

                    # Collect min, max, q01, q99
                    min_list.append(np.array(stats["min"]))
                    max_list.append(np.array(stats["max"]))
                    q01_list.append(np.array(stats["q01"]))
                    q99_list.append(np.array(stats["q99"]))

                # Compute overall mean
                overall_mean = weighted_means.tolist()

                # Compute overall variance and std deviation
                overall_variance = weighted_squares - weighted_means**2
                overall_std = np.sqrt(np.maximum(overall_variance, 0)).tolist()

                # Compute overall min and max per dimension
                # Stack along new axis: (num_tasks, horizon_len, action_dim)
                overall_min = np.min(np.stack(min_list, axis=0), axis=0).tolist()
                overall_max = np.max(np.stack(max_list, axis=0), axis=0).tolist()

                # Compute overall q01 and q99 per dimension
                q01_array = np.stack(
                    q01_list, axis=0
                )  # (num_tasks, horizon_len, action_dim)
                q99_array = np.stack(q99_list, axis=0)
                if percentile_mixing_method == "weighted_average":
                    # Weighted average along task axis
                    weighted_q01 = np.average(
                        q01_array, axis=0, weights=normalized_weights
                    ).tolist()
                    weighted_q99 = np.average(
                        q99_array, axis=0, weights=normalized_weights
                    ).tolist()
                elif percentile_mixing_method == "min_max":
                    weighted_q01 = np.min(q01_array, axis=0).tolist()
                    weighted_q99 = np.max(q99_array, axis=0).tolist()
                else:
                    raise ValueError(
                        f"Invalid percentile mixing method: {percentile_mixing_method}"
                    )
            else:
                # Handle regular stats (1D arrays)
                num_dims = len(first_mean)

                # Initialize accumulators for means and variances
                weighted_means = np.zeros(num_dims)
                weighted_squares = np.zeros(num_dims)

                # Collect min, max, q01, q99 from all tasks
                min_list = []
                max_list = []
                q01_list = []
                q99_list = []

                for task_idx, task_stats in enumerate(per_task_stats):
                    w_i = normalized_weights[task_idx]
                    stats = task_stats[modality]
                    means = np.array(stats["mean"])
                    stds = np.array(stats["std"])

                    # Update weighted sums for mean and variance
                    weighted_means += w_i * means
                    weighted_squares += w_i * (stds**2 + means**2)

                    # Collect min, max, q01, q99
                    min_list.append(stats["min"])
                    max_list.append(stats["max"])
                    q01_list.append(stats["q01"])
                    q99_list.append(stats["q99"])

                # Compute overall mean
                overall_mean = weighted_means.tolist()

                # Compute overall variance and std deviation
                overall_variance = weighted_squares - weighted_means**2
                overall_std = np.sqrt(np.maximum(overall_variance, 0)).tolist()

                # Compute overall min and max per dimension
                overall_min = np.min(np.array(min_list), axis=0).tolist()
                overall_max = np.max(np.array(max_list), axis=0).tolist()

                # Compute overall q01 and q99 per dimension
                # Use weighted average of per-task quantiles
                q01_array = np.array(q01_list)
                q99_array = np.array(q99_list)
                if percentile_mixing_method == "weighted_average":
                    weighted_q01 = np.average(
                        q01_array, axis=0, weights=normalized_weights
                    ).tolist()
                    weighted_q99 = np.average(
                        q99_array, axis=0, weights=normalized_weights
                    ).tolist()
                elif percentile_mixing_method == "min_max":
                    weighted_q01 = np.min(q01_array, axis=0).tolist()
                    weighted_q99 = np.max(q99_array, axis=0).tolist()
                else:
                    raise ValueError(
                        f"Invalid percentile mixing method: {percentile_mixing_method}"
                    )

            # Store the overall statistics for the modality
            overall_stats[modality] = {  # type: ignore
                "min": overall_min,
                "max": overall_max,
                "mean": overall_mean,
                "std": overall_std,
                "q01": weighted_q01,
                "q99": weighted_q99,
            }

        return overall_stats

    @staticmethod
    def merge_metadata(
        metadatas: list[DatasetMetadata],
        dataset_sampling_weights: list[float],
        percentile_mixing_method: str,
    ) -> DatasetMetadata:
        """Merge multiple metadata into one."""
        # Convert to dicts
        metadata_dicts = [metadata.model_dump(mode="json") for metadata in metadatas]
        # Create a new metadata dict
        merged_metadata = {}

        # Check all metadata have the same embodiment tag
        assert all(
            metadata.embodiment_tag == metadatas[0].embodiment_tag
            for metadata in metadatas
        ), "All metadata must have the same embodiment tag"
        merged_metadata["embodiment_tag"] = metadatas[0].embodiment_tag

        # Merge the dataset statistics
        dataset_statistics = {}
        dataset_statistics["state"] = LeRobotMixtureDataset.compute_overall_statistics(
            per_task_stats=[m["statistics"]["state"] for m in metadata_dicts],
            dataset_sampling_weights=dataset_sampling_weights,
            percentile_mixing_method=percentile_mixing_method,
        )
        dataset_statistics["action"] = LeRobotMixtureDataset.compute_overall_statistics(
            per_task_stats=[m["statistics"]["action"] for m in metadata_dicts],
            dataset_sampling_weights=dataset_sampling_weights,
            percentile_mixing_method=percentile_mixing_method,
        )
        merged_metadata["statistics"] = dataset_statistics

        # Merge the modality configs
        modality_configs = defaultdict(set)
        for metadata in metadata_dicts:
            for modality, configs in metadata["modalities"].items():
                modality_configs[modality].add(json.dumps(configs))
        merged_metadata["modalities"] = {}
        for modality, configs in modality_configs.items():
            # Check that all modality configs correspond to the same tag matches
            assert (
                len(configs) == 1
            ), f"Multiple modality configs for modality {modality}: {list(configs)}"
            merged_metadata["modalities"][modality] = json.loads(configs.pop())

        return DatasetMetadata.model_validate(merged_metadata)

    def update_metadata(self, metadata_config: dict) -> None:
        """Merge multiple metadatas into one and set the transforms with the merged metadata.

        Args:
            metadata_config (dict): Configuration for the metadata.
                "percentile_mixing_method": The method to mix the percentiles, either "weighted_average" or "min_max".
                    weighted_average: Use the weighted average of the percentiles using the weight used in sampling the datasets.
                    min_max: Use the min of the 1st percentile and max of the 99th percentile.
        """

        self.merged_metadata: dict[str, DatasetMetadata] = {}
        # Group metadata by tag
        all_metadatas: dict[str, list[DatasetMetadata]] = {}
        for dataset in self.datasets:
            if dataset.tag.value not in all_metadatas:
                all_metadatas[dataset.tag.value] = []
            all_metadatas[dataset.tag.value].append(dataset.metadata)
        for tag, metadatas in all_metadatas.items():
            self.merged_metadata[tag] = self.merge_metadata(
                metadatas=metadatas,
                dataset_sampling_weights=self.dataset_sampling_weights.tolist(),
                percentile_mixing_method=metadata_config["percentile_mixing_method"],
            )
        for dataset in self.datasets:
            dataset.set_transforms_metadata(self.merged_metadata[dataset.tag.value])

    def get_initial_actions(self):
        initial_actions = []
        for dataset in self.datasets:
            if hasattr(dataset, "get_initial_actions"):
                initial_actions.extend(dataset.get_initial_actions())
        return initial_actions


class ShardedLeRobotSingleDataset(LeRobotSingleDataset):
    """
    A single dataset with shards.
    """

    def __init__(
        self,
        *args,
        num_steps_per_shard: int = int(1e4),
        **kwargs,
    ):
        self.args = args
        self.kwargs = kwargs
        super().__init__(*args, **kwargs)
        self.num_steps_per_shard = num_steps_per_shard
        self.all_video_paths = self.get_all_video_paths()
        self.all_parquet_paths = self.get_all_parquet_paths()
        self.sharded_trajectories, self.shard_lengths = self.generate_shards()
        self.frames_to_load = self.get_all_frames_to_load()

        # Set shard caching properties
        self.shard_start_indices: dict[int, int] | None = None
        self.cached_shard: dict[str, np.ndarray] | None = None
        self.cached_df: pd.DataFrame | None = None
        self.frame_indices_map: dict[int, dict[str, np.ndarray]] | None = None
        self._executor = ThreadPoolExecutor(max_workers=0)
        self._cache_job: Future | None = None
        self._shard_lock = threading.Lock() # added

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
        for trajectory_id in self.trajectory_ids:
            if isinstance(trajectory_id, np.integer):
                trajectory_id = trajectory_id.item()
            assert isinstance(
                trajectory_id, int
            ), f"trajectory_id must be an integer, got {type(trajectory_id)}"
            video_paths[trajectory_id] = {}
            for key in self.modality_keys["video"]:
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
            for trajectory_id in self.trajectory_ids
        }

    def generate_shards(self) -> tuple[list[list[int]], np.ndarray]:
        """Generate shards of trajectories. We recommend num_steps_per_shard >> average trajectory length.

        Args:
            num_steps_per_shard (int): The number of steps per shard.

        Returns:
            list[list[str]]: The shards of trajectories.
        """
        sharded_trajectories = [[]]
        curr_num_steps = 0
        curr_shard_index = 0
        discarded_episode_indices = []
        trajectory_ids = self.trajectory_ids
        if self.discard_bad_trajectories:
            discarded_episode_indices = self.lerobot_info_meta.get(
                "discarded_episode_indices", []
            )
            trajectory_ids = [
                trajectory_id
                for trajectory_id in trajectory_ids
                if trajectory_id not in discarded_episode_indices
            ]

        assert (
            len(trajectory_ids) > 0
        ), f"No valid trajectories found for dataset {self.dataset_path}"
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
            f"Generated {len(sharded_trajectories)} shards for dataset {self.dataset_path}"
        )
        return sharded_trajectories, np.array(shard_lengths)

    def get_all_frames_to_load(self):
        """Generate a map of video frame indices to trajectory indices."""
        all_frames_to_load = {}
        for trajectory_id in self.trajectory_ids:
            all_frames_to_load[trajectory_id] = {}
            for key in self.modality_keys["video"]:
                assert key.startswith(
                    "video."
                ), f"Video key must start with 'video.', got {key}"
                filtered_indices = self.step_filter[trajectory_id]
                if len(filtered_indices) > 0:
                    frames_to_load = np.unique(
                        np.concatenate(
                            [
                                i + np.array(self.delta_indices[key])
                                for i in filtered_indices
                            ]
                        )
                    )
                    # Cap within the length of the trajectory and >= 0
                    frames_to_load = frames_to_load[
                        (frames_to_load < self.trajectory_lengths[trajectory_id])
                        & (frames_to_load >= 0)
                    ]
                else:
                    frames_to_load = np.array([])
                all_frames_to_load[trajectory_id][key] = frames_to_load
        return all_frames_to_load

    @staticmethod
    def get_shard(
        trajectory_ids: list[int] | np.ndarray,
        modality_keys: dict,
        video_paths: dict[int, dict[str, Path]],
        parquet_paths: dict[int, Path],
        frames_to_load: dict[int, dict[str, np.ndarray]],
        video_backend: str = "pyav",
        video_backend_kwargs: dict | None = None,
    ) -> tuple[
        dict[str, np.ndarray],
        dict[int, int],
        pd.DataFrame,
        dict[int, dict[str, np.ndarray]],
    ]:
        print("Caching shard")
        start_time = time.time()
        assert (
            "video" in modality_keys
        ), "No video modality found. No need to use caching."
        cached_frames = {}
        trajectory_start_indices = {}
        frame_indices_map = {}
        curr_step_index = 0
        cached_df = None
        curr_frame_index = {key: 0 for key in modality_keys["video"]}
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
            frame_indices_map[trajectory_id] = {}
            for key in modality_keys["video"]:
                # Only load the frames that are needed
                this_frames_to_load = frames_to_load[trajectory_id][key]
                if len(this_frames_to_load) == 0:
                    continue
                load_timestamps = parquet_timestamps[this_frames_to_load]
                assert key.startswith(
                    "video."
                ), f"Video key must start with 'video.', got {key}"
                # Store a mapping that frame_indices_map[trajectory_id][key][frame_index] = index_in_concat_video_frames
                frame_indices_map[trajectory_id][key] = (
                    np.ones(len(parquet_timestamps), dtype=np.int32) * -1
                )
                frame_indices_map[trajectory_id][key][this_frames_to_load] = np.arange(
                    curr_frame_index[key],
                    curr_frame_index[key] + len(this_frames_to_load),
                    dtype=np.int32,
                )
                curr_frame_index[key] += len(this_frames_to_load)
                if key not in cached_frames:
                    cached_frames[key] = []

                with _VIDEO_DECODE_LOCK:
                    frames = get_frames_by_timestamps(
                        video_paths[trajectory_id][key].as_posix(),
                        timestamps=load_timestamps,
                        video_backend=video_backend,
                        video_backend_kwargs=video_backend_kwargs or {},
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
        return cached_frames, trajectory_start_indices, cached_df, frame_indices_map

    def start_cache_shard(self, shard_index: int) -> None:
        """Start caching a shard in a background thread."""
        self._cache_job = self._executor.submit(
            self.get_shard,
            self.sharded_trajectories[shard_index],
            self.modality_keys,
            self.all_video_paths,
            self.all_parquet_paths,
            self.frames_to_load,
            self.video_backend,
            self.video_backend_kwargs,
        )

    def finish_cache_shard(self):
        """Get the cached shard."""
        assert self._cache_job is not None
        (
            self.cached_shard,
            self.shard_start_indices,
            self.cached_df,
            self.frame_indices_map,
        ) = self._cache_job.result()
        self._cache_job = None  # Clear the future to allow memory to be freed

    def delete_cached_shard(self):
        """Delete the cached shard."""
        #del self.cached_shard
        #del self.shard_start_indices
        #del self.cached_df
        with self._shard_lock:
            self.cached_shard = None
            self.shard_start_indices = None
            self.cached_df = None
            self.frame_indices_map = None

    def get_trajectories_in_shard(self) -> list[int]:
        """Get the trajectories in a shard."""
        assert self.shard_start_indices is not None
        return list(self.shard_start_indices.keys())

    def get_video(
        self, trajectory_id: int, key: str, step_indices: np.ndarray
    ) -> np.ndarray:
        """Get the video frames from cached shards for a trajectory by a base index.

        Args:
            trajectory_id (str): The ID of the trajectory.
            key (str): The key of the video.
            base_index (int): The base index of the trajectory.

        Returns:
            np.ndarray: The video frames for the trajectory and frame indices. Shape: (T, H, W, C)
        """
        ## Get the trajectory index
        #trajectory_index = self.get_trajectory_index(trajectory_id)
        ## Ensure the indices are within the valid range
        ## This is equivalent to padding the video with extra frames at the beginning and end
        #step_indices = np.maximum(step_indices, 0)
        #step_indices = np.minimum(
        #    step_indices, self.trajectory_lengths[trajectory_index] - 1
        #)
        ## Calculate the absolute indices
        #assert (
        #    self.shard_start_indices is not None
        #    and self.cached_shard is not None
        #    and trajectory_id in self.shard_start_indices
        #    and self.frame_indices_map is not None
        #    and trajectory_id in self.frame_indices_map
        #    and key in self.frame_indices_map[trajectory_id]
        #), "Shard not cached. Please call `cache_next_shard` and `use_next_shard` first."
        #indices_in_shard = self.frame_indices_map[trajectory_id][key][step_indices]
        #assert np.all(
        #    indices_in_shard != -1
        #), f"Indices in shard are not loaded for {trajectory_id=}, {key=}, {step_indices=}"
        #return self.cached_shard[key][indices_in_shard]
        
        trajectory_index = self.get_trajectory_index(trajectory_id)
        step_indices = np.maximum(step_indices, 0)
        step_indices = np.minimum(
            step_indices, self.trajectory_lengths[trajectory_index] - 1
        )

        # ======= FIX: Copy references safely under Lock =======
        with self._shard_lock:
            if self.cached_shard is None or self.frame_indices_map is None:
                raise RuntimeError("Attempted to read video while shard is deleted/not ready!")
            
            # Copy the references locally. This prevents the background thread 
            # from deleting them while NumPy is doing C-level indexing.
            local_cached_shard = self.cached_shard
            local_frame_indices_map = self.frame_indices_map
            local_shard_start_indices = self.shard_start_indices
        # ======================================================

        assert (
            local_shard_start_indices is not None
            and local_cached_shard is not None
            and trajectory_id in local_shard_start_indices
            and trajectory_id in local_frame_indices_map
            and key in local_frame_indices_map[trajectory_id]
        ), "Shard not cached properly."

        indices_in_shard = local_frame_indices_map[trajectory_id][key][step_indices]
        
        # ======= FIX: Prevent -1 indexing (causes C-level out-of-bounds read) =======
        if np.any(indices_in_shard == -1):
            raise IndexError(f"Indices in shard are not loaded for {trajectory_id=}, {key=}")
        # ============================================================================

        return local_cached_shard[key][indices_in_shard]
        

    def get_trajectory_data(self, trajectory_id: int) -> pd.DataFrame:
        """Get the trajectory data."""
        assert self.cached_df is not None, "Cached dataframe is None"
        traj_data = self.cached_df.loc[self.cached_df["episode_index"] == trajectory_id]
        trajectory_index = self.get_trajectory_index(trajectory_id)
        trajectory_length = self.trajectory_lengths[trajectory_index]
        assert (
            len(traj_data) == trajectory_length
        ), f"Trajectory length mismatch: {len(traj_data)} != {trajectory_length} {self.args} {self.kwargs}"
        indices = traj_data["index"].to_numpy()
        if len(indices) > 0:
            start_index = indices[0]
            expected_indices = np.arange(start_index, start_index + len(indices))
            assert np.array_equal(
                indices, expected_indices
            ), f"[{self}] Index sequence mismatch in trajectory data, {trajectory_id=}"
        return traj_data


class ShardedLeRobotMixtureDataset(LeRobotMixtureDataset, IterableDataset):
    """
    A mixture of multiple datasets. This class samples a single dataset based on the dataset weights and then calls the `__getitem__` method of the sampled dataset.
    It is recommended to modify the single dataset class instead of this class.
    """

    def __init__(
        self,
        data_mixture: list[tuple[LeRobotSingleDataset, float]],
        training: bool,
        balance_dataset_weights: bool = True,
        balance_trajectory_weights: bool = True,
        seed: int = 42,
        shard_sampling_rate: float = 0.5,
        num_shards_to_sample: int = 2**20,
        allow_padding_at_end: bool = False,
    ):
        """
        Initialize the mixture dataset.

        Args:
            data_mixture (list[tuple[ShardedLeRobotSingleDataset, float]]): Datasets and their corresponding weights.
            mode (str): If "train", __iter__ will yield different samples every epoch; if "val" or "test", __iter__ will yield the same sample every epoch.
            balance_dataset_weights (bool): If True, the weight of dataset will be multiplied by the total trajectory length of each dataset.
            balance_trajectory_weights (bool): If True, sample trajectories within a dataset weighted by their length; otherwise, use equal weighting.
            seed (int): Random seed for sampling.
            shard_sampling_rate (float): How much data per shard to sample, in a 0-1 scale.
            num_shards_to_sample (int): The number of shards to sample.
        """
        super().__init__(
            data_mixture=data_mixture,
            training=training,
            balance_dataset_weights=balance_dataset_weights,
            balance_trajectory_weights=balance_trajectory_weights,
            seed=seed,
            allow_padding_at_end=allow_padding_at_end,
        )
        # Add type hint
        self.datasets: list[ShardedLeRobotSingleDataset] = self.datasets
        # Set properties
        self.shard_sampling_rate = shard_sampling_rate
        self.num_shards_to_sample = num_shards_to_sample

        # Calculate shard sampling weights
        all_shard_sampling_weights = []
        all_shards = []
        for dataset_id, (dataset, weight) in enumerate(
            zip(self.datasets, self._dataset_sampling_weights)
        ):
            shard_sampling_weights = dataset.shard_lengths / dataset.shard_lengths.sum()
            all_shard_sampling_weights.append(shard_sampling_weights * weight)
            all_shards.extend(
                [
                    (dataset_id, shard_idx)
                    for shard_idx in range(shard_sampling_weights.shape[0])
                ]
            )
        all_shard_sampling_weights = np.concatenate(all_shard_sampling_weights)
        all_shard_sampling_weights /= all_shard_sampling_weights.sum()
        self._shard_sampling_weights = all_shard_sampling_weights
        self._all_shards = all_shards

        # Generate shards sample schedule for all ranks and workers
        self._shards_sample_schedule = self.generate_shards_sample_schedule()

        # Check shard sampling rate
        assert (
            0 <= shard_sampling_rate <= 1
        ), "Shard sampling rate must be between 0 and 1"

        # Set properties for distributed training
        if dist.is_initialized():
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
        else:
            self.rank = 0
            self.world_size = 1
        self.worker_id = None
        self.num_workers = None

    @property
    def dataset_sampling_weights(self) -> np.ndarray:
        """The dataset sampling weights."""
        return self._dataset_sampling_weights

    @property
    def shard_sampling_weights(self) -> list[np.ndarray]:
        """The weights of each shard."""
        return self._shard_sampling_weights

    @property
    def all_shards(self) -> list[tuple[int, int]]:
        """The shards to sample."""
        return self._all_shards

    @property
    def shards_sample_schedule(self) -> list[tuple[int, int]]:
        """The shards sample schedule.

        Returns:
            list[tuple[int, int]]: The shards to sample, in (dataset_index, shard_index).
        """
        assert (
            self._shards_sample_schedule is not None
        ), "Shards sample schedule not set."
        return self._shards_sample_schedule

    @property
    def trajectory_sampling_weights(self):
        """The trajectory sampling weights."""
        raise ValueError(
            "ShardedRobotMixtureDataset does not support trajectory sampling weights."
        )

    @property
    def primary_dataset_indices(self):
        """The primary dataset indices."""
        raise ValueError(
            "ShardedRobotMixtureDataset does not support primary dataset indices."
        )

    def reset_seed(self, seed: int):
        self.seed = seed
        self._shards_sample_schedule = self.generate_shards_sample_schedule()

    def generate_shards_sample_schedule(self):
        if self.training:
            rng = np.random.default_rng(self.seed)
            sampled_shard_ids = rng.choice(
                len(self.all_shards),
                size=self.num_shards_to_sample,
                p=self.shard_sampling_weights,
            )
            shards_sample_schedule = [self.all_shards[i] for i in sampled_shard_ids]
            rng.shuffle(shards_sample_schedule)
        else:
            shards_sample_schedule = [
                self.all_shards[i % len(self.all_shards)]
                for i in range(self.num_shards_to_sample)
            ]
        return shards_sample_schedule

    def filter_shards_sample_schedule(self):
        """Filter the shards sample schedule for each worker.

        Returns:
            list[tuple[int, int]]: The shards to sample, in (dataset_index, shard_index).
        """
        # Filter shards for each worker
        filtered_schedule = []
        worker_info = get_worker_info()
        # If we have multiple workers, further split shards among them
        if worker_info is not None:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
        else:
            worker_id = 0
            num_workers = 1

        if self.worker_id is None:
            assert self.num_workers is None
            self.worker_id = worker_id
            self.num_workers = num_workers
        else:
            assert (
                self.worker_id == worker_id and self.num_workers == num_workers
            ), "Worker ID or number of workers has been changed since it was set. This is not allowed."

        for i, shard in enumerate(self.shards_sample_schedule):
            if (
                i % (self.world_size * num_workers)
                == self.rank * num_workers + worker_id
            ):
                filtered_schedule.append(shard)
        # print(f"Filtered shards for rank {self.rank}, worker {worker_id}: {filtered_schedule}")
        return filtered_schedule

    def __str__(self) -> str:
        dataset_descriptions = []
        for dataset, weight in zip(self.datasets, self.dataset_sampling_weights):
            shard_lengths = dataset.shard_lengths
            assert len(shard_lengths.shape) == 1, "Shard lengths must be a 1D array"
            num_shards = shard_lengths.shape[0]
            max_shard_length = int(shard_lengths.max())
            min_shard_length = int(shard_lengths.min())
            dataset_description = {
                "Dataset": str(dataset),
                "Sampling weight": float(weight),
                "Num shards": num_shards,
                "Max shard length": max_shard_length,
                "Min shard length": min_shard_length,
            }
            dataset_descriptions.append(dataset_description)
        return yaml.dump(
            {
                "Mixture dataset": dataset_descriptions,
                "Rank": self.rank,
                "World size": self.world_size,
            }
        )  # type: ignore

    def __iter__(self):
        """Iterate over the dataset."""

        # Not supported: balance_trajectory_weights=False
        if not self.balance_trajectory_weights:
            raise NotImplementedError(
                "balance_trajectory_weights=False is not supported. Please use balance_dataset_weights=True instead."
            )

        self._shards_sample_schedule = self.filter_shards_sample_schedule()
        self.curr_shard_index = -1
        self.cache_next_shard()
        rng = np.random.default_rng(self.seed)
        for i, (dataset_index, shard_index) in enumerate(self.shards_sample_schedule):
            self.curr_shard_index += 1
            assert (
                i == self.curr_shard_index
            ), f"Shard index mismatch: {i} != {self.curr_shard_index}"
            dataset = self.datasets[dataset_index]
            wait_start = time.time()
            dataset.finish_cache_shard()
            wait_end = time.time()
            print(
                f"Rank {self.rank}, Worker {self.worker_id}: Wait for shard {shard_index} in dataset {dataset_index} in {wait_end - wait_start:.2f} seconds"
            )
            # Start caching the next shard immediately
            self.cache_next_shard()
            all_steps: list[tuple[int, int]] = []
            for trajectory_id in dataset.get_trajectories_in_shard():
                trajectory_index = dataset.get_trajectory_index(trajectory_id)
                if self.allow_padding_at_end:
                    allowed_length = dataset.trajectory_lengths[trajectory_index]
                else:
                    max_delta_index = dataset.max_delta_index
                    trajectory_length = dataset.trajectory_lengths[trajectory_index]
                    allowed_length = trajectory_length - max_delta_index
                # Get the allowed indices from the step filter
                allowed_indices = dataset.step_filter[trajectory_id]
                # Remove indices that are too large
                allowed_indices = allowed_indices[allowed_indices <= allowed_length]
                for i in allowed_indices:
                    all_steps.append((trajectory_id, i))
            if self.training:
                rng.shuffle(all_steps)
            sampled_steps = all_steps[
                : int(dataset.num_steps_per_shard * self.shard_sampling_rate)
            ]
            for trajectory_id, step_index in sampled_steps:
                # print(
                #     f"Loading step data from rank {self.rank}, worker {self.worker_id}: {dataset_index} {trajectory_id}, {step_index}"
                # )
                indices = {
                    key: delta_indices + step_index
                    for key, delta_indices in dataset.delta_indices.items()
                }
                step_data = dataset.get_step_data(trajectory_id, indices)
                # Skip samples where state or action would be empty
                if step_data is not None:
                    yield dataset.transforms(step_data)  # type: ignore

            # Delete the cached shard and shard start indices to free up memory
            dataset.delete_cached_shard()

    def cache_next_shard(self):
        """Cache the next shard in a background thread."""
        next_dataset_idx, next_shard_idx = self.shards_sample_schedule[
            self.curr_shard_index + 1
        ]
        self.datasets[next_dataset_idx].start_cache_shard(next_shard_idx)

    def __getitem__(self, index: int) -> dict:
        raise NotImplementedError(
            "__getitem__ is not supported for CachedRobotMixtureDataset. Please use __iter__ instead."
        )

    def __len__(self) -> int:
        """The length of the dataset."""
        total_length = 0
        for dataset_idx, _ in self.shards_sample_schedule:
            dataset = self.datasets[dataset_idx]
            total_length += int(dataset.num_steps_per_shard * self.shard_sampling_rate)
        return total_length


if __name__ == "__main__":
    all_transforms ={"oxe_droid": build_transform_pipeline(
        video_keys=video_keys, action_keys=action_keys, state_keys=state_keys
    )}
    

    train_dataset = ShardedLeRobotMixtureDataset.from_mixture_spec(
        mixture_spec=mixture_spec,
        dataset_class=ShardedLeRobotSubLangSingleActionChunkDatasetDROID,
        all_modality_configs=all_modality_configs,
        all_transforms=all_transforms,
        metadata_versions=metadata_versions,
        fps=fps,
        mixture_kwargs=mixture_kwargs,
        dataset_kwargs=dataset_kwargs,
    )
