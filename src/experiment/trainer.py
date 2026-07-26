import transformers
from transformers import BatchFeature  # type: ignore
from transformers.trainer import (
    TRAINER_STATE_NAME,
    TrainerState,
    get_last_checkpoint,
    get_parameter_names,
    is_sagemaker_mp_enabled,
)
import contextlib
import os
from pathlib import Path
import time
from typing import Optional
from src.data.lerobot import (
    ShardedLeRobotSubLangSingleActionChunkDatasetDROID,
    build_transform_pipeline,
)
from src.data.schema import EmbodimentTag
from src.data.zulu_transform import DefaultDataCollator
import torch
import torch.nn as nn
import torch.distributed as dist
import torch._dynamo.config
from torch.utils.data import DataLoader, Dataset, Sampler
from torch.profiler import ProfilerActivity, profile

from src.data.lerobot_mixture import ShardedLeRobotMixtureDataset
from src.configs.train import (
    trainer_conf,
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

LAYERNORM_LAYERS = [
    torch.nn.LayerNorm,
    torch.nn.GroupNorm,
    torch.nn.InstanceNorm1d,
    torch.nn.InstanceNorm2d,
    torch.nn.InstanceNorm3d,
    torch.nn.LocalResponseNorm,
    torch.nn.BatchNorm1d,
    torch.nn.BatchNorm2d,
    torch.nn.BatchNorm3d,
    torch.nn.SyncBatchNorm,
]


class ForceRestart(ValueError):
    pass


class BaseSampler(Sampler):
    """Sampler for dataset, which enables `set_epoch` for Dataset.
    `set_epoch` will be called by huggingface Trainer at the end of each epoch.
    `shuffle` is also supported for training set shuffling
    """

    def __init__(self, data_source: Dataset, shuffle: bool = False, seed: int = 0):
        self.data_source = data_source
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

    def __iter__(self):
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            # must not add rank here, or randomization will be different for each rank
            return iter(torch.randperm(len(self.data_source), generator=g).tolist())
        return iter(range(len(self.data_source)))

    def set_epoch(self, epoch):
        self.epoch = epoch
        if hasattr(self.data_source, "set_epoch"):
            # this is important for dataset
            self.data_source.set_epoch(epoch)

    def __len__(self):
        return len(self.data_source)


class ContextTimer:

    def __init__(self, trainer):
        self.last_key = None
        self.trainer = trainer
        self.start_times = {}
        self.key_stack = []

    def with_label(self, key):
        self.last_key = key
        return self

    def __enter__(self):
        self.key_stack.append(self.last_key)  # Push key to stack
        self.start_times[self.last_key] = time.time()  # Start timing for this key
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        key = self.key_stack.pop()  # Pop key from stack
        diff = time.time() - self.start_times[key]
        self.trainer.log({f"{key}_time": diff})
        # print(f"{key}: {diff:.2f} seconds")


class BaseTrainer(transformers.Trainer):  # type: ignore

    def __init__(self, **kwargs):
        # Increase the cache size limit for torch._dynamo to
        # accommodate videos with different numbers of frames.
        torch._dynamo.config.cache_size_limit = 1000

        self.compute_dtype = kwargs.pop("compute_dtype")
        self.output_dir = kwargs.pop("output_dir")
        self.timer = ContextTimer(self)

        self.world_size = int(os.environ.get("WORLD_SIZE", "1"))
        self.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        self.global_rank = int(os.environ.get("RANK", "0"))
        self.node_rank = int(os.environ.get("NODE_RANK", "0"))

        # Get distributed info
        self.current_step = 0

        # Profiling (legacy per-step profiling)
        self.enable_profiling = kwargs.pop("enable_profiling", False)
        self.profiling_steps = kwargs.pop("profiling_steps", 5)
        # Pop new ProfCallback config options (handled in create_trainer, not here)
        kwargs.pop("enable_prof_callback", None)
        kwargs.pop("profile_start_step", None)
        kwargs.pop("profile_warmup_steps", None)
        kwargs.pop("profile_active_steps", None)
        kwargs.pop("profile_record_shapes", None)
        kwargs.pop("profile_with_stack", None)
        kwargs.pop("profile_memory", None)
        kwargs.pop("msc_profile_url", None)
        kwargs.pop("profile_delete_after_upload", None)
        if self.enable_profiling:
            # Setup profiling directories
            self.profile_dir = Path(self.output_dir) / "profiling"
            self.memory_profile_dir = self.profile_dir / "memory"
            self.torch_profile_dir = self.profile_dir / "torch"

            self.memory_profile_dir.mkdir(exist_ok=True, parents=True)
            self.torch_profile_dir.mkdir(exist_ok=True, parents=True)

            # Start recording the memory history.
            torch.cuda.memory._record_memory_history(max_entries=100000)

        super().__init__(**kwargs)

        self.loss_queues = {}
        self.loss_queue_size = 10

    def _get_train_sampler(self):
        return BaseSampler(self.train_dataset, shuffle=True, seed=self.args.seed)

    def _get_eval_sampler(self, eval_dataset):
        return BaseSampler(eval_dataset, shuffle=False)

    def training_step(self, model, inputs, num_items_in_batch=None):
        enable_profile = (
            self.enable_profiling and self.current_step % self.profiling_steps == 0
        )
        if enable_profile:
            profile_context = profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                record_shapes=True,
                with_stack=True,
            )
        else:
            profile_context = contextlib.nullcontext()

        start_time = time.time()

        with self.timer.with_label("training_step"), profile_context as prof:
            output = super().training_step(model, inputs)

        time_taken = time.time() - start_time
        print(
            f"Rank {self.global_rank} time taken for training_step {self.current_step}: {time_taken:.2f} seconds"
        )

        if enable_profile:
            trace_path = f"{self.torch_profile_dir}/trace_rank_{self.global_rank}_step_{self.current_step}.json.gz"
            print(f"Rank {self.global_rank} exporting torch profile to {trace_path}")
            prof.export_chrome_trace(trace_path)  # type: ignore

            snapshot_path = f"{self.memory_profile_dir}/memory_snapshot_rank_{self.global_rank}_step_{self.current_step}.pickle"
            print(f"Rank {self.global_rank} dumping memory snapshot to {snapshot_path}")
            torch.cuda.memory._dump_snapshot(snapshot_path)

        self.current_step += 1
        return output

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        with self.timer.with_label("model_forward"):
            outputs = model(inputs)
        ### For additional losses, track and log their moving averages
        for key, value in outputs.items():
            if key.endswith("_loss") and key != "loss":
                # Initialize queue if not exists
                if key not in self.loss_queues:
                    self.loss_queues[key] = []

                # Add current loss value to queue
                current_value = value.item() if torch.is_tensor(value) else value
                self.loss_queues[key].append(current_value)

                # Keep only last N values
                if len(self.loss_queues[key]) > self.loss_queue_size:
                    self.loss_queues[key].pop(0)

                # Log average every 10 steps
                if self.state.global_step % self.loss_queue_size == 0: # global step 
                    avg_loss = sum(self.loss_queues[key]) / len(self.loss_queues[key])
                    self.log({f"{key}_avg": avg_loss})

        loss = outputs["loss"]

        return (loss, outputs) if return_outputs else loss

    def create_optimizer(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        if is_sagemaker_mp_enabled():
            return super().create_optimizer()

        opt_model = self.model

        if self.optimizer is None:
            decay_parameters = get_parameter_names(opt_model, LAYERNORM_LAYERS)
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p
                        for n, p in opt_model.named_parameters()
                        if (n in decay_parameters and p.requires_grad)
                    ],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [
                        p
                        for n, p in opt_model.named_parameters()
                        if (n not in decay_parameters and p.requires_grad)
                    ],
                    "weight_decay": 0.0,
                },
            ]

            optimizer_cls, optimizer_kwargs = transformers.Trainer.get_optimizer_cls_and_kwargs(  # type: ignore
                self.args
            )
            self.optimizer = optimizer_cls(
                optimizer_grouped_parameters, **optimizer_kwargs
            )

            # DeepSpeed CPU Adam (ZeRO offload) expects 'bias_correction' in each param group.
            # HuggingFace Trainer's AdamW does not set it, causing KeyError in cpu_adam.step().
            if getattr(self.args, "deepspeed", None):
                for group in self.optimizer.param_groups:
                    group.setdefault("bias_correction", True)

        return self.optimizer

    def save_model(
        self, output_dir: Optional[str] = None, _internal_call: bool = False
    ):
        """
        Production-grade, reflection-based serialization.
        Dynamically detects completely frozen backbones by inspecting the computational graph
        at runtime, omitting them while safely preserving all parameters and buffers of active modules.
        """
        if self.args.should_save:
            if output_dir is None:
                output_dir = self.args.output_dir

            os.makedirs(output_dir, exist_ok=True) # type: ignore

            # 1. Dynamically discover prefixes of submodules that are completely frozen
            # This avoids any hardcoded strings and handles any architecture names seamlessly.
            frozen_prefixes = []
            for name, child in self.model.named_children():
                child_params = list(child.parameters())
                # If a module contains parameters and ALL of them have requires_grad=False, it's frozen
                if len(child_params) > 0 and all(
                    not p.requires_grad for p in child_params
                ):
                    frozen_prefixes.append(f"{name}.")

            # 2. Extract state dict safely
            state_dict = self.model.state_dict()

            # 3. Filter using the dynamically discovered prefixes
            cpu_state_dict = {}
            for key, value in state_dict.items():
                if any(key.startswith(prefix) for prefix in frozen_prefixes):
                    continue
                cpu_state_dict[key] = value.cpu()

            del state_dict  # Immediate VRAM cleanup

            # 4. Serialize using traditional torch.save to handle weight-tying seamlessly
            weight_path = os.path.join(output_dir, "pytorch_model.bin") # type: ignore
            torch.save(cpu_state_dict, weight_path)

            if self.global_rank == 0:
                print(
                    f"Checkpoint saved to {weight_path}. "
                    f"Dynamically stripped frozen modules with prefixes: {frozen_prefixes}"
                )
            return True

    def train(
        self,
        resume_from_checkpoint=None,
        trial=None,
        ignore_keys_for_eval=None,
        **kwargs,
    ):
        """Correctly set self.state from checkpoint so get_train_dataloader can read from it."""
        if resume_from_checkpoint is False:
            resume_from_checkpoint = None

        if isinstance(resume_from_checkpoint, bool) and resume_from_checkpoint:
            resume_from_checkpoint = get_last_checkpoint(self.args.output_dir)
            if resume_from_checkpoint is None:
                raise ValueError(
                    f"No valid checkpoint found in output directory ({self.args.output_dir})"
                )

        if resume_from_checkpoint is not None:
            # In case of repeating the find_executable_batch_size, set `self._train_batch_size` properly
            if isinstance(resume_from_checkpoint, str):
                self.state = TrainerState.load_from_json(
                    os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)
                )
                self._cached_global_step = self.state.global_step
        return super().train(
            resume_from_checkpoint, trial, ignore_keys_for_eval, **kwargs
        )

    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training [`~torch.utils.data.DataLoader`].

        Will use no sampler if `train_dataset` does not implement `__len__`, a random sampler (adapted to distributed
        training if necessary) otherwise.

        Subclass and override this method if you want to inject some custom behavior.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        if not isinstance(train_dataset, (ShardedLeRobotMixtureDataset)):
            return super().get_train_dataloader()

        # During resume, don't skip the data
        self.args.ignore_data_skip = True
        curr_global_step = getattr(self, "_cached_global_step", self.state.global_step)
        print(f"Current global step: {curr_global_step}")
        if curr_global_step > 0:
            new_seed = train_dataset.seed + curr_global_step
            train_dataset.reset_seed(new_seed)
            print(
                f"Resetting seed to {new_seed}. Please note that this will make the experiment non-reproducible."
            )

        print("Creating custom train dataloader")
        # Handle the case where the dataset is an IterableDataset
        data_collator = self.data_collator
        # data_collator = self._get_collator_with_removed_columns(
        #    data_collator, description="training"
        # )

        dataloader_params = {
            "batch_size": self._train_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
        }
        # persistent_workers is only valid when num_workers > 0 (PyTorch raises otherwise)
        if self.args.dataloader_num_workers > 0:
            dataloader_params["persistent_workers"] = (
                self.args.dataloader_persistent_workers
            )

        return DataLoader(train_dataset, **dataloader_params)


class WAMTrainer(BaseTrainer):
    def __init__(self, **kwargs):
        self.benchmark_time = kwargs.pop("benchmark_time", False)
        self.step_timer = None
        self.num_trials = kwargs.pop("num_trials", 10)
        self.curr_trial = 0
        self.all_times = []
        self.start_time = time.time()
        self.restart_max_seconds = kwargs.pop("restart_max_seconds", 0)
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            self.rank = dist.get_rank()
        else:
            self.rank = 0  # Default to main process if not running distributed

        self.micro_global_step = 0

        super().__init__(**kwargs)

    def training_step(self, model, inputs, *args, **kwargs):
        self.micro_global_step += 1

        # if hasattr(self.model.action_head, "global_step"):
        #    self.model.action_head.global_step = self.state.global_step

        if self.benchmark_time:
            if self.state.global_step % 100 == 0:
                if self.step_timer is not None:
                    elapsed_time = time.time() - self.step_timer
                    self.all_times.append(elapsed_time)
                    self.curr_trial += 1
                self.step_timer = time.time()
            if self.curr_trial >= self.num_trials:
                exit(0)
        if self.state.global_step % self.state.save_steps == 1:
            if self.restart_max_seconds > 0:
                cur_time = time.time()
                if (cur_time - self.start_time) > self.restart_max_seconds:
                    raise ForceRestart(
                        f"Exceeded time limit {self.restart_max_seconds} seconds"
                    )
        loss_dict = super().training_step(model, inputs, *args, **kwargs)
        return loss_dict
