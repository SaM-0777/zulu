from abc import ABC
import functools
import json
import logging
import os
from pathlib import Path
import pathlib
import re
import shutil
from typing import Dict, Any, Union
import warnings
import torch
import transformers
import transformers.training_args
from transformers import set_seed  # type: ignore
from transformers.trainer import Trainer
from transformers.trainer_callback import TrainerCallback

from src.configs.train import (
    experiment_conf,
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

from src.data.lerobot import (
    ShardedLeRobotSubLangSingleActionChunkDatasetDROID,
    build_transform_pipeline,
)
from src.data.lerobot_mixture import ShardedLeRobotMixtureDataset
from src.data.schema import EmbodimentTag
from src.data.zulu_transform import DefaultDataCollator
from src.experiment.trainer import WAMTrainer

from src.policies.model import Model


def get_checkpoint_path(output_dir: str, checkpoint_prefix: str = "checkpoint"):
    output_dir = os.path.abspath(output_dir)
    pathlib_dir = pathlib.Path(output_dir)

    if list(pathlib_dir.glob("config.json")):
        # training has been finished
        return output_dir, False
    else:
        try:
            ordering_and_checkpoint_path = []
            glob_checkpoints = [
                str(x)
                for x in pathlib.Path(output_dir).glob(f"{checkpoint_prefix}-*")
                if os.path.isdir(x)
            ]
            for path in glob_checkpoints:
                regex_match = re.match(f".*{checkpoint_prefix}-([0-9]+)", path)
                if regex_match is not None and regex_match.groups() is not None:
                    ordering_and_checkpoint_path.append(
                        (int(regex_match.groups()[0]), path)
                    )
            checkpoints_sorted = sorted(ordering_and_checkpoint_path)
            return checkpoints_sorted[-1][1], True
        except IndexError:
            return None, True


def safe_save_model_for_hf_trainer(trainer: Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    # if trainer.deepspeed:
    #    torch.cuda.synchronize()
    #    trainer.save_model(output_dir, _internal_call=True)
    #    return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


class LossLoggerCallback(TrainerCallback):
    """Callback that writes per-step loss metrics to a JSONL file for offline analysis."""

    def __init__(self, output_path: str):
        self.output_path = output_path

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not state.is_world_process_zero or logs is None:
            return
        entry = {"step": state.global_step}
        for key in ("loss", "dynamics_loss_avg", "action_loss_avg", "learning_rate"):
            if key in logs:
                entry[key] = logs[key]
        if len(entry) > 1:  # more than just "step"
            with open(self.output_path, "a") as f:
                f.write(json.dumps(entry) + "\n")


class CheckpointFormatCallback(TrainerCallback):
    """This callback format checkpoint to make them standalone. For now, it copies all config
    files to /checkpoint-{step}/experiment_cfg/:
    - conf.yaml
    - initial_actions.npz
    - metadata.json
    """

    def __init__(
        self,
        run_name: str,
        exp_cfg_dir: Path | None = None,
        processor_dir: Path | None = None,
    ):
        """
        Args:
            run_name: Name of the experiment run
            exp_cfg_dir: Path to the directory containing all experiment metadata
        """
        self.exp_cfg_dir = exp_cfg_dir
        self.processor_dir = processor_dir

    def on_save(self, args, state, control, **kwargs):
        """Called after the trainer saves a checkpoint."""
        if state.is_world_process_zero:
            checkpoint_dir = Path(args.output_dir) / f"checkpoint-{state.global_step}"

            # Copy experiment config directory if provided
            if self.exp_cfg_dir is not None:
                exp_cfg_dst = checkpoint_dir / self.exp_cfg_dir.name
                if self.exp_cfg_dir.exists():
                    print(
                        f"Copying experiment config directory {self.exp_cfg_dir} to {exp_cfg_dst}"
                    )
                    shutil.copytree(self.exp_cfg_dir, exp_cfg_dst, dirs_exist_ok=True)

            # Copy processor directory if provided
            if self.processor_dir is not None:
                if self.processor_dir.exists():
                    print(
                        f"Copying processor directory {self.processor_dir} to {checkpoint_dir}"
                    )
                    shutil.copytree(
                        self.processor_dir, checkpoint_dir, dirs_exist_ok=True
                    )

            # Copy wandb_config.json if provided
            wandb_config_src = Path(args.output_dir) / "wandb_config.json"
            wandb_config_dst = checkpoint_dir / "wandb_config.json"
            if wandb_config_src.exists():
                print(
                    f"Copying wandb_config.json from {wandb_config_src} to {wandb_config_dst}"
                )
                shutil.copy2(wandb_config_src, wandb_config_dst)


class ProfCallback(TrainerCallback):
    """Callback to manage PyTorch profiler during training.

    Dynamically starts/stops the profiler within a specified session step window.
    After profiling completes, triggers optional S3 upload and removes itself.

    Args:
        profile_dir: Directory to save profile traces
        upload_callback: Optional callback to trigger S3 upload after profiling
        profile_start_step: Session step to start profiling (default: 50)
        profile_end_step: Session step to stop profiling
        warmup_steps: Number of warmup steps for profiler schedule (default: 1)
        active_steps: Number of active profiling steps (default: 5)
        trainer: Trainer instance (required for self-removal after profiling)
        record_shapes: Record tensor shapes in profiler (default: False)
        with_stack: Record Python stack traces (default: True)
        profile_memory: Record memory allocation/deallocation (default: False)
    """

    def __init__(
        self,
        profile_dir,
        upload_callback=None,
        profile_start_step=50,
        profile_end_step=55,
        warmup_steps=1,
        active_steps=5,
        trainer=None,
        record_shapes=False,
        with_stack=True,
        profile_memory=False,
    ):
        self.profile_dir = profile_dir
        self.upload_callback = upload_callback
        self.profile_start_step = profile_start_step
        self.profile_end_step = profile_end_step
        self.warmup_steps = warmup_steps
        self.active_steps = active_steps
        self.trainer = trainer
        self.record_shapes = record_shapes
        self.with_stack = with_stack
        self.profile_memory = profile_memory
        self.upload_triggered = False
        self.starting_global_step = None
        self.session_step = 0
        self.prof = None
        self.profiling_active = False
        self.profiling_complete = False
        self.removed_from_trainer = False

    def on_step_begin(self, args, state, control, **kwargs):
        # Remove callback after upload triggered to eliminate all overhead
        if (
            self.profiling_complete
            and self.upload_triggered
            and not self.removed_from_trainer
        ):
            if self.trainer is not None and hasattr(self.trainer, "callback_handler"):
                try:
                    self.trainer.callback_handler.callbacks.remove(self)
                    self.removed_from_trainer = True
                    logging.info(
                        f"Removed ProfCallback from trainer at global step {state.global_step}"
                    )
                except (ValueError, AttributeError) as e:
                    logging.warning(f"Failed to remove ProfCallback: {e}")
            return

        # Early return if profiling already complete
        if self.profiling_complete:
            return

        # Record starting global step on first call
        if self.starting_global_step is None:
            self.starting_global_step = state.global_step

        # Calculate session step
        self.session_step = state.global_step - self.starting_global_step

        # Start profiler when we reach the profiling window
        if self.session_step == self.profile_start_step and self.prof is None:
            logging.info(
                f"Starting profiler at global step {state.global_step} (session step {self.session_step})"
            )
            self.prof = torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                schedule=torch.profiler.schedule(
                    skip_first=0,
                    wait=0,
                    warmup=self.warmup_steps,
                    active=self.active_steps,
                    repeat=1,
                ),
                profile_memory=self.profile_memory,
                with_stack=self.with_stack,
                record_shapes=self.record_shapes,
                on_trace_ready=torch.profiler.tensorboard_trace_handler(
                    str(self.profile_dir)
                ),
            )
            self.prof.__enter__()
            self.profiling_active = True

    def on_step_end(self, args, state, control, **kwargs):
        # Early return if profiling already complete
        if self.profiling_complete:
            return

        # Recalculate session_step to ensure accuracy
        if self.starting_global_step is not None:
            self.session_step = state.global_step - self.starting_global_step

        # Step profiler if active
        if self.profiling_active and self.prof is not None:
            self.prof.step()

        # Stop profiler when we reach the end of profiling window
        if self.session_step == self.profile_end_step and self.prof is not None:
            self.prof.__exit__(None, None, None)
            self.profiling_active = False

            # Explicitly release profiler resources to minimize CUPTI overhead
            # Combined with TEARDOWN_CUPTI=1 env var for full cleanup
            del self.prof
            self.prof = None

            # Force CUDA synchronization to ensure profiler cleanup completes
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            self.profiling_complete = True
            logging.info(
                f"Profiler stopped and resources released at global step {state.global_step} "
                f"(session step {self.session_step})"
            )

            # Trigger upload if callback provided
            if self.upload_callback:
                logging.info(f"Triggering upload at global step {state.global_step}...")
                self.upload_callback()

            # Mark as ready for callback removal
            self.upload_triggered = True


class BaseExperiment(ABC):
    def __init__(self, output_abs_dir: str, cfg: Any):
        self.output_abs_dir = output_abs_dir
        output_dir = output_abs_dir
        max_steps = cfg.get("max_steps")
        save_total_limit = cfg.get("save_total_limit")
        transforms = cfg.get("transforms")
        video_keys = cfg.get("video_keys")
        state_keys = cfg.get("state_keys")
        action_keys = cfg.get("action_keys")
        
        print(f"Config loaded ")

        assert max_steps > 0, "max_steps must be > 0 for standarized evaluation"
        assert (
            save_total_limit >= 5
        ), "save_total_limit must be >= 5 for standarized evaluation"

        assert transforms is not None, "Evaluation transforms are not provided."
        for tag, transform_cfg in transforms.items():
            try:

                _ = EmbodimentTag(tag)
                transform = {
                    "oxe_droid": build_transform_pipeline(
                        video_keys=video_keys,
                        action_keys=action_keys,
                        state_keys=state_keys,
                    )
                }
            except Exception as e:
                raise ValueError(f"Evaluation transform {tag} is invalid")

        cfg["training_args"]["output_dir"] = output_dir.rstrip("/")
        cfg["training_args"]["run_name"] = "wam_train_001_20260629"
        print(f"Run name {cfg["training_args"]["run_name"]}")

        training_args = transformers.training_args.TrainingArguments(
            **cfg["training_args"]
        )
        set_seed(training_args.seed)

        if "WANDB_PROJECT" not in os.environ:
            os.environ["WANDB_PROJECT"] = "zulu"
        if "WANDB_RUN_ID" not in os.environ:
            runtime_id = os.environ.get("RUNTIME_ID", None)
            if runtime_id:
                os.environ["WANDB_RUN_ID"] = runtime_id
        os.environ["WANDB_DIR"] = output_dir

        output_dir = Path(output_dir)
        exp_cfg_dir = output_dir / "experiment_cfg"
        exp_cfg_dir.mkdir(parents=True, exist_ok=True)

        wandb_config_file = output_dir / "wandb_config.json"
        with open(wandb_config_file, "w") as f:
            json.dump(
                {
                    "project": os.environ.get("WANDB_PROJECT", ""),
                    "run_id": os.environ.get("WANDB_RUN_ID", ""),
                },
                f,
            )

        resume_path, continue_training = get_checkpoint_path(output_dir=str(output_dir))
        if not continue_training:
            print(f"Models is ready under {training_args.output_dir}. Skip training.")
            exit(0)
        if resume_path:
            print(f"Resuming training from {resume_path}")
            resume_from_checkpoint = True
        else:
            # First time training.
            resume_from_checkpoint = False

        self.model = self.create_model(cfg=cfg)
        print(f"model created successfully")
        # if hasattr(model.action_head, "max_steps"):
        #    model.action_head.max_steps = max_steps

        compute_dtype = self.model.dtype

        train_dataset = self.create_train_dataset(cfg=cfg)
        print("Using dataset ", train_dataset)
        assert (
            train_dataset.merge_metadata is not None
        ), "set config.merge=true in order to save the metadata"

        metadata_save_path = exp_cfg_dir / "metadata.json"
        with open(metadata_save_path, "w") as f:
            json.dump(
                {
                    k: v.model_dump(mode="json")
                    for k, v in train_dataset.merged_metadata.items()
                },
                f,
                indent=4,
            )
        print("Successfully dumped metadata")

        val_dataset = self.create_val_dataset(cfg=cfg, model=self.model)
        data_collator = self.create_data_collator(cfg=cfg)
        trainer = self.create_trainer(
            cfg=cfg,
            exp_cfg_dir=exp_cfg_dir,
            model=self.model,
            training_args=training_args,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            data_collator=data_collator,
            compute_dtype=compute_dtype,
        )

        self.cfg = {**cfg}
        self.exp_cfg_dir = exp_cfg_dir
        self.training_args = training_args
        self.resume_from_checkpoint = resume_from_checkpoint
        self.train_dataset = train_dataset
        self.trainer = trainer

        #def gradient_checkpointing_enable(self, *args, **kwargs):
        #    self.model.dit_backbone.gradient_checkpointing = True

        #def gradient_checkpointing_disable(self, *args, **kwargs):
        #    self.model.dit_backbone.gradient_checkpointing = False

    def create_train_dataset(self, cfg: Dict):
        if not torch.distributed.is_initialized():
            torch.distributed.is_initialized()
        all_transforms = {
            "oxe_droid": build_transform_pipeline(
                video_keys=video_keys, action_keys=action_keys, state_keys=state_keys
            )
        }
        dataset_path = os.path.join(
            self.output_abs_dir,
            "data",
            #"dreamzero_droid_first3" # pretraining
            "dreamzero_droid_pickplace" # finetuning
        )
        print(f"Dataset path {dataset_path}")
        mixture_spec[0]["dataset_path"] = {"oxe_droid": [dataset_path]} # manual
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

        return train_dataset

    def create_val_dataset(self, cfg: Dict, model: Any):
        return None

    def create_data_collator(self, cfg: Dict):
        data_collator_config = cfg["data_collator"]
        max_length = data_collator_config["max_length"]
        num_views = data_collator_config["num_views"]
        embodiment_tag_mapping = data_collator_config["embodiment_tag_mapping"]
        collator = DefaultDataCollator(
            max_length=max_length,
            num_views=num_views,
            embodiment_tag_mapping=embodiment_tag_mapping,
        )
        return collator

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

        if model_dtype == "bfloat16":
            dtype = torch.bfloat16
        elif model_dtype == "float16":
            dtype = torch.float16

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
        
        # works on single gpu
        model.dit_backbone = torch.compile(model.dit_backbone, mode="max-autotune") # "default", "reduce-overhead", "max-autotune"
        
        # only compile the blocks of dit
        #for i, block in enumerate(model.dit_backbone.blocks):
        #    model.dit_backbone.blocks[i] = torch.compile(
        #        block,
        #        mode="default",
        #        dynamic=True,
        #        fullgraph=False,
        #    )

        def gradient_checkpointing_enable(self, *args, **kwargs):
            # 'self' here refers to the 'model' instance
            self.dit_backbone.gradient_checkpointing = True

        def gradient_checkpointing_disable(self, *args, **kwargs):
            self.dit_backbone.gradient_checkpointing = False

        model.gradient_checkpointing_enable = gradient_checkpointing_enable.__get__(
            model, model.__class__
        )
        model.gradient_checkpointing_disable = gradient_checkpointing_disable.__get__(
            model, model.__class__
        )

        model.dit_backbone.gradient_checkpointing = cfg["action_head_cfg"].get(
            "use_gradient_checkpointing", False # was True
        )

        return model

    def create_trainer(
        self,
        cfg: Dict,
        exp_cfg_dir: Path,
        model: Model,
        training_args: transformers.training_args.TrainingArguments,
        data_collator: DefaultDataCollator,
        compute_dtype: torch.dtype,
        train_dataset: Any,
        val_dataset=None,
    ):
        if cfg["global_batch_size"] is not None:
            global_bs = cfg["global_batch_size"]
            bs = training_args.per_device_train_batch_size
            grad_acc = self._compute_grad_accum_to_match_global_bs(global_bs, bs)
            training_args.gradient_accumulation_steps = grad_acc
            print(
                f"Set global batch size to {global_bs}, set gradient accumulation steps to {grad_acc}"
            )
        elif cfg["raise_error_if_global_batch_size_not_set"]:
            raise ValueError(
                "global_batch_size is not set. To ensure the scripts can be reproduced regardless of the number of nodes used, please set this."
            )
        else:
            warnings.warn(
                "global_batch_size is not set. This is fine for debugging, but please set this for real experiments."
            )

        trainer_conf = cfg["trainer_conf"]
        tconf = trainer_conf.copy()
        tconf.pop("output_dir", None)
        tconf.pop("model", None)
        tconf.pop("train_dataset", None)
        tconf.pop("eval_dataset", None)
        tconf.pop("compute_dtype", None)

        output_dir = os.path.join(self.output_abs_dir, "checkpoint")
        trainer_partial = functools.partial(
            WAMTrainer,
            **tconf,
            output_dir=output_dir,
            model=model,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_dtype=compute_dtype,
        )

        trainer = trainer_partial(data_collator=data_collator, args=training_args)
        trainer.base_cfg = {**cfg}
        train_dl_len = len(trainer.get_train_dataloader())
        eval_dl_len = (
            len(trainer.get_eval_dataloader())
            if val_dataset is not None
            else "no eval dataloader"
        )

        run_name = cfg["training_args"].get("run_name", "zulu_run_001_20260629")
        ckpt_format_callback = CheckpointFormatCallback(
            run_name=run_name, exp_cfg_dir=exp_cfg_dir
        )
        trainer.add_callback(ckpt_format_callback)

        loss_log_path = Path(output_dir)
        loss_log_path.mkdir(parents=True, exist_ok=True)
        loss_log_path_str = str(loss_log_path / "loss_log.jsonl")
        trainer.add_callback(LossLoggerCallback(output_path=loss_log_path_str))

        if cfg["trainer_conf"].get("enable_prof_callback", False):
            global_rank = int(os.environ.get("RANK", "0"))

            trainer_cfg = cfg["trainer_conf"]

            profile_start_step = trainer_conf.get("profile_start_step", 50)
            profile_warmup_steps = trainer_conf.get("profile_warmup_steps", 1)
            profile_active_steps = trainer_conf.get("profile_active_steps", 5)
            profile_record_shapes = trainer_conf.get("profile_record_shapes", False)
            profile_with_stack = trainer_conf.get(
                "profile_with_stack", False
            )  # Default False to match omni (stack traces add significant file size)
            profile_memory = trainer_cfg.get("profile_memory", False)

            profile_end_step = (
                profile_start_step + profile_warmup_steps + profile_active_steps - 1
            )

            profile_dir = Path(output_dir) / "profiling" / f"rank_{global_rank}"
            profile_dir.mkdir(parents=True, exist_ok=True)

            print(
                f"Profiling enabled: steps {profile_start_step}-{profile_end_step}, "
                f"saving to {profile_dir}"
            )

            trainer.add_callback(
                ProfCallback(
                    profile_dir=profile_dir,
                    upload_callback=None,
                    profile_start_step=profile_start_step,
                    profile_end_step=profile_end_step,
                    warmup_steps=profile_warmup_steps,
                    active_steps=profile_active_steps,
                    trainer=trainer,
                    record_shapes=profile_record_shapes,
                    with_stack=profile_with_stack,
                    profile_memory=profile_memory,
                )
            )

        print(
            f"train dataloader length: {train_dl_len}\n"
            f"eval dataloader length: {eval_dl_len}\n"
            f"train dataset length: {len(trainer.train_dataset)}\n"
            f"GPU memory before training: {torch.cuda.memory_allocated() / 1024 / 1024 / 1024} GB",
            flush=True,
        )
        return trainer

    def train(self):
        # Start training
        output_dir = Path(self.output_abs_dir) / "checkpoint"
        self.trainer.train(resume_from_checkpoint=self.resume_from_checkpoint)
        self.trainer.save_state()
        safe_save_model_for_hf_trainer(trainer=self.trainer, output_dir=str(output_dir))

    def _compute_grad_accum_to_match_global_bs(self, global_bs: int, bs: int):
        if torch.distributed.is_initialized():
            num_devices = torch.distributed.get_world_size()
        else:
            num_devices = int(os.environ.get("WORLD_SIZE", 2))
        per_step_bs = bs * num_devices
        assert global_bs % per_step_bs == 0, f"{global_bs=}, {per_step_bs=}"
        num_grad_accum = global_bs // per_step_bs
        return num_grad_accum


class WAMExperiment(BaseExperiment):
    def __init__(self, cfg: Any):
        # output_dir = os.path.join(os.getcwd(), "zulu_checkpoint")
        output_dir = os.getcwd()
        print(output_dir)
        super().__init__(output_abs_dir=output_dir, cfg=cfg)


def main():
    from accelerate import Accelerator

    accelerator = Accelerator()

    experiment = WAMExperiment(experiment_conf)
    experiment._compute_grad_accum_to_match_global_bs(32, 1)
    experiment.train()
    #pass


if __name__ == "__main__":
    main()
