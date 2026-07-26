from src.data.lerobot import ModalityConfig

mixture_spec = [
    {
        "dataset_path": {
            "oxe_droid": []
        },
        "dataset_weight": 1.0,
        "distribute_weights": False,
    }
]

mixture_kwargs = {
    "training": True,
    "balance_dataset_weights": False,
    "seed": 42,
    "shard_sampling_rate": 0.1, #0.1
}

all_modality_configs = {
    "oxe_droid": {
        "video": ModalityConfig(
            delta_indices=[
                0,
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                10,
                11,
                12,
                13,
                14,
                15,
                16,
                17,
                18,
                19,
                20,
                21,
                22,
                23,
                24,
            ],
            eval_delta_indices=[0],
            modality_keys=[
                "video.exterior_image_1_left",
                "video.exterior_image_2_left",
                "video.wrist_image_left",
            ],
        ),
        "state": ModalityConfig(
            delta_indices=[0],
            modality_keys=["state.joint_position", "state.gripper_position"],
        ),
        "action": ModalityConfig(
            delta_indices=[
                0,
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                10,
                11,
                12,
                13,
                14,
                15,
                16,
                17,
                18,
                19,
                20,
                21,
                22,
                23,
            ],
            modality_keys=["action.joint_position", "action.gripper_position"],
        ),
        "language": ModalityConfig(
            delta_indices=[0],
            modality_keys=[
                "annotation.language.language_instruction",
                "annotation.language.language_instruction_2",
                "annotation.language.language_instruction_3",
            ],
        ),
    },
}

all_transforms = {
    "oxe_droid": {
        "transforms": [
            {
                "apply_to": [
                    "video.exterior_image_1_left",
                    "video.exterior_image_2_left",
                    "video.wrist_image_left",
                ],
            },
            {
                "apply_to": [
                    "video.exterior_image_1_left",
                    "video.exterior_image_2_left",
                    "video.wrist_image_left",
                ],
                "scale": 0.95,
                "mode": "random",
            },
            {
                "apply_to": [
                    "video.exterior_image_1_left",
                    "video.exterior_image_2_left",
                    "video.wrist_image_left",
                ],
                "height": 160,
                "width": 320,
                "interpolation": "linear",
            },
            {
                "apply_to": [
                    "video.exterior_image_1_left",
                    "video.exterior_image_2_left",
                    "video.wrist_image_left",
                ],
                "brightness": 0.1, # 0.3
                "contrast": 0.1, # 0.4
                "saturation": 0.1, # 0.5
                "hue": 0.02, # 0.08
            },
            {
                "apply_to": [
                    "video.exterior_image_1_left",
                    "video.exterior_image_2_left",
                    "video.wrist_image_left",
                ],
            },
            {
                "apply_to": ["state.joint_position", "state.gripper_position"],
            },
            {
                "apply_to": ["state.joint_position", "state.gripper_position"],
                "normalization_modes": {
                    "state.joint_position": "q99",
                    "state.gripper_position": "q99",
                },
            },
            {
                "apply_to": ["action.joint_position", "action.gripper_position"],
            },
            {
                "apply_to": ["action.joint_position", "action.gripper_position"],
                "normalization_modes": {
                    "action.joint_position": "q99",
                    "action.gripper_position": "q99",
                },
            },
            {
                "video_concat_order": [
                    "video.exterior_image_1_left",
                    "video.exterior_image_2_left",
                    "video.wrist_image_left",
                ],
                "state_concat_order": [
                    "state.joint_position",
                    "state.gripper_position",
                ],
                "action_concat_order": [
                    "action.joint_position",
                    "action.gripper_position",
                ],
            },
            {
                "default_instruction": "Perform the default behavior.",
                "language_dropout_prob": 0.0,
                "always_use_default_instruction": False,
                "max_state_dim": 64,
                "max_action_dim": 32,
                "max_length": 200,
                "state_horizon": 1,
                "action_horizon": 24,
                "embodiment_tag_mapping": {
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
            },
        ],
    },
}

metadata_versions = {"oxe_droid": "0221"}

fps = {"yam": 30}

dataset_kwargs = {
    "max_chunk_size": 4,
    "video_backend": "decord",
    "video_backend_kwargs": {
        #"thread_count": 1,  # Force PyAV to use only 1 thread per file
        #"thread_type": "SLICE",  # Prevent PyAV from using FRAME-level threading
    },
    "relative_action": True,
    "relative_action_per_horizon": False,
    "relative_action_keys": ["joint_position"],
}

mixture_kwargs = {
    "training": True,
    "balance_dataset_weights": False,
    "seed": 42,
    "shard_sampling_rate": 0.1,
}

video_keys = [
    "video.exterior_image_1_left",
    "video.exterior_image_2_left",
    "video.wrist_image_left",
]
state_keys = ["state.joint_position", "state.gripper_position"]
action_keys = ["action.joint_position", "action.gripper_position"]


trainer_conf = {
    "output_dir": "./checkpoint",
    #"_partial_": True,
    #"_recursive_": False,
    "callbacks": None,
    "model": "???",
    "train_dataset": "???",
    "compute_dtype": "float32",
    "benchmark_time": False,
    "enable_profiling": False,
    "profiling_steps": 5,
    "enable_prof_callback": False,
    "profile_start_step": 50,
    "profile_warmup_steps": 1,
    "profile_active_steps": 3,
    "profile_record_shapes": False,
    "profile_with_stack": False,
    "profile_memory": False,
}

training_args = {
    "output_dir": "./checkpoint",
    "run_name": "zulu_finetune_pickplace_lora_001", # fine tuning
    "remove_unused_columns": False,
    "gradient_checkpointing": False, # True ; finetuning
    "bf16": True, # must be True
    "fp16": False,
    "tf32": False,
    "per_device_train_batch_size": 1,
    "per_device_eval_batch_size": 64,
    #"gradient_accumulation_steps": 16, # for 2 device
    "gradient_accumulation_steps": 32,
    "dataloader_num_workers": 0, # depends on systme RAM, Must be 8
    "dataloader_pin_memory": False, # dry un only 
    "dataloader_persistent_workers": True,
    "optim": "adamw_torch",
    "adam_beta1": 0.95,
    "adam_beta2": 0.95, # 0.999
    "adam_epsilon": 1e-08,
    "learning_rate": 2e-04, # was 1e-05, 5e-05, 1e-04 (pretraining) ; finetuning
    #"max_grad_norm": None, # 1.0 # finetuning (will see)
    "weight_decay": 0.01, # 1e-05
    "warmup_ratio": 0.05, # was 0.1 ; finetuning
    "lr_scheduler_type": "cosine",
    "logging_steps": 2.0,
    "num_train_epochs": 1000,
    "max_steps": 10000, # was 50000 (pretraining) ; finetuning
    "save_strategy": "steps",
    "save_steps": 500, # was 50 ; finetuning
    "save_total_limit": 10,
    "report_to": "wandb",
    "seed": 42,
    "do_eval": False,
    "ddp_find_unused_parameters": False, # was False ; crash if batch has no action
    "ddp_bucket_cap_mb": 100,
    "torch_compile_mode": None,
}

action_head_cfg = {
    "lora_rank": 8, # was 4 ; finetuning
    "lora_alpha": 16, # was 4; finetuning
    "lora_target_modules": "q,k,v,o,ffn.0,ffn.2", # finetuning
    "init_lora_weights": "kaiming", # finetuning
    "train_architecture": "lora", # finetuning
    "use_gradient_checkpointing": False, # True pretraining

    "num_frames": 33,
    "num_frame_per_block": 8, # was 2
    # "add_pos_embed": True, # not used
    "model_dtype": "bfloat16",
    "max_state_dim": 64,
    "max_action_dim": 32,
    # "action_loss_embodiment_ids": [26, 17, 32],
    "hidden_size": 64,
    "input_embedding_dim": 768,
    # "backbone_embedding_dim": 0,
    # "repa_layer": 8, # not used
    # "repa_coeff": 1.0, # not used
    # "load_pretrained_det_decode_layer_path": None,
    # "freeze_decode_layer": False, # not used
    # "expand_batch": None, # not used
    # "use_vlln": True, # not used
    "self_attention_cfg": {
        "positional_embeddings": None,
        "num_layers": 4,
        "num_attention_heads": 24,
        "attention_head_dim": 64,
        "dropout": 0.2,
        "final_dropout": True,
    },
    "dit_cfg": {
        "model_type": "ti2v",
        "frame_seq_len": 242, # was 50
        "dim": 1024,
        "in_dim": 768,
        "ffn_dim": 4096,
        "out_dim": 768,
        "freq_dim": 256,
        "eps": 1e-06,
        "num_heads": 16,
        "num_layers": 12,
        "max_chunk_size": 4,
        "num_frame_per_block": 8, # was 2
        "num_action_per_block": 24,
        "num_state_per_block": 1,
        "concat_first_frame_latent": False,
    },
    "noise_beta_alpha": 1.5,
    "noise_beta_beta": 1.0,
    "noise_s": 0.999,
    "decouple_video_action_noise": False,
    "video_noise_beta_alpha": 3.0,
    "video_noise_beta_beta": 1.0,
    "target_video_height": 160,
    "target_video_width": 320,
}

data_collator = {
    "max_length": 200,
    "num_views": 3,
    "embodiment_tag_mapping": {
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
}

experiment_conf = {
    "max_steps": 100,
    "save_total_limit": 5,
    "global_batch_size": 32,
    "video_keys": video_keys,
    "state_keys": state_keys,
    "action_keys": action_keys,
    "transforms": all_transforms,
    "trainer_conf": trainer_conf,
    "training_args": training_args,
    "action_head_cfg": action_head_cfg,
    "data_collator": data_collator,
    "relative_action_per_horizon": False,
    "relative_action": True,
    "relative_action_keys": [
        "joint_position"
    ],
    "enable_prof_callback": False
}
