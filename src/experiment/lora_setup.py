from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


def detect_lora_target_modules(
    model: nn.Module,
    strategy: str = "attention_and_ffn",
    verbose: bool = True,
) -> list[str]:
    """
    Dynamically detect which Linear layer names should receive LoRA adapters.

    The function walks the model's named_modules, finds all nn.Linear layers,
    and selects target module names based on the chosen strategy.

    Args:
        model: The model to inspect.
        strategy: Detection strategy. Options:
            - "attention_and_ffn": Target attention projections (q, k, v, o)
              and FFN linear layers. This is the standard LoRA strategy for
              transformers.
            - "all_linear": Target ALL nn.Linear layers in the model (except
              the final output/head layers).
            - "dit_only": Target only Linear layers inside `dit_backbone`.
        verbose: If True, print detected modules and counts.

    Returns:
        A list of module name patterns (suffixes) that PEFT's LoraConfig will
        match against. These are the LAST component(s) of each Linear layer's
        name, deduplicated.

    Example:
        If the model has these Linear layers:
            dit_backbone.blocks.0.self_attn.q  -> nn.Linear
            dit_backbone.blocks.0.self_attn.k  -> nn.Linear
            dit_backbone.blocks.0.ffn.0        -> nn.Linear
            dit_backbone.blocks.0.ffn.2        -> nn.Linear
            dit_backbone.dino_proj             -> nn.Linear

        Then detect_lora_target_modules(model, "attention_and_ffn") returns:
            ["q", "k", "v", "o", "ffn.0", "ffn.2"]
    """
    # Collect all Linear layer names in the model
    linear_names: list[str] = []  # full paths like "dit_backbone.blocks.0.self_attn.q"
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            linear_names.append(name)

    if not linear_names:
        if verbose:
            print("[lora_setup] WARNING: no nn.Linear layers found in model", file=sys.stderr)
        return []

    if verbose:
        print(f"[lora_setup] Found {len(linear_names)} nn.Linear layers in model", file=sys.stderr)

    # Strategy: select which Linear layers to target
    selected_full_names: list[str] = []

    if strategy == "all_linear":
        # Target every Linear layer
        selected_full_names = linear_names

    elif strategy == "dit_only":
        # Only target Linear layers inside dit_backbone
        selected_full_names = [n for n in linear_names if n.startswith("dit_backbone")]

    elif strategy == "attention_and_ffn":
        # Target attention projections (q, k, v, o) and FFN layers (ffn.0, ffn.2)
        # These are the standard LoRA targets for transformer models.
        #
        # We match by checking the LAST 1-2 components of the name.
        # Common patterns in this codebase (from dit.py):
        #   self_attn.q, self_attn.k, self_attn.v, self_attn.o
        #   cross_attn.q, cross_attn.k, cross_attn.v, cross_attn.o
        #   ffn.0, ffn.2  (Sequential: Linear -> GELU -> Linear)
        attention_suffixes = {"q", "k", "v", "o"}
        ffn_patterns = [re.compile(r"ffn\.\d+$")]  # ffn.0, ffn.2, etc.

        for name in linear_names:
            parts = name.split(".")
            last = parts[-1]
            # Check attention projections
            if last in attention_suffixes:
                # Make sure this is actually inside an attention block
                # (not a random Linear named "q" elsewhere)
                parent = ".".join(parts[:-1])
                if "attn" in parent or "attention" in parent:
                    selected_full_names.append(name)
            # Check FFN layers
            elif any(p.search(name) for p in ffn_patterns):
                selected_full_names.append(name)

    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    if not selected_full_names:
        if verbose:
            print(f"[lora_setup] WARNING: strategy '{strategy}' selected 0 modules", file=sys.stderr)
        return []

    # Extract the matchable suffix for PEFT.
    # PEFT's target_modules matches if the module name ENDS WITH any of the
    # provided strings. So we extract the minimal unique suffix.
    #
    # For attention: the suffix is just "q", "k", "v", "o"
    # For FFN: the suffix is "ffn.0", "ffn.2" (we need 2 components to be specific)
    suffixes: set[str] = set()
    for name in selected_full_names:
        parts = name.split(".")
        last = parts[-1]
        if last in ("q", "k", "v", "o"):
            suffixes.add(last)
        elif re.match(r"\d+$", last) and len(parts) >= 2 and parts[-2] == "ffn":
            # ffn.0, ffn.2 — need both components
            suffixes.add(f"ffn.{last}")
        else:
            # For other Linear layers, use the last component
            suffixes.add(last)

    target_modules = sorted(suffixes)

    if verbose:
        print(f"[lora_setup] Strategy: {strategy}", file=sys.stderr)
        print(f"[lora_setup] Selected {len(selected_full_names)} Linear layers", file=sys.stderr)
        print(f"[lora_setup] LoRA target module suffixes: {target_modules}", file=sys.stderr)
        # Print a few example full paths
        print(f"[lora_setup] Example targeted layers:", file=sys.stderr)
        for name in selected_full_names[:5]:
            print(f"  {name}", file=sys.stderr)
        if len(selected_full_names) > 5:
            print(f"  ... and {len(selected_full_names) - 5} more", file=sys.stderr)

    return target_modules


# =============================================================================
# 2. Load pretrained weights into the base model (before LoRA wrapping)
# =============================================================================
def load_pretrained_weights(model: nn.Module, checkpoint_path: str | Path) -> None:
    """
    Load a full pretrained checkpoint into the base model.

    This must be called BEFORE apply_lora(), because PEFT wrapping changes
    the model's parameter names (adds base_model.model. prefix).

    The checkpoint can be:
      - A directory containing pytorch_model.bin or model.safetensors
      - A direct path to a .bin or .safetensors file

    The model's custom load_state_dict (which strips _orig_mod. and module.
    prefixes) is used if available.

    Args:
        model: The base model (NOT yet wrapped with PEFT).
        checkpoint_path: Path to the pretrained checkpoint.
    """
    checkpoint_path = Path(checkpoint_path)
    print(f"\n[lora_setup] Loading pretrained weights from: {checkpoint_path}", file=sys.stderr)

    # Find the actual weights file
    weights_file = None
    if checkpoint_path.is_dir():
        # Try common filenames
        for candidate in ["pytorch_model.bin", "model.safetensors", "pytorch_model.safetensors"]:
            p = checkpoint_path / candidate
            if p.exists():
                weights_file = p
                break
        if weights_file is None:
            # Search recursively
            for p in checkpoint_path.rglob("*.bin"):
                weights_file = p
                break
            if weights_file is None:
                for p in checkpoint_path.rglob("*.safetensors"):
                    weights_file = p
                    break
    else:
        weights_file = checkpoint_path

    if weights_file is None or not weights_file.exists():
        raise FileNotFoundError(
            f"Could not find weights file in: {checkpoint_path}\n"
            f"Looked for: pytorch_model.bin, model.safetensors"
        )

    print(f"[lora_setup] Loading weights file: {weights_file}", file=sys.stderr)

    # Load the state dict
    if str(weights_file).endswith(".safetensors"):
        from safetensors.torch import load_file
        state_dict = load_file(str(weights_file))
    else:
        state_dict = torch.load(str(weights_file), map_location="cpu", weights_only=False)

    print(f"[lora_setup] Checkpoint has {len(state_dict)} keys", file=sys.stderr)

    # Use the model's custom load_state_dict if it handles prefix stripping
    # (the WAM Model class does this — see model.py load_state_dict)
    load_result = model.load_state_dict(state_dict, strict=False)

    missing = len(load_result.missing_keys) if hasattr(load_result, "missing_keys") else 0
    unexpected = len(load_result.unexpected_keys) if hasattr(load_result, "unexpected_keys") else 0
    print(f"[lora_setup] Loaded: {len(state_dict) - unexpected} keys matched", file=sys.stderr)
    print(f"[lora_setup] Missing keys: {missing} (expected for frozen backbone modules)", file=sys.stderr)
    print(f"[lora_setup] Unexpected keys: {unexpected}", file=sys.stderr)

    if unexpected > 0 and unexpected < 50:
        print(f"[lora_setup] Unexpected keys (first 10):", file=sys.stderr)
        for k in load_result.unexpected_keys[:10]:
            print(f"  {k}", file=sys.stderr)


# =============================================================================
# 3. Apply LoRA to the model
# =============================================================================
def apply_lora(
    model: nn.Module,
    lora_rank: int = 32,
    lora_alpha: int = 64,
    target_modules: list[str] | None = None,
    lora_dropout: float = 0.05,
    verbose: bool = True,
) -> nn.Module:
    """
    Freeze the base model and wrap it with PEFT LoRA adapters.

    Args:
        model: The base model (with pretrained weights already loaded).
        lora_rank: LoRA rank (r). Higher = more capacity. Typical: 8-64.
        lora_alpha: LoRA alpha. Standard practice: alpha = 2 * rank.
        target_modules: List of module name suffixes to attach LoRA to.
            If None, will auto-detect using detect_lora_target_modules().
        lora_dropout: Dropout probability for LoRA layers.
        verbose: If True, print trainable parameter counts.

    Returns:
        The PEFT-wrapped model (a PeftModel instance).

    After wrapping:
        - All base model parameters are frozen (requires_grad=False)
        - Only LoRA adapter parameters are trainable
        - The model can be trained with the standard HF Trainer
        - Save with model.save_pretrained(output_dir) to save adapter only
    """
    from peft import LoraConfig, get_peft_model, PeftModel

    # Auto-detect target modules if not provided
    if target_modules is None:
        target_modules = detect_lora_target_modules(model, verbose=verbose)
        if not target_modules:
            raise ValueError(
                "No LoRA target modules detected. Either pass target_modules "
                "explicitly or check that your model has nn.Linear layers."
            )

    if verbose:
        print(f"\n[lora_setup] Applying LoRA:", file=sys.stderr)
        print(f"  rank           : {lora_rank}", file=sys.stderr)
        print(f"  alpha          : {lora_alpha}", file=sys.stderr)
        print(f"  target_modules : {target_modules}", file=sys.stderr)
        print(f"  dropout        : {lora_dropout}", file=sys.stderr)

    # Create LoRA config
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias="none",           # don't train bias terms
        task_type=None,        # not a standard causal LM task
        # Don't use default init — let PEFT use its own (kaiming for A, zero for B)
    )

    # Wrap the model with PEFT
    # This freezes ALL base model parameters and adds LoRA adapters
    model = get_peft_model(model, lora_config)

    # Print trainable parameters for verification
    if verbose:
        model.print_trainable_parameters()
        print(f"\n[lora_setup] LoRA wrapping complete.", file=sys.stderr)
        print(f"[lora_setup] The model is now a PeftModel.", file=sys.stderr)
        print(f"[lora_setup] Only adapter weights will be trained.", file=sys.stderr)
        print(f"[lora_setup] Save with: model.save_pretrained(output_dir)", file=sys.stderr)

    return model


# =============================================================================
# 4. Helper: check if a model is a PeftModel
# =============================================================================
def is_peft_model(model: nn.Module) -> bool:
    """Check if the model has been wrapped with PEFT."""
    try:
        from peft import PeftModel
        return isinstance(model, PeftModel)
    except ImportError:
        return False


# =============================================================================
# 5. Helper: save a model (handles both PEFT and regular models)
# =============================================================================
def save_model(model: nn.Module, output_dir: str | Path) -> None:
    """
    Save a model, handling both PEFT and regular models.

    For PEFT models: saves only the adapter weights (small, ~10-50MB).
    For regular models: saves the full state dict.

    Args:
        model: The model to save.
        output_dir: Directory to save to.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if is_peft_model(model):
        # PEFT model — save adapter only
        print(f"[lora_setup] Saving LoRA adapter to {output_dir}", file=sys.stderr)
        model.save_pretrained(str(output_dir))
        print(f"[lora_setup] Adapter saved. To load: PeftModel.from_pretrained(base_model, '{output_dir}')", file=sys.stderr)
    else:
        # Regular model — save full state dict
        print(f"[lora_setup] Saving full model to {output_dir}", file=sys.stderr)
        state_dict = model.state_dict()
        cpu_state_dict = {k: v.cpu() for k, v in state_dict.items()}
        torch.save(cpu_state_dict, str(output_dir / "pytorch_model.bin"))
        print(f"[lora_setup] Full model saved to {output_dir / 'pytorch_model.bin'}", file=sys.stderr)
