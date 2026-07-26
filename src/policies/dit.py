# import os
# import math
from typing import List, Optional, Tuple
from src.embeds.action_encoder import CategorySpecificMLP, MultiEmbodimentActionEncoder
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

# TRANSFORMER_ENGINE_AVAILABLE = False

# DotProductAttention = None

# try:
#    import transformer_engine  # type: ignore
#    from wam.cudnn_attn import DotProductAttention

#    TRANSFORMER_ENGINE_AVAILABLE = True
# except ModuleNotFoundError:
# TRANSFORMER_ENGINE_AVAILABLE = False


def sinusoidal_embedding_1d(dim: int, position: torch.Tensor) -> torch.Tensor:
    assert dim % 2 == 0
    half = dim // 2
    position = position.type(torch.float64)

    sinusoid = torch.outer(
        position,
        torch.pow(
            10000,
            -torch.arange(half, dtype=position.dtype, device=position.device).div(half),
        ),
    )
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x


def rope_params_no_polar(
    max_seq_len: int, dim: int, theta: float = 10000
) -> torch.Tensor:
    assert dim % 2 == 0
    inv_freq = 1.0 / torch.pow(theta, torch.arange(0, dim, 2).to(torch.float32) / dim)
    t = torch.arange(max_seq_len, dtype=inv_freq.dtype)
    freqs = torch.outer(t, inv_freq)
    emb = torch.stack((freqs.cos(), freqs.sin()), dim=-1).flatten(-2)
    return emb


def rope_action_apply_polar(
    x: torch.Tensor,
    freqs: torch.Tensor,
    freqs_action: torch.Tensor,
    freqs_state: torch.Tensor,
    action_register_length: int | None,
    num_action_per_block: int | None = None,
    num_state_per_block: int | None = None,
) -> torch.Tensor:
    B, seq_len, n, _ = x.shape

    orig_dtype = x.dtype

    # precompute multipliers
    x = torch.view_as_complex(x.to(torch.float64).reshape(B, seq_len, n, -1, 2))

    if action_register_length is not None:
        assert num_action_per_block is not None
        assert num_state_per_block is not None

        chunk_size = action_register_length // (
            num_action_per_block + num_state_per_block
        )

        freqs_1d_action = freqs_action[: chunk_size * num_action_per_block].view(
            chunk_size * num_action_per_block, 1, -1
        )
        freqs_1d_state = freqs_state[: chunk_size * num_state_per_block].view(
            chunk_size * num_state_per_block, 1, -1
        )
        freqs = torch.cat([freqs, freqs_1d_action, freqs_1d_state], dim=0)

    freqs = torch.view_as_complex(
        freqs.to(torch.float64).reshape(*freqs.shape[:-1], -1, 2)
    )
    # apply rotary embedding
    freqs = freqs.unsqueeze(0)
    x = torch.view_as_real(x * freqs).flatten(3)
    return x.to(orig_dtype)


def causal_rope_action_apply_polar(
    x: torch.Tensor,
    freqs: torch.Tensor,
    freqs_action: torch.Tensor,
    freqs_state: torch.Tensor,
    action_register_length: int | None,
    num_action_per_block: int,
    num_state_per_block: int,
    action_state_index: int,
):
    B, seq_len, n, _ = x.shape

    # precompute multipliers
    x = torch.view_as_complex(x.to(torch.float64).reshape(B, seq_len, n, -1, 2))

    if action_register_length is not None:
        assert action_register_length == (num_action_per_block + num_state_per_block)
        freqs_action = freqs_action[
            action_state_index
            * num_action_per_block : (action_state_index + 1)
            * num_action_per_block
        ]
        freqs_state = freqs_state[
            action_state_index
            * num_state_per_block : (action_state_index + 1)
            * num_state_per_block
        ]
        freqs_1d = torch.cat([freqs_action, freqs_state], dim=0).view(
            action_register_length, 1, -1
        )
        freqs = torch.cat([freqs, freqs_1d], dim=0)
    
    #freqs = torch.view_as_complex(
    #    freqs.to(torch.float64).reshape(*freqs.shape[:-1], -1, 2)
    #)

    # apply rotary embedding
    freqs = freqs.unsqueeze(0)
    x = torch.view_as_real(x * freqs).flatten(3)

    return x


class AttentionModule(nn.Module):
    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        dropout_p: float = 0,
        softmax_scale: Optional[float] = None,
        q_scale: Optional[float] = None,
        causal: bool = False,
        window_size: Optional[Tuple[int, int]] = None,
        deterministic: bool = False,
        dtype: Optional[torch.dtype] = None,
        backend: Optional[str] = None,
    ):
        super(AttentionModule, self).__init__()
        self.is_causal = causal
        self.dropout_p = dropout_p
        self.softmax_scale = softmax_scale
        self.dtype = dtype
        # if backend is None:
        #    backend = "torch"
        ## if os.getenv("ATTENTION_BACKEND") is not None:
        ##    backend = os.getenv("ATTENTION_BACKEND")
        ## else:
        ##    backend = "FA2"
        ## if os.getenv("ENABLE_TERSORRT", "FALSE").lower() == "true":
        ##    backend = "torch"
        ## if backend == "TE" and not TRANSFORMER_ENGINE_AVAILABLE:
        ##    backend = "FA2"

        ## assert backend in ["torch", "FA2", "FA3", "TE", "torch_onnx"]
        # self.backend = backend

        # if backend == "torch":

        #    self.attn_func = _torch_impl

        # else:
        #    raise ValueError(f"Invalid backend: {backend}")

    def _torch_impl(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        q_lens: Optional[torch.Tensor] = None,
        k_lens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # out_dtype = q.dtype
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            is_causal=self.is_causal,
            dropout_p=self.dropout_p,
            scale=self.softmax_scale,
        )

        out = out.transpose(1, 2).contiguous()
        return out

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        q_lens: Optional[torch.Tensor] = None,
        k_lens: Optional[torch.Tensor] = None,
    ):
        # if (
        #    self.backend == "torch"
        #    or self.backend == "torch_onnx"
        #    or self.backend == "TE"
        #    and TRANSFORMER_ENGINE_AVAILABLE
        # ):
        #    if q_lens is not None or k_lens is not None:
        #        # warnings.warn()
        #        pass
        #    return self.attn_func(q, k, v, q_lens=None, k_lens=None)
        # else:
        #    return self.attn_func(q, k, v, q_lens=q_lens, k_lens=k_lens)

        return self._torch_impl(q, k, v, q_lens=q_lens, k_lens=k_lens)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super(RMSNorm, self).__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor):
        return self._norm(x.float()).type_as(x) * self.weight

    def _norm(self, x: torch.Tensor):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)


class CausalSelfAttentionBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        frame_seq_len: int,
        local_attn_size: int = 1,
        sink_size: int = 0,
        num_frame_per_block: int = 1,
        qk_norm: bool = True,
        eps: float = 1e-6,
        num_action_per_block: int = 32,
        num_state_per_block: int = 1,
    ):
        super(CausalSelfAttentionBlock, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.local_attn_size = local_attn_size
        self.sink_size = sink_size
        self.num_frame_per_block = num_frame_per_block
        self.qk_norm = qk_norm
        self.eps = eps
        self.max_attention_size = (
            21 * frame_seq_len
            if local_attn_size == -1
            else local_attn_size * frame_seq_len
        )
        self.frame_seq_len = frame_seq_len
        self.num_action_per_block = num_action_per_block
        self.num_state_per_block = num_state_per_block

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = RMSNorm(dim=dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = RMSNorm(dim=dim, eps=eps) if qk_norm else nn.Identity()
        self.attn = (
            AttentionModule(
                num_heads=num_heads,
                head_dim=self.head_dim,
            )
            if qk_norm
            else nn.Identity()
        )
        self.causal_attn = AttentionModule(
            num_heads=num_heads, head_dim=self.head_dim, causal=True
        )

    def _visualize_attention_mask(
        self,
        total_len: int,
        first_image_len: int,
        image_blocks_len: int,
        action_len: int,
        state_len: int,
        num_image_blocks: int,
        num_action_blocks: int,
        num_state_blocks: int,
        num_frame_per_block: int,
        frame_seq_len: int,
        num_action_per_block: int,
        num_state_per_block: int,
    ):
        first_image_start = 0
        first_image_end = first_image_len
        image_blocks_start = first_image_end
        image_blocks_end = image_blocks_start + image_blocks_len
        action_start = image_blocks_end
        action_end = action_start + action_len
        state_start = action_end
        state_end = state_start + state_len

        mask = torch.zeros(total_len, total_len, dtype=torch.bool)
        mask[first_image_start:first_image_end, first_image_start:first_image_end] = (
            True
        )

        for block_idx in range(num_image_blocks):
            block_start = (
                image_blocks_start + block_idx * num_frame_per_block * frame_seq_len
            )
            block_end = (
                image_blocks_start
                + (block_idx + 1) * num_frame_per_block * frame_seq_len
            )

            mask[block_start:block_end, first_image_start:first_image_end] = True
            if self.local_attn_size != -1:
                image_kv_start = max(
                    image_blocks_start, block_end - self.local_attn_size * frame_seq_len
                )
            else:
                image_kv_start = image_blocks_start
            mask[block_start:block_end, image_kv_start:block_end] = True

            # old
            action_block_start = action_start + block_idx * num_action_per_block
            action_block_end = action_start + (block_idx + 1) * num_action_per_block
            mask[block_start:block_end, action_block_start:action_block_end] = True

            state_block_start = state_start + block_idx * num_state_per_block
            state_block_end = state_start + (block_idx + 1) * num_state_per_block
            mask[block_start:block_end, state_block_start:state_block_end] = True

        for block_idx in range(num_action_blocks):
            action_block_start = action_start + block_idx * num_action_per_block
            action_block_end = action_start + (block_idx + 1) * num_action_per_block

            mask[
                action_block_start:action_block_end, first_image_start:first_image_end
            ] = True

            image_block_end = (
                image_blocks_start
                + (block_idx + 1) * num_frame_per_block * frame_seq_len
            )
            if self.local_attn_size != -1:
                image_kv_start = max(
                    image_blocks_start,
                    image_block_end - self.local_attn_size * frame_seq_len,
                )
            else:
                image_kv_start = image_blocks_start
            mask[
                action_block_start:action_block_end, image_kv_start:image_block_end
            ] = True
            mask[
                action_block_start:action_block_end, action_block_start:action_block_end
            ] = True

            state_block_start = state_start + block_idx * num_state_per_block
            state_block_end = state_start + (block_idx + 1) * num_state_per_block
            mask[
                action_block_start:action_block_end, state_block_start:state_block_end
            ] = True

        for block_idx in range(num_state_blocks):
            state_block_start = state_start + block_idx * num_state_per_block
            state_block_end = state_start + (block_idx + 1) * num_state_per_block
            mask[
                state_block_start:state_block_end, state_block_start:state_block_end
            ] = True

        return mask

    def _blockwise_causal_flash_attn(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        frame_seq_len: int,
        num_frame_per_block: int = 1,
        action_horizon: Optional[int] = None,
        state_horizon: Optional[int] = None,
        num_action_per_block: Optional[int] = None,
        num_state_per_block: Optional[int] = None,
        visualize_mask: bool = False,
    ):
        b, total_len, n, d = q.shape # 1 8086 16 64
        has_action_state = action_horizon is not None and state_horizon is not None

        if not has_action_state: # triggers only if there no action or state (training only on frames)
            num_frames = total_len // frame_seq_len # 33
            block_size = frame_seq_len * num_frame_per_block # 242 * 8 = 1936
            num_blocks = (num_frames - 1) // num_frame_per_block # (33 - 1) // 8 = 4 

            if num_blocks <= 0:
                return self.attn(q, k, v)

            if self.local_attn_size == -1:
                return self.causal_attn(q, k, v)

            output = torch.empty_like(q)

            block_starts = [frame_seq_len + i * block_size for i in range(num_blocks)]
            block_ends = [min(start + block_size, total_len) for start in block_starts]
            kv_starts = [
                max(0, end - self.local_attn_size * frame_seq_len) for end in block_ends
            ]

            for block_idx in range(num_blocks):
                block_start = block_starts[block_idx]
                block_end = block_ends[block_idx]
                kv_start = kv_starts[block_idx]

                output[:, block_start:block_end] = self.attn(
                    q[:, block_start:block_end],
                    k[:, kv_start:block_end],
                    v[:, kv_start:block_end],
                )

            return output

        assert action_horizon is not None and state_horizon is not None
        assert num_action_per_block is not None and num_state_per_block is not None

        first_image_len = frame_seq_len # 242
        action_len = action_horizon # 96
        state_len = state_horizon # 4
        image_blocks_len = total_len - first_image_len - action_len - state_len # 8086 - 242 - 96 - 4 = 7744

        num_image_blocks = image_blocks_len // (num_frame_per_block * frame_seq_len) # 7744 // (8 * 242) = 4

        # assert action_horizon % num_image_blocks == 0, (
        #    f"action_horizon={action_horizon} not divisible by "
        #    f"num_image_blocks={num_image_blocks}"
        # )
        # assert state_horizon % num_image_blocks == 0, (
        #    f"state_horizon={state_horizon} not divisible by "
        #    f"num_image_blocks={num_image_blocks}"
        # )

        num_action_blocks = action_horizon // num_action_per_block # 96 // 24 = 4
        num_state_blocks = state_horizon // num_state_per_block # 4 // 1 = 4
        # num_action_per_block = action_horizon // num_image_blocks
        # num_state_per_block = state_horizon // num_image_blocks
        # num_action_blocks = num_image_blocks
        # num_state_blocks = num_image_blocks

        assert num_image_blocks == num_action_blocks == num_state_blocks

        first_image_start = 0
        first_image_end = first_image_len # 242
        image_blocks_start = first_image_end # 242
        image_blocks_end = image_blocks_start + image_blocks_len # 242 + 7744 = 7986 all dino tokens
        action_start = image_blocks_start + image_blocks_len # 242 + 7744 = 7986
        action_end = action_start + action_len # 7986 + 96 = 8082 all action tokens
        state_start = action_end # 8082
        state_end = state_start + state_len # 8082 + 4 = 8086 

        if visualize_mask:
            mask = self._visualize_attention_mask(
                total_len,
                first_image_len,
                image_blocks_len,
                action_len,
                state_len,
                num_image_blocks,
                num_action_blocks,
                num_state_blocks,
                num_frame_per_block,
                frame_seq_len,
                num_action_per_block,
                num_state_per_block,
            )

            print(f"Total length: {total_len}")
            print(
                f"First image: [{first_image_start}:{first_image_end}] (len={first_image_len})"
            )
            print(
                f"Image blocks: [{image_blocks_start}:{image_blocks_end}] (len={image_blocks_len}, num_blocks={num_image_blocks})"
            )
            print(
                f"Action tokens: [{action_start}:{action_end}] (len={action_len}, num_blocks={num_action_blocks})"
            )
            print(
                f"State tokens: [{state_start}:{state_end}] (len={state_len}, num_blocks={num_state_blocks})"
            )
            print(f"Local attention size: {self.local_attn_size}")

            if total_len <= 100:
                for i in range(total_len):
                    row = "".join("1" if mask[i, j] else "." for j in range(total_len))
                    print(f"{i:4d}: {row}")
            else:
                downsample = max(1, total_len // 100)
                print(f"AttentionMask (downsampled by {downsample}x):")
                print(
                    f"Rows=Query tokens, Cols=key tokens (1=can attend), .=cannot attend"
                )
                for i in range(0, total_len, downsample):
                    row = "".join(
                        [
                            "1" if mask[i, j] else "."
                            for j in range(0, total_len, downsample)
                        ]
                    )
                    print(f"{i:4d}: {row}")

        output = torch.empty_like(q) # 1 8086 16 64
        output[:, first_image_start:first_image_end] = self.attn( # frame 0 is attending to itself
            q[:, first_image_start:first_image_end],
            k[:, first_image_start:first_image_end],
            v[:, first_image_start:first_image_end],
        )

        image_block_starts = [ # partition of tokens of 8 frames (1 block)
            image_blocks_start + i * num_frame_per_block * frame_seq_len
            for i in range(num_image_blocks)
        ]
        image_block_ends = [
            image_blocks_start + (i + 1) * num_frame_per_block * frame_seq_len
            for i in range(num_image_blocks)
        ]
        if self.local_attn_size != -1: # 33 != -1
            image_kv_starts = [
                max(image_blocks_start, end - self.local_attn_size * frame_seq_len)
                for end in image_block_ends
            ]
        else:
            image_kv_starts = [image_blocks_start] * num_image_blocks

        action_block_starts = [
            action_start + i * num_action_per_block for i in range(num_action_blocks)
        ]
        action_block_ends = [
            action_start + (i + 1) * num_action_per_block
            for i in range(num_action_blocks)
        ]
        state_block_starts = [
            state_start + i * num_state_per_block for i in range(num_state_blocks)
        ]
        state_block_ends = [
            state_start + (i + 1) * num_state_per_block for i in range(num_state_blocks)
        ]

        for block_idx in range(num_image_blocks):
            block_start = image_block_starts[block_idx] # 242 ; 0
            block_end = image_block_ends[block_idx] # 2178 ; 0
            image_kv_start = image_kv_starts[block_idx] # 242 ; 0
            action_block_start = action_block_starts[block_idx] # 7986 ; 0
            action_block_end = action_block_ends[block_idx] # 8010 ; 0
            state_block_start = state_block_starts[block_idx] # 8082 ; 0
            state_block_end = state_block_ends[block_idx] # 8083 ; 0

            k_context = torch.cat(
                [
                    k[:, first_image_start:first_image_end], # first frame
                    k[:, image_kv_start:block_end], # F1...F8 (8 frames of a single block)
                    k[:, action_block_start:action_block_end], # 24 action token per block
                    k[:, state_block_start:state_block_end], # 1 state token per block
                ],
                dim=1,
            )
            v_context = torch.cat(
                [
                    v[:, first_image_start:first_image_end],
                    v[:, image_kv_start:block_end],
                    v[:, action_block_start:action_block_end],
                    v[:, state_block_start:state_block_end],
                ],
                dim=1,
            )

            output[:, block_start:block_end] = self.attn(
                q[:, block_start:block_end], k_context, v_context
            )

        for block_idx in range(num_action_blocks):
            action_block_start = action_block_starts[block_idx] # 7986 ; 0
            action_block_end = action_block_ends[block_idx] # 8010 ; 0 
            image_block_end = image_block_ends[block_idx] # 2178 ; 0
            state_block_start = state_block_starts[block_idx] # 8082 ; 0
            state_block_end = state_block_ends[block_idx] # 8083 ; 0

            if self.local_attn_size != -1:
                image_kv_start = max(
                    image_blocks_start,
                    image_block_end - self.local_attn_size * frame_seq_len,
                ) # 242
            else:
                image_kv_start = image_blocks_start

            k_context = torch.cat(
                [
                    k[:, first_image_start:first_image_end], # F0 ; 0
                    k[:, image_kv_start:image_block_end], # F1...8 ; 0
                    k[:, action_block_start:action_block_end], # A0...23 ; 0
                    k[:, state_block_start:state_block_end], # S0 ; 0
                ],
                dim=1,
            )
            v_context = torch.cat(
                [
                    v[:, first_image_start:first_image_end],
                    v[:, image_kv_start:image_block_end],
                    v[:, action_block_start:action_block_end],
                    v[:, state_block_start:state_block_end],
                ],
                dim=1,
            )

            output[:, action_block_start:action_block_end] = self.attn(
                q[:, action_block_start:action_block_end], k_context, v_context
            )

        for block_idx in range(num_state_blocks):
            state_block_start = state_block_starts[block_idx] # 8082 ; 0
            state_block_end = state_block_ends[block_idx] # 8083 ; 0

            output[:, state_block_start:state_block_end] = self.attn(
                q[:, state_block_start:state_block_end], 
                k[:, state_block_start:state_block_end],
                v[:, state_block_start:state_block_end],
            )

        return output

    def forward(
        self,
        x: torch.Tensor,
        freqs: torch.Tensor,
        freqs_action: torch.Tensor,
        freqs_state: torch.Tensor,
        action_register_length: int | None,
        kv_cache: torch.Tensor | None = None,
        current_start_frame: int = 0,
        # is_tf: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor | None]:
        b, s = x.shape[:2]
        n, d = self.num_heads, self.head_dim

        def qkv_fn(x):
            q = self.norm_q(self.q(x)).view(b, s, n, d)
            k = self.norm_k(self.k(x)).view(b, s, n, d)
            v = self.v(x).view(b, s, n, d)
            return q, k, v

        q, k, v = qkv_fn(x)
        updated_kv_cache: torch.Tensor | None = None

        if kv_cache is None: # always true 
            roped_query = rope_action_apply_polar(
                x=q,
                freqs=freqs,
                freqs_action=freqs_action,
                freqs_state=freqs_state,
                action_register_length=action_register_length,
                num_action_per_block=self.num_action_per_block,
                num_state_per_block=self.num_state_per_block,
            ).type_as(v) # 1 8086 16 64
            roped_key = rope_action_apply_polar(
                x=k,
                freqs=freqs,
                freqs_action=freqs_action,
                freqs_state=freqs_state,
                action_register_length=action_register_length,
                num_action_per_block=self.num_action_per_block,
                num_state_per_block=self.num_state_per_block,
            ).type_as(v) # 1 8086 16 64

            if action_register_length is not None:
                chunk_size = action_register_length // (
                    self.num_action_per_block + self.num_state_per_block
                )
                action_horizon = chunk_size * self.num_action_per_block
                state_horizon = chunk_size * self.num_state_per_block
            else:
                action_horizon = None
                state_horizon = None

            x = self._blockwise_causal_flash_attn(
                roped_query,
                roped_key,
                v,
                self.frame_seq_len,
                self.num_frame_per_block,
                action_horizon=action_horizon,
                state_horizon=state_horizon,
                num_action_per_block=(
                    self.num_action_per_block if action_register_length else None
                ),
                num_state_per_block=(
                    self.num_state_per_block if action_register_length else None
                ),
                visualize_mask=False,
            )

        else:
            action_state_index = (current_start_frame - 1) // self.num_frame_per_block

            roped_query = causal_rope_action_apply_polar(
                x=q,
                freqs=freqs,
                freqs_action=freqs_action,
                freqs_state=freqs_state,
                action_register_length=action_register_length,
                num_action_per_block=self.num_action_per_block,
                num_state_per_block=self.num_state_per_block,
                action_state_index=action_state_index,
            ).type_as(v)
            roped_key = causal_rope_action_apply_polar(
                x=k,
                freqs=freqs,
                freqs_action=freqs_action,
                freqs_state=freqs_state,
                action_register_length=action_register_length,
                num_action_per_block=self.num_action_per_block,
                num_state_per_block=self.num_state_per_block,
                action_state_index=action_state_index,
            ).type_as(v)

            roped_action_query: torch.Tensor | None = None
            roped_action_key: torch.Tensor | None = None
            action_v: torch.Tensor | None = None

            if action_register_length is not None:
                roped_action_query = roped_query[:, -action_register_length:]
                roped_query = roped_query[:, :-action_register_length]
                roped_action_key = roped_key[:, -action_register_length:]
                roped_key = roped_key[:, :-action_register_length]
                action_v = v[:, -action_register_length:]
                v = v[:, :-action_register_length]
                assert roped_action_query is not None
                assert roped_action_key is not None
                assert action_v is not None

            num_new_tokens = roped_query.shape[1]
            assert roped_key.shape[1] == num_new_tokens
            assert v.shape[1] == num_new_tokens

            updated_kv_cache = kv_cache
            updated_k = updated_kv_cache[0]
            updated_v = updated_kv_cache[1]

            new_k = torch.cat([updated_k, roped_key], dim=1)
            new_v = torch.cat([updated_v, v], dim=1)

            new_k = new_k[:, -self.max_attention_size :]
            new_v = new_v[:, -self.max_attention_size :]

            if action_register_length is not None:
                x = self.attn(
                    torch.cat([roped_query, roped_action_query], dim=1),  # type: ignore
                    torch.cat([new_k, roped_action_key], dim=1),  # type: ignore
                    torch.cat([new_v, action_v], dim=1),  # type: ignore
                )
            else:
                x = self.attn(roped_query, new_k, new_v)
            updated_kv_cache = torch.stack([new_k, new_v], dim=0)

        x = x.flatten(2)
        x = self.o(x)
        return x, updated_kv_cache


class CausalCrossAttentionBlock(nn.Module):

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: Tuple[int, int] = (-1, -1),
        qk_norm: bool = True,
        eps: float = 1e-6,
    ):
        super(CausalCrossAttentionBlock, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.eps = eps
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)

        self.norm_q = RMSNorm(dim=dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = RMSNorm(dim=dim, eps=eps) if qk_norm else nn.Identity()

    def forward(self, x: torch.Tensor, context: torch.Tensor, crossattn_cache=None):
        b, l1, _ = x.shape
        n = self.num_heads
        d = self.head_dim

        q = self.norm_q(self.q(x)).view(b, l1, n, d).transpose(1, 2)

        # i hv some doubts here , we're not using dual pass as the context is only text and NO contextual img included in the context ......
        if crossattn_cache is not None:
            if not crossattn_cache["is_init"]:
                crossattn_cache["is_init"] = True
                k = self.norm_k(self.k(context)).reshape(b, -1, n, d).transpose(1, 2)
                v = self.v(context).reshape(b, -1, n, d).transpose(1, 2)
                crossattn_cache["k"] = k
                crossattn_cache["v"] = v
            else:
                k = crossattn_cache["k"]
                v = crossattn_cache["v"]
        else:
            k = self.norm_k(self.k(context)).reshape(b, -1, n, d).transpose(1, 2)
            v = self.v(context).reshape(b, -1, n, d).transpose(1, 2)

        x_out = F.scaled_dot_product_attention(
            query=q, key=k, value=v, dropout_p=0.0, is_causal=False
        )
        x_out = x_out.transpose(1, 2).contiguous().view(b, l1, self.dim)
        x_out = self.o(x_out)
        return x_out


class CausalAttentionBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        ffn_dim: int,
        num_heads: int,
        frame_seq_len: int,
        local_attn_size: int = 1,
        sink_size: int = 0,
        num_frame_per_block: int = 1,
        qk_norm: bool = True,
        cross_attn_norm: bool = False,
        eps: float = 1e-6,
        num_action_per_block: int = 32,
        num_state_per_block: int = 1,
    ):
        super(CausalAttentionBlock, self).__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.local_attn_size = local_attn_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.self_attn = CausalSelfAttentionBlock(
            dim=dim,
            num_heads=num_heads,
            frame_seq_len=frame_seq_len,
            local_attn_size=local_attn_size,
            sink_size=sink_size,
            num_frame_per_block=num_frame_per_block,
            qk_norm=qk_norm,
            eps=eps,
            num_action_per_block=num_action_per_block,
            num_state_per_block=num_state_per_block,
        )
        self.norm3 = (
            nn.LayerNorm(dim, eps, elementwise_affine=True)
            if cross_attn_norm
            else nn.Identity()
        )
        self.cross_attn = CausalCrossAttentionBlock(
            dim=dim, num_heads=num_heads, window_size=(-1, -1), qk_norm=qk_norm, eps=eps
        )
        self.norm2 = nn.LayerNorm(dim, eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(ffn_dim, dim),
        )
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(
        self,
        x: torch.Tensor,
        e: torch.Tensor,
        freqs: torch.Tensor,
        freqs_action: torch.Tensor,
        freqs_state: torch.Tensor,
        action_register_length: int | None,
        context: torch.Tensor,
        kv_cache: Optional[torch.Tensor] = None,
        crossattn_cache: Optional[torch.Tensor] = None,
        current_start_frame: int = 0,
        # is_tf: bool = True,
    ):
        e_chunk = (self.modulation.unsqueeze(1) + e).chunk(6, dim=2)

        # Align modulation sequence length to x to ensure valid broadcasting
        L = x.shape[1]
        aligned: list[torch.Tensor] = []
        for part in e_chunk:
            L_e = part.shape[1]
            if L_e == L:
                aligned.append(part)
            elif L_e >= L:
                aligned.append(part[:, :L])
            else:
                print(f"Triggered Interleaved")
                repeat = (L + L_e - 1) // L_e
                aligned.append(part.repeat_interleave(repeat, dim=1)[:, :L])
        e_chunk = tuple(aligned)

        y, updated_kv_cache = self.self_attn(
            x=(self.norm1(x) * (1 + e_chunk[1].squeeze(2)) + e_chunk[0].squeeze(2)),
            freqs=freqs,
            freqs_action=freqs_action,
            freqs_state=freqs_state,
            action_register_length=action_register_length,
            kv_cache=kv_cache,
            current_start_frame=current_start_frame,
        )

        x = x + (y * e_chunk[2].squeeze(2))

        def cross_att_ffn(
            x: torch.Tensor, context: torch.Tensor, e_chunk: Tuple[torch.Tensor, ...]
        ):
            x = x + self.cross_attn(self.norm3(x), context)
            y = self.ffn(
                (self.norm2(x) * (1 + e_chunk[4].squeeze(2)) + e_chunk[3].squeeze(2))
            )
            x = x + (y * e_chunk[5].squeeze(2))
            return x

        x = cross_att_ffn(x, context, e_chunk=e_chunk)
        return x, updated_kv_cache


class DiTBackbone(nn.Module):
    def __init__(
        self,
        patch_size: tuple[int, int, int] = (1, 1, 1),
        frame_seq_len: int = 242,
        freq_dim: int = 256,
        text_len: int = 200,
        text_dim: int = 768,
        in_dim: int = 768,  # dino dim
        out_dim: int = 768,
        dim: int = 1024,
        ffn_dim: int = 4096,
        num_heads: int = 16,
        num_layers: int = 12,
        action_dim: int = 32,
        max_state_dim: int = 16,
        max_num_embodiment: int = 32,
        hidden_size=1024,
        num_frame_per_block: int = 1,
        num_action_per_block: int = 32,
        num_state_per_block: int = 1,
        eps: float = 1e-6,
        max_chunk_size: int = 4,
        sink_size: int = 0,
        qk_norm: bool = True,
        cross_attn_norm: bool = True,
        **kwargs,
    ):
        super(DiTBackbone, self).__init__()
        self.patch_size = patch_size
        self.frame_seq_len = frame_seq_len
        self.freq_dim = freq_dim
        self.text_len = text_len
        self.text_dim = text_dim
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.action_dim = action_dim
        self.max_state_dim = max_state_dim
        self.max_num_embodiment = max_num_embodiment
        self.hidden_size = hidden_size
        self.num_frame_per_block = num_frame_per_block
        self.num_action_per_block = num_action_per_block
        self.num_state_per_block = num_state_per_block
        self.local_attn_size = (
            max_chunk_size * num_frame_per_block + 1 if max_chunk_size != -1 else -1
        ) # 4 * 8 + 1 = 33

        # max_num_embodiment = 18

        self.state_encoder = CategorySpecificMLP(
            num_categories=max_num_embodiment,
            input_dim=self.max_state_dim,
            hidden_dim=self.hidden_size,
            output_dim=self.dim,
        )
        self.action_encoder = MultiEmbodimentActionEncoder(
            action_dim=self.action_dim,
            hidden_size=self.dim,
            num_embodiments=max_num_embodiment,
        )
        self.action_decoder = CategorySpecificMLP(
            num_categories=max_num_embodiment,
            input_dim=self.dim,
            hidden_dim=self.hidden_size,
            output_dim=self.action_dim,
        )

        self.dino_input_norm = nn.LayerNorm(in_dim, elementwise_affine=False)
        self.dino_proj = nn.Linear(in_dim, dim)
        self.frame_head = nn.Sequential(
            nn.LayerNorm(dim, eps=eps),
            nn.Linear(dim, out_dim),
        )
        nn.init.zeros_(self.frame_head[-1].weight)  # type: ignore
        nn.init.zeros_(self.frame_head[-1].bias)  # type: ignore

        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim), nn.GELU(approximate="tanh"), nn.Linear(dim, dim)
        )
        self.time_embedding = nn.Sequential(
            nn.Linear(self.freq_dim, self.dim), nn.SiLU(), nn.Linear(self.dim, self.dim)
        )
        self.time_projection = nn.Sequential(
            nn.SiLU(), nn.Linear(self.dim, self.dim * 6)
        )

        self.blocks = nn.ModuleList(
            [
                CausalAttentionBlock(
                    dim=dim,
                    ffn_dim=ffn_dim,
                    num_heads=num_heads,
                    frame_seq_len=frame_seq_len,
                    local_attn_size=self.local_attn_size,
                    sink_size=sink_size,
                    num_frame_per_block=num_frame_per_block,
                    qk_norm=qk_norm,
                    cross_attn_norm=cross_attn_norm,
                    eps=eps,
                    num_action_per_block=num_action_per_block,
                    num_state_per_block=num_state_per_block,
                )
                for _ in range(self.num_layers)
            ]
        )
        # self.head = CausalHead(dim, out_dim, patch_size, eps)

        d = dim // num_heads

        freq_actions = rope_params_no_polar(1024 * 10, d)
        freqs_state = rope_params_no_polar(1024, d)
        freqs = [
            rope_params_no_polar(1024, d - 4 * (d // 6)),
            rope_params_no_polar(1024, 2 * (d // 6)),
            rope_params_no_polar(1024, 2 * (d // 6)),
        ]

        self.register_buffer("freq_actions", freq_actions)
        self.register_buffer("freqs_state", freqs_state)
        self.register_buffer("freqs_0", freqs[0])
        self.register_buffer("freqs_1", freqs[1])
        self.register_buffer("freqs_2", freqs[2])

        self.gradient_checkpointing = False # True

        self.dynamics_loss_weight = 0.5 # was 1.0, 0.1
        self.action_loss_weight = 1.0
        self.use_residual_frame_prediction = True

    def _create_freqs(self, grid_size: Tuple[int, ...], start_frame: int):
        # device = self.patch_embedding.weight.device
        # if any(freq.device != device for freq in self.freqs):
        #    self.freqs = [freq.to(device) for freq in self.freqs]
        # if self.freq_actions.device != device:
        #    self.freq_actions = self.freq_actions.to(device)

        f0 = self.get_buffer("freqs_0")
        f1 = self.get_buffer("freqs_1")
        f2 = self.get_buffer("freqs_2")

        f, h, w = grid_size[-3:]
        freqs = torch.cat(
            [
                f0[start_frame : start_frame + f].view(f, 1, 1, -1).expand(f, h, w, -1),
                f1[:h].view(1, h, 1, -1).expand(f, h, w, -1),
                f2[:w].view(1, 1, w, -1).expand(f, h, w, -1),
            ],
            dim=-1,
        ).reshape(f * h * w, 1, -1)

        return freqs

    def _create_kv_caches(
        self,
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
        frame_seqlen: int,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """
        Initialize a Per-GPU KV cache for the Wan model.
        Use the model's num_heads and head_dim (5B has 24 heads, 14B has 40).
        """
        num_heads = self.num_heads
        head_dim = self.dim // num_heads
        kv_cache1 = []
        kv_cache_neg = []
        for _ in range(self.num_layers):
            kv_cache1.append(
                torch.zeros(
                    [2, batch_size, 0, num_heads, head_dim], dtype=dtype, device=device
                ),
            )
            kv_cache_neg.append(
                torch.zeros(
                    [2, batch_size, 0, num_heads, head_dim], dtype=dtype, device=device
                ),
            )

        return kv_cache1, kv_cache_neg

    def _create_crossattn_caches(
        self,
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> tuple[list[dict], list[dict]]:
        crossattn_cache: list[dict] = []
        crossattn_cache_neg: list[dict] = []

        for _ in range(self.num_layers):
            crossattn_cache.append({"is_init": False, "k": None, "v": None})
            crossattn_cache_neg.append({"is_init": False, "k": None, "v": None})

        return crossattn_cache, crossattn_cache_neg

    def _forward_inference(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        timestep_action: Optional[torch.Tensor],
        context: torch.Tensor,
        seq_len: int,
        grid_size: Tuple[int, int, int],
        kv_cache: list[torch.Tensor],
        crossattn_cache: list[torch.Tensor],
        current_start_frame: int,
        action: Optional[torch.Tensor] = None,
        state: Optional[torch.Tensor] = None,
        embodiment_id: Optional[torch.Tensor] = None,
    ):
        x_dino_raw = self.dino_input_norm(x)  # [B, seq_len, in_dim]
        self.dino_proj.to(dtype=x_dino_raw.dtype, device=x_dino_raw.device)
        x = self.dino_proj(x_dino_raw)  # [2, 7986, 768]
        freqs = self._create_freqs(grid_size=grid_size, start_frame=current_start_frame)

        B = x.shape[0]
        F = timestep.shape[1]  # should be 1 for streaming

        state_features: torch.Tensor | None = None
        action_register_length: int | None = None

        if action is not None:
            if embodiment_id is None:
                embodiment_id = torch.tensor([0], device=action.device).repeat(B)
            action_features = self.action_encoder(  # [B, T, hidden_size]
                actions=action, timesteps=timestep_action, cat_ids=embodiment_id
            )
            action_length = action_features.shape[1]
            state_features = self.state_encoder(x=state, cat_ids=embodiment_id)
            action_register = torch.cat([action_features, state_features], dim=1)  # type: ignore
            action_register_length = action_register.shape[1]
            x = torch.cat([x, action_register], dim=1)
        else:
            action_length = None

        if F <= seq_len:
            repeat = (seq_len + F - 1) // F
            timestep = timestep.repeat_interleave(repeat, dim=1)[:, :seq_len]
        else:
            indices = torch.linspace(
                0, F - 1, seq_len, device=timestep.device, dtype=torch.long
            )
            timestep = timestep[:, indices]

        if action is not None:
            assert timestep_action is not None
            assert state_features is not None
            timestep_action = timestep_action.to(device=x.device)
            stride = timestep_action.shape[1] // state_features.shape[1]
            timestep_state = timestep_action[:, ::stride]
            timestep = torch.cat([timestep, timestep_action, timestep_state], dim=1)

        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, timestep.flatten()).type_as(x)
        )
        e = e.unflatten(dim=0, sizes=(B, -1))
        e0 = self.time_projection(e).unflatten(dim=2, sizes=(6, self.dim))

        context = self.text_embedding(context)

        updated_kv_caches: list[torch.Tensor] = []
        for block_idx, block in enumerate(self.blocks):
            crossattn_cache_block = (
                crossattn_cache[block_idx] if crossattn_cache is not None else None
            )
            x, updated_kv_cache = block(
                x,
                e=e0,
                freqs=freqs,
                freqs_action=self.freq_actions,
                freqs_state=self.freqs_state,
                context=context,
                action_register_length=action_register_length,
                kv_cache=kv_cache[block_idx],
                crossattn_cache=crossattn_cache_block,
                current_start_frame=current_start_frame,
            )
            updated_kv_caches.append(updated_kv_cache)

        x_frame = x[:, :seq_len]  # [B, seq_len, dim]
        frame_delta = self.frame_head(x_frame)  # [B, seq_len, out_dim]

        if self.use_residual_frame_prediction:
            """
            Residual prediction: head predicts delta from current frame tokens
            x_dino_raw is the current-frame input; the shifted loss target
            makes the model learn the CHANGE between consecutive frames
            """
            frame_pred = x_dino_raw + frame_delta
        else:
            frame_pred = frame_delta

        if action is not None:
            assert action_length is not None
            action_noise_pred = x[:, seq_len : seq_len + action_length]
            action_noise_pred = self.action_decoder(action_noise_pred, embodiment_id)
        else:
            action_noise_pred = None

        return frame_pred, action_noise_pred, updated_kv_caches

    def _forward_train(
        self,
        x: torch.Tensor, # dino tokens
        timestep: torch.Tensor, # dino fake timestep
        timestep_action: Optional[torch.Tensor], # 
        context: torch.Tensor, # B 200 768 200 tokens for language this is input_ids
        seq_len: int, # 7986 33 frames, each frame has 242 tokens => 242 * 33 = 7986
        grid_size: Tuple[int, int, int], # 33 11 22
        action: Optional[torch.Tensor] = None, # noisy action for the given timestep_action
        state: Optional[torch.Tensor] = None, # state
        embodiment_id: Optional[torch.Tensor] = None, # 0
    ):        
        x_dino_raw = self.dino_input_norm(x)  # [B, seq_len, in_dim]
        x = self.dino_proj(x_dino_raw)  # [2, 7986, 768]
        freqs = self._create_freqs(grid_size=grid_size, start_frame=0)

        assert x.shape[1] == seq_len

        B = x.shape[0] # 1
        F = timestep.shape[1] # 33
        
        state_features: torch.Tensor | None = None
        action_register_length: int | None = None

        if action is not None:
            if embodiment_id is None:
                embodiment_id = torch.tensor([0], device=action.device).repeat(B)
            action_features = self.action_encoder(  # [B, T, hidden_size]
                actions=action, timesteps=timestep_action, cat_ids=embodiment_id
            )
            action_length = action_features.shape[1]
            state_features = self.state_encoder(x=state, cat_ids=embodiment_id)
            action_register = torch.cat([action_features, state_features], dim=1)  # type: ignore
            action_register_length = action_register.shape[1]
            x = torch.cat([x, action_register], dim=1)
        else:
            action_length = None

        # dino timestep
        timestep = timestep.to(device=x.device)
        timestep = timestep.unsqueeze(-1).expand(B, F, seq_len // F).reshape(B, -1)
        # timestep_original = timestep.clone()  # keep the original

        if action is not None:
            assert timestep_action is not None
            assert state_features is not None
            timestep_action = timestep_action.to(device=x.device)
            stride = timestep_action.shape[1] // state_features.shape[1] # 24
            timestep_state = timestep_action[:, ::stride]
            timestep = torch.cat([timestep, timestep_action, timestep_state], dim=1)

        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, timestep.flatten()).type_as(x)
        )
        e = e.unflatten(dim=0, sizes=(B, -1))
        e0 = self.time_projection(e).unflatten(dim=2, sizes=(6, self.dim))

        context = self.text_embedding(context)

        kwargs = dict(
            e=e0,
            freqs=freqs,
            freqs_action=self.freq_actions,
            freqs_state=self.freqs_state,
            action_register_length=action_register_length,
            context=context,
        )

        def create_custom_forward(module):
            def custom_forward(*inputs, **kwargs):
                outputs, updated_kv_cache = module(*inputs, **kwargs)
                assert updated_kv_cache is None
                return outputs

            return custom_forward

        for block in self.blocks:
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                checkpoint = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block), x, **kwargs, use_reentrant=False  # type: ignore
                )
                x = checkpoint  # type: ignore
            else:
                x, _ = block(x, **kwargs)

        x_frame = x[:, :seq_len]  # [B, seq_len, dim] only dino tokens 
        frame_delta = self.frame_head(x_frame)  # [B, seq_len, out_dim]

        if self.use_residual_frame_prediction:
            """
            Residual prediction: head predicts delta from current frame tokens
            x_dino_raw is the current-frame input; the shifted loss target
            makes the model learn the CHANGE between consecutive frames
            """
            frame_pred = x_dino_raw + frame_delta
        else:
            frame_pred = frame_delta

        # if clean_x is not None:
        #    x = x[:, clean_x.shape[1] :]
        if action is not None:
            assert action_length is not None
            action_noise_pred = x[:, seq_len : seq_len + action_length]
            action_noise_pred = self.action_decoder(action_noise_pred, embodiment_id)
        else:
            action_noise_pred = None

        # x_frame = x[:, :seq_len]
        # e_frame = e[:, :seq_len]

        # x_frame = self.head(x_frame, e_frame.unsqueeze(2))
        # frame_noise_pred = self.unpatchify(x_frame, grid_size)

        return frame_pred, action_noise_pred

    def forward(self, *args, **kwargs):
        if kwargs.get("kv_cache", None) is not None:
            return self._forward_inference(*args, **kwargs)
        else:
            return self._forward_train(*args, **kwargs)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        nn.init.xavier_uniform_(self.dino_proj.weight)
        if self.dino_proj.bias is not None:
            nn.init.zeros_(self.dino_proj.bias)

        for m in self.frame_head.modules():
            if isinstance(m, nn.Linear):
                nn.init.zeros_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        for m in self.text_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
        for m in self.time_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)

        # nn.init.zeros_(self.head.head.weight)
        # nn.init.zeros_(self.action_decoder.mlp[-1].weight)
