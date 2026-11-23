"""In-container LLM training benchmark for the CVS PyTorch Training suite.

Launched by torchrun on every node. Builds a HuggingFace causal LM on the meta
device, shards it with FSDP2, trains on pre-tokenized text, and has rank 0
write one JSON artifact holding every metric the suite gates or records.
"""

import argparse
import contextlib
import datetime
import importlib.metadata
import json
import math
import os
import re
import select
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F


STARTUP_OK = "PYTORCH_TRAINING_STARTUP_OK"
STARTUP_FAILED = "PYTORCH_TRAINING_STARTUP_FAILED"
WARMUP_OK = "PYTORCH_TRAINING_WARMUP_OK"
RUN_OK = "PYTORCH_TRAINING_OK "
# CodeCarbon's first reading waits on a hardware probe and its last on the final
# power sample, each a few seconds; this bounds both against a wedged worker.
CODECARBON_WORKER_TIMEOUT_S = 120

# A prime larger than any window count, so stepping by it visits every window
# of a token file once before repeating, in an order unrelated to file layout.
_WINDOW_STRIDE = 1_000_000_007


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--workload", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--model-path", required=True)
    p.add_argument("--init", choices=["random", "pretrained"], default="random")
    p.add_argument("--precision", choices=["BF16", "FP8", "MXFP8", "MXFP4"], default="BF16")
    p.add_argument("--seq-len", type=int, required=True)
    p.add_argument("--micro-batch-size", type=int, required=True)
    p.add_argument("--gradient-accumulation-steps", type=int, required=True)
    p.add_argument("--activation-checkpointing-every", type=int, default=1)
    p.add_argument("--master-weights", choices=["fp32", "bf16"], default="fp32")
    p.add_argument("--attn-implementation", default="sdpa")
    p.add_argument("--compile", action="store_true")
    p.add_argument("--loss-chunk-tokens", type=int, default=0)
    p.add_argument("--data-mode", choices=["tokens", "synthetic"], default="tokens")
    p.add_argument("--train-tokens", default="")
    p.add_argument("--eval-tokens", default="")
    p.add_argument("--warmup-steps", type=int, default=0)
    p.add_argument("--steps", type=int, required=True)
    p.add_argument("--max-duration-seconds", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--run-mode", required=True)
    p.add_argument("--phase", required=True)
    p.add_argument("--eval-enabled", action="store_true")
    p.add_argument("--eval-every-steps", type=int, default=0)
    p.add_argument("--eval-batches", type=int, default=8)
    p.add_argument("--milestone-steps", default="")
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--min-learning-rate-ratio", type=float, default=0.1)
    p.add_argument("--lr-schedule", choices=["constant", "cosine"], default="cosine")
    p.add_argument("--lr-warmup-steps", type=int, default=0)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--adam-beta1", type=float, default=0.9)
    p.add_argument("--adam-beta2", type=float, default=0.95)
    p.add_argument("--grad-clip-norm", type=float, default=1.0)
    p.add_argument("--fsdp-strategy", choices=["full_shard", "hybrid_shard"], default="full_shard")
    p.add_argument("--gpus-per-node", type=int, required=True)
    p.add_argument("--peak-tflops-per-gpu", type=float, required=True)
    p.add_argument("--training-flops-per-token", type=float, default=0.0)
    p.add_argument("--checkpoint-dir", required=True)
    p.add_argument("--keep-checkpoint", action="store_true")
    p.add_argument("--resume-steps", type=int, default=2)
    p.add_argument("--loss-curve-sample-every", type=int, default=1)
    p.add_argument("--loss-curve-minimum-points", type=int, default=2)
    p.add_argument("--convergence-train-loss", type=float, default=None)
    p.add_argument("--convergence-eval-loss", type=float, default=None)
    p.add_argument("--codecarbon-enabled", action="store_true")
    p.add_argument("--codecarbon-required", action="store_true")
    p.add_argument("--codecarbon-measure-power-secs", type=int, default=5)
    p.add_argument("--codecarbon-country-iso-code", default="")
    p.add_argument("--export-hf-dir", default="")
    p.add_argument("--output", required=True)
    p.add_argument("--collective-timeout-seconds", type=int, default=600)
    p.add_argument("--startup-timeout-seconds", type=int, default=300)
    p.add_argument("--setup-timeout-seconds", type=int, default=1800)
    return p.parse_args()


# ---------------------------------------------------------------- statistics


def _percentile(values, pct):
    ordered = sorted(float(v) for v in values)
    rank = (len(ordered) - 1) * pct / 100.0
    low, high = math.floor(rank), math.ceil(rank)
    if low == high:
        return ordered[low]
    return ordered[low] + (ordered[high] - ordered[low]) * (rank - low)


def _loss_curve_decreased(losses, minimum_points):
    if len(losses) < minimum_points or not all(math.isfinite(v) for v in losses):
        return False
    x_mean = (len(losses) - 1) / 2.0
    y_mean = sum(losses) / len(losses)
    numerator = sum((i - x_mean) * (v - y_mean) for i, v in enumerate(losses))
    denominator = sum((i - x_mean) ** 2 for i in range(len(losses)))
    return denominator > 0 and numerator / denominator < 0


# ---------------------------------------------------------------------- data


class TokenWindows:
    """Fixed-length windows over a flat uint32 token file.

    The window for a global sample index is a pure function of that index, so
    replaying a step after a checkpoint reload feeds it the same tokens.
    """

    def __init__(self, path, seq_len):
        self.tokens = np.memmap(path, dtype=np.uint32, mode="r")
        self.seq_len = seq_len
        self.count = (len(self.tokens) - 1) // seq_len
        if self.count < 1:
            raise ValueError(f"{path} holds {len(self.tokens)} tokens, fewer than one {seq_len}-token window")

    def window(self, sample_index, shuffle=True):
        slot = (sample_index * _WINDOW_STRIDE) % self.count if shuffle else sample_index % self.count
        start = slot * self.seq_len
        return np.asarray(self.tokens[start : start + self.seq_len + 1], dtype=np.int64)

    def batch(self, sample_indices, device, shuffle=True):
        rows = np.stack([self.window(index, shuffle) for index in sample_indices])
        tensor = torch.from_numpy(rows).to(device, non_blocking=True)
        return tensor[:, :-1], tensor[:, 1:]


class SyntheticTokens:
    def __init__(self, vocab_size, seq_len, seed):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.seed = seed

    def batch(self, sample_indices, device, shuffle=True):
        del shuffle
        generator = torch.Generator().manual_seed(self.seed * 1_000_003 + int(sample_indices[0]))
        rows = torch.randint(0, self.vocab_size, (len(sample_indices), self.seq_len + 1), generator=generator)
        rows = rows.to(device, non_blocking=True)
        return rows[:, :-1], rows[:, 1:]


def _micro_batch_indices(step, micro, args, rank, world_size):
    first = ((step * args.gradient_accumulation_steps + micro) * world_size + rank) * args.micro_batch_size
    return list(range(first, first + args.micro_batch_size))


# --------------------------------------------------------------------- model


def _decoder_layers(model):
    return list(_layer_container(model))


def _expert_count(config):
    for name in ("num_local_experts", "num_experts", "n_routed_experts"):
        value = getattr(config, name, None)
        if value:
            return int(value)
    return 0


def training_flops_per_token(model, config, seq_len):
    """6 x active non-embedding parameters plus the attention score and value matmuls.

    Attention counts the full causal context per layer (the TorchTitan and
    Megatron convention), clipped to the window on sliding-window layers. MoE
    expert weights count at top_k / num_experts, since a token visits only
    its routed experts.
    """
    total = expert = embed = 0
    for name, param in model.named_parameters():
        count = param.numel()
        total += count
        if re.search(r"\.experts\.", name):
            expert += count
        if name.endswith("embed_tokens.weight") or name.endswith("wte.weight"):
            embed += count
    active = total
    if not getattr(config, "tie_word_embeddings", False):
        active -= embed
    experts = _expert_count(config)
    top_k = getattr(config, "num_experts_per_tok", None)
    if expert and experts and top_k:
        active -= expert * (1.0 - float(top_k) / experts)
    layers = int(config.num_hidden_layers)
    heads = int(config.num_attention_heads)
    head_dim = getattr(config, "head_dim", None) or config.hidden_size // heads
    qk_dim = (getattr(config, "qk_nope_head_dim", 0) or 0) + (getattr(config, "qk_rope_head_dim", 0) or 0)
    qk_dim = qk_dim or head_dim
    v_dim = getattr(config, "v_head_dim", None) or head_dim
    layer_types = getattr(config, "layer_types", None) or ["full_attention"] * layers
    window = getattr(config, "sliding_window", None)
    attention = 0.0
    for layer_type in layer_types:
        context = min(seq_len, window) if layer_type == "sliding_attention" and window else seq_len
        attention += 6.0 * heads * (qk_dim + v_dim) * context
    return 6.0 * active + attention, int(total), float(active)


def _float8_filter(module, fqn):
    # The output head and router stay in high precision: torchao's own
    # recipes skip them, and their shapes are rarely multiples of 16.
    if not isinstance(module, torch.nn.Linear):
        return False
    if "lm_head" in fqn or fqn.endswith(".gate") or "router" in fqn:
        return False
    return module.in_features % 16 == 0 and module.out_features % 16 == 0


def _apply_precision(model, args):
    if args.precision == "BF16":
        return {"recipe": "bf16"}
    if args.precision == "FP8":
        from torchao.float8 import Float8LinearConfig, convert_to_float8_training

        config = Float8LinearConfig.from_recipe_name("tensorwise")
        convert_to_float8_training(model, config=config, module_filter_fn=_float8_filter)
        converted = sum(1 for m in model.modules() if type(m).__name__ == "Float8Linear")
        if not converted:
            raise RuntimeError("FP8 requested but no Linear layer was converted to Float8Linear")
        return {"recipe": "torchao-float8-tensorwise", "float8_linears": converted}
    from torchao.prototype.mx_formats import MXLinearConfig
    from torchao.quantization import quantize_

    element = torch.float8_e4m3fn if args.precision == "MXFP8" else torch.float4_e2m1fn_x2
    quantize_(model, config=MXLinearConfig(elem_dtype=element, block_size=32), filter_fn=_float8_filter)
    return {"recipe": f"torchao-mx-{args.precision.lower()}"}


def _layer_container(model):
    for path in ("model.layers", "model.decoder.layers", "transformer.h"):
        parent, _, attr = path.rpartition(".")
        node = model
        for part in parent.split("."):
            node = getattr(node, part, None)
            if node is None:
                break
        if node is not None and hasattr(node, attr):
            return getattr(node, attr)
    raise RuntimeError(f"cannot locate the decoder layers of {type(model).__name__}")


def _apply_activation_checkpointing(model, every):
    """Recompute every ``every``-th decoder layer in backward; 0 disables.

    Checkpointing a subset spends spare HBM on stored activations instead of
    a second forward through every layer.
    """
    if every <= 0:
        return
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper

    layers = _layer_container(model)
    for index in range(len(layers)):
        if index % every == 0:
            layers[index] = checkpoint_wrapper(layers[index], preserve_rng_state=False)


def _has_experts(module):
    return any(hasattr(m, "experts") for m in module.modules())


def _compile_layer(layer):
    """Compile one decoder layer, leaving any MoE expert block eager.

    HF MoE blocks loop over experts in Python with data-dependent token
    counts, so compiling them recompiles on every new routing and a
    DeepSeek-V2-Lite warmup step never finished.
    """
    inner = getattr(layer, "_checkpoint_wrapped_module", layer)
    if not _has_experts(inner):
        layer.compile()
        return
    for child in inner.children():
        if not _has_experts(child):
            child.compile()


def _touch_all_experts(module, inputs, output):
    # HF MoE blocks skip experts that received no tokens, leaving their
    # gradients None on that rank only; FSDP2 then reduce-scatters a
    # different gradient set per rank and the collectives deadlock. A zero
    # term gives every expert a (zero) gradient on every rank.
    anchor = sum(p.reshape(-1)[0] for p in module.experts.parameters()) * 0.0
    if isinstance(output, tuple):
        return (output[0] + anchor.to(output[0].dtype),) + tuple(output[1:])
    return output + anchor.to(output.dtype)


def _register_expert_anchors(model):
    for module in model.modules():
        if isinstance(getattr(module, "experts", None), torch.nn.Module):
            module.register_forward_hook(_touch_all_experts)


def _init_random_weights(model, config):
    std = float(getattr(config, "initializer_range", 0.02) or 0.02)
    touched = set()
    with torch.no_grad():
        for module in model.modules():
            name = type(module).__name__.lower()
            if isinstance(module, (torch.nn.Linear, torch.nn.Embedding)) or hasattr(module, "in_features"):
                weight = getattr(module, "weight", None)
                if weight is not None:
                    weight.normal_(mean=0.0, std=std)
                    touched.add(id(weight))
                bias = getattr(module, "bias", None)
                if bias is not None:
                    bias.zero_()
                    touched.add(id(bias))
            elif "norm" in name:
                for pname, param in module.named_parameters(recurse=False):
                    if pname == "weight":
                        param.fill_(1.0)
                    else:
                        param.zero_()
                    touched.add(id(param))
        # Fused expert banks (GPT-OSS), attention sinks, and the like.
        for param in model.parameters():
            if id(param) in touched:
                continue
            if param.dim() >= 2:
                param.normal_(mean=0.0, std=std)
            else:
                param.zero_()


def _load_pretrained(model, args, rank):
    from safetensors.torch import load_file
    from torch.distributed.checkpoint.state_dict import StateDictOptions, set_model_state_dict

    state = {}
    if rank == 0:
        files = sorted(Path(args.model_path).glob("*.safetensors"))
        if not files:
            raise FileNotFoundError(f"init=pretrained but {args.model_path} has no safetensors weights")
        dtype = torch.float32 if args.master_weights == "fp32" else torch.bfloat16
        for path in files:
            for key, value in load_file(str(path)).items():
                state[key] = value.to(dtype)
    options = StateDictOptions(full_state_dict=True, broadcast_from_rank0=True, strict=False)
    result = set_model_state_dict(model, state, options=options)
    missing = [k for k in getattr(result, "missing_keys", []) if "lm_head" not in k]
    if missing:
        raise RuntimeError(f"pretrained weights are missing {len(missing)} tensors, e.g. {missing[:5]}")


def build_model(args, device, mesh, rank):
    from accelerate import init_empty_weights
    from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
    from transformers import AutoConfig, AutoModelForCausalLM

    config = AutoConfig.from_pretrained(args.model_path)
    # A quantized checkpoint's config (Llama-3.3-70B-FP8, GPT-OSS MXFP4) would
    # make from_config build inference-only quantized layers.
    if hasattr(config, "quantization_config"):
        delattr(config, "quantization_config")
    config.use_cache = False
    dtype = torch.float32 if args.master_weights == "fp32" else torch.bfloat16
    # Parameters on meta so no rank ever holds the full model; buffers (rotary
    # inv_freq) stay real on the CPU because to_empty would discard them.
    with init_empty_weights(include_buffers=False):
        model = AutoModelForCausalLM.from_config(
            config, torch_dtype=dtype, attn_implementation=args.attn_implementation
        )
    buffers = {name: buf.detach().clone() for name, buf in model.named_buffers()}
    flops, total_params, active_params = training_flops_per_token(model, config, args.seq_len)
    precision_info = _apply_precision(model, args)
    _register_expert_anchors(model)
    _apply_activation_checkpointing(model, args.activation_checkpointing_every)
    policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
    for layer in _decoder_layers(model):
        if args.compile:
            # Per layer, as TorchTitan does: one graph per block reuses the
            # compiled code for every layer and fuses the norm, rotary, and
            # gated-MLP elementwise kernels that eager HF code runs one by one.
            _compile_layer(layer)
        fully_shard(layer, mesh=mesh, mp_policy=policy)
    fully_shard(model, mesh=mesh, mp_policy=policy)
    model.to_empty(device=device)
    with torch.no_grad():
        for name, buf in model.named_buffers():
            if name in buffers:
                buf.copy_(buffers[name].to(device=buf.device, dtype=buf.dtype))
    if args.init == "pretrained":
        _load_pretrained(model, args, rank)
    else:
        torch.manual_seed(args.seed)
        _init_random_weights(model, config)
    model.train()
    info = {
        "architecture": type(model).__name__,
        "parameters_total": total_params,
        "parameters_active": active_params,
        "vocab_size": int(config.vocab_size),
        **precision_info,
    }
    return model, config, flops, info


def build_mesh(args, world_size):
    from torch.distributed.device_mesh import init_device_mesh

    nodes = world_size // args.gpus_per_node
    if args.fsdp_strategy == "hybrid_shard" and nodes > 1:
        return init_device_mesh("cuda", (nodes, args.gpus_per_node), mesh_dim_names=("replicate", "shard"))
    return init_device_mesh("cuda", (world_size,), mesh_dim_names=("shard",))


def build_optimizer(model, args):
    decay, no_decay = [], []
    for param in model.parameters():
        if not param.requires_grad:
            continue
        (decay if param.dim() >= 2 else no_decay).append(param)
    groups = [
        {"params": decay, "weight_decay": args.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(
        groups, lr=args.learning_rate, betas=(args.adam_beta1, args.adam_beta2), eps=1e-8, foreach=True
    )
    total = max(1, args.warmup_steps + args.steps)

    def factor(step):
        if args.lr_schedule == "constant":
            return 1.0
        if args.lr_warmup_steps and step < args.lr_warmup_steps:
            return float(step + 1) / args.lr_warmup_steps
        progress = min(1.0, (step - args.lr_warmup_steps) / max(1, total - args.lr_warmup_steps))
        floor = args.min_learning_rate_ratio
        return floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, factor)
    return optimizer, scheduler


# ------------------------------------------------------------------ training


def _cross_entropy_sum(logits, labels):
    return F.cross_entropy(logits.float(), labels, reduction="sum")


def _cross_entropy(logits, labels, chunk_tokens=0):
    flat = logits.reshape(-1, logits.size(-1))
    targets = labels.reshape(-1)
    if chunk_tokens <= 0 or flat.size(0) <= chunk_tokens:
        return F.cross_entropy(flat.float(), targets)
    from torch.utils.checkpoint import checkpoint

    # Recomputed in backward so only one chunk's fp32 logits and softmax
    # are alive at a time.
    total = torch.zeros((), dtype=torch.float32, device=flat.device)
    for start in range(0, flat.size(0), chunk_tokens):
        end = start + chunk_tokens
        total = total + checkpoint(_cross_entropy_sum, flat[start:end], targets[start:end], use_reentrant=False)
    return total / (targets != -100).sum().clamp(min=1)


def train_step(model, optimizer, scheduler, data, step, args, rank, world_size, device):
    accumulated = torch.zeros((), dtype=torch.float32, device=device)
    for micro in range(args.gradient_accumulation_steps):
        inputs, labels = data.batch(_micro_batch_indices(step, micro, args, rank, world_size), device)
        logits = model(input_ids=inputs, use_cache=False).logits
        loss = _cross_entropy(logits, labels, args.loss_chunk_tokens) / args.gradient_accumulation_steps
        del logits
        loss.backward()
        accumulated += loss.detach()
    if args.grad_clip_norm > 0:
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
        if hasattr(grad_norm, "full_tensor"):
            grad_norm = grad_norm.full_tensor()
    else:
        grad_norm = torch.zeros((), device=device)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    scheduler.step()
    dist.all_reduce(accumulated, op=dist.ReduceOp.AVG)
    return float(accumulated.item()), float(grad_norm)


def evaluation_windows(window_count, eval_batches, world_size):
    """Forward passes per rank over distinct held-out windows, one window each.

    Every rank must run the same number of forwards because FSDP all-gathers
    inside each one, so the window count is rounded down to a multiple of
    the world size rather than repeating windows.
    """
    per_rank = min(eval_batches, window_count // world_size)
    if per_rank < 1:
        raise ValueError(f"held-out text has {window_count} windows, fewer than the {world_size} ranks")
    return per_rank


def evaluate(model, data, args, rank, world_size, device):
    model.eval()
    totals = torch.zeros(2, dtype=torch.float64, device=device)
    with torch.no_grad():
        for batch in range(evaluation_windows(data.count, args.eval_batches, world_size)):
            index = batch * world_size + rank
            inputs, labels = data.batch([index], device, shuffle=False)
            logits = model(input_ids=inputs, use_cache=False).logits
            totals[0] += _cross_entropy(logits, labels, args.loss_chunk_tokens).double() * labels.numel()
            totals[1] += labels.numel()
            del logits
    model.train()
    dist.all_reduce(totals)
    return float(totals[0] / totals[1]), int(totals[1])


# ---------------------------------------------------------------- checkpoint


def _flatten(node, prefix=""):
    if isinstance(node, dict):
        for key in sorted(node, key=str):
            yield from _flatten(node[key], f"{prefix}/{key}")
    elif isinstance(node, (list, tuple)):
        for index, value in enumerate(node):
            yield from _flatten(value, f"{prefix}[{index}]")
    else:
        yield prefix, node


_INT_VIEW = {1: torch.uint8, 2: torch.int16, 4: torch.int32, 8: torch.int64}


def _fingerprint(state, device):
    """Exact bit-pattern digest of this rank's shard of a state dict.

    Holding a reference copy of a 70B model's fp32 weights and Adam moments
    would need about 100 GB of host memory per rank, so equality is decided
    on a positional integer checksum of the raw bits instead.
    """
    digest = torch.zeros(2, dtype=torch.int64, device=device)
    chunk = 1 << 26
    scalars = []
    for index, (path, value) in enumerate(_flatten(state)):
        if not torch.is_tensor(value):
            scalars.append(f"{path}={value!r}")
            continue
        local = value.to_local() if hasattr(value, "to_local") else value
        flat = local.detach().reshape(-1)
        if flat.numel() == 0:
            continue
        bits = flat.contiguous().view(_INT_VIEW[flat.element_size()]).to(device)
        for start in range(0, bits.numel(), chunk):
            part = bits[start : start + chunk].to(torch.int64)
            weights = torch.arange(start + 1, start + part.numel() + 1, device=device, dtype=torch.int64) % 65521
            digest[0] += part.sum() * (index + 1)
            digest[1] += (part * weights).sum()
    return digest.tolist(), scalars


def _share_dcp_metadata(directory, rank, local_rank):
    """Give every node the .metadata file that only rank 0 writes.

    Each node writes its shards to its own local disk, so the reload on a
    node other than rank 0's would find its shards but no metadata.
    """
    meta = Path(directory) / ".metadata"
    payload = [meta.read_bytes() if rank == 0 else None]
    dist.broadcast_object_list(payload, src=0)
    if local_rank == 0 and not meta.exists():
        meta.write_bytes(payload[0])
    dist.barrier()


_REPLICATED_FILE = "replicated.pt"


def _split_replicated(node):
    """Split a state dict into its DTensor shards and everything else.

    DCP deduplicates replicated entries (Adam step counts, param groups) and
    spreads them over the ranks' files, so on node-local disks a reload needs
    files that only another node holds. Saving just the shards through DCP
    keeps every rank's reads in its own file.
    """
    if isinstance(node, dict):
        sharded, replicated = {}, {}
        for key, value in node.items():
            s, r = _split_replicated(value)
            if s is not None:
                sharded[key] = s
            if r is not None:
                replicated[key] = r
        return sharded or None, replicated or None
    if hasattr(node, "to_local"):
        return node, None
    return None, node


def _merge_replicated(sharded, replicated, template):
    if isinstance(template, dict):
        return {
            key: _merge_replicated((sharded or {}).get(key), (replicated or {}).get(key), value)
            for key, value in template.items()
        }
    if hasattr(template, "to_local"):
        return sharded
    if torch.is_tensor(template):
        return replicated.to(device=template.device, dtype=template.dtype)
    return replicated


def _local_checkpoint_bytes(directory, rank):
    return sum(p.stat().st_size for p in Path(directory).glob(f"__{rank}_*.distcp"))


def checkpoint_roundtrip(model, optimizer, scheduler, data, step, args, rank, local_rank, world_size, device):
    """Save, run K steps, reload, rerun the same K steps, and compare.

    The replayed losses match the uninterrupted ones only if model weights,
    optimizer moments, step counts, and the LR schedule all came back intact.
    """
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict

    directory = Path(args.checkpoint_dir) / f"step_{step}"
    if local_rank == 0:
        shutil.rmtree(directory, ignore_errors=True)
        directory.mkdir(parents=True, exist_ok=True)
    dist.barrier()
    model_state, optim_state = get_state_dict(model, optimizer)
    before, before_scalars = _fingerprint({"model": model_state, "optim": optim_state}, device)
    schedule = scheduler.state_dict()
    torch.cuda.synchronize()
    dist.barrier()
    start = time.perf_counter()
    sharded, replicated = _split_replicated({"model": model_state, "optim": optim_state})
    # The overlapping loader stages through pinned host buffers, which ran
    # out (hipErrorOutOfMemory) on Qwen3-30B-A3B's ~55k expert tensors.
    writer = dcp.FileSystemWriter(str(directory), per_thread_copy_ahead=0)
    dcp.save(sharded, storage_writer=writer)
    if local_rank == 0:
        torch.save(replicated, directory / _REPLICATED_FILE)
    dist.barrier()
    save_seconds = time.perf_counter() - start
    _share_dcp_metadata(directory, rank, local_rank)
    size = torch.tensor([_local_checkpoint_bytes(directory, rank)], dtype=torch.float64, device=device)
    dist.all_reduce(size)
    del model_state, optim_state, sharded, replicated

    uninterrupted = [
        train_step(model, optimizer, scheduler, data, step + k, args, rank, world_size, device)[0]
        for k in range(args.resume_steps)
    ]

    torch.cuda.synchronize()
    dist.barrier()
    start = time.perf_counter()
    model_state, optim_state = get_state_dict(model, optimizer)
    template = {"model": model_state, "optim": optim_state}
    sharded, _ = _split_replicated(template)
    dcp.load(sharded, checkpoint_id=str(directory))
    replicated = torch.load(directory / _REPLICATED_FILE, map_location="cpu", weights_only=False)
    restored = _merge_replicated(sharded, replicated, template)
    set_state_dict(model, optimizer, model_state_dict=restored["model"], optim_state_dict=restored["optim"])
    scheduler.load_state_dict(schedule)
    torch.cuda.synchronize()
    dist.barrier()
    load_seconds = time.perf_counter() - start
    model_state, optim_state = get_state_dict(model, optimizer)
    after, after_scalars = _fingerprint({"model": model_state, "optim": optim_state}, device)
    del model_state, optim_state, restored, template, sharded, replicated
    verdict = torch.tensor([1.0 if (before == after and before_scalars == after_scalars) else 0.0], device=device)
    dist.all_reduce(verdict, op=dist.ReduceOp.MIN)

    replayed = [
        train_step(model, optimizer, scheduler, data, step + k, args, rank, world_size, device)[0]
        for k in range(args.resume_steps)
    ]
    timings = torch.tensor([save_seconds, load_seconds], dtype=torch.float64, device=device)
    dist.all_reduce(timings, op=dist.ReduceOp.MAX)
    dist.barrier()
    if local_rank == 0 and not args.keep_checkpoint:
        shutil.rmtree(directory, ignore_errors=True)
    return (
        {
            "checkpoint_save_seconds": float(timings[0]),
            "checkpoint_load_seconds": float(timings[1]),
            "checkpoint_io_seconds": float(timings[0] + timings[1]),
            "checkpoint_size_gb": float(size.item()) / 1e9,
            "checkpoint_state_match": float(verdict.item()),
            "checkpoint_loss_delta": max(abs(a - b) for a, b in zip(uninterrupted, replayed)),
        },
        uninterrupted,
        replayed,
    )


def export_hf(model, config, args, rank):
    """Gather full weights to rank 0 and write a HF checkpoint for lm-eval."""
    from safetensors.torch import save_file
    from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict

    state = get_model_state_dict(model, options=StateDictOptions(full_state_dict=True, cpu_offload=True))
    if rank == 0:
        out = Path(args.export_hf_dir)
        shutil.rmtree(out, ignore_errors=True)
        out.mkdir(parents=True)
        tensors = {key: value.to(torch.bfloat16).contiguous() for key, value in state.items()}
        if getattr(config, "tie_word_embeddings", False):
            tensors.pop("lm_head.weight", None)
        save_file(tensors, str(out / "model.safetensors"), metadata={"format": "pt"})
        config.save_pretrained(out)
        for item in Path(args.model_path).iterdir():
            if item.name.startswith(("tokenizer", "special_tokens", "vocab", "merges", "generation_config")):
                shutil.copy2(item, out / item.name)
    del state
    dist.barrier()


# ---------------------------------------------------------------- codecarbon


# The tracker polls AMDSMI from a background thread. Inside rank 0 that thread
# corrupted the trainer's heap within minutes on long MoE and multi-node runs,
# so it runs in its own process and only exchanges JSON lines with the trainer.
_CODECARBON_WORKER = """
import json, os, sys
result = os.fdopen(os.dup(1), "w")
os.dup2(2, 1)
import amdsmi
from codecarbon import EmissionsTracker
amdsmi.amdsmi_init()
count = len(amdsmi.amdsmi_get_processor_handles())
tracker = EmissionsTracker(**json.loads(sys.argv[1]))
tracker.start()
print(json.dumps({"gpu_count": count}), file=result, flush=True)
sys.stdin.read()
emissions = tracker.stop()
energy = getattr(getattr(tracker, "final_emissions_data", None), "energy_consumed", None)
print(json.dumps({"emissions": emissions, "energy": energy}), file=result, flush=True)
"""


def _read_worker_line(proc, timeout_s):
    ready, _, _ = select.select([proc.stdout], [], [], timeout_s)
    line = proc.stdout.readline() if ready else ""
    if not line:
        proc.kill()
        raise RuntimeError(f"CodeCarbon worker gave no result (exit code {proc.wait()})")
    return json.loads(line)


def _start_codecarbon(args, rank, local_gpu_count):
    payload = {"version": None, "tracker": "amdsmi", "gpu_count": 0, "active": False, "reason": "disabled"}
    tracker = None
    if rank == 0 and args.codecarbon_enabled:
        try:
            version = importlib.metadata.version("codecarbon")
            if version != "3.2.4":
                raise RuntimeError(f"CodeCarbon 3.2.4 required, found {version}")
            kwargs = {
                "measure_power_secs": args.codecarbon_measure_power_secs,
                "tracking_mode": "machine",
                "gpu_ids": list(range(local_gpu_count)),
                "log_level": "error",
                "save_to_file": False,
            }
            if args.codecarbon_country_iso_code:
                kwargs["country_iso_code"] = args.codecarbon_country_iso_code
            tracker = subprocess.Popen(
                [sys.executable, "-c", _CODECARBON_WORKER, json.dumps(kwargs)],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                text=True,
            )
            gpu_count = _read_worker_line(tracker, CODECARBON_WORKER_TIMEOUT_S)["gpu_count"]
            if gpu_count < local_gpu_count:
                raise RuntimeError(f"AMDSMI found {gpu_count} GPUs, expected {local_gpu_count}")
            payload.update({"version": version, "gpu_count": gpu_count, "active": True, "reason": ""})
        except Exception as exc:
            if tracker is not None:
                tracker.kill()
                tracker = None
            payload["reason"] = f"{type(exc).__name__}: {exc}"
    state = [payload]
    dist.broadcast_object_list(state, src=0)
    if args.codecarbon_enabled and args.codecarbon_required and not state[0]["active"]:
        raise RuntimeError(f"required CodeCarbon tracking unavailable: {state[0]['reason']}")
    return tracker, state[0]


def _stop_codecarbon(tracker, payload, rank):
    if rank == 0 and tracker is not None:
        try:
            tracker.stdin.close()
            result = _read_worker_line(tracker, CODECARBON_WORKER_TIMEOUT_S)
            tracker.wait(timeout=CODECARBON_WORKER_TIMEOUT_S)
            emissions, energy = result["emissions"], result["energy"]
            if emissions is None or not math.isfinite(float(emissions)):
                raise RuntimeError(f"invalid emissions result: {emissions!r}")
            payload["emissions_kg_co2eq"] = float(emissions)
            if energy is not None and math.isfinite(float(energy)) and float(energy) > 0:
                payload["energy_kwh"] = float(energy)
        except Exception as exc:
            tracker.kill()
            payload.update({"active": False, "reason": f"{type(exc).__name__}: {exc}"})
    state = [payload]
    dist.broadcast_object_list(state, src=0)
    return state[0]


# ----------------------------------------------------------- process control


def _start_distributed(device, local_rank, collective_timeout_s, startup_timeout_s):
    """Bring up NCCL and prove it with one all-reduce, or leave at once.

    About one 32-rank start in ten on MI325X segfaults a rank inside
    libdrm_amdgpu during the NCCL connect, and the survivors then wait in
    teardown until the launch timeout. STARTUP_FAILED lets the launcher tell
    a failed start from a failed run and relaunch.
    """

    def fail(reason):
        print(f"{STARTUP_FAILED} local_rank={local_rank} {reason}", flush=True)
        sys.stderr.flush()
        os._exit(1)

    deadline = threading.Timer(startup_timeout_s, fail, args=(f"no progress in {startup_timeout_s}s",))
    deadline.daemon = True
    deadline.start()
    try:
        dist.init_process_group(
            backend="nccl", device_id=device, timeout=datetime.timedelta(seconds=collective_timeout_s)
        )
        probe = torch.ones(1, device=device)
        dist.all_reduce(probe)
        torch.cuda.synchronize()
    except Exception as exc:
        fail(f"{type(exc).__name__}: {exc}".splitlines()[0])
    deadline.cancel()
    if local_rank == 0:
        print(STARTUP_OK, flush=True)


def _wait_for_all_ranks(name, timeout_s):
    """Meet every rank through the rendezvous store rather than through NCCL.

    Rank 0 reading 140 GB of pretrained weights takes minutes while the other
    ranks are idle; waiting here keeps them out of the collective timeout.
    """
    store = dist.distributed_c10d._get_default_store()
    key = f"pytorch_training_ready/{name}"
    store.add(key, 1)
    deadline = time.monotonic() + timeout_s
    while store.add(key, 0) < dist.get_world_size():
        if time.monotonic() > deadline:
            raise RuntimeError(f"not every rank finished {name} within {timeout_s}s")
        time.sleep(0.5)


def _finish(rank, world_size, artifact_error, timeout_s):
    """Wait until every rank is done with the GPU, then leave.

    A rank that exits straight after its last collective can cut off data its
    peers are still receiving, so exit is coordinated through the store.
    """
    store = dist.distributed_c10d._get_default_store()
    if rank == 0:
        store.set("cvs_artifact_error", artifact_error)
    torch.cuda.synchronize()
    store.add("cvs_ranks_done", 1)
    deadline = time.monotonic() + timeout_s
    while store.add("cvs_ranks_done", 0) < world_size:
        if time.monotonic() > deadline:
            raise RuntimeError(f"ranks did not all finish within {timeout_s}s of rank {rank}")
        time.sleep(0.2)
    error = store.get("cvs_artifact_error").decode()
    if error:
        raise RuntimeError(f"result artifact failed on rank 0: {error}")
    # HIP static destructors segfault a rank at interpreter exit after a clean
    # run; the artifact is already on disk.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


def _versions():
    out = {"torch": torch.__version__, "hip": getattr(torch.version, "hip", None)}
    for name in ("transformers", "torchao", "accelerate"):
        with contextlib.suppress(importlib.metadata.PackageNotFoundError):
            out[name] = importlib.metadata.version(name)
    return out


# ---------------------------------------------------------------------- main


def run(args, rank, local_rank, world_size, device):
    torch.manual_seed(args.seed)
    mesh = build_mesh(args, world_size)
    setup_start = time.perf_counter()
    model, config, derived_flops, model_info = build_model(args, device, mesh, rank)
    _wait_for_all_ranks("model", args.setup_timeout_seconds)
    setup_seconds = time.perf_counter() - setup_start
    flops_per_token = args.training_flops_per_token or derived_flops
    optimizer, scheduler = build_optimizer(model, args)
    if args.data_mode == "tokens":
        data = TokenWindows(args.train_tokens, args.seq_len)
        eval_data = TokenWindows(args.eval_tokens, args.seq_len) if args.eval_enabled else None
    else:
        data = SyntheticTokens(model_info["vocab_size"], args.seq_len, args.seed)
        eval_data = None

    tokens_per_step = args.micro_batch_size * args.seq_len * args.gradient_accumulation_steps * world_size
    milestones = [int(s) for s in args.milestone_steps.split(",") if s]
    loss_series, evaluations, step_times = [], [], []
    learning_rates = []
    step = 0

    for _ in range(args.warmup_steps):
        train_step(model, optimizer, scheduler, data, step, args, rank, world_size, device)
        step += 1
    torch.cuda.synchronize()
    dist.barrier()
    if rank == 0:
        print(WARMUP_OK, flush=True)

    tracker, codecarbon = _start_codecarbon(args, rank, args.gpus_per_node)
    train_start = time.perf_counter()
    measured = 0
    while True:
        if args.run_mode == "soak":
            if time.perf_counter() - train_start >= args.max_duration_seconds:
                break
        elif measured >= args.steps:
            break
        elif args.max_duration_seconds and time.perf_counter() - train_start >= args.max_duration_seconds:
            break
        learning_rates.append(scheduler.get_last_lr()[0])
        begin = time.perf_counter()
        loss, grad_norm = train_step(model, optimizer, scheduler, data, step, args, rank, world_size, device)
        elapsed = time.perf_counter() - begin
        step += 1
        measured += 1
        step_times.append(elapsed)
        if not math.isfinite(loss):
            raise RuntimeError(f"training loss became {loss} at step {step}")
        if measured == 1 or measured % args.loss_curve_sample_every == 0 or step in milestones:
            loss_series.append(
                {
                    "step": step,
                    "loss": loss,
                    "grad_norm": grad_norm,
                    "time_seconds": time.perf_counter() - train_start,
                }
            )
        if rank == 0 and (measured <= 3 or measured % 10 == 0):
            print(f"step {step} loss {loss:.4f} grad_norm {grad_norm:.3f} step_ms {elapsed * 1000:.1f}", flush=True)
        if eval_data is not None and args.eval_every_steps and step % args.eval_every_steps == 0:
            eval_loss, _ = evaluate(model, eval_data, args, rank, world_size, device)
            evaluations.append(
                {"step": step, "eval_loss": eval_loss, "time_seconds": time.perf_counter() - train_start}
            )
    train_seconds = time.perf_counter() - train_start
    codecarbon = _stop_codecarbon(tracker, codecarbon, rank)

    peak = torch.tensor(
        [torch.cuda.max_memory_allocated(device), torch.cuda.max_memory_reserved(device)],
        dtype=torch.float64,
        device=device,
    )
    dist.all_reduce(peak, op=dist.ReduceOp.MAX)
    measured_seconds = sum(step_times)
    tokens_per_sec = tokens_per_step * measured / measured_seconds
    tflops_per_gpu = tokens_per_sec / world_size * flops_per_token / 1e12
    losses = [p["loss"] for p in loss_series]
    metrics = {
        "tokens_per_sec": tokens_per_sec,
        "tokens_per_sec_per_gpu": tokens_per_sec / world_size,
        "tflops_per_sec_per_gpu": tflops_per_gpu,
        "mfu_pct": tflops_per_gpu / args.peak_tflops_per_gpu * 100.0,
        "step_time_ms_mean": measured_seconds / measured * 1000.0,
        "step_time_ms_p50": _percentile(step_times, 50) * 1000.0,
        "step_time_ms_p95": _percentile(step_times, 95) * 1000.0,
        "peak_memory_allocated_mb": float(peak[0]) / 2**20,
        "peak_memory_reserved_mb": float(peak[1]) / 2**20,
        "loss_initial": losses[0],
        "loss_final": losses[-1],
        "learning_rate_initial": learning_rates[0],
        "learning_rate_final": scheduler.get_last_lr()[0],
        "loss_curve_points": float(len(losses)),
        "loss_curve_decreased": 1.0 if _loss_curve_decreased(losses, args.loss_curve_minimum_points) else 0.0,
    }
    by_step = {p["step"]: p["loss"] for p in loss_series}
    for milestone in milestones:
        if milestone in by_step:
            metrics[f"loss_step_{milestone}"] = by_step[milestone]

    if eval_data is not None:
        eval_loss, eval_tokens = evaluate(model, eval_data, args, rank, world_size, device)
        evaluations.append({"step": step, "eval_loss": eval_loss, "time_seconds": train_seconds})
        metrics.update(
            {
                "eval_loss": eval_loss,
                "perplexity": math.exp(min(eval_loss, 700.0)),
                "eval_tokens": float(eval_tokens),
                "eval_completed": 1.0,
            }
        )
    hit = None
    if args.convergence_train_loss is not None or args.convergence_eval_loss is not None:
        candidates = []
        if args.convergence_train_loss is not None:
            first = next((p for p in loss_series if p["loss"] <= args.convergence_train_loss), None)
            candidates.append(first)
        if args.convergence_eval_loss is not None:
            first = next((e for e in evaluations if e["eval_loss"] <= args.convergence_eval_loss), None)
            candidates.append(first)
        if all(c is not None for c in candidates):
            hit = max(candidates, key=lambda c: c["step"])
    if hit is not None:
        metrics["convergence_step"] = float(hit["step"])
        metrics["convergence_time_seconds"] = float(hit["time_seconds"])

    checkpoint, uninterrupted, replayed = checkpoint_roundtrip(
        model, optimizer, scheduler, data, step, args, rank, local_rank, world_size, device
    )
    metrics.update(checkpoint)
    if args.export_hf_dir:
        export_hf(model, config, args, rank)

    return {
        "workload": args.workload,
        "model": args.model,
        "precision": args.precision,
        "init": args.init,
        "seq_len": args.seq_len,
        "micro_batch_size": args.micro_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "global_batch_size": args.micro_batch_size * args.gradient_accumulation_steps * world_size,
        "world_size": world_size,
        "num_nodes": world_size // args.gpus_per_node,
        "phase": args.phase,
        "run_mode": args.run_mode,
        "data_mode": args.data_mode,
        "fsdp_strategy": args.fsdp_strategy,
        "activation_checkpointing_every": args.activation_checkpointing_every,
        "master_weights": args.master_weights,
        "attn_implementation": args.attn_implementation,
        "loss_chunk_tokens": args.loss_chunk_tokens,
        "training_flops_per_token": flops_per_token,
        "derived_flops_per_token": derived_flops,
        "peak_tflops_per_gpu": args.peak_tflops_per_gpu,
        "completed_steps": measured,
        "warmup_steps": args.warmup_steps,
        "processed_tokens": tokens_per_step * measured,
        "train_seconds": train_seconds,
        "setup_seconds": setup_seconds,
        "model_info": model_info,
        "versions": _versions(),
        "codecarbon": codecarbon,
        "checkpoint_losses": {"uninterrupted": uninterrupted, "replayed": replayed},
        "loss_time_series": loss_series,
        "evaluations": evaluations,
        "metrics": metrics,
    }


def main():
    args = _parse_args()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    _start_distributed(device, local_rank, args.collective_timeout_seconds, args.startup_timeout_seconds)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    artifact = run(args, rank, local_rank, world_size, device)
    artifact_error = ""
    if rank == 0:
        try:
            out = Path(args.output)
            out.parent.mkdir(parents=True, exist_ok=True)
            tmp = out.with_suffix(".tmp")
            tmp.write_text(json.dumps(artifact, indent=2, sort_keys=True))
            os.replace(tmp, out)
            print(RUN_OK + json.dumps(artifact["metrics"]), flush=True)
        except Exception as exc:
            artifact_error = f"{type(exc).__name__}: {exc}"
    _finish(rank, world_size, artifact_error, args.collective_timeout_seconds)


if __name__ == "__main__":
    main()
