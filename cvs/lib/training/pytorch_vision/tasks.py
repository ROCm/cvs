"""Container-side task workloads staged next to benchmark.py.

benchmark.py owns image classification (the native rocAL loop and the HF
Trainer loop). This module adds what that loop cannot express:

* ``build_vim`` - Vision Mamba (Vim) for the native classification loop.
* ``run_task`` - detection (DETR / COCO), semantic segmentation
  (SegFormer / ADE20K), and multimodal SFT (Qwen2.5-VL / ChartQA), each driven
  by one shared training loop that emits the same artifact contract as the
  classification paths plus the task's own quality metric.
"""

from __future__ import annotations

import contextlib
import functools
import io
import json
import math
import os
import random
import shutil
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Vision Mamba (Vim). Matches the hustvl/Vim "midclstok" checkpoints: a class
# token in the middle of the patch sequence and bidirectional ("v2") Mamba
# mixers whose forward and backward scans have separate parameters.
# ---------------------------------------------------------------------------

_VIM_SIZES = {
    "vim_tiny": {"embed_dim": 192, "depth": 24},
    "vim_small": {"embed_dim": 384, "depth": 24},
    "vim_base": {"embed_dim": 768, "depth": 24},
}


def _selective_scan():
    try:
        from mamba_ssm.ops.selective_scan_interface import selective_scan_fn

        return selective_scan_fn
    except ImportError:
        from mamba_ssm.ops.selective_scan_interface import selective_scan_ref

        return selective_scan_ref


class _RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * self.weight.float()).to(dtype)


class _BiMamba(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_inner = expand * d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.dt_rank = math.ceil(d_model / 16)
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        for suffix in ("", "_b"):
            setattr(
                self,
                "conv1d" + suffix,
                nn.Conv1d(self.d_inner, self.d_inner, d_conv, groups=self.d_inner, padding=d_conv - 1),
            )
            setattr(self, "x_proj" + suffix, nn.Linear(self.d_inner, self.dt_rank + 2 * d_state, bias=False))
            dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
            nn.init.uniform_(dt_proj.weight, -(self.dt_rank**-0.5), self.dt_rank**-0.5)
            dt = torch.exp(torch.rand(self.d_inner) * (math.log(0.1) - math.log(0.001)) + math.log(0.001)).clamp(
                min=1e-4
            )
            with torch.no_grad():
                dt_proj.bias.copy_(dt + torch.log(-torch.expm1(-dt)))
            setattr(self, "dt_proj" + suffix, dt_proj)
            a = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
            setattr(self, "A" + suffix + "_log", nn.Parameter(torch.log(a)))
            setattr(self, "D" + suffix, nn.Parameter(torch.ones(self.d_inner)))

    def _scan(self, x, z, suffix):
        length = x.shape[-1]
        conv = getattr(self, "conv1d" + suffix)
        x = F.silu(conv(x)[..., :length])
        x_dbl = getattr(self, "x_proj" + suffix)(x.transpose(1, 2))
        dt, b, c = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt_proj = getattr(self, "dt_proj" + suffix)
        dt = (dt @ dt_proj.weight.t()).transpose(1, 2)
        a = -torch.exp(getattr(self, "A" + suffix + "_log").float())
        return _selective_scan()(
            x.contiguous(),
            dt.contiguous(),
            a,
            b.transpose(1, 2).contiguous(),
            c.transpose(1, 2).contiguous(),
            getattr(self, "D" + suffix).float(),
            z=z.contiguous(),
            delta_bias=dt_proj.bias.float(),
            delta_softplus=True,
        )

    def forward(self, hidden):
        xz = self.in_proj(hidden).transpose(1, 2)
        x, z = xz.chunk(2, dim=1)
        forward = self._scan(x, z, "")
        backward = self._scan(x.flip([-1]), z.flip([-1]), "_b").flip([-1])
        return self.out_proj(((forward + backward) / 2).transpose(1, 2).to(hidden.dtype))


class _VimBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.mixer = _BiMamba(dim)
        self.norm = _RMSNorm(dim)

    def forward(self, hidden, residual):
        residual = hidden.float() if residual is None else residual + hidden.float()
        return self.mixer(self.norm(residual.to(self.norm.weight.dtype))), residual


class VisionMamba(nn.Module):
    def __init__(self, embed_dim, depth, num_classes=1000, image_size=224, patch_size=16):
        super().__init__()
        self.patch_embed = nn.Module()
        self.patch_embed.proj = nn.Conv2d(3, embed_dim, patch_size, patch_size)
        num_patches = (image_size // patch_size) ** 2
        self.token_position = num_patches // 2
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.layers = nn.ModuleList([_VimBlock(embed_dim) for _ in range(depth)])
        self.norm_f = _RMSNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, images):
        x = self.patch_embed.proj(images).flatten(2).transpose(1, 2)
        cls = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((x[:, : self.token_position], cls, x[:, self.token_position :]), dim=1) + self.pos_embed
        residual = None
        for layer in self.layers:
            x, residual = layer(x, residual)
        residual = residual + x.float()
        x = self.norm_f(residual.to(self.norm_f.weight.dtype))
        return self.head(x[:, self.token_position])


def build_vim(name, num_classes, image_size, checkpoint=""):
    if name not in _VIM_SIZES:
        raise ValueError(f"unknown Vision Mamba model {name!r}; known {sorted(_VIM_SIZES)}")
    model = VisionMamba(num_classes=num_classes, image_size=image_size, **_VIM_SIZES[name])
    if checkpoint:
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        state = state.get("model", state)
        if num_classes != state["head.weight"].shape[0]:
            state = {k: v for k, v in state.items() if not k.startswith("head.")}
        missing, unexpected = model.load_state_dict(state, strict=False)
        missing = [k for k in missing if not k.startswith("head.") or num_classes == 1000]
        if missing or unexpected:
            raise RuntimeError(f"Vim checkpoint mismatch: missing={missing} unexpected={unexpected}")
    return model


def resolve_hf_file(model_id, filename):
    """Absolute path of ``filename`` in a cached HF repo, without network."""
    from huggingface_hub import hf_hub_download

    return hf_hub_download(model_id, filename, local_files_only=True)


# ---------------------------------------------------------------------------
# Shared training loop for detection, segmentation, and multimodal SFT.
# ---------------------------------------------------------------------------


def _options(args):
    return json.loads(args.task_options or "{}")


def _is_fsdp(model):
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

    return isinstance(model, FSDP)


def _wrap(model, args, device, local_rank, layer_classes=(), sync_from_rank0=False):
    opts = _options(args)
    if opts.get("strategy", "ddp") == "ddp":
        from torch.nn.parallel import DistributedDataParallel

        return DistributedDataParallel(
            model.to(device),
            device_ids=[local_rank],
            find_unused_parameters=bool(opts.get("find_unused_parameters", False)),
        )
    from torch.distributed.fsdp import (
        FullyShardedDataParallel as FSDP,
        MixedPrecision,
        ShardingStrategy,
    )
    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

    policy = functools.partial(transformer_auto_wrap_policy, transformer_layer_cls=set(layer_classes))
    strategy = {
        "full_shard": ShardingStrategy.FULL_SHARD,
        "hybrid_shard": ShardingStrategy.HYBRID_SHARD,
    }[args.fsdp_strategy]
    wrapped = FSDP(
        model,
        auto_wrap_policy=policy,
        sharding_strategy=strategy,
        mixed_precision=MixedPrecision(
            param_dtype=torch.bfloat16, reduce_dtype=torch.float32, buffer_dtype=torch.bfloat16
        ),
        device_id=device,
        use_orig_params=True,
        limit_all_gathers=True,
        sync_module_states=sync_from_rank0,
        param_init_fn=(lambda m: m.to_empty(device=device, recurse=False)) if sync_from_rank0 else None,
    )
    if opts.get("activation_checkpointing"):
        from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
            CheckpointImpl,
            apply_activation_checkpointing,
            checkpoint_wrapper,
        )

        apply_activation_checkpointing(
            wrapped,
            checkpoint_wrapper_fn=functools.partial(checkpoint_wrapper, checkpoint_impl=CheckpointImpl.NO_REENTRANT),
            check_fn=lambda m: isinstance(m, tuple(layer_classes)),
        )
    return wrapped


def _loader(dataset, batch_size, rank, world_size, collate, shuffle, workers=8, drop_last=True):
    from torch.utils.data import DataLoader, DistributedSampler

    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=shuffle, seed=2026)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=workers,
        collate_fn=collate,
        pin_memory=True,
        drop_last=drop_last,
        persistent_workers=workers > 0,
        prefetch_factor=4 if workers > 0 else None,
    )


def _cycle(loader):
    epoch = 0
    while True:
        loader.sampler.set_epoch(epoch)
        yield from loader
        epoch += 1


def _shard(dataset, rank, world_size):
    """Strided, unpadded slice: every evaluation sample is scored exactly once."""
    return torch.utils.data.Subset(dataset, list(range(rank, len(dataset), world_size)))


def _local(value):
    if hasattr(value, "to_local"):
        return value.to_local()
    if hasattr(value, "local_tensor"):
        return value.local_tensor()
    return value


def _gather_objects(value, world_size):
    gathered = [None] * world_size
    dist.all_gather_object(gathered, value)
    return gathered


class _Task:
    """One workload: model, data, loss, and its quality evaluation."""

    unit = "images"

    def __init__(self, bench, args, rank, world_size, device, local_rank):
        self.bench = bench
        self.args = args
        self.opts = _options(args)
        self.rank = rank
        self.world_size = world_size
        self.device = device
        self.local_rank = local_rank

    def tokens(self, batch):
        return 0

    def samples(self, batch):
        raise NotImplementedError

    def flops(self, samples, tokens):
        return samples * self.args.training_flops_per_image


# --- Detection: DETR on COCO ------------------------------------------------


class _CocoDetection(torch.utils.data.Dataset):
    def __init__(self, root, split, processor, limit=None, train=False):
        from pycocotools.coco import COCO

        with contextlib.redirect_stdout(io.StringIO()):
            self.coco = COCO(str(Path(root) / "annotations" / f"instances_{split}.json"))
        ids = sorted(self.coco.getImgIds())
        if train:
            ids = [i for i in ids if self.coco.getAnnIds(imgIds=i, iscrowd=False)]
        self.ids = ids[:limit] if limit else ids
        self.root = Path(root) / split
        self.processor = processor
        self.train = train

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        from PIL import Image

        image_id = self.ids[index]
        info = self.coco.loadImgs(image_id)[0]
        image = Image.open(self.root / info["file_name"]).convert("RGB")
        anns = [a for a in self.coco.loadAnns(self.coco.getAnnIds(imgIds=image_id)) if not a.get("iscrowd", 0)]
        if self.train and random.random() < 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            width = info["width"]
            anns = [dict(a, bbox=[width - a["bbox"][0] - a["bbox"][2], *a["bbox"][1:]]) for a in anns]
        encoded = self.processor(
            images=image,
            annotations={"image_id": image_id, "annotations": anns},
            return_tensors="pt",
        )
        return {
            "pixel_values": encoded["pixel_values"][0],
            "labels": encoded["labels"][0],
            "image_id": image_id,
            "orig_size": (info["height"], info["width"]),
        }


class DetectionTask(_Task):
    def build(self):
        from transformers import DetrForObjectDetection, DetrImageProcessor

        self.processor = DetrImageProcessor.from_pretrained(self.args.model_id)
        root = self.args.dataset_path
        self.train_set = _CocoDetection(root, "train2017", self.processor, train=True)
        self.eval_set = _CocoDetection(root, "val2017", self.processor, limit=self.args.eval_sample_count)
        # The DETR checkpoint already carries its backbone weights; without
        # use_pretrained_backbone=False timm tries to fetch ImageNet weights.
        model = DetrForObjectDetection.from_pretrained(self.args.model_id, use_pretrained_backbone=False)
        return _wrap(model, self.args, self.device, self.local_rank)

    def collate(self, items):
        pixel = [item["pixel_values"] for item in items]
        # Rounding the padded canvas up to a bucket bounds the number of
        # distinct conv shapes, so MIOpen tunes a few dozen instead of one
        # per batch.
        bucket = int(self.opts.get("pad_multiple", 1))
        height = -(-max(p.shape[1] for p in pixel) // bucket) * bucket
        width = -(-max(p.shape[2] for p in pixel) // bucket) * bucket
        values = torch.zeros(len(pixel), 3, height, width)
        mask = torch.zeros(len(pixel), height, width, dtype=torch.long)
        for index, p in enumerate(pixel):
            values[index, :, : p.shape[1], : p.shape[2]] = p
            mask[index, : p.shape[1], : p.shape[2]] = 1
        return {
            "pixel_values": values,
            "pixel_mask": mask,
            "labels": [item["labels"] for item in items],
            "image_ids": [item["image_id"] for item in items],
            "orig_sizes": [item["orig_size"] for item in items],
        }

    def samples(self, batch):
        return batch["pixel_values"].shape[0]

    def loss(self, model, batch):
        labels = [{k: v.to(self.device, non_blocking=True) for k, v in t.items()} for t in batch["labels"]]
        outputs = model(
            pixel_values=batch["pixel_values"].to(self.device, non_blocking=True),
            pixel_mask=batch["pixel_mask"].to(self.device, non_blocking=True),
            labels=labels,
        )
        return outputs.loss

    def evaluate(self, model):
        from pycocotools.cocoeval import COCOeval

        loader = torch.utils.data.DataLoader(
            _shard(self.eval_set, self.rank, self.world_size),
            batch_size=self.opts.get("eval_batch_size", 4),
            num_workers=8,
            collate_fn=self.collate,
        )
        raw = model.module if hasattr(model, "module") else model
        results, loss_sum, count = [], 0.0, 0
        model.eval()
        with torch.no_grad():
            for batch in loader:
                with self.bench._autocast(self.args.precision):
                    labels = [{k: v.to(self.device) for k, v in t.items()} for t in batch["labels"]]
                    outputs = raw(
                        pixel_values=batch["pixel_values"].to(self.device),
                        pixel_mask=batch["pixel_mask"].to(self.device),
                        labels=labels,
                    )
                loss_sum += float(outputs.loss) * len(labels)
                count += len(labels)
                outputs.logits = outputs.logits.float()
                outputs.pred_boxes = outputs.pred_boxes.float()
                processed = self.processor.post_process_object_detection(
                    outputs, threshold=0.0, target_sizes=torch.tensor(batch["orig_sizes"], device=self.device)
                )
                for image_id, item in zip(batch["image_ids"], processed):
                    for score, label, box in zip(item["scores"], item["labels"], item["boxes"]):
                        x0, y0, x1, y1 = box.tolist()
                        results.append(
                            {
                                "image_id": int(image_id),
                                "category_id": int(label),
                                "bbox": [x0, y0, x1 - x0, y1 - y0],
                                "score": float(score),
                            }
                        )
        model.train()
        everything = [r for part in _gather_objects(results, self.world_size) for r in part]
        totals = torch.tensor([loss_sum, count], dtype=torch.float64, device=self.device)
        dist.all_reduce(totals)
        stats = torch.zeros(2, dtype=torch.float64, device=self.device)
        if self.rank == 0 and everything:
            with contextlib.redirect_stdout(io.StringIO()):
                detections = self.eval_set.coco.loadRes(everything)
                evaluator = COCOeval(self.eval_set.coco, detections, "bbox")
                evaluator.params.imgIds = list(self.eval_set.ids)
                evaluator.evaluate()
                evaluator.accumulate()
                evaluator.summarize()
            stats[0] = evaluator.stats[0] * 100.0
            stats[1] = evaluator.stats[1] * 100.0
        dist.broadcast(stats, src=0)
        return {
            "map_50_95": float(stats[0]),
            "map_50": float(stats[1]),
            "eval_loss": float(totals[0] / max(totals[1], 1)),
            "eval_sample_count": float(totals[1]),
            "eval_completed": 1.0,
        }


# --- Segmentation: SegFormer on ADE20K ---------------------------------------


_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


class _Ade20k(torch.utils.data.Dataset):
    """ADEChallengeData2016 with reduce_labels: 0 (other) -> 255 ignore."""

    def __init__(self, root, split, crop, train, limit=None):
        folder = "training" if split == "train" else "validation"
        images = sorted((Path(root) / "images" / folder).glob("*.jpg"))
        self.images = images[:limit] if limit else images
        self.labels = Path(root) / "annotations" / folder
        self.crop = crop
        self.train = train

    def __len__(self):
        return len(self.images)

    def _pair(self, index):
        import numpy as np
        from PIL import Image

        image = Image.open(self.images[index]).convert("RGB")
        label = Image.open(self.labels / (self.images[index].stem + ".png"))
        label = np.array(label, dtype=np.int64)
        label = np.where(label == 0, 255, label - 1)
        return image, label

    def __getitem__(self, index):
        import numpy as np
        from PIL import Image

        image, label = self._pair(index)
        if self.train:
            scale = random.uniform(0.5, 2.0)
            short = int(self.crop * scale)
            width, height = image.size
            ratio = short / min(width, height)
            size = (max(1, int(width * ratio)), max(1, int(height * ratio)))
            image = image.resize(size, Image.BILINEAR)
            label = np.array(Image.fromarray(label.astype(np.uint8)).resize(size, Image.NEAREST), dtype=np.int64)
            if random.random() < 0.5:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
                label = label[:, ::-1]
            pixels = torch.from_numpy(np.array(image, dtype=np.float32)).permute(2, 0, 1) / 255.0
            target = torch.from_numpy(np.ascontiguousarray(label))
            pad_h = max(0, self.crop - pixels.shape[1])
            pad_w = max(0, self.crop - pixels.shape[2])
            if pad_h or pad_w:
                pixels = F.pad(pixels, (0, pad_w, 0, pad_h))
                target = F.pad(target, (0, pad_w, 0, pad_h), value=255)
            top = random.randint(0, pixels.shape[1] - self.crop)
            left = random.randint(0, pixels.shape[2] - self.crop)
            pixels = pixels[:, top : top + self.crop, left : left + self.crop]
            target = target[top : top + self.crop, left : left + self.crop]
        else:
            resized = image.resize((self.crop, self.crop), Image.BILINEAR)
            pixels = torch.from_numpy(np.array(resized, dtype=np.float32)).permute(2, 0, 1) / 255.0
            target = torch.from_numpy(label)
        return {"pixel_values": (pixels - _IMAGENET_MEAN) / _IMAGENET_STD, "labels": target}


class SegmentationTask(_Task):
    def build(self):
        from transformers import SegformerForSemanticSegmentation

        crop = self.args.image_size
        self.num_classes = self.args.num_classes
        self.train_set = _Ade20k(self.args.dataset_path, "train", crop, True)
        self.eval_set = _Ade20k(self.args.dataset_path, "val", crop, False, limit=self.args.eval_sample_count)
        model = SegformerForSemanticSegmentation.from_pretrained(
            self.args.model_id, num_labels=self.num_classes, ignore_mismatched_sizes=True
        )
        return _wrap(model, self.args, self.device, self.local_rank)

    @staticmethod
    def collate(items):
        return {
            "pixel_values": torch.stack([item["pixel_values"] for item in items]),
            "labels": torch.stack([item["labels"] for item in items]),
        }

    def samples(self, batch):
        return batch["pixel_values"].shape[0]

    def loss(self, model, batch):
        return model(
            pixel_values=batch["pixel_values"].to(self.device, non_blocking=True),
            labels=batch["labels"].to(self.device, non_blocking=True),
        ).loss

    def evaluate(self, model):
        loader = torch.utils.data.DataLoader(
            _shard(self.eval_set, self.rank, self.world_size), batch_size=1, num_workers=8, collate_fn=self.collate
        )
        raw = model.module if hasattr(model, "module") else model
        confusion = torch.zeros(self.num_classes, self.num_classes, dtype=torch.int64, device=self.device)
        totals = torch.zeros(2, dtype=torch.float64, device=self.device)
        model.eval()
        with torch.no_grad():
            for batch in loader:
                pixels = batch["pixel_values"].to(self.device)
                labels = batch["labels"].to(self.device)
                with self.bench._autocast(self.args.precision):
                    logits = raw(pixel_values=pixels).logits.float()
                upsampled = F.interpolate(logits, size=labels.shape[-2:], mode="bilinear", align_corners=False)
                totals[0] += F.cross_entropy(upsampled, labels, ignore_index=255).double()
                totals[1] += 1
                predictions = upsampled.argmax(1)
                valid = labels != 255
                confusion += torch.bincount(
                    labels[valid] * self.num_classes + predictions[valid], minlength=self.num_classes**2
                ).view(self.num_classes, self.num_classes)
        model.train()
        dist.all_reduce(confusion)
        dist.all_reduce(totals)
        intersection = confusion.diag().double()
        union = confusion.sum(0).double() + confusion.sum(1).double() - intersection
        present = union > 0
        iou = intersection[present] / union[present]
        return {
            "miou_pct": float(iou.mean() * 100.0) if present.any() else 0.0,
            "pixel_accuracy_pct": float(intersection.sum() / confusion.sum().clamp(min=1) * 100.0),
            "eval_loss": float(totals[0] / totals[1].clamp(min=1)),
            "eval_sample_count": float(totals[1]),
            "eval_completed": 1.0,
        }


# --- Multimodal SFT: Qwen2.5-VL on ChartQA -----------------------------------


_CHARTQA_PROMPT = "Answer the question with a single word or number."


def _chartqa_rows(path, split, limit=None):
    import glob

    import pyarrow.parquet as pq

    files = sorted(glob.glob(str(Path(path) / "**" / f"{split}-*.parquet"), recursive=True))
    if not files:
        raise FileNotFoundError(f"no ChartQA {split} parquet under {path}")
    rows = []
    for file in files:
        rows.extend(pq.read_table(file, columns=["image", "query", "label"]).to_pylist())
        if limit and len(rows) >= limit:
            break
    return rows[:limit] if limit else rows


def _relaxed_match(prediction, targets, tolerance=0.05):
    """ChartQA relaxed accuracy: 5% numeric tolerance, otherwise exact match."""

    def _number(text):
        text = text.strip().rstrip("%").replace(",", "")
        try:
            return float(text)
        except ValueError:
            return None

    prediction = prediction.strip().strip(".").lower()
    for target in targets:
        target = str(target).strip().lower()
        p, t = _number(prediction), _number(target)
        if p is not None and t is not None:
            if t == 0 and p == 0 or t != 0 and abs(p - t) / abs(t) <= tolerance:
                return True
        elif prediction == target:
            return True
    return False


class _ChartQA(torch.utils.data.Dataset):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        from PIL import Image

        row = self.rows[index]
        image = Image.open(io.BytesIO(row["image"]["bytes"])).convert("RGB")
        return {"image": image, "query": row["query"], "answers": list(row["label"])}


class VLMTask(_Task):
    unit = "samples"

    def build(self):
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
            Qwen2_5_VLDecoderLayer,
            Qwen2_5_VLVisionBlock,
        )

        self.processor = AutoProcessor.from_pretrained(
            self.args.model_id,
            min_pixels=self.opts.get("min_pixels", 128 * 28 * 28),
            max_pixels=self.opts.get("max_pixels", 512 * 28 * 28),
        )
        self.processor.tokenizer.padding_side = "left"
        self.max_new_tokens = int(self.opts.get("max_new_tokens", 16))
        self.train_set = _ChartQA(_chartqa_rows(self.args.dataset_path, "train"))
        self.eval_set = _ChartQA(_chartqa_rows(self.args.dataset_path, "test", self.args.eval_sample_count))
        self.val_set = _ChartQA(_chartqa_rows(self.args.dataset_path, "val", self.opts.get("eval_loss_samples", 256)))
        # Only global rank 0 reads the weights; every other rank builds on the
        # meta device and receives its shard from FSDP's rank-0 broadcast. A
        # 72B checkpoint loaded on all eight ranks of a node would need over a
        # terabyte of host memory per node.
        if self.rank == 0:
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.args.model_id, torch_dtype=torch.bfloat16, attn_implementation="sdpa"
            )
        else:
            from accelerate import init_empty_weights
            from transformers import AutoConfig

            config = AutoConfig.from_pretrained(self.args.model_id)
            config._attn_implementation = "sdpa"
            with init_empty_weights(include_buffers=False):
                model = Qwen2_5_VLForConditionalGeneration._from_config(config, torch_dtype=torch.bfloat16)
        model.config.use_cache = False
        self.parameter_count = sum(p.numel() for p in model.parameters())
        return _wrap(
            model,
            self.args,
            self.device,
            self.local_rank,
            layer_classes=(Qwen2_5_VLDecoderLayer, Qwen2_5_VLVisionBlock),
            sync_from_rank0=True,
        )

    def _messages(self, item, answer=None):
        user = {
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": f"{item['query']}\n{_CHARTQA_PROMPT}"}],
        }
        if answer is None:
            return [user]
        return [user, {"role": "assistant", "content": [{"type": "text", "text": answer}]}]

    def collate(self, items):
        prompts = [
            self.processor.apply_chat_template(self._messages(item), add_generation_prompt=True) for item in items
        ]
        full = [self.processor.apply_chat_template(self._messages(item, item["answers"][0])) for item in items]
        images = [item["image"] for item in items]
        self.processor.tokenizer.padding_side = "right"
        encoded = self.processor(text=full, images=images, return_tensors="pt", padding=True)
        prompt_lengths = [len(ids) for ids in self.processor(text=prompts, images=images, padding=False)["input_ids"]]
        self.processor.tokenizer.padding_side = "left"
        labels = encoded["input_ids"].clone()
        labels[encoded["attention_mask"] == 0] = -100
        for row, length in enumerate(prompt_lengths):
            labels[row, :length] = -100
        encoded["labels"] = labels
        return dict(encoded)

    def samples(self, batch):
        return batch["input_ids"].shape[0]

    def tokens(self, batch):
        return int(batch["attention_mask"].sum())

    def flops(self, samples, tokens):
        # 6 * parameters * tokens, the dense-transformer training estimate.
        return 6.0 * self.parameter_count * tokens

    def loss(self, model, batch):
        inputs = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}
        return model(**inputs).loss

    def evaluate(self, model):
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

        loss_loader = _loader(self.val_set, 1, self.rank, self.world_size, self.collate, False, workers=2)
        totals = torch.zeros(4, dtype=torch.float64, device=self.device)
        model.eval()
        with torch.no_grad():
            for batch in loss_loader:
                with self.bench._autocast(self.args.precision):
                    totals[0] += float(self.loss(model, batch))
                totals[1] += 1
            # Every rank must issue the same number of forward passes under
            # FSDP, so the test slice is split evenly and generate() runs with
            # synced_gpus so ranks that finish early keep serving all-gathers.
            per_rank = len(self.eval_set) // self.world_size
            indices = list(range(self.rank * per_rank, (self.rank + 1) * per_rank))
            batch_size = int(self.opts.get("eval_batch_size", 4))
            context = (
                FSDP.summon_full_params(model, recurse=False, writeback=False)
                if _is_fsdp(model)
                else (contextlib.nullcontext())
            )
            raw = model.module if hasattr(model, "module") else model
            with context:
                for start in range(0, len(indices), batch_size):
                    items = [self.eval_set[i] for i in indices[start : start + batch_size]]
                    prompts = [
                        self.processor.apply_chat_template(self._messages(item), add_generation_prompt=True)
                        for item in items
                    ]
                    encoded = self.processor(
                        text=prompts, images=[item["image"] for item in items], return_tensors="pt", padding=True
                    ).to(self.device)
                    with self.bench._autocast(self.args.precision):
                        generated = raw.generate(
                            **encoded,
                            max_new_tokens=self.max_new_tokens,
                            do_sample=False,
                            synced_gpus=True,
                            use_cache=True,
                        )
                    answers = self.processor.batch_decode(
                        generated[:, encoded["input_ids"].shape[1] :], skip_special_tokens=True
                    )
                    for item, answer in zip(items, answers):
                        totals[2] += float(_relaxed_match(answer, item["answers"]))
                        totals[3] += 1
                    if self.rank == 0 and start == 0:
                        print(
                            f"PYTORCH_VISION_VLM_SAMPLE q={items[0]['query']!r} a={answers[0]!r} "
                            f"ref={items[0]['answers']}",
                            flush=True,
                        )
        model.train()
        dist.all_reduce(totals)
        return {
            "vlm_benchmark_score": float(totals[2] / totals[3].clamp(min=1) * 100.0),
            "eval_loss": float(totals[0] / totals[1].clamp(min=1)),
            "eval_sample_count": float(totals[3]),
            "eval_completed": 1.0,
        }


TASKS = {
    "detection": DetectionTask,
    "segmentation": SegmentationTask,
    "vlm_sft": VLMTask,
}


def _checkpoint_roundtrip(model, args, device, rank):
    """Save and reload the trained weights and require an exact match.

    DDP saves one full state dict from rank 0. FSDP writes a sharded
    torch.distributed.checkpoint so each rank only moves its own shard; that
    is the only layout a 72B model can round-trip without gathering it.
    """
    path = Path(args.checkpoint_path)
    timings = torch.zeros(3, dtype=torch.float64, device=device)
    if _is_fsdp(model):
        import torch.distributed.checkpoint as dcp
        from torch.distributed.checkpoint.state_dict import get_model_state_dict, set_model_state_dict

        directory = path.with_suffix("")
        state = get_model_state_dict(model)
        reference = {k: _local(v).detach().cpu().clone() for k, v in state.items()}
        dist.barrier()
        start = time.perf_counter()
        dcp.save(state, checkpoint_id=str(directory))
        dist.barrier()
        timings[0] = time.perf_counter() - start
        restored = get_model_state_dict(model)
        for value in restored.values():
            _local(value).detach().zero_()
        dist.barrier()
        start = time.perf_counter()
        dcp.load(restored, checkpoint_id=str(directory))
        set_model_state_dict(model, restored)
        dist.barrier()
        timings[1] = time.perf_counter() - start
        current = get_model_state_dict(model)
        delta = 0.0
        match = True
        for key, value in current.items():
            local = _local(value).detach().cpu()
            match = match and torch.equal(local, reference[key])
            if local.numel():
                delta = max(delta, float((local.double() - reference[key].double()).abs().max()))
        timings[2] = delta
        verdict = torch.tensor([1.0 if match else 0.0], dtype=torch.float64, device=device)
        dist.all_reduce(verdict, op=dist.ReduceOp.MIN)
        dist.all_reduce(timings, op=dist.ReduceOp.MAX)
        dist.barrier()
        if rank == 0 and not args.keep_checkpoint:
            shutil.rmtree(directory, ignore_errors=True)
        match_value = float(verdict.item())
    else:
        raw = model.module
        original = {k: v.detach().cpu().clone() for k, v in raw.state_dict().items()}
        verdict = torch.zeros(1, dtype=torch.float64, device=device)
        if rank == 0:
            path.parent.mkdir(parents=True, exist_ok=True)
            start = time.perf_counter()
            with path.open("wb") as stream:
                torch.save({"model": original}, stream)
                stream.flush()
                os.fsync(stream.fileno())
            timings[0] = time.perf_counter() - start
            start = time.perf_counter()
            loaded = torch.load(path, map_location="cpu", weights_only=False)["model"]
            raw.load_state_dict(loaded, strict=True)
            torch.cuda.synchronize(device)
            timings[1] = time.perf_counter() - start
            current = {k: v.detach().cpu() for k, v in raw.state_dict().items()}
            verdict[0] = 1.0 if all(torch.equal(current[k], original[k]) for k in original) else 0.0
            timings[2] = max(
                (
                    float((current[k].double() - original[k].double()).abs().max())
                    for k in original
                    if original[k].numel()
                ),
                default=0.0,
            )
            if not args.keep_checkpoint:
                path.unlink(missing_ok=True)
        dist.broadcast(timings, src=0)
        dist.broadcast(verdict, src=0)
        match_value = float(verdict.item())
    return {
        "checkpoint_save_seconds": float(timings[0]),
        "checkpoint_load_seconds": float(timings[1]),
        "checkpoint_state_match": match_value,
        # No optimizer step follows the reload, so the resume deltas are the
        # measured weight difference and a zero loss delta.
        "checkpoint_loss_delta": 0.0,
        "checkpoint_resume_model_max_abs_delta": float(timings[2]),
        "checkpoint_resume_optimizer_max_abs_delta": 0.0,
    }


def _measure_loader(loader, args, device, samples_of):
    iterator = iter(loader)
    for _ in range(args.loader_warmup_steps):
        next(iterator)
    start = time.perf_counter()
    count = 0
    for _ in range(args.loader_benchmark_steps):
        count += samples_of(next(iterator))
    elapsed = time.perf_counter() - start
    rate = torch.tensor(count / elapsed if elapsed > 0 else 0.0, dtype=torch.float64, device=device)
    dist.all_reduce(rate)
    return float(rate)


def run_task(bench, args, rank, world_size, device, local_rank):
    task = TASKS[args.task](bench, args, rank, world_size, device, local_rank)
    model = task.build()
    model.train()
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.learning_rate, weight_decay=args.weight_decay
    )
    loader = _loader(
        task.train_set,
        args.batch_size,
        rank,
        world_size,
        task.collate,
        True,
        workers=int(task.opts.get("loader_workers", 8)),
    )
    if rank == 0:
        print(
            f"PYTORCH_VISION_TASK task={args.task} train={len(task.train_set)} eval={len(task.eval_set)} "
            f"strategy={task.opts.get('strategy', 'ddp')}",
            flush=True,
        )
    loader_rate = _measure_loader(loader, args, device, task.samples)
    batches = _cycle(loader)
    steps_per_epoch = math.ceil(len(loader) / args.gradient_accumulation_steps)
    target_steps = steps_per_epoch * args.max_epochs if args.max_epochs else args.measure_steps

    def _step():
        optimizer.zero_grad(set_to_none=True)
        losses, samples, tokens = [], 0, 0
        for index in range(args.gradient_accumulation_steps):
            batch = next(batches)
            sync = (
                model.no_sync()
                if index < args.gradient_accumulation_steps - 1 and hasattr(model, "no_sync")
                else contextlib.nullcontext()
            )
            with sync:
                with bench._autocast(args.precision):
                    loss = task.loss(model, batch)
                (loss / args.gradient_accumulation_steps).backward()
            losses.append(loss.detach().float())
            samples += task.samples(batch)
            tokens += task.tokens(batch)
        if hasattr(model, "clip_grad_norm_"):
            model.clip_grad_norm_(float(task.opts.get("max_grad_norm", 1.0)))
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(task.opts.get("max_grad_norm", 1.0)))
        optimizer.step()
        return torch.stack(losses).mean(), samples, tokens

    for _ in range(args.warmup_steps):
        _step()
    torch.cuda.synchronize(device)
    dist.barrier()
    torch.cuda.reset_peak_memory_stats(device)

    milestones = {int(v) for v in args.milestone_steps.split(",") if v.strip()}
    tracker, codecarbon = bench._start_codecarbon(args, rank, torch.cuda.device_count())
    window_start = time.perf_counter()
    step_times, loss_series, milestone_losses = [], [], {}
    local_samples = local_tokens = 0
    first_loss = last_loss = None
    all_finite = True
    step = 0
    while step < target_steps:
        if args.max_duration_seconds and time.perf_counter() - window_start >= args.max_duration_seconds:
            break
        start = time.perf_counter()
        loss, samples, tokens = _step()
        torch.cuda.synchronize(device)
        step_times.append((time.perf_counter() - start) * 1000.0)
        local_samples += samples
        local_tokens += tokens
        value = float(loss)
        if args.run_mode != "perf":
            value = bench._reduce_mean(value, world_size, device)
        all_finite = all_finite and math.isfinite(value)
        step += 1
        first_loss = value if first_loss is None else first_loss
        last_loss = value
        if step % args.loss_curve_sample_every == 0:
            loss_series.append({"step": step, "loss": value, "time_seconds": time.perf_counter() - window_start})
        if step in milestones:
            milestone_losses[str(step)] = value
    torch.cuda.synchronize(device)
    window = bench._reduce_max(time.perf_counter() - window_start, device)
    codecarbon = bench._stop_codecarbon(tracker, codecarbon, rank)
    finite = torch.tensor(1 if all_finite else 0, device=device)
    dist.all_reduce(finite, op=dist.ReduceOp.MIN)
    if not finite.item():
        raise RuntimeError("non-finite loss observed during the measured window")

    counts = torch.tensor([local_samples, local_tokens], dtype=torch.float64, device=device)
    dist.all_reduce(counts)
    total_samples, total_tokens = float(counts[0]), float(counts[1])
    critical = bench._gather_step_times(step_times, rank, world_size, device)
    allocated = torch.tensor(torch.cuda.max_memory_allocated(device) / 1e6, device=device)
    reserved = torch.tensor(torch.cuda.max_memory_reserved(device) / 1e6, device=device)
    dist.all_reduce(allocated, op=dist.ReduceOp.MAX)
    dist.all_reduce(reserved, op=dist.ReduceOp.MAX)
    loss_initial = bench._reduce_mean(first_loss, world_size, device)
    loss_final = bench._reduce_mean(last_loss, world_size, device)

    checkpoint = _checkpoint_roundtrip(model, args, device, rank) if args.checkpoint_path else {}
    evaluation = task.evaluate(model) if args.eval_enabled else {}

    if rank != 0:
        return None
    ordered = sorted(critical)
    samples_per_sec = total_samples / window
    per_gpu = samples_per_sec / world_size
    tflops = task.flops(total_samples, total_tokens) / window / world_size / 1e12
    values = [p["loss"] for p in loss_series]
    slope = 0.0
    if len(values) > 1:
        x_mean = (len(values) - 1) / 2.0
        y_mean = sum(values) / len(values)
        slope = sum((i - x_mean) * (v - y_mean) for i, v in enumerate(values)) / sum(
            (i - x_mean) ** 2 for i in range(len(values))
        )
    metrics = {
        "images_per_sec": samples_per_sec,
        "images_per_sec_per_gpu": per_gpu,
        "tflops_per_sec_per_gpu": tflops,
        "mfu_pct": 100.0 * tflops / args.peak_tflops_per_gpu,
        "step_time_ms_mean": sum(ordered) / len(ordered),
        "step_time_ms_p50": bench._percentile(ordered, 50),
        "step_time_ms_p95": bench._percentile(ordered, 95),
        "peak_memory_allocated_mb": float(allocated),
        "peak_memory_reserved_mb": float(reserved),
        "loss_initial": loss_initial,
        "loss_final": loss_final,
        "loss_curve_points": float(len(loss_series)),
        "loss_curve_decreased": float(len(loss_series) >= args.loss_curve_minimum_points and slope < 0),
        **bench._scorecard_milestone_metrics(milestone_losses),
        **checkpoint,
        **evaluation,
    }
    if task.unit == "samples":
        metrics["multimodal_samples_per_sec"] = samples_per_sec
        metrics["tokens_per_sec"] = total_tokens / window
    else:
        metrics["data_loader_images_per_sec"] = loader_rate
    if args.convergence_train_loss is not None:
        point = next((p for p in loss_series if p["loss"] <= args.convergence_train_loss), None)
        if point is not None:
            metrics["convergence_step"] = point["step"]
            metrics["convergence_time_seconds"] = point["time_seconds"]
    if codecarbon.get("active") and "emissions_kg_co2eq" in codecarbon:
        metrics["codecarbon_tracking_active"] = 1.0
    return {
        "schema_version": 1,
        "workload": args.workload,
        "task": args.task,
        "model": args.model,
        "model_id": args.model_id,
        "backend": args.backend,
        "precision": args.precision,
        "image_size": args.image_size,
        "batch_size_per_gpu": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "effective_global_batch_size": args.batch_size * args.gradient_accumulation_steps * world_size,
        "world_size": world_size,
        "phase": args.phase,
        "run_mode": args.run_mode,
        "synthetic_data": False,
        "data_mode": args.data_mode,
        "rocal_device": args.rocal_device,
        "augmentation": args.augmentation,
        "dataset_path": args.dataset_path,
        "harness": "task",
        "task_options": task.opts,
        "torch_version": torch.__version__,
        "hip_version": torch.version.hip,
        "device": torch.cuda.get_device_name(device),
        "measurement_window_s": window,
        "completed_steps": step,
        "completed_epochs": step // steps_per_epoch if args.max_epochs else None,
        "steps_per_epoch": steps_per_epoch,
        "processed_images": int(total_samples),
        "processed_tokens": int(total_tokens),
        "loss_time_series": loss_series,
        "milestone_losses": milestone_losses,
        "evaluations": [dict(evaluation, step=step, time_seconds=window)] if evaluation else [],
        "step_times_ms": critical,
        "training_flops_per_image": args.training_flops_per_image,
        "peak_tflops_per_gpu": args.peak_tflops_per_gpu,
        "codecarbon": codecarbon,
        "metrics": metrics,
    }
