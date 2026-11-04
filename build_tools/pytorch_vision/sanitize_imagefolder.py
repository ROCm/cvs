"""Find and re-encode ImageFolder files that rocAL cannot decode.

rocAL decodes every ``.JPEG`` with TurboJPEG. A file whose payload is not a
JPEG aborts the whole training process: ImageNet-1k train ships one PNG named
``.JPEG`` (n02105855_2933, class index 230), and it crashes rocAL partway
through any run long enough to reach it. CMYK/YCCK JPEGs are flagged too.

rocAL decodes into a buffer of at most 1000x1000 (``max_decoded_width`` and
``max_decoded_height``), and TurboJPEG can only scale down by up to 1/8, so a
side longer than 8000 pixels overruns that buffer and segfaults inside
libjpeg. ImageNet-1k train has one such file (9331x2105, class index 718).

The scan reads only file headers. With ``--fix`` each flagged file is copied
to ``--backup-dir`` (same relative path) and rewritten in place as an RGB JPEG.

usage:
    python sanitize_imagefolder.py <dataset root> [--fix --backup-dir DIR] [--workers N]
"""

import argparse
import os
import shutil
from multiprocessing import Pool
from pathlib import Path

MAX_SIDE = 8 * 1000
SOF = {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}
HEADER_BYTES = 1 << 20


def classify(path):
    """Return a problem string, or None for a 1- or 3-component JPEG."""
    try:
        with open(path, "rb") as handle:
            data = handle.read(HEADER_BYTES)
    except OSError as exc:
        return f"unreadable:{exc.errno}"
    if data[:2] != b"\xff\xd8":
        return "not_jpeg:" + data[:8].hex()
    index = 2
    while index + 4 <= len(data):
        if data[index] != 0xFF:
            return None
        marker = data[index + 1]
        if marker == 0xFF:
            index += 1
            continue
        if marker in SOF:
            if index + 10 > len(data):
                return None
            components = data[index + 9]
            if components not in (1, 3):
                return f"components:{components}"
            height = int.from_bytes(data[index + 5 : index + 7], "big")
            width = int.from_bytes(data[index + 7 : index + 9], "big")
            return f"oversize:{width}x{height}" if max(width, height) > MAX_SIDE else None
        index += 2 + int.from_bytes(data[index + 2 : index + 4], "big")
    return None


def _check(path):
    return path, classify(path)


def _files(root):
    for split in ("train", "val"):
        split_dir = Path(root) / split
        if not split_dir.is_dir():
            continue
        for class_dir in os.scandir(split_dir):
            if class_dir.is_dir():
                for item in os.scandir(class_dir.path):
                    yield item.path


def _reencode(path, root, backup_dir):
    from PIL import Image

    backup = Path(backup_dir) / Path(path).relative_to(root)
    backup.parent.mkdir(parents=True, exist_ok=True)
    if not backup.exists():
        shutil.copy2(path, backup)
    with Image.open(backup) as image:
        image = image.convert("RGB")
        if max(image.size) > MAX_SIDE:
            image.thumbnail((MAX_SIDE, MAX_SIDE), Image.LANCZOS)
        image.save(path, format="JPEG", quality=95)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("root")
    parser.add_argument("--fix", action="store_true")
    parser.add_argument("--backup-dir")
    parser.add_argument("--workers", type=int, default=32)
    args = parser.parse_args()
    if args.fix and not args.backup_dir:
        parser.error("--fix requires --backup-dir")

    flagged = []
    total = 0
    with Pool(args.workers) as pool:
        for path, problem in pool.imap_unordered(_check, _files(args.root), chunksize=256):
            total += 1
            if problem:
                flagged.append(path)
                print(f"{problem} {path}", flush=True)
    for path in flagged if args.fix else []:
        _reencode(path, args.root, args.backup_dir)
        print(f"fixed {path}", flush=True)
    print(f"SCANNED {total} flagged={len(flagged)} fixed={len(flagged) if args.fix else 0}", flush=True)


if __name__ == "__main__":
    main()
