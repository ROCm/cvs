"""Tokenize a parquet text corpus into flat uint32 token files, once per tokenizer.

Output goes to ``<cache-dir>/<tokenizer digest>/<split>.bin`` with a JSON
sidecar. A finished split is reused as long as its sidecar records the same
source files and tokenizer, so only the first run of a model family pays for
tokenization.
"""

import argparse
import hashlib
import json
import os
import re
import socket
from pathlib import Path

import numpy as np


DATA_OK = "PYTORCH_TRAINING_DATA "
# WikiText marks each article with a level-one heading line such as " = Valkyria Chronicles III = ".
_WIKITEXT_ARTICLE = re.compile(r"^ = [^=].* = \n?$")


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--cache-dir", required=True)
    p.add_argument("--splits", required=True, help="comma-separated split names")
    p.add_argument("--text-column", default="text")
    p.add_argument("--append-eos", action="store_true")
    return p.parse_args()


def tokenizer_digest(tokenizer_dir):
    digest = hashlib.sha256()
    for path in sorted(Path(tokenizer_dir).iterdir()):
        if path.name.startswith(("tokenizer", "vocab", "merges", "special_tokens")) and path.is_file():
            digest.update(path.name.encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()[:16]


def split_files(source, split):
    files = sorted(Path(source).glob(f"{split}-*.parquet")) or sorted(Path(source).glob(f"{split}*.parquet"))
    if not files:
        raise FileNotFoundError(f"no {split} parquet files under {source}")
    return files


def documents(files, text_column):
    """Yield one string per document, grouping WikiText lines into articles."""
    import pyarrow.parquet as pq

    article = []
    wikitext = None
    for path in files:
        for text in pq.read_table(path, columns=[text_column]).column(text_column).to_pylist():
            text = text or ""
            if wikitext is None and text.strip():
                wikitext = bool(_WIKITEXT_ARTICLE.match(text))
            if not wikitext:
                if text.strip():
                    yield text
                continue
            if _WIKITEXT_ARTICLE.match(text) and article:
                yield "".join(article)
                article = []
            article.append(text)
    if article:
        yield "".join(article)


def tokenize_split(files, tokenizer, text_column, append_eos, out_path):
    eos = tokenizer.eos_token_id
    if append_eos and eos is None:
        raise ValueError("append_eos requested but the tokenizer has no EOS token")
    # Host and pid in the name: nodes sharing one cache directory write their
    # own temporary file, and the identical results replace each other.
    tmp = out_path.with_name(f"{out_path.name}.{socket.gethostname()}.{os.getpid()}.tmp")
    count = 0
    batch = []
    with tmp.open("wb") as stream:

        def flush():
            nonlocal count
            encoded = tokenizer(batch, add_special_tokens=False)["input_ids"]
            for ids in encoded:
                if append_eos:
                    ids = ids + [eos]
                array = np.asarray(ids, dtype=np.uint32)
                array.tofile(stream)
                count += array.size
            batch.clear()

        for doc in documents(files, text_column):
            batch.append(doc)
            if len(batch) >= 1024:
                flush()
        if batch:
            flush()
    os.replace(tmp, out_path)
    return count


def main():
    args = _parse_args()
    from transformers import AutoTokenizer

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    if len(tokenizer) >= 2**32:
        raise ValueError("vocabulary does not fit uint32 token ids")
    digest = tokenizer_digest(args.tokenizer)
    out_dir = Path(args.cache_dir) / digest
    out_dir.mkdir(parents=True, exist_ok=True)
    result = {"tokenizer_digest": digest, "splits": {}}
    for split in [s for s in args.splits.split(",") if s]:
        files = split_files(args.source, split)
        source_id = [[f.name, f.stat().st_size] for f in files]
        out_path = out_dir / f"{split}.bin"
        meta_path = out_dir / f"{split}.json"
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        reusable = (
            out_path.exists()
            and meta.get("source_files") == source_id
            and meta.get("append_eos") == args.append_eos
            and out_path.stat().st_size == 4 * int(meta.get("tokens", -1))
        )
        if not reusable:
            tokens = tokenize_split(files, tokenizer, args.text_column, args.append_eos, out_path)
            meta = {
                "tokens": tokens,
                "source_files": source_id,
                "append_eos": args.append_eos,
                "tokenizer": args.tokenizer,
                "tokenizer_digest": digest,
                "vocab_size": len(tokenizer),
            }
            tmp = meta_path.with_name(f"{meta_path.name}.{socket.gethostname()}.{os.getpid()}.tmp")
            tmp.write_text(json.dumps(meta, indent=2))
            os.replace(tmp, meta_path)
        result["splits"][split] = {"path": str(out_path), "tokens": int(meta["tokens"]), "reused": reusable}
    print(DATA_OK + json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
