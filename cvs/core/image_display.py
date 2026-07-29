'''Container image tag / digest display strings (shared by orchestration and reports).'''

from __future__ import annotations


def format_image_display(*, image_tag: str = "", image_digest: str = "", image_id: str = "") -> str:
    digest = image_digest or image_id
    if digest:
        short = digest
        if "@" in short:
            short = short.split("@", 1)[1]
        if short.startswith("sha256:") and len(short) > 19:
            short = f"{short[:19]}\u2026"
        if image_tag:
            return f"{image_tag} @ {short}"
        return short
    return image_tag or "\u2014"
