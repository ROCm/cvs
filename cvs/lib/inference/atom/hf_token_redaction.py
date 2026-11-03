'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

Redact Hugging Face token values from ATOM logs and exception text.
Paths such as ``.hf_token`` are left unchanged.
'''

import logging
import re

_SECRET_ASSIGN_RE = re.compile(
    r"(?i)((?:HF_TOKEN|HUGGING_FACE_HUB_TOKEN|HUGGINGFACE_HUB_TOKEN|hf_token)\s*=\s*)"
    r"(?:'[^']*'|\"[^\"]*\"|\S+)"
)
_SECRET_MAPPING_RE = re.compile(
    r"""(?i)(['"])(HF_TOKEN|HUGGING_FACE_HUB_TOKEN|HUGGINGFACE_HUB_TOKEN|hf_token)\1(\s*:\s*)(['"])[^'"]*\4"""
)
# Real hub tokens are much longer than package names such as ``hf_transfer``.
# ``.hf_token`` stays because ``token`` is only 5 characters.
_HF_VALUE_RE = re.compile(r"\bhf_[A-Za-z0-9]{30,}\b")


def redact_hf_token_text(text):
    """Replace Hugging Face token values. Paths like ``.hf_token`` stay."""
    if not isinstance(text, str) or not text:
        return text
    text = _SECRET_ASSIGN_RE.sub(r"\1<redacted>", text)
    text = _SECRET_MAPPING_RE.sub(r"\1\2\1\3\4<redacted>\4", text)
    return _HF_VALUE_RE.sub("<redacted>", text)


class HfTokenLogFilter(logging.Filter):
    """Redact Hugging Face tokens from log records emitted during ATOM runs."""

    def filter(self, record):
        try:
            rendered = record.getMessage()
        except (TypeError, ValueError):
            rendered = str(record.msg)
        record.msg = redact_hf_token_text(rendered)
        record.args = ()
        return True


def install_hf_token_log_redaction():
    """Install token redaction on the root logger and its current handlers.

    Child-logger records reach the root handlers without passing through the
    root logger's own filters, so each handler needs the filter too. Call again
    after pytest attaches log handlers.
    """
    root = logging.getLogger()
    filt = next((existing for existing in root.filters if isinstance(existing, HfTokenLogFilter)), None)
    if filt is None:
        filt = HfTokenLogFilter()
        root.addFilter(filt)
    for handler in root.handlers:
        if not any(isinstance(existing, HfTokenLogFilter) for existing in handler.filters):
            handler.addFilter(filt)
