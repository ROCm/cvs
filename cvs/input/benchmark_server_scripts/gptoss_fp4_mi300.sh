#!/usr/bin/env bash
# Alias for sites that reference gptoss_fp4_mi300.sh (MI300) instead of mi300x.
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${here}/fixed_seq_len/gptoss_fp4_mi300x.sh" "$@"
