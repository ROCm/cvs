"""
GPU idle-state readiness check.

Samples ``amd-smi metric`` a few times before any preflight workload touches
the GPUs. Utilization uses the lowest reading across samples, so a monitoring
agent's brief spike does not fail an idle GPU while a busy GPU stays above the
limit. Temperature uses the highest reading.
"""

import json
import shlex

from cvs.lib.preflight.host_readiness_base import HostReadinessCheck, findings, to_number

SAMPLE_SEP = '===CVS_IDLE_SAMPLE==='
PROCESS_SEP = '===CVS_IDLE_PROCESS==='
AMD_SMI_MISSING = 'CVS_AMD_SMI_MISSING'

# Only the sections read here; a bare 'amd-smi metric' also reads --xgmi-err, which resets those counters.
_METRIC_ARGS = 'metric --usage --mem-usage --temperature --json'

_UTIL_FIELDS = (
    ('gfx_activity', 'GFX activity', 'gfx_util_max_pct'),
    ('umc_activity', 'memory activity', 'mem_util_max_pct'),
)


def _value(entry, *keys):
    """Numeric amd-smi field: accepts ``{'value': x, 'unit': u}``, a bare number, or N/A (None)."""
    current = entry
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    if isinstance(current, dict):
        current = current.get('value')
    return to_number(current)


def gpu_entries(payload):
    if isinstance(payload, dict) and 'gpu_data' in payload:
        payload = payload['gpu_data']
    if isinstance(payload, dict):
        payload = [payload]
    return [entry for entry in payload or [] if isinstance(entry, dict) and 'gpu' in entry]


def split_probe_output(output):
    """Return ``(sample_texts, process_text)``."""
    text = str(output or '')
    head, _, process_text = text.partition(PROCESS_SEP)
    samples = [chunk.strip() for chunk in head.split(SAMPLE_SEP)[1:]]
    return samples, process_text.strip()


def _load_json(text):
    """Parse amd-smi JSON, skipping warning text printed around it (which may itself contain brackets)."""
    decoder = json.JSONDecoder()
    for start, char in enumerate(text):
        if char in '[{':
            try:
                return decoder.raw_decode(text, start)[0]
            except json.JSONDecodeError:
                continue
    return None


def parse_processes(text):
    """Return ``(processes, readable)``; processes are ``{'gpu', 'name', 'pid'}`` dicts."""
    if not text:
        return [], False
    if 'No running processes' in text and _load_json(text) is None:
        return [], True
    payload = _load_json(text)
    if payload is None:
        return [], False
    processes = []
    for entry in gpu_entries(payload):
        for item in entry.get('process_list') or []:
            info = item.get('process_info') if isinstance(item, dict) else None
            if isinstance(info, dict) and info.get('pid') not in (None, '', 'N/A'):
                processes.append({'gpu': entry.get('gpu'), 'name': str(info.get('name') or 'N/A'), 'pid': info['pid']})
    return processes, True


class GpuIdleStateCheck(HostReadinessCheck):
    """Verify every GPU is idle, nearly empty, thermally sane, and not held by a process."""

    def __init__(
        self,
        orch,
        samples=3,
        interval_sec=2,
        gfx_util_max_pct=5,
        mem_util_max_pct=5,
        vram_used_max_pct=5,
        temp_max_c=None,
        fail_on_gpu_processes=True,
        allowed_procs='',
        on_mismatch='fail',
        config_dict=None,
    ):
        super().__init__(orch, on_mismatch, config_dict)
        self.samples = max(1, int(samples))
        self.interval_sec = max(0, int(interval_sec))
        self.limits = {'gfx_util_max_pct': gfx_util_max_pct, 'mem_util_max_pct': mem_util_max_pct}
        self.vram_used_max_pct = vram_used_max_pct
        self.temp_max_c = dict(temp_max_c or {})
        self.fail_on_gpu_processes = bool(fail_on_gpu_processes)
        self.allowed_procs = {name.strip() for name in str(allowed_procs or '').split(',') if name.strip()}
        self.probe_timeout = 120 + self.samples * self.interval_sec

    def build_command(self):
        return (
            'if command -v amd-smi >/dev/null 2>&1; then _cvs_smi=$(command -v amd-smi); '
            'elif [ -x /opt/rocm/bin/amd-smi ]; then _cvs_smi=/opt/rocm/bin/amd-smi; '
            f'else echo {AMD_SMI_MISSING}; exit 0; fi\n'
            '_cvs_i=0\n'
            f'while [ "$_cvs_i" -lt {self.samples} ]; do\n'
            f'  echo {shlex.quote(SAMPLE_SEP)}\n'
            f'  "$_cvs_smi" {_METRIC_ARGS} 2>&1\n'
            '  _cvs_i=$((_cvs_i + 1))\n'
            f'  [ "$_cvs_i" -lt {self.samples} ] && sleep {self.interval_sec}\n'
            'done\n'
            f'echo {shlex.quote(PROCESS_SEP)}\n'
            '"$_cvs_smi" process --json 2>&1\n'
        )

    @staticmethod
    def aggregate(sample_payloads):
        """Fold samples into ``{gpu: {'gfx_activity', 'umc_activity', 'vram_used_pct', 'temperature': {...}}}``."""
        gpus = {}
        for payload in sample_payloads:
            for entry in gpu_entries(payload):
                gpu = gpus.setdefault(
                    str(entry['gpu']), {'gfx_activity': [], 'umc_activity': [], 'vram_used_pct': [], 'temperature': {}}
                )
                for field, _, _ in _UTIL_FIELDS:
                    reading = _value(entry, 'usage', field)
                    if reading is not None:
                        gpu[field].append(reading)
                used = _value(entry, 'mem_usage', 'used_vram')
                total = _value(entry, 'mem_usage', 'total_vram')
                if used is not None and total:
                    gpu['vram_used_pct'].append(100.0 * used / total)
                for sensor, reading in (entry.get('temperature') or {}).items():
                    number = to_number(reading.get('value')) if isinstance(reading, dict) else to_number(reading)
                    if number is not None:
                        gpu['temperature'].setdefault(sensor, []).append(number)
        folded = {}
        for gpu, readings in gpus.items():
            folded[gpu] = {
                'gfx_activity': min(readings['gfx_activity']) if readings['gfx_activity'] else None,
                'umc_activity': min(readings['umc_activity']) if readings['umc_activity'] else None,
                'vram_used_pct': round(min(readings['vram_used_pct']), 2) if readings['vram_used_pct'] else None,
                'temperature': {sensor: max(values) for sensor, values in readings['temperature'].items()},
            }
        return folded

    def _gpu_mismatches(self, gpu, metrics, warnings):
        mismatches = []
        for field, label, limit_key in _UTIL_FIELDS:
            limit = self.limits.get(limit_key)
            value = metrics[field]
            if limit is None:
                continue
            if value is None:
                warnings.append(f"GPU {gpu}: {label} is not reported by amd-smi")
            elif value > limit:
                mismatches.append(f"GPU {gpu}: {label} {value:g}% is above the idle limit of {limit:g}%")
        if self.vram_used_max_pct is not None and metrics['vram_used_pct'] is not None:
            if metrics['vram_used_pct'] > self.vram_used_max_pct:
                mismatches.append(
                    f"GPU {gpu}: VRAM {metrics['vram_used_pct']:g}% used is above the idle limit of {self.vram_used_max_pct:g}%"
                )
        for sensor, limit in self.temp_max_c.items():
            reading = metrics['temperature'].get(sensor)
            if limit is not None and reading is not None and reading > limit:
                mismatches.append(f"GPU {gpu}: {sensor} temperature {reading:g} C is above {limit:g} C")
        return mismatches

    def evaluate_node(self, node, output):
        text = str(output or '')
        if AMD_SMI_MISSING in text:
            return findings(errors=[f"amd-smi was not found on {node}"])
        sample_texts, process_text = split_probe_output(text)
        payloads = [_load_json(chunk) for chunk in sample_texts]
        parsed = [payload for payload in payloads if payload is not None]
        if not parsed:
            first = text.strip().splitlines()[:1]
            return findings(
                errors=[f"Could not parse amd-smi metric output on {node}: {first[0] if first else 'empty'}"]
            )

        warnings = []
        if len(parsed) < len(sample_texts):
            warnings.append(f"{len(sample_texts) - len(parsed)} of {len(sample_texts)} amd-smi samples were unreadable")
        gpus = self.aggregate(parsed)
        if not gpus:
            return findings(errors=[f"amd-smi reported no GPUs on {node}"])

        mismatches = []
        for gpu in sorted(gpus, key=lambda key: (len(key), key)):
            mismatches.extend(self._gpu_mismatches(gpu, gpus[gpu], warnings))

        processes, readable = parse_processes(process_text)
        if not readable:
            warnings.append("Could not read the GPU process list from 'amd-smi process'")
        foreign = [proc for proc in processes if proc['name'] not in self.allowed_procs]
        if self.fail_on_gpu_processes:
            for proc in foreign:
                mismatches.append(f"GPU {proc['gpu']}: process {proc['name']} (pid {proc['pid']}) is using the GPU")

        details = {'gpus': gpus, 'processes': processes, 'samples': len(parsed)}
        return findings(mismatches, [], warnings, details)
