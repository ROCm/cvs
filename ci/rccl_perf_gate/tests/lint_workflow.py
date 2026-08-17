#!/usr/bin/env python3
"""Syntax-check every shell `run:` block in a GitHub Actions workflow.

A broken run-block is invisible until the job executes it, which on a
self-hosted GPU pipeline means burning an allocation to discover a typo.
"""
import os
import re
import subprocess
import sys
import tempfile

import yaml

path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/wf.yml"
doc = yaml.safe_load(open(path))

checked = 0
bad = 0
for job_name, job in doc["jobs"].items():
    for i, step in enumerate(job.get("steps", [])):
        body = step.get("run")
        if not body:
            continue
        name = step.get("name", "step%d" % i)
        shell = step.get("shell", "bash")
        if "python" in shell:
            print("  skip [%s] %s (shell=%s)" % (job_name, name, shell))
            continue
        checked += 1
        # Actions expands ${{ ... }} before bash sees it. Substitute a literal so
        # we are checking the shell, not the templating.
        src = re.sub(r"\$\{\{[^}]*\}\}", "X", body)
        with tempfile.NamedTemporaryFile("w", suffix=".sh", delete=False) as fh:
            fh.write(src)
            tmp = fh.name
        res = subprocess.run(["bash", "-n", tmp], capture_output=True, text=True)
        os.unlink(tmp)
        if res.returncode != 0:
            bad += 1
            print("  FAIL [%s] %s\n%s" % (job_name, name, res.stderr))
        else:
            print("  ok   [%s] %s" % (job_name, name))

print("\n%d run-blocks checked, %d with syntax errors" % (checked, bad))
sys.exit(1 if bad else 0)
