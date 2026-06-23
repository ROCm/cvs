# InferenceX ATOM single-node variants (DTNI layout)

Each subdirectory is one workload × GPU arch:

- `config.json` — run recipe (sweep, container, ATOM server args)
- `threshold.json` — pass/fail specs per sweep cell

Run example (MI300X W1):

```bash
cvs run inferencex_atom_single \
  --cluster_file input/cluster_file/mi300x_atom_single.json \
  --config_file input/dtni/inferencex_atom_single/deepseek_r1_fp8_mi300x_atom_perf/config.json \
  --html=/tmp/inferencex_atom_w1_mi300x.html -vvv -s
```

Set `container.image` / `container.name` and cluster node IPs before first lab run.
Pin the ATOM docker tag in the variant README when calibrating thresholds (see plan §4).
