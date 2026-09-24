"""Explicit configurations for offline Aorta tests."""


def variant_dict():
    return {
        "schema_version": 1,
        "paths": {"shared_fs": "/scratch/tester", "models_dir": "/models", "log_dir": "/logs", "hf_token_file": ""},
        "model": {"id": "test", "remote": 0},
        "container": {
            "name": "aorta_test",
            "image": "test-image",
            "runtime": {"name": "docker", "args": {"volumes": ["/scratch/tester/repo:/mnt"], "ipc": "host"}},
            "env": {"NCCL_MAX_NCHANNELS": "112"},
        },
        "aorta_path": "/scratch/tester/repo",
        "gpus_per_node": 2,
        "thresholds": {"expected_results": {"max_avg_iteration_ms": 12000, "min_compute_ratio": 0.01}},
        "analysis": {"enable_tracelens": False},
    }
