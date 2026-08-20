import unittest

from cvs.parsers.schemas import PytorchXditFluxConfigFile, PytorchXditWanConfigFile


def _wan_benchmark_params():
    return {
        "wan22_i2v_a14b": {
            "prompt": "test prompt",
            "size": "720*1280",
            "frame_num": 81,
            "num_benchmark_steps": 5,
            "compile": True,
            "torchrun_nproc": 8,
            "ulysses_size": 8,
            "ring_size": 2,
            "expected_results": {"auto": {"max_avg_total_time_s": 15.0}},
        }
    }


def _flux_benchmark_params():
    return {
        "flux1_dev_t2i": {
            "prompt": "A cat",
            "seed": 42,
            "num_inference_steps": 25,
            "max_sequence_length": 256,
            "no_use_resolution_binning": True,
            "warmup_steps": 1,
            "warmup_calls": 5,
            "num_repetitions": 25,
            "height": 1024,
            "width": 1024,
            "ulysses_degree": 8,
            "ring_degree": 2,
            "use_torch_compile": True,
            "torchrun_nproc": 8,
            "expected_results": {"auto": {"max_avg_pipe_time_s": 12.0}},
        }
    }


class TestPytorchXditDistributedSchemas(unittest.TestCase):
    def test_wan_config_accepts_example_nccl_hints(self):
        raw = {
            "config": {
                "hf_home": "/home/user",
                "output_base_dir": "/home/user/out",
                "nnodes": 2,
                "master_addr": "10.0.0.1",
                "master_port": 29500,
                "_example_nccl_ib_hca": "rdma0,rdma1,rdma2,rdma3,rdma4,rdma5,rdma6,rdma7",
                "nccl_ib_hca": "rdma0",
                "_example_nccl_socket_ifname": "eno0",
                "nccl_socket_ifname": "eno0",
                "_example_gloo_socket_ifname": "eno0",
                "gloo_socket_ifname": "eno0",
            },
            "benchmark_params": _wan_benchmark_params(),
        }

        validated = PytorchXditWanConfigFile.model_validate(raw)

        self.assertEqual(
            validated.config._example_nccl_ib_hca,
            "rdma0,rdma1,rdma2,rdma3,rdma4,rdma5,rdma6,rdma7",
        )
        self.assertEqual(validated.config._example_nccl_socket_ifname, "eno0")
        self.assertEqual(validated.config._example_gloo_socket_ifname, "eno0")

    def test_flux_config_accepts_example_nccl_hints(self):
        raw = {
            "config": {
                "hf_home": "/home/user",
                "output_base_dir": "/home/user/out",
                "nnodes": 2,
                "master_addr": "10.0.0.1",
                "master_port": 29500,
                "_example_nccl_ib_hca": "rdma0,rdma1,rdma2,rdma3,rdma4,rdma5,rdma6,rdma7",
                "nccl_ib_hca": "rdma0",
                "_example_nccl_socket_ifname": "eno0",
                "nccl_socket_ifname": "eno0",
                "_example_gloo_socket_ifname": "eno0",
                "gloo_socket_ifname": "eno0",
            },
            "benchmark_params": _flux_benchmark_params(),
        }

        validated = PytorchXditFluxConfigFile.model_validate(raw)

        self.assertEqual(
            validated.config._example_nccl_ib_hca,
            "rdma0,rdma1,rdma2,rdma3,rdma4,rdma5,rdma6,rdma7",
        )


if __name__ == "__main__":
    unittest.main()
