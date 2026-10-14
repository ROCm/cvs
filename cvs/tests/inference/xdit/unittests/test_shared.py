"""Unit tests for xDiT pytest wiring helpers."""

import base64
import json
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from cvs.lib import globals
from cvs.lib.inference.xdit.xdit_benchmark_job import cleanup_staged_artifacts
from cvs.tests.inference.xdit import conftest
from cvs.tests.inference.xdit._shared import (
    Lifecycle,
    _SecretValue,
    _attach_benchmark_artifacts,
    _is_flux2_workload,
    _output_dirs_by_host,
    _output_parser,
    _report_dimensions,
    _report_threshold,
    benchmark_params_from_variant,
    clip_score_stage,
    hf_remote_download_enabled,
    hf_token_from_variant,
    inference_from_variant,
    log_topology,
    parse_thresholds_stage,
    resolve_execution_hosts,
    scoped_cluster_dict,
    suite_spec,
    verify_model_stage,
)


class _Variant:
    def __init__(self):
        self.inference = {"model_repo": "/models/flux"}
        self.benchmark_params = {"flux1_dev_t2i": {}}


class TestVariantHelpers(unittest.TestCase):
    def test_reads_object_variant(self):
        variant = _Variant()

        self.assertEqual(inference_from_variant(variant), variant.inference)
        self.assertEqual(benchmark_params_from_variant(variant), variant.benchmark_params)

    def test_reads_mapping_variant(self):
        variant = {"config": {"model_repo": "/models/wan"}, "benchmark_params": {"wan22_i2v_a14b": {}}}

        self.assertEqual(inference_from_variant(variant), variant["config"])
        self.assertEqual(benchmark_params_from_variant(variant), variant["benchmark_params"])

    def test_reads_hf_token_from_paths_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            token_path = os.path.join(tmp, ".hf_token")
            with open(token_path, "w", encoding="utf-8") as handle:
                handle.write("hf_from_config\n")
            variant = SimpleNamespace(
                paths=SimpleNamespace(hf_token_file=token_path),
                inference={"hf_token_file": token_path},
            )
            token = hf_token_from_variant(variant)
            self.assertEqual(token, "hf_from_config")
            self.assertIsInstance(token, _SecretValue)
            self.assertEqual(repr(token), "<redacted>")
            self.assertNotIn("hf_from_config", repr(token))

    def test_missing_hf_token_file_returns_empty(self):
        variant = SimpleNamespace(
            paths=SimpleNamespace(hf_token_file="/missing/.hf_token"),
            inference={"hf_token_file": "/missing/.hf_token"},
        )
        self.assertEqual(hf_token_from_variant(variant), "")


class TestSuiteSpec(unittest.TestCase):
    def test_flux_single(self):
        self.assertEqual(
            suite_spec("cvs.tests.inference.xdit.xdit_flux_dev_single"),
            {"family": "flux", "distributed": False, "diffusers": False},
        )

    def test_wan_diffusers_distributed(self):
        self.assertEqual(
            suite_spec("xdit_wan22_14b_diffusers_distributed"),
            {"family": "wan", "distributed": True, "diffusers": True},
        )


class TestReportResults(unittest.TestCase):
    def test_flux_dimensions_use_shape_steps_and_nnodes(self):
        variant = SimpleNamespace(
            model=SimpleNamespace(id="black-forest-labs/FLUX.1-dev"),
            inference={"nnodes": 2},
        )

        values = _report_dimensions(
            variant,
            {
                "height": 1024,
                "width": 768,
                "num_inference_steps": 25,
                "num_repetitions": 25,
                "torchrun_nproc": 8,
            },
            {"family": "flux", "distributed": True, "diffusers": False},
        )

        self.assertEqual(values[:5], ("black-forest-labs/FLUX.1-dev", "1024x768", 25, "diffusers", 2))
        self.assertEqual(values[5], "SIZE=1024x768,STEPS=25,BENCH=25")

    def test_distributed_nnodes_use_execution_hosts_when_nnodes_missing(self):
        variant = SimpleNamespace(
            model=SimpleNamespace(id="FLUX.1-dev"),
            inference={"_execution_hosts": ["10.0.0.1", "10.0.0.2"]},
        )

        values = _report_dimensions(
            variant,
            {
                "height": 1024,
                "width": 1024,
                "num_inference_steps": 50,
                "num_repetitions": 25,
                "torchrun_nproc": 8,
            },
            {"family": "flux", "distributed": True, "diffusers": False},
        )

        self.assertEqual(values[4], 2)
        self.assertEqual(values[5], "SIZE=1024x1024,STEPS=50,BENCH=25")

    def test_wan_dimensions_use_size_frames_inference_and_benchmark_steps(self):
        variant = SimpleNamespace(
            model=SimpleNamespace(id="Wan-AI/Wan2.2-I2V-A14B-Diffusers"),
            inference={"nnodes": 1},
        )

        values = _report_dimensions(
            variant,
            {
                "size": "720*1280",
                "frame_num": 81,
                "num_inference_steps": 40,
                "num_benchmark_steps": 1,
            },
            {"family": "wan", "distributed": False, "diffusers": True},
        )

        self.assertEqual(values[:5], ("Wan-AI/Wan2.2-I2V-A14B-Diffusers", "720*1280", 40, "diffusers", 1))
        self.assertEqual(values[5], "SIZE=720*1280,FRAMES=81,STEPS=40,BENCH=1")

    def test_report_threshold_uses_gpu_then_auto_fallback(self):
        thresholds = {
            "mi300x": {"max_avg_pipe_time_s": 3.0},
            "auto": {"max_avg_pipe_time_s": 10.0},
        }

        self.assertEqual(_report_threshold(thresholds, "mi300x", "avg_pipe_time_s"), {"kind": "max", "value": 3.0})
        self.assertEqual(_report_threshold(thresholds, "other", "avg_pipe_time_s"), {"kind": "max", "value": 10.0})
        self.assertEqual(
            _report_threshold({"auto": {"min_clip_score": 20.0}}, "mi355", "clip_score"),
            {"kind": "min", "value": 20.0},
        )


class _RecordingReportManager:
    def __init__(self):
        self.is_enabled = True
        self.added = []

    def add_html_to_report(self, html_file, link_name=None, request=None, dest_name=None, **kwargs):
        self.added.append((html_file, link_name, dest_name, kwargs.get("track_in_reports", True)))
        return dest_name


class TestAttachBenchmarkArtifacts(unittest.TestCase):
    def test_copies_flux_and_wan_artifacts_per_host(self):
        with tempfile.TemporaryDirectory() as tmp:
            flux = os.path.join(tmp, "flux_node_outputs", "results")
            wan = os.path.join(tmp, "wan_node_outputs", "outputs")
            os.makedirs(flux)
            os.makedirs(wan)
            flux_timing = os.path.join(flux, "timing.json")
            flux_clip = os.path.join(flux, "clipscore.json")
            flux_png = os.path.join(flux, "flux_result.png")
            wan_video = os.path.join(wan, "video.mp4")
            wan_rank = os.path.join(wan, "rank0_step0.json")
            ignored = os.path.join(flux, "debug.log")
            for path in (flux_timing, flux_clip, flux_png, wan_video, wan_rank, ignored):
                with open(path, "w", encoding="utf-8") as handle:
                    handle.write("x")

            request = SimpleNamespace(config=SimpleNamespace(_html_report_manager=_RecordingReportManager()))
            flux_copied = _attach_benchmark_artifacts(request, "10.0.0.1", os.path.join(tmp, "flux_node_outputs"))
            wan_copied = _attach_benchmark_artifacts(request, "10.0.0.2", os.path.join(tmp, "wan_node_outputs"))

        self.assertEqual(
            flux_copied,
            [
                "10.0.0.1_results_clipscore.json",
                "10.0.0.1_results_flux_result.png",
                "10.0.0.1_results_timing.json",
            ],
        )
        self.assertEqual(
            wan_copied,
            ["10.0.0.2_outputs_rank0_step0.json", "10.0.0.2_outputs_video.mp4"],
        )
        self.assertTrue(all("debug.log" not in name for name in flux_copied))
        for _path, link_name, _dest, tracked in request.config._html_report_manager.added:
            self.assertIsNone(link_name)
            self.assertFalse(tracked)

    def test_skips_empty_media_placeholders(self):
        with tempfile.TemporaryDirectory() as tmp:
            os.makedirs(os.path.join(tmp, "results"))
            with open(os.path.join(tmp, "results", "timing.json"), "w", encoding="utf-8") as handle:
                handle.write("[]")
            with open(os.path.join(tmp, "results", "flux_0.png"), "wb") as handle:
                handle.write(b"")
            request = SimpleNamespace(config=SimpleNamespace(_html_report_manager=_RecordingReportManager()))

            copied = _attach_benchmark_artifacts(request, "10.0.0.1", tmp)

        self.assertEqual(copied, ["10.0.0.1_results_timing.json"])

    def test_skips_when_html_reporting_is_disabled(self):
        request = SimpleNamespace(config=SimpleNamespace())
        self.assertEqual(_attach_benchmark_artifacts(request, "10.0.0.1", "/missing"), [])


class TestParseThresholdsStage(unittest.TestCase):
    def test_reports_node_path_when_parser_reads_staged_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            results = os.path.join(tmp, "results")
            os.makedirs(results)
            with open(os.path.join(results, "timing.json"), "w", encoding="utf-8") as handle:
                handle.write('[{"pipe_time": 1.25}]')
            with open(os.path.join(results, "flux_0.png"), "wb") as handle:
                handle.write(b"x")
            node_dir = "/home/user/flux_host_outputs"
            variant = SimpleNamespace(
                inference={
                    "_test_output_dirs_by_node": {"10.0.0.1": tmp},
                    "_test_node_output_dirs_by_node": {"10.0.0.1": node_dir},
                },
                benchmark_params={
                    "flux1_dev_t2i": {
                        "num_repetitions": 1,
                        "ulysses_degree": 1,
                        "ring_degree": 1,
                        "height": 1024,
                        "width": 1024,
                        "num_inference_steps": 4,
                        "expected_results": {"auto": {"max_avg_pipe_time_s": 10}},
                    }
                },
                thresholds={},
            )
            lifecycle = Lifecycle()
            request = SimpleNamespace(node=SimpleNamespace(nodeid="test_parse"), config=SimpleNamespace())

            parse_thresholds_stage(
                variant,
                "auto",
                {"family": "flux", "distributed": False, "diffusers": False},
                lifecycle,
                request,
            )

        self.assertEqual(lifecycle.results[0][5], node_dir)
        self.assertEqual(next(iter(lifecycle.report_results.values()))["10.0.0.1"]["output_dir"], node_dir)
        self.assertNotIn(tmp, lifecycle.results[0][5])

    def test_parse_failure_names_node_output_path(self):
        variant = SimpleNamespace(
            inference={
                "_test_output_dirs_by_node": {"10.0.0.1": "/tmp/xdit-results-gone"},
                "_test_node_output_dirs_by_node": {"10.0.0.1": "/home/user/flux_host_outputs"},
            },
            benchmark_params={
                "flux1_dev_t2i": {
                    "num_repetitions": 1,
                    "ulysses_degree": 1,
                    "ring_degree": 1,
                    "expected_results": {},
                }
            },
            thresholds={},
        )
        lifecycle = Lifecycle()
        request = SimpleNamespace(node=SimpleNamespace(nodeid="test_parse"), config=SimpleNamespace())

        with patch("cvs.tests.inference.xdit._shared.update_test_result"):
            parse_thresholds_stage(
                variant,
                "auto",
                {"family": "flux", "distributed": False, "diffusers": False},
                lifecycle,
                request,
            )

        message = globals.error_list[-1]
        self.assertIn("/home/user/flux_host_outputs", message)
        self.assertNotIn("xdit-results-gone", message)


class TestOutputDirsByHost(unittest.TestCase):
    def test_prefers_per_node_map(self):
        lifecycle = Lifecycle()
        inference = {
            "_test_output_dirs_by_node": {"node-a": "/out/a", "node-b": "/out/b"},
            "_test_output_dir": "/out/a",
        }

        self.assertEqual(_output_dirs_by_host(inference, lifecycle), {"node-a": "/out/a", "node-b": "/out/b"})

    def test_falls_back_to_benchmark_host(self):
        lifecycle = Lifecycle()
        lifecycle.benchmark_host = "node-a"

        self.assertEqual(_output_dirs_by_host({"_test_output_dir": "/out/a"}, lifecycle), {"node-a": "/out/a"})

    def test_collapses_hosts_that_share_one_output_dir(self):
        lifecycle = Lifecycle()
        lifecycle.benchmark_host = "10.32.80.110"
        inference = {
            "_test_output_dirs_by_node": {
                "10.32.80.110": "/out/flux_rank0_outputs",
                "10.32.80.111": "/out/flux_rank0_outputs",
            }
        }

        self.assertEqual(
            _output_dirs_by_host(inference, lifecycle),
            {"10.32.80.110": "/out/flux_rank0_outputs"},
        )


class TestLogTopology(unittest.TestCase):
    def _messages(self, variant, cluster_dict, spec):
        with patch("cvs.tests.inference.xdit._shared.log") as mock_log:
            log_topology(variant, cluster_dict, spec)
        return [str(call.args[0]) % tuple(call.args[1:]) for call in mock_log.info.call_args_list]

    def test_distributed_flux_lists_ranks_and_world_size(self):
        variant = SimpleNamespace(
            inference={"nnodes": 2, "master_addr": "10.0.0.1", "master_port": 29502},
            benchmark_params={
                "flux1_dev_t2i": {"torchrun_nproc": 8, "ulysses_degree": 8, "ring_degree": 2},
            },
        )
        cluster = {"node_dict": {"10.0.0.1": {}, "10.0.0.2": {}}}

        messages = self._messages(variant, cluster, {"family": "flux", "distributed": True, "diffusers": False})

        self.assertIn("Distributed FLUX topology", messages)
        self.assertIn("  rank 0 -> 10.0.0.1 (8 GPUs)", messages)
        self.assertIn("  rank 1 -> 10.0.0.2 (8 GPUs)", messages)
        self.assertIn("Total GPU ranks (world_size): 16 = 2 nodes × 8 nproc", messages)
        self.assertIn("Rendezvous: 10.0.0.1:29502", messages)
        self.assertIn("Parallelism check: PASS (product 16 == world_size 16)", messages)

    def test_single_node_lists_one_job_per_host(self):
        variant = SimpleNamespace(
            inference={"_execution_hosts": ["10.0.0.1", "10.0.0.2"]},
            benchmark_params={
                "flux1_dev_t2i": {"torchrun_nproc": 8, "ulysses_degree": 8, "ring_degree": 1},
            },
        )

        messages = self._messages(variant, {}, {"family": "flux", "distributed": False, "diffusers": False})

        self.assertIn("Single-node FLUX topology", messages)
        self.assertIn("Independent jobs: 2 (one per node)", messages)
        self.assertIn("Total GPU ranks (world_size): 8 = 1 nodes × 8 nproc", messages)
        self.assertNotIn("Rendezvous: :29500", messages)

    def test_wan_uses_ulysses_and_ring_sizes(self):
        variant = SimpleNamespace(
            inference={"_execution_hosts": ["10.0.0.1"]},
            benchmark_params={"wan22_i2v_a14b": {"torchrun_nproc": 8, "ulysses_size": 8}},
        )

        messages = self._messages(variant, {}, {"family": "wan", "distributed": False, "diffusers": False})

        self.assertIn("Single-node WAN topology", messages)
        self.assertIn("xDiT parallel layout: ulysses=8 × ring=1 = 8", messages)


class TestUlyssesRing(unittest.TestCase):
    def test_flux_reads_degree_keys(self):
        from cvs.tests.inference.xdit._shared import _ulysses_ring

        self.assertEqual(
            _ulysses_ring(
                {"ulysses_degree": 8, "ring_degree": 1},
                {"family": "flux"},
            ),
            (8, 1),
        )

    def test_wan_reads_size_keys(self):
        from cvs.tests.inference.xdit._shared import _ulysses_ring

        self.assertEqual(
            _ulysses_ring(
                {"torchrun_nproc": 8, "ulysses_size": 4, "ring_size": 2},
                {"family": "wan"},
            ),
            (4, 2),
        )


class TestHostScoping(unittest.TestCase):
    def setUp(self):
        self.cluster = {
            "node_dict": {
                "node-a": {"mgmt_ip": "node-a"},
                "node-b": {"mgmt_ip": "node-b"},
                "unused": {"mgmt_ip": "unused"},
            },
            "username": "tester",
        }

    def test_single_uses_every_cluster_node(self):
        hosts = resolve_execution_hosts(self.cluster, {}, distributed=False)

        self.assertEqual(hosts, ["node-a", "node-b", "unused"])

    def test_single_ignores_benchmark_serv_node(self):
        hosts = resolve_execution_hosts(
            self.cluster,
            {"benchmark_serv_node": "node-b"},
            distributed=False,
        )

        self.assertEqual(hosts, ["node-a", "node-b", "unused"])

    def test_scoped_cluster_excludes_unrelated_hosts(self):
        scoped = scoped_cluster_dict(self.cluster, ["node-a", "node-b"])

        self.assertEqual(list(scoped["node_dict"]), ["node-a", "node-b"])
        self.assertEqual(scoped["head_node_dict"], {"mgmt_ip": "node-a"})
        self.assertEqual(scoped["username"], "tester")

    def test_distributed_takes_first_nnodes_from_cluster(self):
        hosts = resolve_execution_hosts(
            self.cluster,
            {"nnodes": 2, "server_node_list": ["unused", "node-b"], "benchmark_serv_node": "node-b"},
            distributed=True,
        )

        self.assertEqual(hosts, ["node-a", "node-b"])

    def test_distributed_rejects_nnodes_larger_than_cluster(self):
        with self.assertRaisesRegex(ValueError, "requests 4 nodes"):
            resolve_execution_hosts(self.cluster, {"nnodes": 4}, distributed=True)

    def test_distributed_rejects_nnodes_less_than_2(self):
        with self.assertRaisesRegex(ValueError, "nnodes >= 2"):
            resolve_execution_hosts(self.cluster, {"nnodes": 1}, distributed=True)

    def test_empty_cluster_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "could not resolve an execution host"):
            resolve_execution_hosts({"node_dict": {}}, {}, distributed=False)


class TestConftestHelpers(unittest.TestCase):
    def test_deep_merge_preserves_cluster_runtime_fields(self):
        merged = conftest._deep_merge(
            {"runtime": {"name": "docker", "args": {"network": "host"}}, "lifetime": "per_run"},
            {"runtime": {"args": {"ipc": "host"}}, "image": "xdit:latest"},
        )

        self.assertEqual(merged["runtime"]["name"], "docker")
        self.assertEqual(merged["runtime"]["args"], {"network": "host", "ipc": "host"})
        self.assertEqual(merged["image"], "xdit:latest")


class TestVerifyModelStage(unittest.TestCase):
    def test_hf_remote_download_reads_model_remote(self):
        self.assertFalse(hf_remote_download_enabled(SimpleNamespace(model=SimpleNamespace(remote=0))))
        self.assertTrue(hf_remote_download_enabled(SimpleNamespace(model=SimpleNamespace(remote=1))))
        self.assertTrue(hf_remote_download_enabled(SimpleNamespace(inference={"model_remote": 1})))

    def test_repo_id_does_not_download_when_remote_is_zero(self):
        orch = MagicMock()
        orch.hosts = ["n1"]
        orch.all.exec.return_value = {"n1": ""}
        variant = SimpleNamespace(
            model=SimpleNamespace(id="org/wan", remote=0),
            inference={"model_repo": "org/wan", "hf_home": "/cache/hf"},
        )
        with (
            patch("cvs.tests.inference.xdit._shared.download_hf_snapshot") as download,
            patch("cvs.tests.inference.xdit._shared.fail_test") as fail,
        ):
            verify_model_stage(orch, variant, {"family": "wan", "diffusers": False}, MagicMock(), MagicMock())
        download.assert_not_called()
        fail.assert_called()
        self.assertIn("no downloads are performed by this test", fail.call_args.args[0])

    def test_repo_id_downloads_when_remote_is_one(self):
        orch = MagicMock()
        orch.hosts = ["n1"]

        orch.all.exec.return_value = {"n1": "MODEL_OK"}
        orch.exec.return_value = {"n1": "CLIP_OK"}
        variant = SimpleNamespace(
            model=SimpleNamespace(id="org/wan", remote=1),
            inference={
                "model_repo": "org/wan",
                "hf_home": "/cache/hf",
                "hf_home_container": "/hf_home",
            },
        )
        with (
            patch(
                "cvs.tests.inference.xdit._shared.download_hf_snapshot",
                return_value=({"/hf_home/hub/snap": "/hf_home/hub/snap"}, []),
            ) as download,
            patch("cvs.tests.inference.xdit._shared.fail_test") as fail,
            patch(
                "cvs.tests.inference.xdit._shared.verify_required_checks_on_nodes",
                return_value=None,
            ),
        ):
            verify_model_stage(orch, variant, {"family": "wan", "diffusers": False}, MagicMock(), MagicMock())
        download.assert_called_once()
        fail.assert_not_called()
        probe = orch.exec.call_args.args[0]
        self.assertIn("HF_HOME=/hf_home", probe)
        self.assertNotIn("snapshot_download", str(orch.all.exec.call_args.args[0]))

    def test_missing_clip_model_fails_even_when_thresholds_are_not_enforced(self):
        orch = MagicMock()
        orch.hosts = ["n1"]

        orch.all.exec.return_value = {"n1": "MODEL_OK"}
        orch.exec.return_value = {"n1": "CLIP_MISSING not in cache"}
        variant = SimpleNamespace(
            model=SimpleNamespace(id="org/wan", remote=1),
            inference={
                "model_repo": "org/wan",
                "hf_home": "/cache/hf",
                "hf_home_container": "/hf_home",
            },
            benchmark_params={"wan22_i2v_a14b": {"clip_model": "/models/clip"}},
            enforce_thresholds=False,
        )
        with (
            patch(
                "cvs.tests.inference.xdit._shared.download_hf_snapshot",
                return_value=({"/hf_home/hub/snap": "/hf_home/hub/snap"}, []),
            ),
            patch("cvs.tests.inference.xdit._shared.fail_test") as fail,
            patch(
                "cvs.tests.inference.xdit._shared.verify_required_checks_on_nodes",
                return_value=None,
            ),
        ):
            verify_model_stage(orch, variant, {"family": "wan", "diffusers": False}, MagicMock(), MagicMock())

        fail.assert_called_once()
        message = fail.call_args.args[0]
        self.assertIn("/models/clip", message)
        self.assertIn("does not download", message)
        self.assertIn("enforce_thresholds", message)
        probe = orch.exec.call_args.args[0]
        self.assertIn("HF_HOME=/hf_home", probe)
        self.assertIn("/models/clip", probe)
        self.assertNotIn("snapshot_download", str(orch.all.exec.call_args.args[0]))


class TestOutputParserRequirements(unittest.TestCase):
    def test_flux1_requires_num_repetitions(self):
        variant = SimpleNamespace(model=SimpleNamespace(id="black-forest-labs/FLUX.1-dev"), inference={})
        params = {"num_repetitions": 25}
        parser = _output_parser(
            {"family": "flux", "distributed": False, "diffusers": False},
            params,
            "/out",
            variant,
        )
        self.assertFalse(_is_flux2_workload(variant, params))
        self.assertEqual(parser.expected_repetitions, 25)

    def test_flux2_does_not_require_flux1_repetition_count(self):
        variant = SimpleNamespace(model=SimpleNamespace(id="black-forest-labs/FLUX.2-dev"), inference={})
        params = {"model_type": "flux2", "num_repetitions": 25}
        parser = _output_parser(
            {"family": "flux", "distributed": False, "diffusers": False},
            params,
            "/out",
            variant,
        )
        self.assertTrue(_is_flux2_workload(variant, params))
        self.assertIsNone(parser.expected_repetitions)

    def test_native_wan_requires_num_benchmark_steps(self):
        variant = SimpleNamespace(inference={})
        parser = _output_parser(
            {"family": "wan", "distributed": False, "diffusers": False},
            {"num_benchmark_steps": 5, "require_video_artifact": True},
            "/out",
            variant,
        )
        self.assertEqual(parser.expected_steps, 5)

    def test_wan_diffusers_does_not_use_native_step_count(self):
        variant = SimpleNamespace(inference={})
        parser = _output_parser(
            {"family": "wan", "distributed": False, "diffusers": True},
            {"num_benchmark_steps": 5, "num_repetitions": 3},
            "/out",
            variant,
        )
        self.assertFalse(hasattr(parser, "expected_steps"))


class TestGpuTypeFromDeviceProbe(unittest.TestCase):
    def test_mi325_device_id_is_not_scored_as_mi300x(self):
        output = "Device ID: 0x74a5\nDevice ID: 0x74a5\n"
        self.assertEqual(conftest.gpu_type_from_device_probe(output), "mi325")

    def test_mi355_device_id_still_selects_mi355(self):
        output = "Device ID: 0x75a3\n"
        self.assertEqual(conftest.gpu_type_from_device_probe(output), "mi355")

    def test_mi350_and_mi300x_device_ids(self):
        self.assertEqual(conftest.gpu_type_from_device_probe("Device ID: 0x75a0\n"), "mi350")
        self.assertEqual(conftest.gpu_type_from_device_probe("Device ID: 0x74a1\n"), "mi300x")

    def test_unrecognized_device_id_stays_mi300x(self):
        self.assertEqual(conftest.gpu_type_from_device_probe("Device ID: 0x1234\n"), "mi300x")

    def test_probe_skips_non_amd_drm_cards(self):
        self.assertIn("${f%/*}/vendor", conftest._GPU_DEVICE_ID_PROBE)
        self.assertIn("0x1002", conftest._GPU_DEVICE_ID_PROBE)


class _ClipOrch:
    def __init__(self, by_host, collect=None):
        self.by_host = by_host
        self.collect = collect or {}
        self.hosts_tried = []
        self.cmds = []

    def exec(self, cmd, hosts=None, timeout=None, detailed=False, print_console=True):
        del timeout, print_console
        host = hosts[0]
        self.hosts_tried.append(host)
        self.cmds.append(cmd)
        if "XDIT_ARTIFACTS_BEGIN" in cmd:
            value = self.collect.get(host, "")
            if detailed:
                value = {"output": value, "exit_code": 0 if value else 1}
            return {host: value}
        value = self.by_host[host]
        if detailed and not isinstance(value, dict):
            value = {"output": value, "exit_code": 0 if "CVS_CLIP_SCORE " in str(value) else 1}
        return {host: value}


def _clip_variant(output_dirs, thresholds=None):
    return SimpleNamespace(
        inference={
            "_test_container_output_dirs_by_node": output_dirs,
            "_test_node_output_dirs_by_node": output_dirs,
            "_test_output_dirs_by_node": output_dirs,
        },
        benchmark_params={
            "flux1_dev_t2i": {
                "prompt": "A small cat",
                "height": 1024,
                "width": 1024,
                "num_inference_steps": 25,
                "num_repetitions": 1,
                "ulysses_degree": 8,
                "ring_degree": 1,
                "expected_results": thresholds or {"auto": {"max_avg_pipe_time_s": 10.0, "min_clip_score": 20.0}},
            }
        },
        thresholds={},
        enforce_thresholds=True,
        model=SimpleNamespace(id="black-forest-labs/FLUX.1-dev"),
    )


def _wan_clip_variant(output_dirs, diffusers=False, thresholds=None):
    timing = "max_avg_pipe_time_s" if diffusers else "max_avg_total_time_s"
    model_id = "Wan-AI/Wan2.2-I2V-A14B-Diffusers" if diffusers else "Wan-AI/Wan2.2-I2V-A14B"
    return SimpleNamespace(
        inference={
            "_test_container_output_dirs_by_node": output_dirs,
            "_test_node_output_dirs_by_node": output_dirs,
            "_test_output_dirs_by_node": output_dirs,
        },
        benchmark_params={
            "wan22_i2v_a14b": {
                "prompt": "A cat walking",
                "size": "720*1280",
                "frame_num": 81,
                "num_inference_steps": 40,
                "num_benchmark_steps": 1,
                "torchrun_nproc": 8,
                "ulysses_size": 8,
                "ring_size": 1,
                "expected_results": thresholds or {"auto": {timing: 300.0, "min_mean_frame_clip_score": 20.0}},
            }
        },
        thresholds={},
        enforce_thresholds=True,
        model=SimpleNamespace(id=model_id),
    )


def _remote_clipscore_text(payload):
    blob = json.dumps(payload).encode("utf-8")
    items = [{"rel": "results/clipscore.json", "b64": base64.b64encode(blob).decode("ascii")}]
    return f"XDIT_ARTIFACTS_BEGIN\n{json.dumps(items)}\nXDIT_ARTIFACTS_END\n"


class TestClipScoreStage(unittest.TestCase):
    def setUp(self):
        globals.error_list = []

    def tearDown(self):
        cleanup_staged_artifacts()

    def test_records_passing_score_from_container_output(self):
        payload = {"clip_score": 27.5, "image_count": 1, "model": "openai/clip-vit-large-patch14", "scores": []}
        orch = _ClipOrch({"10.0.0.1": f"loading\nCVS_CLIP_SCORE {json.dumps(payload)}\n"})
        variant = _clip_variant({"10.0.0.1": "/outputs/flux_host_outputs"})
        lifecycle = Lifecycle()
        request = SimpleNamespace(node=SimpleNamespace(nodeid="test_clip_score"), config=SimpleNamespace())

        clip_score_stage(
            orch,
            variant,
            "auto",
            {"family": "flux", "distributed": False, "diffusers": False},
            lifecycle,
            request,
        )

        self.assertEqual(globals.error_list, [])
        self.assertEqual(lifecycle.results[0][6], "clip_score")
        self.assertEqual(lifecycle.results[0][7], 27.5)
        self.assertTrue(lifecycle.results[0][8])
        self.assertEqual(variant.thresholds["SIZE=1024x1024,STEPS=25,BENCH=1"]["clip_score"]["kind"], "min")

    def test_tries_next_host_when_shared_directory_has_no_image(self):
        payload = {"clip_score": 22.0, "image_count": 1, "model": "openai/clip-vit-large-patch14"}
        orch = _ClipOrch(
            {
                "10.0.0.1": {"output": "No FLUX PNG images found", "exit_code": 1},
                "10.0.0.2": {"output": f"CVS_CLIP_SCORE {json.dumps(payload)}", "exit_code": 0},
            }
        )
        variant = _clip_variant(
            {
                "10.0.0.1": "/outputs/flux_rank0_outputs",
                "10.0.0.2": "/outputs/flux_rank0_outputs",
            }
        )
        lifecycle = Lifecycle()
        lifecycle.benchmark_host = "10.0.0.1"
        request = SimpleNamespace(node=SimpleNamespace(nodeid="test_clip_score"), config=SimpleNamespace())

        clip_score_stage(
            orch,
            variant,
            "auto",
            {"family": "flux", "distributed": True, "diffusers": False},
            lifecycle,
            request,
        )

        self.assertEqual(orch.hosts_tried[:2], ["10.0.0.1", "10.0.0.2"])
        self.assertEqual(orch.hosts_tried[-1], "10.0.0.2")
        self.assertEqual(globals.error_list, [])
        self.assertEqual(lifecycle.results[0][0], "10.0.0.1")
        self.assertEqual(lifecycle.results[0][7], 22.0)

    def test_failure_below_floor_is_recorded(self):
        payload = {"clip_score": 11.0, "image_count": 1, "model": "openai/clip-vit-large-patch14"}
        orch = _ClipOrch({"10.0.0.1": {"output": f"CVS_CLIP_SCORE {json.dumps(payload)}", "exit_code": 0}})
        variant = _clip_variant({"10.0.0.1": "/outputs/flux_host_outputs"})
        lifecycle = Lifecycle()
        request = SimpleNamespace(node=SimpleNamespace(nodeid="test_clip_score"), config=SimpleNamespace())

        with patch("cvs.tests.inference.xdit._shared.update_test_result"):
            clip_score_stage(
                orch,
                variant,
                "auto",
                {"family": "flux", "distributed": False, "diffusers": False},
                lifecycle,
                request,
            )

        self.assertTrue(lifecycle.failed)
        self.assertTrue(any("11.00" in message for message in globals.error_list))

    def test_reads_clipscore_file_when_output_dir_is_not_mounted(self):
        payload = {"clip_score": 27.5, "image_count": 1, "model": "openai/clip-vit-base-patch32"}
        orch = _ClipOrch(
            {"10.0.0.1": f"CVS_CLIP_SCORE {json.dumps(payload)}\n"},
            collect={"10.0.0.1": _remote_clipscore_text(payload)},
        )
        variant = _clip_variant({"10.0.0.1": "/outputs/flux_host_outputs"})
        lifecycle = Lifecycle()
        copied = []

        def _attach(request, host, output_dir):
            del request, host
            copied.append(output_dir)
            return []

        request = SimpleNamespace(node=SimpleNamespace(nodeid="test_clip_score"), config=SimpleNamespace())
        with patch("cvs.tests.inference.xdit._shared._attach_benchmark_artifacts", _attach):
            clip_score_stage(
                orch,
                variant,
                "auto",
                {"family": "flux", "distributed": False, "diffusers": False},
                lifecycle,
                request,
            )

        self.assertEqual(globals.error_list, [])
        self.assertTrue(any("XDIT_ARTIFACTS_BEGIN" in cmd for cmd in orch.cmds))
        self.assertIn("/outputs/flux_host_outputs", orch.cmds[-1])
        self.assertEqual(len(copied), 1)
        with open(os.path.join(copied[0], "results", "clipscore.json"), encoding="utf-8") as handle:
            saved = json.load(handle)
        self.assertEqual(saved["clip_score"], 27.5)
        self.assertFalse(os.path.exists("/outputs/flux_host_outputs"))

    def test_wan_native_records_mean_frame_score(self):
        payload = {
            "mean_frame_clip_score": 27.5,
            "min_frame_clip_score": 21.0,
            "sample_count": 8,
            "frame_indices": [0, 11, 23, 34, 46, 57, 69, 80],
            "model": "openai/clip-vit-base-patch32",
            "video": "video.mp4",
        }
        orch = _ClipOrch(
            {"10.0.0.1": f"CVS_CLIP_SCORE {json.dumps(payload)}\n"},
            collect={"10.0.0.1": _remote_clipscore_text(payload)},
        )
        variant = _wan_clip_variant({"10.0.0.1": "/outputs/wan_host_outputs"})
        lifecycle = Lifecycle()
        copied = []

        def _attach(request, host, output_dir):
            del request, host
            copied.append(output_dir)
            return []

        request = SimpleNamespace(node=SimpleNamespace(nodeid="test_clip_score"), config=SimpleNamespace())
        with patch("cvs.tests.inference.xdit._shared._attach_benchmark_artifacts", _attach):
            clip_score_stage(
                orch,
                variant,
                "auto",
                {"family": "wan", "distributed": False, "diffusers": False},
                lifecycle,
                request,
            )

        self.assertEqual(globals.error_list, [])
        self.assertIn("--sample-frames 8", orch.cmds[0])
        self.assertIn("--video-name video.mp4", orch.cmds[0])
        self.assertEqual(lifecycle.results[0][6], "mean_frame_clip_score")
        self.assertEqual(lifecycle.results[0][7], 27.5)
        self.assertTrue(lifecycle.results[0][8])
        cell = "SIZE=720*1280,FRAMES=81,STEPS=40,BENCH=1"
        self.assertEqual(variant.thresholds[cell]["mean_frame_clip_score"]["kind"], "min")
        self.assertEqual(len(copied), 1)
        with open(os.path.join(copied[0], "results", "clipscore.json"), encoding="utf-8") as handle:
            saved = json.load(handle)
        self.assertEqual(saved["mean_frame_clip_score"], 27.5)

    def test_wan_diffusers_command_uses_i2v_video(self):
        payload = {
            "mean_frame_clip_score": 22.0,
            "sample_count": 8,
            "model": "openai/clip-vit-base-patch32",
            "video": "video_i2v.mp4",
        }
        orch = _ClipOrch({"10.0.0.1": f"CVS_CLIP_SCORE {json.dumps(payload)}\n"})
        variant = _wan_clip_variant({"10.0.0.1": "/outputs/wan_host_outputs"}, diffusers=True)
        lifecycle = Lifecycle()
        request = SimpleNamespace(node=SimpleNamespace(nodeid="test_clip_score"), config=SimpleNamespace())

        clip_score_stage(
            orch,
            variant,
            "auto",
            {"family": "wan", "distributed": False, "diffusers": True},
            lifecycle,
            request,
        )

        self.assertEqual(globals.error_list, [])
        self.assertIn("--video-name video_i2v.mp4", orch.cmds[0])
        self.assertEqual(lifecycle.results[0][7], 22.0)

    def test_wan_frame_score_below_floor_is_recorded(self):
        payload = {"mean_frame_clip_score": 11.0, "sample_count": 8, "model": "openai/clip-vit-base-patch32"}
        orch = _ClipOrch({"10.0.0.1": {"output": f"CVS_CLIP_SCORE {json.dumps(payload)}", "exit_code": 0}})
        variant = _wan_clip_variant({"10.0.0.1": "/outputs/wan_host_outputs"})
        lifecycle = Lifecycle()
        request = SimpleNamespace(node=SimpleNamespace(nodeid="test_clip_score"), config=SimpleNamespace())

        with patch("cvs.tests.inference.xdit._shared.update_test_result"):
            clip_score_stage(
                orch,
                variant,
                "auto",
                {"family": "wan", "distributed": False, "diffusers": False},
                lifecycle,
                request,
            )

        self.assertTrue(lifecycle.failed)
        self.assertTrue(any("11.00" in message for message in globals.error_list))

    def test_wan_missing_video_fails(self):
        orch = _ClipOrch({"10.0.0.1": {"output": "No WAN video video.mp4 found under /outputs/wan", "exit_code": 1}})
        variant = _wan_clip_variant({"10.0.0.1": "/outputs/wan_host_outputs"})
        lifecycle = Lifecycle()
        request = SimpleNamespace(node=SimpleNamespace(nodeid="test_clip_score"), config=SimpleNamespace())

        with patch("cvs.tests.inference.xdit._shared.update_test_result"):
            clip_score_stage(
                orch,
                variant,
                "auto",
                {"family": "wan", "distributed": False, "diffusers": False},
                lifecycle,
                request,
            )

        self.assertTrue(lifecycle.failed)
        self.assertTrue(any("No WAN video" in message for message in globals.error_list))


if __name__ == "__main__":
    unittest.main()
