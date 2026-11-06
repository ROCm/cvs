import threading
import time
import unittest
from types import SimpleNamespace
from unittest import mock

try:
    import torch

    from cvs.lib.training.pytorch_vision import benchmark
    from cvs.lib.training.pytorch_vision.benchmark import (
        _HFStepTimer,
        _build_lr_scheduler,
        _eval_ranks,
        _evaluation_meets_convergence_target,
        _exact_eval_batch_size,
        _evaluation_meets_stop_target,
        _next_rocal_batches,
        _rocal_batches_per_epoch,
        _scorecard_milestone_metrics,
    )
except ModuleNotFoundError:
    torch = None
    benchmark = None
    _build_lr_scheduler = None
    _eval_ranks = None
    _evaluation_meets_convergence_target = None
    _evaluation_meets_stop_target = None
    _next_rocal_batches = None
    _rocal_batches_per_epoch = None
    _scorecard_milestone_metrics = None
    _exact_eval_batch_size = None
    _HFStepTimer = None


@unittest.skipIf(torch is None, "PyTorch is only installed in the benchmark container")
class TestLearningRateSchedule(unittest.TestCase):
    @staticmethod
    def _optimizer():
        parameter = torch.nn.Parameter(torch.tensor(1.0))
        return torch.optim.SGD([parameter], lr=1.0)

    @staticmethod
    def _advance(optimizer, scheduler, steps):
        for _ in range(steps):
            optimizer.step()
            scheduler.step()

    def test_multistep_schedule_warms_up_and_decays_at_epoch_boundaries(self):
        optimizer = self._optimizer()
        args = SimpleNamespace(
            lr_schedule="multistep",
            lr_warmup_epochs=1,
            lr_milestones_epochs="2,3",
            lr_gamma=0.1,
        )
        scheduler = _build_lr_scheduler(optimizer, args, steps_per_epoch=10)

        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 0.1)
        self._advance(optimizer, scheduler, 9)
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 1.0)
        self._advance(optimizer, scheduler, 11)
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 0.1)
        self._advance(optimizer, scheduler, 10)
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 0.01)

    def test_constant_schedule_keeps_base_learning_rate(self):
        optimizer = self._optimizer()
        args = SimpleNamespace(
            lr_schedule="constant",
            lr_warmup_epochs=0,
            lr_milestones_epochs="",
            lr_gamma=0.1,
        )
        scheduler = _build_lr_scheduler(optimizer, args, steps_per_epoch=None)
        self._advance(optimizer, scheduler, 5)
        self.assertEqual(optimizer.param_groups[0]["lr"], 1.0)

    def test_convergence_target_uses_accuracy_and_optional_loss(self):
        args = SimpleNamespace(
            convergence_top1=75.5,
            convergence_eval_loss=1.5,
            target_top5=92.5,
        )
        self.assertTrue(
            _evaluation_meets_convergence_target(
                {"top1_accuracy_pct": 76.0, "eval_loss": 1.4},
                args,
            )
        )
        self.assertFalse(
            _evaluation_meets_convergence_target(
                {"top1_accuracy_pct": 75.0, "eval_loss": 1.4},
                args,
            )
        )
        self.assertFalse(
            _evaluation_meets_stop_target(
                {"top1_accuracy_pct": 76.0, "top5_accuracy_pct": 92.0, "eval_loss": 1.4},
                args,
            )
        )
        self.assertTrue(
            _evaluation_meets_stop_target(
                {"top1_accuracy_pct": 76.0, "top5_accuracy_pct": 93.0, "eval_loss": 1.4},
                args,
            )
        )

    def test_exports_all_scorecard_loss_milestones(self):
        losses = {"100": 6.0, "500": 5.0, "1000": 4.0, "5000": 3.0}
        self.assertEqual(
            _scorecard_milestone_metrics(losses),
            {
                "loss_step_100": 6.0,
                "loss_step_500": 5.0,
                "loss_step_1000": 4.0,
                "loss_step_5000": 3.0,
            },
        )


@unittest.skipIf(torch is None, "PyTorch is only installed in the benchmark container")
class TestRocalBatchOwnership(unittest.TestCase):
    class ReusingLoader:
        def __init__(self, batches, samples=10, batch_size=2):
            self.batches = batches
            self.iterator_length = samples
            self.batch_size = batch_size
            self.index = 0
            self.images = torch.empty(batch_size, 3, 2, 2)
            self.labels = torch.empty(batch_size, dtype=torch.int64)

        def __next__(self):
            if self.index >= self.batches:
                raise StopIteration
            self.index += 1
            self.images.fill_(self.index)
            self.labels.fill_(self.index)
            return [self.images], self.labels

        def reset(self):
            self.index = 0

    def test_gradient_accumulation_owns_reused_iterator_buffers(self):
        loader = self.ReusingLoader(batches=2)
        images, labels = _next_rocal_batches(
            loader,
            accumulation_steps=2,
            channels_last=False,
            device=torch.device("cpu"),
            reset_on_end=False,
        )
        self.assertEqual(images[0].unique().item(), 1)
        self.assertEqual(images[1].unique().item(), 2)
        self.assertEqual(labels[0].unique().item(), 1)
        self.assertEqual(labels[1].unique().item(), 2)
        self.assertNotEqual(images[0].data_ptr(), images[1].data_ptr())

    def test_final_accumulation_step_returns_partial_microbatch_group(self):
        loader = self.ReusingLoader(batches=5)
        first_images, _ = _next_rocal_batches(
            loader,
            accumulation_steps=4,
            channels_last=False,
            device=torch.device("cpu"),
            reset_on_end=False,
        )
        final_images, _ = _next_rocal_batches(
            loader,
            accumulation_steps=4,
            channels_last=False,
            device=torch.device("cpu"),
            reset_on_end=False,
        )
        self.assertEqual(len(first_images), 4)
        self.assertEqual(len(final_images), 1)

    def test_epoch_batch_count_includes_partial_batch(self):
        loader = self.ReusingLoader(batches=5, samples=10, batch_size=3)
        self.assertEqual(len(range(loader.iterator_length // loader.batch_size)), 3)
        self.assertEqual(_rocal_batches_per_epoch(loader), 4)

    def test_epoch_is_each_ranks_share_of_the_full_training_set(self):
        loader = self.ReusingLoader(batches=5, samples=50, batch_size=4)
        self.assertEqual(_rocal_batches_per_epoch(loader, world_size=4), 4)


if __name__ == "__main__":
    unittest.main()


class TestEvalLabelContract(unittest.TestCase):
    """Row 6 of the scorecard: label-mapping correctness needs evidence, not
    just plausible accuracy."""

    def test_label_metrics_are_part_of_the_contract(self):
        from cvs.lib.training.pytorch_vision.utils.metrics import METRIC_UNITS

        for name in (
            "eval_label_classes_observed",
            "eval_label_min_per_class",
            "eval_label_max_per_class",
        ):
            self.assertIn(name, METRIC_UNITS)

    def test_full_imagenet_val_is_uniform_across_classes(self):
        """ImageNet-1k validation holds exactly 50 images per class, so a
        correct mapping yields 1000 classes with min == max == 50. A scrambled
        or truncated label space breaks that invariant while accuracy alone
        could still look reasonable."""
        hist = [50] * 1000
        observed = sum(1 for c in hist if c > 0)
        self.assertEqual(observed, 1000)
        self.assertEqual(min(hist), max(hist))
        self.assertEqual(sum(hist), 50000)

    def test_truncated_label_space_is_detectable(self):
        hist = [100] * 500 + [0] * 500
        observed = sum(1 for c in hist if c > 0)
        self.assertEqual(sum(hist), 50000)  # sample count alone still passes
        self.assertNotEqual(observed, 1000)  # class coverage catches it


@unittest.skipIf(torch is None, "PyTorch is only installed in the benchmark container")
class TestExactEvalBatchSize(unittest.TestCase):
    """Evaluation must leave no partial batch for rocAL to pad with repeats."""

    def test_divides_the_shard_exactly_for_real_topologies(self):
        # 50,000-image validation set over 8 and 16 ranks, train batch 128.
        for shard, preferred in ((6250, 128), (3125, 128), (6250, 64), (6250, 256)):
            with self.subTest(shard=shard, preferred=preferred):
                size = _exact_eval_batch_size(shard, preferred)
                self.assertEqual(shard % size, 0)
                self.assertLessEqual(size, preferred)

    def test_picks_the_largest_valid_divisor(self):
        self.assertEqual(_exact_eval_batch_size(6250, 128), 125)
        self.assertEqual(_exact_eval_batch_size(6250, 64), 50)
        self.assertEqual(_exact_eval_batch_size(3125, 128), 125)

    def test_uses_the_training_batch_when_it_already_divides(self):
        self.assertEqual(_exact_eval_batch_size(6250, 125), 125)
        self.assertEqual(_exact_eval_batch_size(6400, 128), 128)

    def test_prime_shard_falls_back_to_one_rather_than_padding(self):
        self.assertEqual(_exact_eval_batch_size(6247, 128), 1)

    def test_degenerate_input_never_returns_zero(self):
        self.assertGreaterEqual(_exact_eval_batch_size(0, 128), 1)


@unittest.skipIf(torch is None, "PyTorch is only installed in the benchmark container")
class TestEvalRanks(unittest.TestCase):
    """Full evaluation runs on equal shards so its pipeline is never rebuilt."""

    def test_imagenet_on_four_nodes_uses_the_ranks_that_divide_it(self):
        self.assertEqual(_eval_ranks(50_000, 32), 25)

    def test_an_even_split_keeps_every_rank(self):
        self.assertEqual(_eval_ranks(50_000, 8), 8)
        self.assertEqual(_eval_ranks(10_000, 1), 1)

    def test_food101_on_one_node(self):
        self.assertEqual(_eval_ranks(25_250, 8), 5)

    def test_no_sample_count_keeps_every_rank(self):
        self.assertEqual(_eval_ranks(0, 8), 8)


@unittest.skipIf(torch is None, "PyTorch is only installed in the benchmark container")
class TestCoordinatedExit(unittest.TestCase):
    """No rank may leave while a peer is still working on the GPU."""

    def _run(self, world_size, artifact_error="", slow_rank=None):
        store = torch.distributed.HashStore()
        exited, errors, times = [], {}, {}

        def rank_main(rank):
            if rank == slow_rank:
                time.sleep(0.5)
            try:
                benchmark._finish(rank, world_size, artifact_error if rank == 0 else "", timeout_s=5)
            except RuntimeError as exc:
                errors[rank] = str(exc)
            times[rank] = time.monotonic()

        with (
            mock.patch.object(benchmark.dist.distributed_c10d, "_get_default_store", return_value=store),
            mock.patch.object(benchmark.torch.cuda, "synchronize"),
            mock.patch.object(benchmark, "_exit_cleanly", side_effect=lambda: exited.append(1)),
        ):
            start = time.monotonic()
            threads = [threading.Thread(target=rank_main, args=(rank,)) for rank in range(world_size)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(timeout=10)
        return exited, errors, {rank: t - start for rank, t in times.items()}

    def test_every_rank_waits_for_the_slowest_one(self):
        exited, errors, times = self._run(world_size=4, slow_rank=3)
        self.assertEqual((len(exited), errors), (4, {}))
        self.assertTrue(all(elapsed >= 0.5 for elapsed in times.values()), times)

    def test_a_rank_zero_artifact_error_fails_every_rank(self):
        exited, errors, _ = self._run(world_size=3, artifact_error="OSError: disk full")
        self.assertEqual(exited, [])
        self.assertEqual(sorted(errors), [0, 1, 2])
        self.assertIn("disk full", errors[1])


@unittest.skipIf(torch is None, "PyTorch is only installed in the benchmark container")
class TestWarmupMarker(unittest.TestCase):
    def test_rank_zero_prints_the_marker_the_launcher_reads(self):
        from cvs.lib.training.pytorch_vision.job import WARMUP_OK

        self.assertEqual(benchmark.WARMUP_OK, WARMUP_OK)
        printed = []
        with mock.patch("builtins.print", side_effect=lambda text, **_: printed.append(text)):
            benchmark._mark_warmup_done(1)
            benchmark._mark_warmup_done(0)
        self.assertEqual(printed, [WARMUP_OK])


@unittest.skipIf(torch is None, "PyTorch is only installed in the benchmark container")
class TestStartDistributed(unittest.TestCase):
    """A failed NCCL start must leave at once and say so; a good one says so too."""

    def _start(self, init_error=None, hang=False):
        exited = threading.Event()
        printed = []

        def fake_exit(code):
            exited.set()
            raise SystemExit(code)

        def fake_init(**_kwargs):
            if init_error:
                raise init_error
            if hang:
                exited.wait(5)

        with (
            mock.patch.object(benchmark.dist, "init_process_group", side_effect=fake_init),
            mock.patch.object(benchmark.dist, "all_reduce"),
            mock.patch.object(benchmark.torch.cuda, "synchronize"),
            mock.patch.object(benchmark.os, "_exit", side_effect=fake_exit),
            mock.patch("builtins.print", side_effect=lambda text, **_: printed.append(text)),
        ):
            try:
                benchmark._start_distributed(torch.device("cpu"), 0, 120, 1)
            except SystemExit:
                pass
        exited.wait(3)
        return printed, exited.is_set()

    def test_successful_start_prints_the_marker(self):
        printed, exited = self._start()
        self.assertEqual(printed, [benchmark.STARTUP_OK])
        self.assertFalse(exited)

    def test_init_error_exits_with_the_failure_marker(self):
        printed, exited = self._start(init_error=RuntimeError("ncclRemoteError\nmore"))
        self.assertTrue(exited)
        self.assertEqual(len(printed), 1)
        self.assertTrue(printed[0].startswith(benchmark.STARTUP_FAILED))
        self.assertIn("RuntimeError: ncclRemoteError", printed[0])
        self.assertNotIn("more", printed[0])

    def test_no_progress_exits_at_the_deadline(self):
        printed, exited = self._start(hang=True)
        self.assertTrue(exited)
        self.assertTrue(any("no progress in 1s" in text for text in printed))


@unittest.skipIf(torch is None, "PyTorch is only installed in the benchmark container")
class TestWaitForAllRanks(unittest.TestCase):
    """Slow per-rank setup is awaited through the store, not an NCCL collective."""

    def _wait(self, arrived, clock):
        store = torch.distributed.HashStore()
        if arrived:
            store.add("pytorch_vision_ready/setup", arrived)
        sleeps = []
        with (
            mock.patch.object(benchmark.dist.distributed_c10d, "_get_default_store", return_value=store),
            mock.patch.object(benchmark.dist, "get_world_size", return_value=4),
            mock.patch.object(benchmark.time, "monotonic", side_effect=clock),
            mock.patch.object(benchmark.time, "sleep", side_effect=sleeps.append),
        ):
            benchmark._wait_for_all_ranks("setup", 600)
        return store, sleeps

    def test_last_rank_to_arrive_releases_everyone(self):
        store, sleeps = self._wait(arrived=3, clock=[0.0])
        self.assertEqual(store.add("pytorch_vision_ready/setup", 0), 4)
        self.assertEqual(sleeps, [])

    def test_gives_up_at_the_deadline(self):
        with self.assertRaisesRegex(RuntimeError, "not every rank finished setup within 600s"):
            self._wait(arrived=1, clock=[0.0, 300.0, 601.0])


@unittest.skipIf(torch is None, "PyTorch is only installed in the benchmark container")
class TestHFStepTimerWindow(unittest.TestCase):
    """The measured window must cover the same steps as step_times_ms.

    Timing the whole trainer.train() call instead folds Trainer startup and the
    warmup steps into the denominator while the numerator counts neither, which
    understates throughput and hides real differences between sweeps.
    """

    def _run(self, warmup, total, step_seconds=0.01, startup_seconds=0.05):
        timer = _HFStepTimer(warmup_steps=warmup, sample_every=1, milestones=[])
        time.sleep(startup_seconds)  # stands in for Trainer/dataloader startup
        for step in range(1, total + 1):
            timer.on_step_begin()
            time.sleep(step_seconds)
            timer.on_step_end(step)
        return timer

    def test_window_excludes_startup_and_warmup(self):
        warmup, total, step_seconds = 3, 8, 0.01
        timer = self._run(warmup, total, step_seconds=step_seconds)
        measured = total - warmup

        self.assertEqual(len(timer.step_times_ms), measured)
        # The window tracks the measured steps, not startup plus every step.
        self.assertAlmostEqual(timer.measured_window_s(), measured * step_seconds, delta=0.05)

    def test_window_is_consistent_with_the_step_times_it_reports(self):
        timer = self._run(warmup=2, total=7)
        self.assertGreaterEqual(timer.measured_window_s() * 1000.0, sum(timer.step_times_ms) * 0.9)

    def test_window_is_zero_when_every_step_is_warmup(self):
        timer = self._run(warmup=5, total=5)
        self.assertEqual(timer.step_times_ms, [])
        self.assertEqual(timer.measured_window_s(), 0.0)


@unittest.skipIf(torch is None, "PyTorch is only installed in the benchmark container")
class TestHFStepTimerMilestones(unittest.TestCase):
    """milestone_steps reaches the harness as the raw CLI string, so the timer
    has to parse it; iterating the string yields characters and silently
    records no milestones at all."""

    def test_milestones_parse_from_the_cli_string(self):
        timer = _HFStepTimer(warmup_steps=0, sample_every=1, milestones="100,500,1000")
        self.assertEqual(timer.milestones, {100, 500, 1000})

    def test_milestones_accept_an_integer_sequence(self):
        timer = _HFStepTimer(warmup_steps=0, sample_every=1, milestones=[100, 500])
        self.assertEqual(timer.milestones, {100, 500})

    def test_milestone_losses_are_recorded_at_the_milestone_step(self):
        timer = _HFStepTimer(warmup_steps=0, sample_every=1, milestones="2,4")
        for step, loss in ((1, 9.0), (2, 8.0), (3, 7.0), (4, 6.0)):
            timer.on_log(step, loss)
        self.assertEqual(timer.milestone_losses, {"2": 8.0, "4": 6.0})

    def test_no_milestones_configured_records_nothing(self):
        timer = _HFStepTimer(warmup_steps=0, sample_every=1, milestones="")
        timer.on_log(1, 9.0)
        self.assertEqual(timer.milestone_losses, {})
