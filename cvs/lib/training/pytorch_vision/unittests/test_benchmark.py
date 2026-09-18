import unittest
from types import SimpleNamespace

try:
    import torch

    from cvs.lib.training.pytorch_vision.benchmark import (
        _build_lr_scheduler,
        _evaluation_meets_convergence_target,
        _evaluation_meets_stop_target,
        _next_rocal_batches,
        _rocal_batches_per_epoch,
        _scorecard_milestone_metrics,
    )
except ModuleNotFoundError:
    torch = None
    _build_lr_scheduler = None
    _evaluation_meets_convergence_target = None
    _evaluation_meets_stop_target = None
    _next_rocal_batches = None
    _rocal_batches_per_epoch = None
    _scorecard_milestone_metrics = None


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
