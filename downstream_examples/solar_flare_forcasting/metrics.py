import torch

from torch.distributed import all_reduce, ReduceOp
from surya.utils.distributed import is_dist_avail_and_initialized


class DistributedClassificationMetrics:
    """
    We store internally true positives, false positives, true negatives and false negatives.

    Then
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        f1 = 2*tp / (2*tp + fp + fn) = 2 * precision * recall / (precision + recall)
    """

    def __init__(self, *args, threshold: float = 0.5, **kwargs):
        assert 0.0 < threshold < 1.0, "`threshold` should lie in (0, 1)."

        self.metrics = [
            "accuracy",
            "precision",
            "recall",
            "f1",
            "tp",
            "tn",
            "fp",
            "fn",
            "tss",
            "hss",
        ]
        self.threshold = threshold
        self._counts_tp_tn_fp_fn = torch.zeros(4)

    @property
    def tp(self):
        return self._counts_tp_tn_fp_fn[0]

    @property
    def tn(self):
        return self._counts_tp_tn_fp_fn[1]

    @property
    def fp(self):
        return self._counts_tp_tn_fp_fn[2]

    @property
    def fn(self):
        return self._counts_tp_tn_fp_fn[3]

    def compute_and_reset(self):
        if is_dist_avail_and_initialized():
            all_reduce(self._counts_tp_tn_fp_fn, op=ReduceOp.SUM)

        result = {
            "tp": self.tp,
            "tn": self.tn,
            "fp": self.fp,
            "fn": self.fn,
        }
        result["accuracy"] = ((self.tp + self.tn) / (self.tp + self.tn + self.fp + self.fn)).cpu()
        result["precision"] = (self.tp / (self.tp + self.fp)).cpu()
        result["recall"] = (self.tp / (self.tp + self.fn)).cpu()
        result["f1"] = (2 * self.tp / (2 * self.tp + self.fp + self.fn)).cpu()
        result["tss"] = (self.tp / (self.tp + self.fn) - self.fp / (self.fp + self.tn)).cpu()

        n = self.tn + self.fp
        p = self.tp + self.fn
        result["hss"] = (
            2.0
            * (self.tp * self.tn - self.fn * self.fp)
            / (p * (self.fn + self.tn) + (self.tp + self.fp) * n)
        ).cpu()

        self.reset()

        return result

    def update(self, prediction, target):
        if self._counts_tp_tn_fp_fn.device != prediction.device:
            self._counts_tp_tn_fp_fn = self._counts_tp_tn_fp_fn.to(device=prediction.device)

        prediction = (prediction > self.threshold).to(dtype=torch.int)
        target = target.to(dtype=torch.int)

        self._counts_tp_tn_fp_fn[0] += torch.logical_and(prediction == 1, target == 1).sum()
        self._counts_tp_tn_fp_fn[1] += torch.logical_and(prediction == 0, target == 0).sum()
        self._counts_tp_tn_fp_fn[2] += torch.logical_and(prediction == 1, target == 0).sum()
        self._counts_tp_tn_fp_fn[3] += torch.logical_and(prediction == 0, target == 1).sum()

    def reset(self):
        self._counts_tp_tn_fp_fn = torch.zeros_like(self._counts_tp_tn_fp_fn)
