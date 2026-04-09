from collections import defaultdict
import math
from typing import List, Optional, Union

import numpy as np
import pandas as pd


class SegmentationMetricsFactory:
    """'Factory' class to accumulate results and compute metrics for segmentation tasks

    :param n_classes: Number of classes to evaluate
    :type n_classes: int
    """

    METRIC_NAMES = [
        "tp",
        "fp",
        "fn",
        "tn",
        "precision",
        "recall",
        "accuracy",
        "f1_score",
        "iou",
    ]

    def __init__(self, n_classes: int):
        self.n_classes = n_classes
        self.confusion_matrix = np.zeros((n_classes, n_classes), dtype=np.int64)

    def update(
        self, pred: np.ndarray, gt: np.ndarray, valid_mask: Optional[np.ndarray] = None
    ):
        """Accumulate results for a new batch

        :param pred: Array containing prediction
        :type pred: np.ndarray
        :param gt: Array containing ground truth
        :type gt: np.ndarray
        :param valid_mask: Binary mask where False elements will be igonred, defaults to None
        :type valid_mask: Optional[np.ndarray], optional
        """
        if pred.shape != gt.shape:
            raise ValueError(
                "Prediction and GT shapes don't match: "
                f"pred.shape={pred.shape}, gt.shape={gt.shape}"
            )
        if not np.issubdtype(pred.dtype, np.integer):
            raise TypeError(f"Prediction should be integer, got dtype={pred.dtype}")
        if not np.issubdtype(gt.dtype, np.integer):
            raise TypeError(f"GT should be integer, got dtype={gt.dtype}")

        if valid_mask is not None:
            if not isinstance(valid_mask, np.ndarray):
                raise TypeError(
                    "valid_mask should be a numpy.ndarray when provided, "
                    f"got type={type(valid_mask).__name__}"
                )
            if valid_mask.shape != gt.shape:
                raise ValueError(
                    "valid_mask shape does not match GT shape: "
                    f"valid_mask.shape={valid_mask.shape}, gt.shape={gt.shape}"
                )
            if not (
                np.issubdtype(valid_mask.dtype, np.bool_)
                or np.issubdtype(valid_mask.dtype, np.integer)
            ):
                raise TypeError(
                    "valid_mask should have a boolean/integer dtype, "
                    f"got dtype={valid_mask.dtype}"
                )
            valid_mask = valid_mask.astype(bool, copy=False)

        # Build mask of valid elements
        mask = (gt >= 0) & (gt < self.n_classes)
        if valid_mask is not None:
            mask &= valid_mask

        # Update confusion matrix
        new_entry = np.bincount(
            self.n_classes * gt[mask].astype(int) + pred[mask].astype(int),
            minlength=self.n_classes**2,
        )
        self.confusion_matrix += new_entry.reshape(self.n_classes, self.n_classes)

    def get_metric_names(self) -> List[str]:
        """Get available metric names

        :return: List of available metric names
        :rtype: List[str]
        """
        return self.METRIC_NAMES

    def get_confusion_matrix(self) -> np.ndarray:
        """Get confusion matrix

        :return: Confusion matrix
        :rtype: np.ndarray
        """
        return self.confusion_matrix

    def get_tp(self, per_class: bool = True) -> Union[np.ndarray, int]:
        """True Positives

        :param per_class: Return per class TP, defaults to True
        :type per_class: bool, optional
        :return: True Positives
        :rtype: Union[np.ndarray, int]
        """
        tp = np.diag(self.confusion_matrix)
        return tp if per_class else int(np.nansum(tp))

    def get_fp(self, per_class: bool = True) -> Union[np.ndarray, int]:
        """False Positives

        :param per_class: Return per class FP, defaults to True
        :type per_class: bool, optional
        :return: False Positives
        :rtype: Union[np.ndarray, int]
        """
        fp = self.confusion_matrix.sum(axis=0) - np.diag(self.confusion_matrix)
        return fp if per_class else int(np.nansum(fp))

    def get_fn(self, per_class: bool = True) -> Union[np.ndarray, int]:
        """False negatives

        :param per_class: Return per class FN, defaults to True
        :type per_class: bool, optional
        :return: False Negatives
        :rtype: Union[np.ndarray, int]
        """
        fn = self.confusion_matrix.sum(axis=1) - np.diag(self.confusion_matrix)
        return fn if per_class else int(np.nansum(fn))

    def get_tn(self, per_class: bool = True) -> Union[np.ndarray, int]:
        """True negatives

        :param per_class: Return per class TN, defaults to True
        :type per_class: bool, optional
        :return: True Negatives
        :rtype: Union[np.ndarray, int]
        """
        total = self.confusion_matrix.sum()
        tn = total - (self.get_tp() + self.get_fp() + self.get_fn())
        return tn if per_class else int(np.nansum(tn))

    @staticmethod
    def _safe_div(
        numerator: Union[np.ndarray, float, int],
        denominator: Union[np.ndarray, float, int],
    ) -> Union[np.ndarray, float]:
        """Safely divide and return NaN for zero/invalid denominators."""
        if np.isscalar(denominator):
            scalar_denominator = float(np.asarray(denominator, dtype=np.float64))
            if scalar_denominator <= 0:
                return math.nan
            scalar_numerator = float(np.asarray(numerator, dtype=np.float64))
            return scalar_numerator / scalar_denominator

        numerator_arr = np.asarray(numerator, dtype=np.float64)
        denominator_arr = np.asarray(denominator, dtype=np.float64)
        result = np.full_like(denominator_arr, np.nan, dtype=np.float64)
        np.divide(
            numerator_arr,
            denominator_arr,
            out=result,
            where=denominator_arr > 0,
        )
        return result

    def get_precision(self, per_class: bool = True) -> Union[np.ndarray, float]:
        """Precision = TP / (TP + FP)

        :param per_class: Return per class precision, defaults to True
        :type per_class: bool, optional
        :return: Precision value (per class if per_class=True, otherwise global)
        :rtype: Union[np.ndarray, float]
        """
        tp = self.get_tp(per_class)
        fp = self.get_fp(per_class)
        denominator = tp + fp
        return self._safe_div(tp, denominator)

    def get_recall(self, per_class: bool = True) -> Union[np.ndarray, float]:
        """Recall = TP / (TP + FN)

        :param per_class: Return per class recall, defaults to True
        :type per_class: bool, optional
        :return: Recall value (per class if per_class=True, otherwise global)
        :rtype: Union[np.ndarray, float]
        """
        tp = self.get_tp(per_class)
        fn = self.get_fn(per_class)
        denominator = tp + fn
        return self._safe_div(tp, denominator)

    def get_accuracy(self, per_class: bool = True) -> Union[np.ndarray, float]:
        """Accuracy = (TP + TN) / (TP + FP + FN + TN)

        :param per_class: Return per class accuracy, defaults to True
        :type per_class: bool, optional
        :return: Accuracy value (per class if per_class=True, otherwise global)
        :rtype: Union[np.ndarray, float]
        """
        tp = self.get_tp(per_class)
        fp = self.get_fp(per_class)
        fn = self.get_fn(per_class)
        tn = self.get_tn(per_class)
        total = tp + fp + fn + tn

        if np.isscalar(total):
            return float((tp + tn) / total) if total > 0 else math.nan
        else:
            return np.where(total > 0, (tp + tn) / total, np.nan)

    def get_f1_score(self, per_class: bool = True) -> Union[np.ndarray, float]:
        """F1-score = 2 * (Precision * Recall) / (Precision + Recall)

        :param per_class: Return per class F1 score, defaults to True
        :type per_class: bool, optional
        :return: F1-score value (per class if per_class=True, otherwise global)
        :rtype: Union[np.ndarray, float]
        """
        precision = self.get_precision(per_class)
        recall = self.get_recall(per_class)
        denominator = precision + recall
        numerator = 2 * (precision * recall)
        return self._safe_div(numerator, denominator)

    def get_iou(self, per_class: bool = True) -> Union[np.ndarray, float]:
        """IoU = TP / (TP + FP + FN)

        :param per_class: Return per class IoU, defaults to True
        :type per_class: bool, optional
        :return: IoU value (per class if per_class=True, otherwise global)
        :rtype: Union[np.ndarray, float]
        """
        tp = self.get_tp(per_class)
        fp = self.get_fp(per_class)
        fn = self.get_fn(per_class)
        union = tp + fp + fn
        return self._safe_div(tp, union)

    def get_averaged_metric(
        self, metric_name: str, method: str, weights: Optional[np.ndarray] = None
    ) -> float:
        """Get average metric value

        :param metric: Name of the metric to compute
        :type metric: str
        :param method: Method to use for averaging ('macro', 'micro' or 'weighted')
        :type method: str
        :param weights: Weights for weighted averaging, defaults to None
        :type weights: Optional[np.ndarray], optional
        :return: Average metric value
        :rtype: float
        """
        metric = getattr(self, f"get_{metric_name}")
        if method == "macro":
            return float(np.nanmean(metric(per_class=True)))
        if method == "micro":
            return float(metric(per_class=False))
        if method == "weighted":
            if weights is None:
                raise ValueError("Weights should be provided for weighted averaging")

            metric_values = np.asarray(metric(per_class=True), dtype=np.float64)
            weights_array = np.asarray(weights, dtype=np.float64)

            if weights_array.shape != metric_values.shape:
                raise ValueError(
                    "Weights shape does not match number of classes: "
                    f"weights.shape={weights_array.shape}, expected={metric_values.shape}"
                )
            if not np.all(np.isfinite(weights_array)):
                raise ValueError("Weights should contain finite values")

            valid_metric_mask = np.isfinite(metric_values)
            valid_weight_sum = float(np.sum(weights_array[valid_metric_mask]))
            if valid_weight_sum <= 0:
                return math.nan

            weighted_sum = float(
                np.sum(metric_values[valid_metric_mask] * weights_array[valid_metric_mask])
            )
            return weighted_sum / valid_weight_sum
        raise ValueError(f"Unknown method {method}")

    def get_metric_per_name(
        self, metric_name: str, per_class: bool = True
    ) -> Union[np.ndarray, float, int]:
        """Get metric value by name

        :param metric_name: Name of the metric to compute
        :type metric_name: str
        :param per_class: Return per class metric, defaults to True
        :type per_class: bool, optional
        :return: Metric value
        :rtype: Union[np.ndarray, float, int]
        """
        return getattr(self, f"get_{metric_name}")(per_class=per_class)


def get_metrics_dataframe(
    metrics_factory: SegmentationMetricsFactory, ontology: dict
) -> pd.DataFrame:
    """Build a DataFrame with all metrics (global and per class) plus confusion matrix

    :param metrics_factory: Properly updated SegmentationMetricsFactory object
    :type metrics_factory: SegmentationMetricsFactory
    :param ontology: Ontology dictionary
    :type ontology: dict
    :return: DataFrame with all metrics
    :rtype: pd.DataFrame
    """
    # Build results dataframe
    results = defaultdict(dict)

    # Add per class and global metrics
    for metric in metrics_factory.get_metric_names():
        per_class = metrics_factory.get_metric_per_name(metric, per_class=True)

        for class_name, class_data in ontology.items():
            results[class_name][metric] = float(per_class[class_data["idx"]])

        if metric not in ["tp", "fp", "fn", "tn"]:
            for avg_method in ["macro", "micro"]:
                results[avg_method][metric] = metrics_factory.get_averaged_metric(
                    metric, avg_method
                )

    # Add confusion matrix
    for class_name_a, class_data_a in ontology.items():
        for class_name_b, class_data_b in ontology.items():
            results[class_name_a][class_name_b] = metrics_factory.confusion_matrix[
                class_data_a["idx"], class_data_b["idx"]
            ]

    return pd.DataFrame(results)
