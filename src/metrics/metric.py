import torch
from torchmetrics import Metric
from mmseg.evaluation.metrics.iou_metric import IoUMetric
from sklearn.metrics import fbeta_score, recall_score, precision_score
import numpy as np
from collections import OrderedDict
import logging
from prettytable import PrettyTable

logging.getLogger("mmengine").setLevel(logging.ERROR)


class BF2score(Metric):

    is_differentiable: bool = False
    higher_is_better: bool = True
    full_state_update: bool = False

    def __init__(self, thershold: float = 0.90):
        super().__init__()
        self.add_state("container", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")
        self.thershold = thershold

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        assert preds.shape == target.shape
        score_container = list()
        for index in range(preds.shape[0]):
            score_container.append(
                fbeta_score(
                    target[index].flatten().detach().cpu(),
                    preds[index].flatten().detach().cpu(),
                    average="macro",
                    beta=2,
                    zero_division=1,
                )
            )
        score_container = torch.Tensor(score_container)
        gt_thershold = score_container.gt(self.thershold)
        self.container += torch.sum(gt_thershold)
        self.total += preds.shape[0]

    def compute(self):
        return self.container / self.total * 100


class BPAscore(Metric):

    is_differentiable: bool = False
    higher_is_better: bool = True
    full_state_update: bool = False

    def __init__(self, thershold: float = 0.90):
        super().__init__()
        self.add_state("container", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")
        self.thershold = thershold

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        assert preds.shape == target.shape
        score_container = list()
        for index in range(preds.shape[0]):
            score_container.append(
                recall_score(
                    target[index].flatten().detach().cpu(),
                    preds[index].flatten().detach().cpu(),
                    average="macro",
                    zero_division=1,
                )
            )
        score_container = torch.Tensor(score_container)
        gt_thershold = score_container.gt(self.thershold)
        self.container += torch.sum(gt_thershold)
        self.total += preds.shape[0]

    def compute(self):
        return self.container / self.total * 100


class BUAscore(Metric):

    is_differentiable: bool = False
    higher_is_better: bool = True
    full_state_update: bool = False

    def __init__(self, thershold: float = 0.90):
        super().__init__()
        self.add_state("container", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")
        self.thershold = thershold

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        assert preds.shape == target.shape
        score_container = list()
        for index in range(preds.shape[0]):
            score_container.append(
                precision_score(
                    target[index].flatten().detach().cpu(),
                    preds[index].flatten().detach().cpu(),
                    average="macro",
                    zero_division=1,
                )
            )
        score_container = torch.Tensor(score_container)
        gt_thershold = score_container.gt(self.thershold)
        self.container += torch.sum(gt_thershold)
        self.total += preds.shape[0]

    def compute(self):
        return self.container / self.total * 100


class MMSeg(Metric):
    def __init__(
        self,
        classes,
        iou_metrics=["mIoU"],
        ignore_index=255,
        nan_to_num=None,
        beta=1,
        prefix="train",
        per_classses=False,
    ):
        super().__init__()
        self.add_state("results", default=[])
        self.prefix = prefix
        self.ignore_index = ignore_index
        self.nan_to_num = nan_to_num
        self.beta = beta
        self.num_classes = len(classes)
        self.iou_metrics = iou_metrics
        self.per_classses = per_classses
        self.metrics = IoUMetric(
            classes=classes,
            iou_metrics=iou_metrics,
            ignore_index=ignore_index,
            prefix=prefix,
        )
        self.metrics._dataset_meta = dict(classes=classes)

    def update(self, preds, target):
        self.results.append(
            self.metrics.intersect_and_union(
                preds,
                target,
                ignore_index=self.ignore_index,
                num_classes=self.num_classes,
            )
        )

    def compute(self):
        results = self.results
        self.metrics.results = []
        results = tuple(zip(*results))
        assert len(results) == 4

        total_area_intersect = sum(results[0])
        total_area_union = sum(results[1])
        total_area_pred_label = sum(results[2])
        total_area_label = sum(results[3])
        ret_metrics = self.metrics.total_area_to_metrics(
            total_area_intersect,
            total_area_union,
            total_area_pred_label,
            total_area_label,
            self.iou_metrics,
            self.nan_to_num,
            self.beta,
        )

        class_names = self.metrics.dataset_meta["classes"]

        # summary table
        ret_metrics_summary = OrderedDict(
            {
                ret_metric: np.round(np.nanmean(ret_metric_value) * 100, 2)
                for ret_metric, ret_metric_value in ret_metrics.items()
            }
        )
        metrics = dict()
        for key, val in ret_metrics_summary.items():
            if key == "aAcc":
                metrics[self.prefix + "/"+key] = val
            else:
                metrics[self.prefix+"/"+"m" + key] = val

        # each class table
        ret_metrics.pop("aAcc", None)
        ret_metrics_class = OrderedDict(
            {
                ret_metric: np.round(ret_metric_value * 100, 2)
                for ret_metric, ret_metric_value in ret_metrics.items()
            }
        )
        ret_metrics_class.update({"Class": class_names})
        ret_metrics_class.move_to_end("Class", last=False)
        if self.per_classses:
            class_table_data = PrettyTable()
            for key, val in ret_metrics_class.items():
                class_table_data.add_column(key, val)
            return metrics, class_table_data

        return metrics


# 测试代码
if __name__ == "__main__":
    classes = ["class_0", "class_1", "class_2", "class_3", "class_4"]
    miou_metric = MMSeg(
        classes=classes, iou_metrics=["mIoU", "mDice", "mFscore"], per_classses=True
    )

    # 模拟预测和目标
    preds = torch.tensor([[0, 1, 2], [1, 2, 0]]).to("cuda:3")
    target = torch.tensor([[0, 1, 1], [1, 2, 0]]).to("cuda:3")

    miou_metric.update(preds, target)
    result, table = miou_metric.compute()

    print(result)
    print(table)
