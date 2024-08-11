import torch
from torchmetrics import Metric
from sklearn.metrics import fbeta_score, recall_score, precision_score

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
            score_container.append(fbeta_score(target[index].flatten().detach().cpu(), preds[index].flatten().detach().cpu(), average='macro', beta=2, zero_division=1))
        score_container = torch.Tensor(score_container)
        gt_thershold = score_container.gt(self.thershold)        
        self.container += torch.sum(gt_thershold)
        self.total += preds.shape[0]

    def compute(self):
        return self.container/self.total*100

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
            score_container.append(recall_score(target[index].flatten().detach().cpu(), preds[index].flatten().detach().cpu(), average='macro', zero_division=1))
        score_container = torch.Tensor(score_container)
        gt_thershold = score_container.gt(self.thershold)        
        self.container += torch.sum(gt_thershold)
        self.total += preds.shape[0]

    def compute(self):
        return self.container/self.total*100


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
            score_container.append(precision_score(target[index].flatten().detach().cpu(), preds[index].flatten().detach().cpu(), average='macro', zero_division=1))
        score_container = torch.Tensor(score_container)
        gt_thershold = score_container.gt(self.thershold)        
        self.container += torch.sum(gt_thershold)
        self.total += preds.shape[0]

    def compute(self):
        return self.container/self.total*100