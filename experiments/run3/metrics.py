import torch
from pytorch_lightning.metrics import Metric


class Accuracy(Metric):
    def __init__(self, dist_sync_on_step=False, compute_on_step=True):
        super().__init__(dist_sync_on_step=dist_sync_on_step, compute_on_step=compute_on_step)
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds, target):
        correct = 0
        total = 0
        
        for pred, tar in zip(preds.detach(), target.detach()):
            correct += int(pred.argmax() == tar)
            total += 1
            
        self.correct += correct
        self.total += total

    def compute(self):
        result = self.correct.float() / self.total
        return result
