import torch
from pytorch_lightning.metrics import Metric


class TopN(Metric):
    def __init__(self, topn, target_label=0, dist_sync_on_step=False, compute_on_step=True):
        super().__init__(dist_sync_on_step=dist_sync_on_step, compute_on_step=compute_on_step)

        self.topn = topn
        self.target_label = target_label
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds, target):
        target = target[0]
        correct = 0
        total = 0
        
        for pred, tar in zip(preds.detach(), target.detach()):
            pred = pred.flatten()
            tar = tar.flatten()
            top_vals = torch.topk(pred, self.topn, largest=False)
            #print(top_vals)
            #print(tar[top_vals.indices])
            correct += torch.sum(tar[top_vals.indices] == self.target_label)
            total += self.topn
            
        self.correct += correct
        self.total += total

    def compute(self):
        result = self.correct.float() / self.total
        return result
