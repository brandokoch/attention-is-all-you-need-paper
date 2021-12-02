import torch
from utils.custom_enumerator import enumerateWithEstimate
import wandb
import copy
import logging

# Configure Logging
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

def noop(*a, **k):
    return None

class Learner:
    def __init__(self, model, train_dl, val_dl, loss_func, cbs, opt, sched=None, device='cuda'):
        self.model=model
        self.train_dl=train_dl
        self.val_dl=val_dl
        self.loss_func=loss_func
        self.cbs=cbs
        self.opt=opt
        self.sched=sched
        self.device=device

        self.cur_step=1

        self.best_val_loss=float('inf')
        self.best_model_state_dict=copy.deepcopy(self.model.state_dict())

        for cb in cbs: 
            cb.learner=self

    def one_batch(self):
        self('before_batch') 
        self.xb,self.yb=self.batch
        self.preds=self.model(self.xb,self.yb)
        self('before_loss')
        self.loss=self.loss_func(
            self.preds.reshape(-1, self.preds.size(-1)), # Reshaping for loss
            self.yb[:,1:].contiguous().view(-1) # Shifting right (without BOS)
        )
        self('after_loss')
        if self.model.training:
            self.loss.backward()
            if self.cur_step % wandb.config.GRAD_ACCUMULATION_STEPS == 0:
                self.opt.step()
                if self.sched != None:
                    self.sched.step()
                self.opt.zero_grad()
            self.cur_step+=1
        self('after_batch')
        
    def one_epoch(self, is_train):
        self('before_epoch')
        self.model.training=is_train

        if self.model.training:
            self.model.train()
        else:
            self.model.eval()

        dl=self.train_dl if is_train else self.val_dl
        for self.batch_idx,self.batch in enumerate(dl):
            self.one_batch()
        self('after_epoch')

    def fit(self, n_epochs):
        self('before_fit')
        self.n_epochs=n_epochs

        for self.epoch_idx in enumerateWithEstimate(range(n_epochs), desc_str="Training status"):
            self.one_epoch(is_train=True)
            with torch.no_grad():
                self.one_epoch(is_train=False)
        self('after_fit')

    def __call__(self, cb_method_name):
        for cb in self.cbs:
            getattr(cb, cb_method_name, noop)()

