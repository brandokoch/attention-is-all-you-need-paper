import os
import copy

import torch
import torch.nn as nn
import numpy as np
from tokenizers import Tokenizer
from nltk.translate.bleu_score import corpus_bleu

# Configure Logging
import wandb
import logging
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

class CheckpointSaver():

    def __init__(self, epoch_cnt):
        self.epoch_cnt = epoch_cnt
    
    def after_epoch(self):
        # Save model every 'epoch_cnt' epochs
        if not self.learner.model.training and self.learner.epoch_idx % self.epoch_cnt == 0:
            epoch_ckpt_pth=os.path.join(wandb.config.RUNS_FOLDER_PTH,wandb.config.RUN_NAME,f'model_ckpt_epoch{self.learner.epoch_idx}.pt')
            torch.save(self.learner.model.state_dict(), epoch_ckpt_pth)
    
        # Save best model
        best_model_ckpt_pth=os.path.join(wandb.config.RUNS_FOLDER_PTH,wandb.config.RUN_NAME,f'model_ckpt_best.pt')
        torch.save(self.learner.best_model_state_dict, best_model_ckpt_pth)


class TrackExample():

    def before_fit(self):
        tokenizer_pth=os.path.join(wandb.config.RUNS_FOLDER_PTH,wandb.config.RUN_NAME,'tokenizer.json')
        self.tokenizer = Tokenizer.from_file(tokenizer_pth)
        self.table=wandb.Table(columns=['train_x','train_y','train_y_pred','val_x','val_y','val_y_pred'])
        
        # Extract a training set example
        x_train,y_train=next(iter(self.learner.train_dl))
        train_example_x=x_train[0].numpy()
        train_example_y=y_train[0].numpy()

        # Extract a validation set example
        x_val,y_val=next(iter(self.learner.val_dl))
        val_example_x=x_val[0].numpy()
        val_example_y=y_val[0].numpy()

        # Convert to text
        self.train_example_x_text=self.tokenizer.decode(train_example_x, skip_special_tokens=False)
        self.train_example_y_text=self.tokenizer.decode(train_example_y, skip_special_tokens=False)

        self.val_example_x_text=self.tokenizer.decode(val_example_x, skip_special_tokens=False)
        self.val_example_y_text=self.tokenizer.decode(val_example_y, skip_special_tokens=False)

    def after_epoch(self):
        if not self.learner.model.training:
            train_example_y_pred_text=self.learner.model.translate(self.train_example_x_text, self.tokenizer)
            val_example_y_pred_text=self.learner.model.translate(self.val_example_x_text, self.tokenizer)

            log.info(f"""Tracking Example progress:
            Train Example x:     \t{ self.train_example_x_text}
            Train Example y:     \t{ self.train_example_y_text}
            Train Example y_pred:\t{ train_example_y_pred_text}
            ---------------------
            Val Example x:       \t{ self.val_example_x_text}
            Val Example y:       \t{ self.val_example_y_text}
            Val Example y_pred:  \t{ val_example_y_pred_text}
            """
            )


class TrackBleu():

    def before_fit(self):
        tokenizer_pth=os.path.join(wandb.config.RUNS_FOLDER_PTH,wandb.config.RUN_NAME,'tokenizer.json')
        self.tokenizer = Tokenizer.from_file(tokenizer_pth)
        
    def before_epoch(self):
        self.preds_text_tokens=[]
        self.yb_text_tokens=[]
        self.xb_text_tokens=[]

    def after_batch(self):
        if not self.learner.model.training:

            preds=self.learner.preds.detach().cpu()
            preds=nn.functional.log_softmax(preds, dim=-1)
            preds=preds.argmax(dim=-1).squeeze(-1)

            preds_text=self.tokenizer.decode_batch(preds.numpy(), skip_special_tokens=False)
            xb_text=self.tokenizer.decode_batch(self.learner.xb.detach().cpu().numpy(), skip_special_tokens=False)
            yb_text=self.tokenizer.decode_batch(self.learner.yb.detach().cpu().numpy(), skip_special_tokens=False)

            preds_text_tokens=[t for t in preds_text]
            xb_text_tokens=[t for t in xb_text]
            yb_text_tokens=[t for t in yb_text]

            self.preds_text_tokens+=preds_text_tokens
            self.xb_text_tokens+=xb_text_tokens
            self.yb_text_tokens+=yb_text_tokens

    def after_epoch(self):
        if not self.learner.model.training:
            yb_text_tokens_for_bleu=[[item] for item in self.yb_text_tokens]
            bleu=corpus_bleu(yb_text_tokens_for_bleu,self.preds_text_tokens)
            wandb.log({'bleu': bleu}, step=self.learner.cur_step)


class MoveToDeviceCallback():

    def before_batch(self):
        if self.learner.device=='cuda':
            try:
                self.learner.batch = (self.learner.batch[0].to('cuda'), self.learner.batch[1].to('cuda'))
            except Exception as e:
                log.error(
                    "Exception occurred: Can't move the batch to GPU", exc_info=True)

    def before_fit(self):
        if self.learner.device=='cuda':
            try:
                self.learner.model = self.learner.model.to('cuda')
            except Exception as e:
                log.error(
                    "Exception occurred: Can't move the model to GPU", exc_info=True)


class TrackLoss():

    def before_epoch(self):
        self.batch_cnt = 0
        self.loss_sum = 0

    def after_batch(self):

        self.batch_cnt += 1
        loss = self.learner.loss

        loss = loss.detach().cpu()
        self.loss_sum += loss

        # Tracking train loss by batch
        if self.learner.model.training:
            wandb.log({'batch':self.learner.batch_idx}, step=self.learner.cur_step)
            wandb.log({'epoch':self.learner.epoch_idx}, step=self.learner.cur_step)
            wandb.log({'Loss/Train': loss.item()}, step=self.learner.cur_step)

            if self.learner.sched!=None:
                lr= self.learner.sched.get_last_lr()
                wandb.log({'Lr': lr[0]}, step=self.learner.cur_step)

    def after_epoch(self):

        # Calculate avg epoch loss
        avg_loss = self.loss_sum/self.batch_cnt
        avg_loss=avg_loss.item()

        # Log
        if self.learner.model.training:
            log.info(f"Epoch: {self.learner.epoch_idx} | Training | Loss: {avg_loss:.5f}")
            wandb.log({'Loss_Avg/Train': avg_loss}, step=self.learner.cur_step)
        else:
            log.info(f"Epoch: {self.learner.epoch_idx} | Validation | Loss: {avg_loss:.5f}")
            wandb.log({'Loss_Avg/Val': avg_loss}, step=self.learner.cur_step)

            if avg_loss<self.learner.best_val_loss:
                log.info(f"Loss/Val high score, remembering state_dict.")
                self.learner.best_val_loss = avg_loss
                self.learner.best_model_state_dict=copy.deepcopy(self.learner.model.state_dict())


