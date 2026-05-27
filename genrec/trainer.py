import json
import math
import os
from collections import OrderedDict, defaultdict
from logging import getLogger
from numbers import Number

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from tqdm import tqdm
from transformers.optimization import get_scheduler

from genrec.evaluator import Evaluator
from genrec.model import AbstractModel
from genrec.tokenizer import AbstractTokenizer
from genrec.utils import config_for_log, get_file_name, get_total_steps, log


class Trainer:
    """
    A class that handles the training process for a model.

    Args:
        config (dict): The configuration parameters for training.
        model (AbstractModel): The model to be trained.
        tokenizer (AbstractTokenizer): The tokenizer used for tokenizing the data.

    Attributes:
        config (dict): The configuration parameters for training.
        model (AbstractModel): The model to be trained.
        evaluator (Evaluator): The evaluator used for evaluating the model.
        logger (Logger): The logger used for logging training progress.
        project_dir (str): The directory path for saving tensorboard logs.
        accelerator (Accelerator): The accelerator used for distributed training
        saved_model_ckpt (str): The file path for saving the trained model checkpoint.

    Methods:
        fit(train_dataloader, val_dataloader): Trains the model using the provided training and validation dataloaders.
        evaluate(dataloader, split='test'): Evaluate the model on the given dataloader.
        end(): Ends the training process and releases any used resources.
    """

    def __init__(self, config: dict, model: AbstractModel, tokenizer: AbstractTokenizer):
        self.config = config
        self.model = model
        self.accelerator = config['accelerator']
        self.evaluator = Evaluator(config, tokenizer)
        self.logger = getLogger()

        self.saved_model_ckpt = os.path.join(
            self.config['ckpt_dir'],
            get_file_name(self.config, suffix='.pth')
        )
        os.makedirs(os.path.dirname(self.saved_model_ckpt), exist_ok=True)

        final_metrics_dir = os.path.join(self.config['log_dir'], 'final_metrics')
        self.final_metrics_file = self.config.get('final_metrics_file') or os.path.join(
            final_metrics_dir,
            get_file_name(self.config, suffix='_final_test_metrics.json')
        )


    def fit(self, train_dataloader, val_dataloader):
        """
        Trains the model using the provided training and validation dataloaders.

        Args:
            train_dataloader: The dataloader for training data.
            val_dataloader: The dataloader for validation data.
        """
        optimizer = AdamW(
            self.model.parameters(),
            lr=self.config['lr'],
            weight_decay=self.config['weight_decay']
        )

        total_n_steps = get_total_steps(self.config, train_dataloader)
        if total_n_steps == 0:
            self.log('No training steps needed.')
            return None, None

        scheduler = get_scheduler(
            name=self.config["scheduler"],
            optimizer=optimizer,
            num_warmup_steps=self.config['warmup_steps'],
            num_training_steps=total_n_steps,
        )

        self.model, optimizer, train_dataloader, val_dataloader, scheduler = self.accelerator.prepare(
            self.model, optimizer, train_dataloader, val_dataloader, scheduler
        )
        self.accelerator.init_trackers(
            project_name=get_file_name(self.config, suffix=''),
            config=config_for_log(self.config),
            init_kwargs={"tensorboard": {"flush_secs": 60}},
        )

        n_epochs = np.ceil(total_n_steps / (len(train_dataloader) * self.accelerator.num_processes)).astype(int)
        best_epoch = 0
        best_val_score = -1

        for epoch in range(n_epochs):
            # Training
            self.model.train()
            total_loss = 0.0
            train_progress_bar = tqdm(
                train_dataloader,
                total=len(train_dataloader),
                desc=f"Training - [Epoch {epoch + 1}]",
            )
            for batch in train_progress_bar:
                optimizer.zero_grad()
                outputs = self.model(batch)
                loss = outputs.loss
                self.accelerator.backward(loss)
                if self.config['max_grad_norm'] is not None:
                    clip_grad_norm_(self.model.parameters(), self.config['max_grad_norm'])
                optimizer.step()
                scheduler.step()
                total_loss = total_loss + loss.item()

            self.accelerator.log({"Loss/train_loss": total_loss / len(train_dataloader)}, step=epoch + 1)
            self.log(f'[Epoch {epoch + 1}] Train Loss: {total_loss / len(train_dataloader)}')
        
            if self.config.get("token_weight_protocol", "").lower().strip() == "mha":
                self.accelerator.log({"Model/mha_temp": self.model.aggregator.log_temp[0].exp().item()}, step=epoch + 1)
                self.log(f'[Epoch {epoch + 1}] MHA Temp: {self.model.aggregator.log_temp[0].exp().item()}')

            # Evaluation
            if (epoch + 1) % self.config['eval_interval'] == 0:
                all_results = self.evaluate(val_dataloader, split='val')
                if self.accelerator.is_main_process:
                    for key in all_results:
                        self.accelerator.log({f"Val_Metric/{key}": all_results[key]}, step=epoch + 1)
                    self.log(f'[Epoch {epoch + 1}] Val Results: {all_results}')
                val_score = all_results[self.config['val_metric']]
                if val_score > best_val_score:
                    best_val_score = val_score
                    best_epoch = epoch + 1
                    if self.accelerator.is_main_process:
                        model_to_save = self.accelerator.unwrap_model(self.model)
                        if hasattr(model_to_save, '_orig_mod'):
                            model_to_save = model_to_save._orig_mod
                        torch.save(model_to_save.state_dict(), self.saved_model_ckpt)
                        self.log(f'[Epoch {epoch + 1}] Saved model checkpoint to {self.saved_model_ckpt}')

                if self.config['patience'] is not None and epoch + 1 - best_epoch >= self.config['patience']:
                    self.log(f'Early stopping at epoch {epoch + 1}')
                    break
        self.log(f'Best epoch: {best_epoch}, Best val score: {best_val_score}')
        return best_epoch, best_val_score
    
    
    @staticmethod
    def _json_safe_value(value):
        """Convert metric values to strict JSON-compatible Python objects."""
        if isinstance(value, torch.Tensor):
            value = value.item() if value.numel() == 1 else value.tolist()
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, Number):
            if isinstance(value, bool):
                return value
            if isinstance(value, int):
                return value
            value = float(value)
            return None if math.isnan(value) or math.isinf(value) else value
        if isinstance(value, dict):
            return {key: Trainer._json_safe_value(val) for key, val in value.items()}
        if isinstance(value, (list, tuple)):
            return [Trainer._json_safe_value(item) for item in value]
        return value

    def save_final_metrics_json(self, results, step=None, epoch=None, split='test'):
        """Save the final evaluation metrics to a standalone JSON file."""
        if not self.accelerator.is_main_process:
            return

        final_metrics_dir = os.path.dirname(self.final_metrics_file)
        if final_metrics_dir:
            os.makedirs(final_metrics_dir, exist_ok=True)

        payload = OrderedDict([
            ('step', self._json_safe_value(step)),
            ('epoch', self._json_safe_value(epoch)),
            ('split', split),
            ('metrics', OrderedDict(
                (key, self._json_safe_value(value)) for key, value in results.items()
            )),
        ])
        with open(self.final_metrics_file, 'w') as f:
            json.dump(payload, f, indent=2, allow_nan=False)
            f.write('\n')

        self.log(f"Saved final {split} metrics JSON to {self.final_metrics_file}")

    def evaluate(self, dataloader, split='test'):
        """
        Evaluate the model on the given dataloader.

        Args:
            dataloader (torch.utils.data.DataLoader): The dataloader to evaluate on.
            split (str, optional): The split name. Defaults to 'test'.

        Returns:
            OrderedDict: A dictionary containing the evaluation results.
        """
        self.model.eval()

        all_results = defaultdict(list)
        val_progress_bar = tqdm(
            dataloader,
            total=len(dataloader),
            desc=f"Eval - {split}",
        )
        for batch in val_progress_bar:
            with torch.no_grad():
                batch = {k: v.to(self.accelerator.device) for k, v in batch.items()}
                if self.config['use_ddp']: # ddp, gather data from all devices for evaluation
                    model_out = self.model.module.generate(batch, n_return_sequences=self.evaluator.maxk)
                    model_out['labels'] = batch['labels']
                    all_outs = self.accelerator.gather_for_metrics(model_out)
                    all_labels = all_outs.pop('labels', None)
                    results = self.evaluator.calculate_metrics(all_outs, all_labels)
                else:
                    model_out = self.model.generate(batch, n_return_sequences=self.evaluator.maxk)
                    results = self.evaluator.calculate_metrics(model_out, batch['labels'])

                for key, value in results.items():
                    all_results[key].append(value)

        output_results = OrderedDict()
        for metric in self.config['metrics']:
            for k in self.config['topk']:
                key = f"{metric}@{k}"
                output_results[key] = torch.cat(all_results[key]).mean().item()
        return output_results

    def end(self):
        """
        Ends the training process and releases any used resources
        """
        self.accelerator.end_training()

    def log(self, message, level='info'):
        return log(message, self.config['accelerator'], self.logger, level=level)
