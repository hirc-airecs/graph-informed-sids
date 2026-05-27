import json
from logging import getLogger
from typing import Union
import torch
import os
from accelerate import Accelerator
from torch.utils.data import DataLoader, Subset
import math

from genrec.dataset import AbstractDataset
from genrec.model import AbstractModel
from genrec.tokenizer import AbstractTokenizer
from genrec.utils import get_config, init_seed, init_logger, init_device, \
    get_dataset, get_tokenizer, get_model, get_trainer, log


class Pipeline:
    def __init__(
        self,
        model_name: Union[str, AbstractModel],
        dataset_name: Union[str, AbstractDataset],
        checkpoint_path: str = None,
        tokenizer: AbstractTokenizer = None,
        trainer = None,
        config_dict: dict = None,
        config_file: str = None,
        test_nrows: bool = False,
        only_init_datasets: bool = False,
        resume_training: bool = False
    ):
        self.config = get_config(
            model_name=model_name,
            dataset_name=dataset_name,
            config_file=config_file,
            config_dict=config_dict
        )
        # Automatically set devices and ddp
        self.config['device'], self.config['use_ddp'] = init_device()
        self.checkpoint_path = checkpoint_path
        self.config['test_nrows'] = test_nrows
        self.config['resume_training'] = resume_training

        # Accelerator
        self.project_dir = os.path.join(
            self.config['tensorboard_log_dir'],
            self.config["dataset"],
            self.config["model"]
        )
        self.accelerator = Accelerator(log_with='tensorboard', project_dir=self.project_dir)
        self.config['accelerator'] = self.accelerator

        # Seed and Logger
        init_seed(self.config['rand_seed'], self.config['reproducibility'])
        init_logger(self.config)
        self.logger = getLogger()
        self.log(f'Device: {self.config["device"]}')
        self.log(f'Config:\n{json.dumps({k: str(v) for k, v in self.config.items()}, indent=4)}')

        # Dataset
        self.raw_dataset = get_dataset(dataset_name)(self.config)
        self.log(self.raw_dataset)
        self.split_datasets = self.raw_dataset.split()

        # Tokenizer
        if tokenizer is not None:
            self.tokenizer = tokenizer(self.config, self.raw_dataset)
        else:
            assert isinstance(model_name, str), 'Tokenizer must be provided if model_name is not a string.'
            self.tokenizer = get_tokenizer(model_name)(self.config, self.raw_dataset)
        self.tokenized_datasets = self.tokenizer.tokenize(self.split_datasets)

        if only_init_datasets:
            return

        # Model
        with self.accelerator.main_process_first():
            self.model = get_model(model_name)(self.config, self.raw_dataset, self.tokenizer)
            if checkpoint_path is not None:
                self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.config['device'], weights_only=False))
                self.log(f'Loaded model checkpoint from {checkpoint_path}')
        self.log(self.model)
        self.log(self.model.n_parameters)

        # Trainer
        if trainer is not None:
            self.trainer = trainer
        else:
            self.trainer = get_trainer(model_name)(self.config, self.model, self.tokenizer)

    def _maybe_limit_validation_dataset(self, val_dataset):
        val_eval_user_limit = self.config.get('val_eval_user_limit')
        if val_eval_user_limit is None:
            return val_dataset

        val_dataset_size = len(val_dataset)
        if isinstance(val_eval_user_limit, float) and 0 < val_eval_user_limit < 1:
            val_eval_user_limit = max(1, math.ceil(val_dataset_size * val_eval_user_limit))
        else:
            val_eval_user_limit = int(val_eval_user_limit)

        if val_eval_user_limit <= 0 or val_eval_user_limit >= val_dataset_size:
            return val_dataset

        generator = torch.Generator()
        generator.manual_seed(int(self.config['rand_seed']))
        indices = torch.randperm(val_dataset_size, generator=generator)[:val_eval_user_limit].tolist()
        self.log(
            f'Validation evaluation limited to {val_eval_user_limit} randomly sampled '
            f'users/sequences out of {val_dataset_size} using rand_seed={self.config["rand_seed"]}.'
        )
        return Subset(val_dataset, indices)

    def run(self):
        # DataLoader
        train_dataloader = DataLoader(
            self.tokenized_datasets['train'],
            batch_size=self.config['train_batch_size'],
            shuffle=True,
            collate_fn=self.tokenizer.collate_fn['train'],
            num_workers=self.config.get('num_workers', 0)
        )
        val_dataset = self._maybe_limit_validation_dataset(self.tokenized_datasets['val'])

        val_dataloader = DataLoader(
            val_dataset,
            batch_size=self.config['eval_batch_size'],
            shuffle=False,
            collate_fn=self.tokenizer.collate_fn['val'],
            num_workers=self.config.get('num_workers', 0)
        )
        test_dataloader = DataLoader(
            self.tokenized_datasets['test'],
            batch_size=self.config['eval_batch_size'],
            shuffle=False,
            collate_fn=self.tokenizer.collate_fn['test'],
            num_workers=self.config.get('num_workers', 0)
        )

        if self.config.get('val_on_test', False):
            self.log('validate_on_test is enabled: using the test dataloader for periodic validation sanity checks.', level='warning')
            val_dataloader = test_dataloader
            if getattr(self.trainer, 'do_fine_grained_eval', False):
                self.trainer.split_datasets['val'] = self.trainer.split_datasets['test']
                self.trainer.case_labels['val'] = self.trainer.case_labels['test']
                self.trainer.fine_grained_ratios['val'] = self.trainer.fine_grained_ratios['test']

        if self.checkpoint_path is None or self.config['resume_training']:
            best_epoch, best_val_score = self.trainer.fit(train_dataloader, val_dataloader)

            self.accelerator.wait_for_everyone()

            if self.config['use_ddp']:
                # make sure all processes have the same checkpoint path
                import torch.distributed as dist
                ckpt_path_container = [self.trainer.saved_model_ckpt]
                dist.broadcast_object_list(ckpt_path_container, src=0)
                self.trainer.saved_model_ckpt = ckpt_path_container[0]

            self.model = self.accelerator.unwrap_model(self.model)
            self.model.load_state_dict(torch.load(self.trainer.saved_model_ckpt))

        self.model, test_dataloader = self.accelerator.prepare(
            self.model, test_dataloader
        )
        if self.accelerator.is_main_process and self.checkpoint_path is None:
            self.log(f'Loaded best model checkpoint from {self.trainer.saved_model_ckpt}')

        test_results = self.trainer.evaluate(test_dataloader)
        self.trainer.save_final_metrics_json(
            test_results
        )

        if self.accelerator.is_main_process:
            for key in test_results:
                self.accelerator.log({f'Test_Metric/{key}': test_results[key]})
        self.log(f'Test Results: {test_results}')

        self.trainer.end()
        return {
            'best_epoch': best_epoch,
            'best_val_score': best_val_score,
            'test_results': test_results,
        }

    def log(self, message, level='info'):
        return log(message, self.config['accelerator'], self.logger, level=level)
