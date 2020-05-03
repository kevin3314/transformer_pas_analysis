import re
import math
import datetime
from typing import List

import numpy as np
import torch
from sklearn.metrics import f1_score

from base import BaseTrainer
from writer.prediction_writer import PredictionKNPWriter
from scorer import Scorer


class Trainer(BaseTrainer):
    """
    Trainer class
    Note:
        Inherited from BaseTrainer.
    """
    def __init__(self, model, loss, metrics, optimizer, config,
                 data_loader, valid_kwdlc_data_loader, valid_kc_data_loader, valid_commonsense_data_loader,
                 lr_scheduler=None):
        super().__init__(model, loss, metrics, optimizer, config)
        self.config = config
        self.data_loader = data_loader
        self.valid_kwdlc_data_loader = valid_kwdlc_data_loader
        self.valid_kc_data_loader = valid_kc_data_loader
        self.valid_commonsense_data_loader = valid_commonsense_data_loader
        self.lr_scheduler = lr_scheduler
        self.log_step = math.ceil(data_loader.n_samples / np.sqrt(data_loader.batch_size) / 200)

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch
        :param epoch: Current training epoch.
        :return: A log that contains all information you want to save.
        Note:
            If you have additional information to record, for example:
                > additional_log = {"x": x, "y": y}
            merge it with log before return. i.e.
                > log = {**log, **additional_log}
                > return log
            The metrics in log must have the key 'metrics'.
        """
        self.model.train()

        total_loss = 0
        for batch_idx, batch in enumerate(self.data_loader):
            batch = tuple(t.to(self.device) for t in batch)
            input_ids, input_mask, segment_ids, target, ng_token_mask, deps, task = batch

            self.optimizer.zero_grad()
            # (b, seq, case, seq) or tuple
            output = self.model(input_ids, input_mask, segment_ids, ng_token_mask, deps, target)
            loss = self.loss(output, target, deps, task)
            loss.backward()
            self.optimizer.step()

            self.writer.set_step((epoch - 1) * len(self.data_loader) + batch_idx)
            self.writer.add_scalar('lr', self.lr_scheduler.get_lr()[0])
            self.writer.add_scalar('loss', loss.item())
            total_loss += loss.item() * input_ids.size(0)

            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} [{}/{} ({:.0f}%)] Time: {} Loss: {:.6f}'.format(
                    epoch,
                    batch_idx * self.data_loader.batch_size,
                    self.data_loader.n_samples,
                    100.0 * batch_idx / len(self.data_loader),
                    datetime.datetime.now().strftime('%H:%M:%S'),
                    loss.item()))

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

        log = {
            'loss': total_loss / self.data_loader.n_samples,
        }

        if self.valid_kwdlc_data_loader is not None:
            val_log = self._valid_epoch(epoch, self.valid_kwdlc_data_loader, 'kwdlc')
            log.update(**{'val_kwdlc_'+k: v for k, v in val_log.items()})

        if self.valid_kc_data_loader is not None:
            val_log = self._valid_epoch(epoch, self.valid_kc_data_loader, 'kc')
            log.update(**{'val_kc_'+k: v for k, v in val_log.items()})

        if self.valid_commonsense_data_loader is not None:
            val_log = self._valid_epoch(epoch, self.valid_commonsense_data_loader, 'commonsense')
            log.update(**{'val_commonsense_'+k: v for k, v in val_log.items()})

        return log

    def _valid_epoch(self, epoch, data_loader, label):
        """
        Validate after training an epoch
        :return: A log that contains information about validation
        Note:
            The validation metrics in log must have the key 'val_metrics'.
        """
        self.model.eval()
        total_loss = 0
        arguments_set: List[List[List[int]]] = []
        contingency_set: List[int] = []
        with torch.no_grad():
            for batch_idx, batch in enumerate(data_loader):
                batch = tuple(t.to(self.device) for t in batch)
                input_ids, input_mask, segment_ids, target, ng_token_mask, deps, task = batch

                # (b, seq, case, seq) or tuple
                output = self.model(input_ids, input_mask, segment_ids, ng_token_mask, deps, target)
                if self.config['arch']['type'] == 'MultitaskDepModel':
                    scores = output[0]  # (b, seq, case, seq)
                elif re.match(r'(CaseInteractionModel2|Refinement|Duplicate)', self.config['arch']['type']):
                    scores = output[-1]  # (b, seq, case, seq)
                elif self.config['arch']['type'] == 'CommonsenseModel':
                    scores = output[0]  # (b, seq, case, seq)
                    contingency_set += output[1].gt(0.5).int().tolist()
                else:
                    scores = output  # (b, seq, case, seq)

                if label in ('kwdlc', 'kc'):
                    arguments_set += torch.argmax(scores, dim=3).tolist()  # (b, seq, case)

                # computing loss, metrics on valid set
                loss = self.loss(output, target, deps, task)
                total_loss += loss.item() * input_ids.size(0)

                self.writer.set_step((epoch - 1) * len(data_loader) + batch_idx, 'valid')
                self.writer.add_scalar(f'loss_{label}', loss.item())

                if batch_idx % self.log_step == 0:
                    self.logger.debug('Validation [{}/{} ({:.0f}%)] Time: {}'.format(
                        batch_idx * data_loader.batch_size,
                        data_loader.n_samples,
                        100.0 * batch_idx / len(data_loader),
                        datetime.datetime.now().strftime('%H:%M:%S')))

        log = {f'loss': total_loss / data_loader.n_samples}

        if label in ('kwdlc', 'kc'):
            prediction_writer = PredictionKNPWriter(data_loader.dataset, self.logger)
            documents_pred = prediction_writer.write(arguments_set, None)

            scorer = Scorer(documents_pred, data_loader.dataset.documents,
                            target_cases=data_loader.dataset.target_cases,
                            target_exophors=data_loader.dataset.target_exophors,
                            coreference=data_loader.dataset.coreference,
                            kc=data_loader.dataset.kc)

            val_metrics = self._eval_metrics(scorer.result_dict(), label)

            log.update(dict(zip([met.__name__ for met in self.metrics], val_metrics)))
        elif label == 'commonsense':
            log['f1'] = self._eval_commonsense(contingency_set)

        return log

    def _eval_commonsense(self, contingency_set: List[int]) -> float:
        gold = [f.label for f in self.valid_commonsense_data_loader.dataset.features]
        f1 = f1_score(gold, contingency_set)
        self.writer.add_scalar(f'commonsense_f1', f1)
        return f1

    def _eval_metrics(self, result: dict, label: str):
        f1_metrics = np.zeros(len(self.metrics))
        for i, metric in enumerate(self.metrics):
            f1_metrics[i] += metric(result)
            self.writer.add_scalar(f'{label}_{metric.__name__}', f1_metrics[i])
        return f1_metrics
