import math
import datetime
from typing import List

import numpy as np
import torch

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
                 data_loader, valid_kwdlc_data_loader, valid_kc_data_loader, lr_scheduler=None):
        super().__init__(model, loss, metrics, optimizer, config)
        self.config = config
        self.data_loader = data_loader
        self.valid_kwdlc_data_loader = valid_kwdlc_data_loader
        self.valid_kc_data_loader = valid_kc_data_loader
        # self.do_validation = (self.valid_kwdlc_data_loader is not None) or (self.valid_kc_data_loader is not None)
        self.lr_scheduler = lr_scheduler
        self.log_step = math.ceil(data_loader.n_samples / np.sqrt(data_loader.batch_size) / 200)

    def _eval_metrics(self, result: dict, label: str):
        f1_metrics = np.zeros(len(self.metrics))
        for i, metric in enumerate(self.metrics):
            f1_metrics[i] += metric(result)
            self.writer.add_scalar('{}_{}'.format(label, metric.__name__), f1_metrics[i])
        return f1_metrics

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
        # total_metrics = np.zeros(len(self.metrics))
        for batch_idx, batch in enumerate(self.data_loader):
            batch = tuple(t.to(self.device) for t in batch)
            input_ids, input_mask, arguments_ids, ng_token_mask, deps = batch

            self.optimizer.zero_grad()
            output = self.model(input_ids, input_mask, ng_token_mask, deps)  # (b, seq, case, seq) or tuple
            loss = self.loss(output, arguments_ids, deps)
            loss.backward()
            self.optimizer.step()

            self.writer.set_step((epoch - 1) * len(self.data_loader) + batch_idx)
            self.writer.add_scalar('lr', self.lr_scheduler.get_lr()[0])
            self.writer.add_scalar('loss', loss.item())
            total_loss += loss.item() * input_ids.size(0)
            # total_metrics += self._eval_metrics(output, target)

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
            # 'metrics': (total_metrics / len(self.data_loader)).tolist()
        }

        if self.valid_kwdlc_data_loader is not None:
            val_log = self._valid_epoch(epoch, self.valid_kwdlc_data_loader, 'kwdlc')
            log.update(**{'val_kwdlc_'+k: v for k, v in val_log.items()})

        if self.valid_kc_data_loader is not None:
            val_log = self._valid_epoch(epoch, self.valid_kc_data_loader, 'kc')
            log.update(**{'val_kc_'+k: v for k, v in val_log.items()})

        return log

    def _valid_epoch(self, epoch, valid_data_loader, label):
        """
        Validate after training an epoch
        :return: A log that contains information about validation
        Note:
            The validation metrics in log must have the key 'val_metrics'.
        """
        self.model.eval()
        total_val_loss = 0
        arguments_sets: List[List[List[int]]] = []
        with torch.no_grad():
            for batch_idx, batch in enumerate(valid_data_loader):
                batch = tuple(t.to(self.device) for t in batch)
                input_ids, input_mask, arguments_ids, ng_token_mask, deps = batch

                output = self.model(input_ids, input_mask, ng_token_mask, deps)  # (b, seq, case, seq) or tuple
                scores = output if isinstance(output, torch.Tensor) else output[-1]

                arguments_set = torch.argmax(scores, dim=3)[:, :, :arguments_ids.size(2)]  # (b, seq, case)
                arguments_sets += arguments_set.tolist()

                # computing loss, metrics on valid set
                loss = self.loss(output, arguments_ids, deps)
                total_val_loss += loss.item() * input_ids.size(0)

                self.writer.set_step((epoch - 1) * len(valid_data_loader) + batch_idx, 'valid')
                self.writer.add_scalar(f'loss_{label}', loss.item())

        # prediction_output_dir = self.config.save_dir / 'valid_out_knp'
        prediction_writer = PredictionKNPWriter(valid_data_loader.dataset, self.logger)
        documents_pred = prediction_writer.write(arguments_sets, None)

        scorer = Scorer(documents_pred, valid_data_loader.dataset.documents,
                        valid_data_loader.dataset.target_exophors,
                        coreference=valid_data_loader.dataset.coreference,
                        kc=valid_data_loader.dataset.kc,
                        eval_eventive_noun=False)
        # scorer.write_html(self.config.save_dir / f'result_{label}.html')
        # scorer.export_txt(self.config.save_dir / f'result_{label}.txt')

        val_metrics = self._eval_metrics(scorer.result_dict(), label)

        # add histogram of model parameters to the tensorboard
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins='auto')

        log = {f'loss_{label}': total_val_loss / valid_data_loader.n_samples}
        log.update(dict(zip([met.__name__ for met in self.metrics], val_metrics)))

        return log
