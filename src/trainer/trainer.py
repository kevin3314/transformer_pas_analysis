import numpy as np
from typing import List

import torch

from base import BaseTrainer
from model.metric import PredictionKNPWriter
from scorer import Scorer


class Trainer(BaseTrainer):
    """
    Trainer class
    Note:
        Inherited from BaseTrainer.
    """
    def __init__(self, model, loss, metrics, optimizer, config, data_loader, valid_data_loader,
                 lr_scheduler=None):
        super().__init__(model, loss, metrics, optimizer, config)
        self.config = config
        self.data_loader = data_loader
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))

    def _eval_metrics(self, result: dict):
        f1_metrics = np.zeros(len(self.metrics))
        for i, metric in enumerate(self.metrics):
            f1_metrics[i] += metric(result)
            self.writer.add_scalar('{}'.format(metric.__name__), f1_metrics[i])
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
        for batch_idx, (input_ids, input_mask, arguments_ids, ng_arg_mask, deps) in enumerate(self.data_loader):
            input_ids = input_ids.to(self.device)          # (b, seq)
            input_mask = input_mask.to(self.device)        # (b, seq)
            arguments_ids = arguments_ids.to(self.device)  # (b, seq, case)
            ng_arg_mask = ng_arg_mask.to(self.device)      # (b, seq, seq)
            deps = deps.to(self.device)                    # (b ,seq, seq)

            self.optimizer.zero_grad()
            output = self.model(input_ids, input_mask, ng_arg_mask, deps)  # (b, seq, case(+1), seq)
            loss = self.loss(output, arguments_ids, deps)
            loss.backward()
            self.optimizer.step()

            self.writer.set_step((epoch - 1) * len(self.data_loader) + batch_idx)
            self.writer.add_scalar('loss', loss.item())
            total_loss += loss.item() * input_ids.size(0)
            # total_metrics += self._eval_metrics(output, target)

            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}'.format(
                    epoch,
                    batch_idx * self.data_loader.batch_size,
                    self.data_loader.n_samples,
                    100.0 * batch_idx / len(self.data_loader),
                    loss.item()))

        log = {
            'loss': total_loss / self.data_loader.n_samples,
            # 'metrics': (total_metrics / len(self.data_loader)).tolist()
        }

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log = {**log, **val_log}

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch
        :return: A log that contains information about validation
        Note:
            The validation metrics in log must have the key 'val_metrics'.
        """
        self.model.eval()
        total_val_loss = 0
        # total_val_metrics = np.zeros(len(self.metrics))
        arguments_sets: List[List[List[int]]] = []
        with torch.no_grad():
            for batch_idx, (input_ids, input_mask, arguments_ids, ng_arg_mask, deps)\
                    in enumerate(self.valid_data_loader):
                input_ids = input_ids.to(self.device)          # (b, seq)
                input_mask = input_mask.to(self.device)        # (b, seq)
                arguments_ids = arguments_ids.to(self.device)  # (b, seq, case)
                ng_arg_mask = ng_arg_mask.to(self.device)      # (b, seq, seq)
                deps = deps.to(self.device)                    # (b, seq, seq)

                output = self.model(input_ids, input_mask, ng_arg_mask, deps)  # (b, seq, case(+1) seq)

                arguments_set = torch.argmax(output, dim=3)[:, :, :arguments_ids.size(2)]  # (b, seq, case)
                arguments_sets += arguments_set.tolist()

                # computing loss, metrics on test set
                loss = self.loss(output, arguments_ids, deps)
                total_val_loss += loss.item() * input_ids.size(0)

                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                self.writer.add_scalar('loss', loss.item())
                # total_val_metrics += self._eval_metrics(output, target)
                # self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

        prediction_output_dir = self.config.save_dir / 'valid_out_knp'
        prediction_writer = PredictionKNPWriter(self.valid_data_loader.dataset,
                                                self.config['valid_dataset']['args'],
                                                self.logger)
        documents_pred = prediction_writer.write(arguments_sets, prediction_output_dir)

        scorer = Scorer(documents_pred, self.valid_data_loader.dataset.reader)
        scorer.write_html(self.config.save_dir / 'result.html')

        val_metrics = self._eval_metrics(scorer.result_dict())

        # add histogram of model parameters to the tensorboard
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins='auto')

        return {
            'val_loss': total_val_loss / self.valid_data_loader.n_samples,
            'val_metrics': val_metrics
        }
