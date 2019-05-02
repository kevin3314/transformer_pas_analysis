import numpy as np
import torch
from torchvision.utils import make_grid
from base import BaseTrainer


class Trainer(BaseTrainer):
    """
    Trainer class
    Note:
        Inherited from BaseTrainer.
    """
    def __init__(self, model, loss, metrics, optimizer, config,
                 data_loader, valid_data_loader=None, lr_scheduler=None):
        super(Trainer, self).__init__(model, loss, metrics, optimizer, config)
        self.config = config
        self.data_loader = data_loader
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))

    def _eval_metrics(self, output, target):
        acc_metrics = np.zeros(len(self.metrics))
        for i, metric in enumerate(self.metrics):
            acc_metrics[i] += metric(output, target)
            self.writer.add_scalar('{}'.format(metric.__name__), acc_metrics[i])
        return acc_metrics

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
        for batch_idx, (input_ids, input_mask, segment_ids, arguments_set, ng_arg_ids_set) in enumerate(self.data_loader):
            input_ids = input_ids.to(self.device)            # (b, seq)
            input_mask = input_mask.to(self.device)          # (b, seq)
            segment_ids = segment_ids.to(self.device)        # (b, seq)
            arguments_set = arguments_set.to(self.device)    # (b, seq, case)
            ng_arg_ids_set = ng_arg_ids_set.to(self.device)  # (b, seq, seq)

            self.optimizer.zero_grad()
            output = self.model(input_ids, input_mask, segment_ids, arguments_set=arguments_set, ng_arg_ids_set=ng_arg_ids_set)
            # loss = self.loss(output, input_ids)
            loss = output
            loss.backward()
            self.optimizer.step()

            self.writer.set_step((epoch - 1) * len(self.data_loader) + batch_idx)
            self.writer.add_scalar('loss', loss.item())
            total_loss += loss.item()
            # total_metrics += self._eval_metrics(output, target)

            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}'.format(
                    epoch,
                    batch_idx * self.data_loader.batch_size,
                    self.data_loader.n_samples,
                    100.0 * batch_idx / len(self.data_loader),
                    loss.item()))
                # self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

        log = {
            'loss': total_loss / len(self.data_loader),
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
        # total_val_loss = 0
        total_val_metrics = np.zeros(len(self.metrics))
        with torch.no_grad():
            for batch_idx, (input_ids, input_mask, segment_ids, ng_arg_ids_set) in enumerate(
                    self.valid_data_loader):
                input_ids = input_ids.to(self.device)  # (b, seq)
                input_mask = input_mask.to(self.device)  # (b, seq)
                segment_ids = segment_ids.to(self.device)  # (b, seq)
                ng_arg_ids_set = ng_arg_ids_set.to(self.device)  # (b, seq, seq)
                #
                # ret_dict = self.model(input_ids, input_mask, segment_ids, ng_arg_ids_set=ng_arg_ids_set)
                # arguments_set = ret_dict["arguments_set"][i].detach().cpu().tolist()

                # loss = self.loss(output, input_ids)

                # self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                # self.writer.add_scalar('loss', loss.item())
                # total_val_loss += loss.item()
                # total_val_metrics += self._eval_metrics(output, target)
                # self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

        # add histogram of model parameters to the tensorboard
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins='auto')

        return {
            # 'val_loss': total_val_loss / len(self.valid_data_loader),
            'val_metrics': (total_val_metrics / len(self.valid_data_loader)).tolist()
        }
