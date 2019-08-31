import argparse
import collections
import random

import pytorch_transformers.optimization as module_optim
import torch
import numpy

import data_loader.data_loaders as module_loader
import data_loader.dataset as module_dataset
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser
from trainer import Trainer


def main(config):
    torch.manual_seed(42)
    random.seed(42)
    numpy.random.seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    logger = config.get_logger('train')

    # setup data_loader instances
    train_dataset = config.initialize('train_dataset', module_dataset)
    train_data_loader = config.initialize('train_data_loader', module_loader, train_dataset)
    valid_dataset = config.initialize('valid_dataset', module_dataset)
    valid_data_loader = config.initialize('valid_data_loader', module_loader, valid_dataset)

    # build model architecture, then print to console
    model = config.initialize('arch', module_arch)
    model.expand_vocab(train_dataset.num_special_tokens)  # same as that in dataset.py. TODO: consider resume case
    logger.info(model)

    # get function handles of loss and metrics
    loss = getattr(module_loss, config['loss'])
    metrics = [getattr(module_metric, met) for met in config['metrics']]

    # prepare optimizer
    param_optimizer = filter(lambda np: np[1].requires_grad, model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = config.initialize('optimizer', module_optim, optimizer_grouped_parameters)

    lr_scheduler = config.initialize('lr_scheduler', module_optim, optimizer)

    trainer = Trainer(model, loss, metrics, optimizer,
                      config=config,
                      data_loader=train_data_loader,
                      valid_data_loader=valid_data_loader,
                      lr_scheduler=lr_scheduler)

    trainer.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', default=None, type=str,
                        help='config file path (default: None)')
    parser.add_argument('-r', '--resume', default=None, type=str,
                        help='path to latest checkpoint (default: None)')
    parser.add_argument('-d', '--device', default='', type=str,
                        help='indices of GPUs to enable (default: "")')
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed for initialization')

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target=('optimizer', 'args', 'lr')),
        CustomArgs(['--bs', '--batch_size'], type=int, target=('data_loader', 'args', 'batch_size'))
    ]
    main(config=ConfigParser(parser, options))
