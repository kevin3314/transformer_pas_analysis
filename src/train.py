import argparse
import collections
import random

import transformers.optimization as module_optim
import torch
import numpy as np

import data_loader.data_loaders as module_loader
import data_loader.dataset as module_dataset
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from utils.parse_config import ConfigParser
from trainer import Trainer
from base.base_model import BaseModel


def main(config: ConfigParser, args: argparse.Namespace):
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

    logger = config.get_logger('train')

    # setup data_loader instances
    if config['train_kwdlc_dataset']['args']['path'] is not None:
        train_dataset = config.init_obj('train_kwdlc_dataset', module_dataset, logger=logger)
        if config['train_kc_dataset']['args']['path'] is not None:
            train_dataset += config.init_obj('train_kc_dataset', module_dataset, logger=logger)
    else:
        train_dataset = config.init_obj('train_kc_dataset', module_dataset, logger=logger)
    train_data_loader = config.init_obj('train_data_loader', module_loader, train_dataset)
    valid_kwdlc_data_loader = None
    valid_kc_data_loader = None
    if config['valid_kwdlc_dataset']['args']['path'] is not None:
        valid_kwdlc_dataset = config.init_obj('valid_kwdlc_dataset', module_dataset, logger=logger)
        valid_kwdlc_data_loader = config.init_obj('valid_data_loader', module_loader, valid_kwdlc_dataset)
    if config['valid_kc_dataset']['args']['path'] is not None:
        valid_kc_dataset = config.init_obj('valid_kc_dataset', module_dataset, logger=logger)
        valid_kc_data_loader = config.init_obj('valid_data_loader', module_loader, valid_kc_dataset)

    # build model architecture, then print to console
    model: BaseModel = config.init_obj('arch', module_arch, vocab_size=train_dataset.expanded_vocab_size)
    logger.info(model)

    # get function handles of loss and metrics
    loss = getattr(module_loss, config['loss'])
    metrics = [getattr(module_metric, met) for met in config['metrics']]

    # build optimizer, learning rate scheduler
    trainable_named_params = filter(lambda x: x[1].requires_grad, model.named_parameters())
    no_decay = ('bias', 'LayerNorm.weight')
    weight_decay = config['optimizer']['args']['weight_decay']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in trainable_named_params if not any(nd in n for nd in no_decay)],
         'weight_decay': weight_decay},
        {'params': [p for n, p in trainable_named_params if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = config.init_obj('optimizer', module_optim, optimizer_grouped_parameters)

    lr_scheduler = config.init_obj('lr_scheduler', module_optim, optimizer)

    trainer = Trainer(model, loss, metrics, optimizer,
                      config=config,
                      data_loader=train_data_loader,
                      valid_kwdlc_data_loader=valid_kwdlc_data_loader,
                      valid_kc_data_loader=valid_kc_data_loader,
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
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size')
    ]
    main(ConfigParser.from_parser(parser, options), parser.parse_args())
