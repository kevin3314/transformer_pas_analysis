import os
import sys
import argparse
from typing import List, Optional, Dict
from logging import Logger

import torch
import torch.nn as nn

import data_loader.data_loaders as module_loader
import data_loader.dataset as module_dataset
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser
from data_loader.dataset import PasExample, InputFeatures


def output_pas_analysis(items: List[str],
                        cases: List[str],
                        arguments_set: List[List[int]],
                        features: InputFeatures,
                        tok_to_special: Dict[int, str],
                        coreference: bool,
                        logger: Logger):
    target_token_index = features.orig_to_tok_index[int(items[0]) - 1]
    target_arguments = arguments_set[target_token_index]

    if items[5] != "_":
        # ガ:55%C,ヲ:57,ニ:NULL,ガ２:NULL
        orig_arguments = {arg_string.split(":", 1)[0]: arg_string.split(":", 1)[1]
                          for arg_string in items[5].split(",")}

        argument_strings = []
        for case, argument_index in zip(cases, target_arguments):
            if coreference is True and case == "=":
                continue

            if "%C" in orig_arguments[case]:
                argument_string = orig_arguments[case]
            else:
                # special
                if argument_index in tok_to_special:
                    argument_string = tok_to_special[argument_index]
                elif features.tok_to_orig_index[argument_index] is None:
                    # [SEP] or [CLS]
                    logger.warning("Choose [SEP] as an argument. Tentatively, change it to NULL.")
                    argument_string = "NULL"
                else:
                    argument_string = features.tok_to_orig_index[argument_index] + 1

            argument_strings.append(case + ":" + str(argument_string))

        items[5] = ",".join(argument_strings)

    if coreference is True and items[6] == "MASKED":
        argument_index = target_arguments[-1]
        # special
        if argument_index in tok_to_special:
            argument_string = tok_to_special[argument_index]
        else:
            argument_string = features.tok_to_orig_index[argument_index] + 1
        items[6] = str(argument_string)

    return items


def write_predictions(all_examples: List[PasExample],
                      all_features: List[InputFeatures],
                      arguments_sets: List[List[List[int]]],
                      output_prediction_file: Optional[str],
                      tok_to_special: Dict[int, str],
                      cases: List[str],
                      coreference: bool,
                      logger: Logger):
    """Write final predictions to the file."""
    if output_prediction_file is not None:
        logger.info(f"Writing predictions to: {output_prediction_file}")

    if coreference is True:
        cases.append("=")

    with open(output_prediction_file, "w") if output_prediction_file is not None else sys.stdout as writer:
        for example, feature, arguments_set in zip(all_examples, all_features, arguments_sets):
            if example.comment is not None:
                writer.write("{}\n".format(example.comment))

            for line in example.lines:
                items = line.split("\t")
                items = output_pas_analysis(items, cases, arguments_set, feature, tok_to_special, coreference, logger)
                writer.write("\t".join(items) + "\n")

            writer.write("\n")


def main(config):
    logger = config.get_logger('test')

    # setup data_loader instances
    dataset = config.initialize('test_dataset', module_dataset)
    data_loader = config.initialize('test_data_loader', module_loader, dataset)

    # build model architecture
    model = config.initialize('arch', module_arch)
    model.expand_vocab(num_expand_vocab=5)  # same as that in dataset.py. TODO: consider resume case
    logger.info(model)

    # get function handles of loss and metrics
    loss_fn = getattr(module_loss, config['loss'])
    # metric_fns = [getattr(module_metric, met) for met in config['metrics']]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    logger.info('Loading checkpoint: {} ...'.format(config.resume))
    checkpoint = torch.load(config.resume, map_location=device)
    state_dict = checkpoint['state_dict']
    if config['n_gpu'] > 1:
        model = nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    model = model.to(device)
    model.eval()

    total_loss = 0.0
    # total_metrics = torch.zeros(len(metric_fns))
    arguments_sets: List[List[List[int]]] = []
    with torch.no_grad():
        for batch_idx, (input_ids, input_mask, arguments_ids, ng_arg_mask) in enumerate(data_loader):
            input_ids = input_ids.to(device)          # (b, seq)
            input_mask = input_mask.to(device)        # (b, seq)
            arguments_ids = arguments_ids.to(device)  # (b, seq, case)
            ng_arg_mask = ng_arg_mask.to(device)      # (b, seq, seq)

            output = model(input_ids, input_mask, ng_arg_mask)  # (b, seq, case, seq)

            arguments_set = torch.argmax(output, dim=3)  # (b, seq, case)
            arguments_sets += arguments_set.tolist()

            # computing loss, metrics on test set
            loss = loss_fn(output, arguments_ids)
            total_loss += loss.item() * input_ids.size(0)
            # for i, metric in enumerate(metric_fns):
            #     total_metrics[i] += metric(ret_dict) * batch_size

    output_prediction_file: str = os.path.join(config.save_dir, 'test_out.conll')
    special_tokens: List[str] = config['test_dataset']['args']['special_tokens']
    num_special_tokens: int = len(special_tokens)
    max_seq_length: int = config['test_dataset']['args']['max_seq_length']
    tok_to_special: Dict[int, str] = {i + max_seq_length - num_special_tokens: token for i, token
                                      in enumerate(special_tokens)}
    cases = config['test_dataset']['args']['cases']
    write_predictions(dataset.pas_examples, dataset.features, arguments_sets, output_prediction_file,
                      tok_to_special=tok_to_special, cases=cases,
                      coreference=config['test_dataset']['args']['coreference'], logger=logger)

    log = {'loss': total_loss / data_loader.n_samples}
    # log.update({
    #     met.__name__: total_metrics[i].item() / n_samples for i, met in enumerate(metric_fns)
    # })
    logger.info(log)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-r', '--resume', default=None, type=str,
                        help='path to latest checkpoint (default: None)')
    parser.add_argument('-d', '--device', default=None, type=str,
                        help='indices of GPUs to enable (default: all)')

    main(ConfigParser(parser, timestamp=False))
