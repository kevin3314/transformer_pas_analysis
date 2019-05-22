import os
import sys
import argparse
from typing import List, Optional
from logging import Logger

import torch

import data_loader.data_loaders as module_loader
import data_loader.dataset as module_dataset
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser
from data_loader.dataset import PasExample
from data_loader.input_features import InputFeatures


def output_pas_analysis(items: List[str],
                        cases: List[str],
                        arguments_set: List[List[int]],
                        features: InputFeatures,
                        line_num: int,
                        max_seq_length: int,
                        num_expand_vocab: int,
                        special_tokens: List[str],
                        coreference: bool,
                        logger: Logger):
    if items[5] != "_":
        # ガ:55%C,ヲ:57,ニ:NULL,ガ２:NULL
        orig_arguments = {arg_string.split(":", 1)[0]: arg_string.split(":", 1)[1]
                          for arg_string in items[5].split(",")}
        argument_strings = []

        for case, argument_string in zip(cases, arguments_set[features.orig_to_tok_index[line_num] + 1]):
            if coreference is True and case == "=":
                continue

            if "%C" in orig_arguments[case]:
                argument_string = orig_arguments[case]
            else:
                # special
                if argument_string >= max_seq_length - num_expand_vocab:
                    argument_string = special_tokens[argument_string - max_seq_length + num_expand_vocab]
                else:
                    # [SEP]
                    if len(features.tok_to_orig_index) + 1 == argument_string:
                        logger.warning("Choose [SEP] as an argument. Tentatively, change it to NULL.")
                        argument_string = "NULL"
                    else:
                        argument_string = features.tok_to_orig_index[argument_string - 1] + 1

            argument_strings.append(case + ":" + str(argument_string))

        items[5] = ",".join(argument_strings)

    if coreference is True and items[6] == "MASKED":
        argument_string = arguments_set[features.orig_to_tok_index[line_num] + 1][-1]
        # special
        if argument_string >= max_seq_length - num_expand_vocab:
            argument_string = special_tokens[argument_string - max_seq_length + num_expand_vocab]
        else:
            argument_string = features.tok_to_orig_index[argument_string - 1] + 1
        items[6] = str(argument_string)

    return items


def write_predictions(all_examples: List[PasExample],
                      all_features: List[InputFeatures],
                      arguments_sets: List[List[List[int]]],
                      output_prediction_file: Optional[str],
                      max_seq_length: int,
                      cases: List[str],
                      num_expand_vocab: int,
                      special_tokens: List[str],
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

            for line_num, line in enumerate(example.lines):
                items = line.split("\t")
                items = output_pas_analysis(items, cases, arguments_set, feature, line_num,
                                            max_seq_length, num_expand_vocab, special_tokens, coreference, logger)
                writer.write("\t".join(items) + "\n")

            writer.write("\n")


def main(config, resume):
    logger = config.get_logger('test')

    # setup data_loader instances
    dataset = config.initialize('test_dataset', module_dataset)
    data_loader = config.initialize('test_data_loader', module_loader, dataset)

    # build model architecture
    model = config.initialize('arch', module_arch)
    model.expand_vocab(num_expand_vocab=5)  # same as that in dataset.py. TODO: consider resume case
    logger.info(model)

    # get function handles of loss and metrics
    # loss_fn = getattr(module_loss, config['loss'])
    # metric_fns = [getattr(module_metric, met) for met in config['metrics']]

    logger.info('Loading checkpoint: {} ...'.format(resume))
    checkpoint = torch.load(resume)
    state_dict = checkpoint['state_dict']
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    # total_loss = 0.0
    # total_metrics = torch.zeros(len(metric_fns))

    arguments_sets: List[List[List[int]]] = []
    for batch_idx, (input_ids, input_mask, segment_ids, example_indices, ng_arg_ids_set) in enumerate(data_loader):
        input_ids = input_ids.to(device)  # (b, seq)
        input_mask = input_mask.to(device)  # (b, seq)
        segment_ids = segment_ids.to(device)  # (b, seq)
        ng_arg_ids_set = ng_arg_ids_set.to(device)  # (b, seq, seq)

        with torch.no_grad():
            arguments_set = model(input_ids, input_mask, segment_ids, ng_arg_ids_set)  # (b, seq, case)

        # for i, example_index in enumerate(example_indices):
        arguments_sets += arguments_set.detach().cpu().tolist()
        # arguments_sets.append(arguments_set)

        # computing loss, metrics on test set
        # loss = loss_fn(output, target)
        # batch_size = input_ids.size(0)
        # total_loss += loss.item() * batch_size
        # for i, metric in enumerate(metric_fns):
        #     total_metrics[i] += metric(ret_dict) * batch_size

    # output_prediction_file = os.path.join(config.save_dir, 'predictions.txt')
    output_prediction_file = 'test_out.conll'
    special_tokens = config['test_dataset']['args']['special_tokens']
    cases = config['test_dataset']['args']['cases']
    write_predictions(dataset.pas_examples, dataset.features, arguments_sets, output_prediction_file,
                      config['test_dataset']['args']['max_seq_length'],
                      cases=cases, num_expand_vocab=len(special_tokens), special_tokens=special_tokens,
                      coreference=config['test_dataset']['args']['coreference'], logger=logger)

    # n_samples = len(data_loader.sampler)
    # log = {'loss': total_loss / n_samples}
    log = {}
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

    main(config=ConfigParser(parser, timestamp=False), resume=parser.parse_args().resume)
