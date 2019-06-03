import sys
from logging import Logger
from typing import List, Optional, Dict

# import torch

from data_loader.dataset import PasExample, InputFeatures

#
# def _parse_result(result_str: str, metric_name: str, cases: List[str]):
#     result_lines: List[str] = result_str.split('\n')
#     start_idx: int = result_lines.index(metric_name) + 1
#     f1_dic: Dict[str: float] = {}
#     for idx, line in list(enumerate(result_lines))[start_idx:]:
#         if line in cases:
#             f1_line = result_lines[idx + 3]
#             assert f1_line.startswith('F:')
#             f1_dic[line] = int(f1_line[2:])
#             cases.remove(line)
#         if not cases:
#             break
#     return f1_dic


def case_analysis_f1_ga(result: dict):
    return result['ガ']['case_analysis']['F']


def case_analysis_f1_wo(result: dict):
    return result['ヲ']['case_analysis']['F']


def case_analysis_f1_ni(result: dict):
    return result['ニ']['case_analysis']['F']


def case_analysis_f1_ga2(result: dict):
    return result['ガ２']['case_analysis']['F']


def case_analysis_f1(result: dict):
    return result['all_case']['case_analysis']['F']


def zero_anaphora_f1_ga(result: dict):
    return result['ガ']['anaphora_all']['F']


def zero_anaphora_f1_wo(result: dict):
    return result['ヲ']['anaphora_all']['F']


def zero_anaphora_f1_ni(result: dict):
    return result['ニ']['anaphora_all']['F']


def zero_anaphora_f1_ga2(result: dict):
    return result['ガ２']['anaphora_all']['F']


def zero_anaphora_f1(result: dict):
    return result['all_case']['anaphora_all']['F']


def zero_anaphora_f1_inter(result: dict):
    return result['all_case']['anaphora_inter_sentential']['F']


def zero_anaphora_f1_intra(result: dict):
    return result['all_case']['anaphora_intra_sentential']['F']


def zero_anaphora_f1_writer_reader(result: dict):
    return result['all_case']['anaphora_writer_reader']['F']


# def my_metric(output, target):
#     with torch.no_grad():
#         pred = torch.argmax(output, dim=1)
#         assert pred.shape[0] == len(target)
#         correct = 0
#         correct += torch.sum(pred == target).item()
#     return correct / len(target)
#
#
# def my_metric2(output, target, k=3):
#     with torch.no_grad():
#         pred = torch.topk(output, k, dim=1)[1]
#         assert pred.shape[0] == len(target)
#         correct = 0
#         for i in range(k):
#             correct += torch.sum(pred[:, i] == target).item()
#     return correct / len(target)


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


def write_prediction(all_examples: List[PasExample],
                     all_features: List[InputFeatures],
                     arguments_sets: List[List[List[int]]],
                     output_prediction_file: Optional[str],
                     dataset_config: dict,
                     logger: Logger):
    """Write final predictions to the file."""
    special_tokens: List[str] = dataset_config['special_tokens']
    max_seq_length: int = dataset_config['max_seq_length']
    tok_to_special: Dict[int, str] = {i + max_seq_length - len(special_tokens): token for i, token
                                      in enumerate(special_tokens)}
    cases = dataset_config['cases']
    coreference = dataset_config['coreference']

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
