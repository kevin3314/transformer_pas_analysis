import sys
import logging
import argparse

from writer.prediction_writer import PredictionKNPWriter
from kwdlc_reader import Document
from analyzer import Analyzer


def main(args):
    logger = logging.getLogger(__name__)
    analyzer = Analyzer(args.model, device=args.device, logger=logger, bertknp=args.use_bertknp)

    if args.input is not None:
        input_string = args.input
    else:
        input_string = ''.join(sys.stdin.readlines())

    arguments_set, dataset = analyzer.analyze(input_string)

    prediction_writer = PredictionKNPWriter(dataset, logger)
    if args.tab is True:
        prediction_writer.write(arguments_set, sys.stdout)
    else:
        document_pred: Document = prediction_writer.write(arguments_set, None)[0]
        for sid in document_pred.sid2sentence.keys():
            document_pred.draw_tree(sid, sys.stdout)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-m', '--model', '-r', '--resume', required=True, type=str,
                        help='path to trained checkpoint')
    parser.add_argument('-d', '--device', default='', type=str,
                        help='indices of GPUs to enable (default: all)')
    parser.add_argument('--input', default=None, type=str,
                        help='sentences to analysis (if not specified, use stdin)')
    parser.add_argument('-tab', action='store_true', default=False,
                        help='whether to output details')
    parser.add_argument('--use-bertknp', action='store_true', default=False,
                        help='use BERTKNP in base phrase segmentation and parsing')
    # parser.add_argument('-c', '--config', default=None, type=str,
    #                     help='config file path (default: None)')
    main(parser.parse_args())
