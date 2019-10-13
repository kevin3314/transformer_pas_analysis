import os
import io
import sys
import logging
import argparse
from xmlrpc.server import SimpleXMLRPCServer

from writer.prediction_writer import PredictionKNPWriter
from analyzer import Analyzer

logger = logging.getLogger(__name__)


def show_usage():
    print('Usage:')
    print('$ env HOST=<HOST> PORT=<PORT> pipenv run server.py')


def analyze_raw_data_from_client(knp_result: str):
    arguments_set, dataset = analyzer.analyze_from_knp(knp_result)

    prediction_writer = PredictionKNPWriter(dataset, logger)
    with io.StringIO() as string:
        _ = prediction_writer.write(arguments_set, string)
        knp_result = string.getvalue()
    return knp_result


if __name__ == '__main__':
    HOST = os.environ.get('HOST')
    PORT = os.environ.get('PORT')
    if not all((HOST, PORT)):
        show_usage()
        sys.exit()

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, type=str,
                        help='path to trained checkpoint')
    args = parser.parse_args()

    analyzer = Analyzer(args.model, device='', logger=logger, bertknp=True)

    server = SimpleXMLRPCServer((HOST, int(PORT)))
    server.register_function(analyze_raw_data_from_client)
    server.serve_forever()
