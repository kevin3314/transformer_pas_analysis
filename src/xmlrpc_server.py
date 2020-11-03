import io
import logging
import argparse
from pathlib import Path
from datetime import datetime
from xmlrpc.server import SimpleXMLRPCServer

from prediction.prediction_writer import PredictionKNPWriter
from analyzer import Analyzer
from utils.parse_config import ConfigParser

logger = logging.getLogger(__name__)


def analyze_raw_data_from_client(knp_result: str):
    log_dir = Path('log') / datetime.now().strftime(r'%Y%m%d_%H%M%S')
    arguments_set, dataset = analyzer.analyze_from_knp(knp_result, knp_dir=log_dir)

    prediction_writer = PredictionKNPWriter(dataset, logger)
    with io.StringIO() as string:
        _ = prediction_writer.write(arguments_set, string, skip_untagged=False)
        knp_result = string.getvalue()
    with log_dir.joinpath('pas.knp').open('wt') as f:
        f.write(knp_result)
    return knp_result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--resume', '-m', '--model', default=None, type=str,
                        help='path to trained checkpoint')
    parser.add_argument('--ens', '--ensemble', default=None, type=str,
                        help='path to directory where checkpoints to ensemble exist')
    parser.add_argument('-d', '--device', default='', type=str,
                        help='indices of GPUs to enable (default: all)')
    parser.add_argument('-c', '--config', default=None, type=str,
                        help='config file path (default: None)')
    parser.add_argument('--host', default='0.0.0.0', type=str,
                        help='host ip address (default: 0.0.0.0)')
    parser.add_argument('--port', default=12345, type=int,
                        help='host port number (default: 12345)')
    args = parser.parse_args()
    config = ConfigParser.from_args(args, run_id='')
    analyzer = Analyzer(config, logger=logger)

    server = SimpleXMLRPCServer((args.host, args.port))
    server.register_function(analyze_raw_data_from_client)
    server.serve_forever()
