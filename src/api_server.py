import io
import html
import logging
import textwrap
import argparse
from pathlib import Path
from datetime import datetime

from flask import Flask, request, jsonify, make_response
from kyoto_reader import Document

from prediction.prediction_writer import PredictionKNPWriter
from analyzer import Analyzer
from predict import draw_tree
from utils.parse_config import ConfigParser

app = Flask(__name__)

logger = logging.getLogger(__name__)


@app.route('/api')
def api():
    input_string = request.args['input']
    log_dir = Path('log') / datetime.now().strftime(r'%Y%m%d_%H%M%S')

    arguments_set, dataset = analyzer.analyze(input_string, knp_dir=log_dir)

    prediction_writer = PredictionKNPWriter(dataset, logger)
    with io.StringIO() as string:
        document: Document = prediction_writer.write(arguments_set, string, skip_untagged=False)[0]
        knp_result: str = string.getvalue()
    with log_dir.joinpath('pas.knp').open('wt') as f:
        f.write(knp_result)

    html_string = textwrap.dedent('''
        <style type="text/css">
        pre {
            font-family: "ＭＳ ゴシック", "Osaka-Mono", "Osaka-等幅", "さざなみゴシック", "Sazanami Gothic", DotumChe,
            GulimChe, BatangChe, MingLiU, NSimSun, Terminal;
            white-space: pre;
        }
        </style>
        ''')
    html_string += '<pre>\n'
    for sid in document.sid2sentence.keys():
        with io.StringIO() as string:
            draw_tree(document, sid, dataset.target_cases, dataset.bridging, dataset.coreference, string, html=True)
            tree_string = string.getvalue()
        logger.info('output:\n' + tree_string)
        html_string += tree_string
    html_string += '</pre>\n'

    return make_response(jsonify({
        "input": analyzer.sanitize_string(input_string),
        "output": [
            {'result': html_string},
            {'results in a KNP format': html.escape(knp_result).replace('\n', '<br>')}
        ]
    }))


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
    parser.add_argument('--remote-knp', action='store_true', default=False,
                        help='Use KNP running on remote host. '
                             'Make sure you specify host address and port in analyzer/config.ini')
    args = parser.parse_args()
    config = ConfigParser.from_args(args, run_id='')
    analyzer = Analyzer(config, logger=logger, remote_knp=args.remote_knp)

    app.run(host=args.host, port=args.port, debug=False, threaded=False)
