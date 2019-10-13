import io
import logging
import textwrap
import argparse

from flask import Flask, request, jsonify, make_response

from writer.prediction_writer import PredictionKNPWriter
from kwdlc_reader import Document
from analyzer import Analyzer


app = Flask(__name__)

logger = logging.getLogger(__name__)


@app.route('/api')
def api():
    input_string = analyzer.sanitize_string(request.args['input'])
    logger.info(f'input: {input_string}')

    arguments_set, dataset = analyzer.analyze(input_string)

    prediction_writer = PredictionKNPWriter(dataset, logger)
    document_pred: Document = prediction_writer.write(arguments_set, None)[0]

    html_string = textwrap.dedent('''
        <style type="text/css">
        <!--
        td {font-size: 11pt;}
        td {border: 1px solid #606060;}
        td {vertical-align: top;}
        pre {font-family: "ＭＳ ゴシック","Osaka-Mono","Osaka-等幅","さざなみゴシック","Sazanami Gothic",DotumChe,GulimChe,
        BatangChe,MingLiU, NSimSun, Terminal; white-space:pre;}
        -->
        </style>
        ''')
    html_string += '<pre>\n'
    for sid in document_pred.sid2sentence.keys():
        with io.StringIO() as string:
            document_pred.draw_tree(sid, string)
            tree_string = string.getvalue()
        logger.info('output:\n' + tree_string)
        html_string += tree_string
    html_string += '</pre>\n'

    return make_response(jsonify({
        "input": input_string,
        "output": html_string
    }))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', '-r', '--resume', required=True, type=str,
                        help='path to trained checkpoint')
    # parser.add_argument('-c', '--config', default=None, type=str,
    #                     help='config file path (default: None)')
    parser.add_argument('--host', default='0.0.0.0', type=str,
                        help='host ip address (default: 0.0.0.0)')
    parser.add_argument('--port', default=12345, type=int,
                        help='host port number (default: 12345)')
    parser.add_argument('--use-bertknp', action='store_true', default=False,
                        help='use BERTKNP in base phrase segmentation and parsing')
    args = parser.parse_args()

    analyzer = Analyzer(args.model, device='', logger=logger, bertknp=args.use_bertknp)

    app.run(host=args.host, port=args.port, debug=False, threaded=False)
