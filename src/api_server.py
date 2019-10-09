import io
import json
import logging
import textwrap
import argparse
from pathlib import Path
from collections import OrderedDict

from flask import Flask, request, jsonify, make_response
import torch
from pyknp import KNP

from data_loader.dataset import PASDataset
from writer.prediction_writer import PredictionKNPWriter
from kwdlc_reader import Document
import model.model as module_arch


app = Flask(__name__)

logger = logging.getLogger(__name__)


@app.route('/api')
def api():
    inp = ''.join(request.args['input'].split())  # remove space character

    input_sentences = [s.strip() + '。' for s in inp.strip('。').split('。')]
    logger.info(f'input: ' + ''.join(input_sentences))

    knp = KNP(option='-tab')
    knp_string = ''.join(knp.parse(input_sentence).all() for input_sentence in input_sentences)

    dataset_config: dict = config['test_kwdlc_dataset']['args']
    dataset_config['path'] = None
    dataset_config['knp_string'] = knp_string
    dataset = PASDataset(**dataset_config)

    with torch.no_grad():
        input_ids, input_mask, arguments_ids, ng_arg_mask, deps = dataset[0]
        input_ids = torch.tensor(input_ids).to(device).unsqueeze(0)          # (1, seq)
        input_mask = torch.tensor(input_mask).to(device).unsqueeze(0)        # (1, seq)
        arguments_ids = torch.tensor(arguments_ids).to(device).unsqueeze(0)  # (1, seq, case)
        ng_arg_mask = torch.tensor(ng_arg_mask).to(device).unsqueeze(0)      # (1, seq, seq)
        deps = torch.tensor(deps).to(device).unsqueeze(0)                    # (1, seq, seq)

        output = model(input_ids, input_mask, ng_arg_mask, deps)  # (1, seq, case, seq)

        arguments_set = torch.argmax(output, dim=3)[:, :, :arguments_ids.size(2)]  # (1, seq, case)

    prediction_writer = PredictionKNPWriter(dataset, logger)
    document_pred: Document = prediction_writer.write(arguments_set.tolist(), None)[0]

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
        "input": ''.join(input_sentences),
        "output": html_string
    }))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, type=str,
                        help='path to trained checkpoint')
    parser.add_argument('--host', default='0.0.0.0', type=str,
                        help='host ip address (default: 0.0.0.0)')
    parser.add_argument('--port', default=12345, type=int,
                        help='host port number (default: 12345)')
    args = parser.parse_args()

    device = torch.device('cpu')

    config_path = Path(args.model).parent / 'config.json'
    with config_path.open() as handle:
        config = json.load(handle, object_hook=OrderedDict)

    # build model architecture
    model_args = dict(config['arch']['args'])
    model = getattr(module_arch, config['arch']['type'])(**model_args)
    coreference = config['test_kwdlc_dataset']['args']['coreference']
    model.expand_vocab(len(config['test_kwdlc_dataset']['args']['exophors']) + 1 + int(coreference))  # NULL and (NA)

    # prepare model for testing
    logger.info(f'Loading checkpoint: {args.model} ...')
    state_dict = torch.load(args.model, map_location=device)['state_dict']
    model.load_state_dict({k.replace('module.', ''): v for k, v in state_dict.items()})
    model.eval()

    app.run(host=args.host, port=args.port, debug=False, threaded=False)
