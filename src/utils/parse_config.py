import logging
import os
from datetime import datetime
from functools import reduce, partial
from operator import getitem
from pathlib import Path

from logger import setup_logging
from utils import read_json, write_json

LOG_LEVELS = {
    0: logging.WARNING,
    1: logging.INFO,
    2: logging.DEBUG
}


class ConfigParser:
    def __init__(self, config, resume=None, run_id=None):
        """
        class to parse configuration json file. Handles hyperparameters for training, initializations of modules,
        checkpoint saving and logging module.
        :param config: Dict containing configurations, hyperparameters for training. contents of `config.json` file
         for example.
        :param resume: String, path to the checkpoint being loaded.
        :param run_id: Unique Identifier for training processes. Used to save checkpoints and training log.
         Timestamp is being used as default
        """
        # load config file and apply modification
        self._config = config
        self.resume = resume
        self._save_dir = None

        if self.config['trainer']['save_dir'] == '':
            return

        # set save_dir where trained model and log will be saved.
        exper_name = self.config['name']
        if run_id is None:  # use timestamp as default run-id
            run_id = datetime.now().strftime(r'%m%d_%H%M%S')
        self._save_dir = Path(self.config['trainer']['save_dir']) / exper_name / run_id
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # save updated config file to the checkpoint dir
        write_json(self.config, self.save_dir / 'config.json')

        # configure logging module
        setup_logging(self.log_dir, log_config='src/logger/logger_config.json')

    @classmethod
    def from_parser(cls, parser, options=None, run_id=None, inherit_save_dir=False):
        """
        Initialize this class from some cli arguments. Used in train, test.
        """
        if options is None:
            options = []
        for opt in options:
            parser.add_argument(*opt.flags, default=None, type=opt.type)
        args = parser.parse_args()

        if args.device is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = args.device

        resume = ensemble = None
        if args.resume is not None:
            resume = Path(args.resume)
            cfg_file = resume.parent / 'config.json'
            if inherit_save_dir is False and run_id is None:
                run_id = str(resume.parent.name)
        elif getattr(args, 'ens', None) is not None:
            cfg_file = next(Path(args.ens).glob('*/config.json'))
            ensemble = args.ens
            if run_id is None:
                run_id = ''
        elif args.config is not None:
            cfg_file = Path(args.config)
        else:
            raise ValueError("Configuration file need to be specified. Add '-c config.json', for example.")

        config = read_json(cfg_file)
        if args.config and (resume or ensemble):
            config.update(read_json(args.config))
            if resume is not None and config['name'] != str(resume.parent.parent.name):
                if run_id is None:
                    run_id = str(resume.parent.parent.name)

        # parse custom cli options into dictionary
        modification = {opt.target: getattr(args, _get_opt_name(opt.flags)) for opt in options}
        config = _update_config(config, modification)
        return cls(config, resume=resume, run_id=run_id)

    def init_obj(self, name, module, *args, **kwargs):
        """
        Finds a function handle with the name given as 'type' in config, and returns the
        instance initialized with corresponding arguments given.

        `object = config.init_obj('name', module, a, b=1)`
        is equivalent to
        `object = module.name(a, b=1)`
        """
        module_name = self[name]['type']
        module_args = dict(self[name]['args'])
        assert all([k not in module_args for k in kwargs]), 'Overwriting kwargs given in config file is not allowed'
        module_args.update(kwargs)
        return getattr(module, module_name)(*args, **module_args)

    def init_ftn(self, name, module, *args, **kwargs):
        """
        Finds a function handle with the name given as 'type' in config, and returns the
        function with given arguments fixed with functools.partial.

        `function = config.init_ftn('name', module, a, b=1)`
        is equivalent to
        `function = lambda *args, **kwargs: module.name(a, *args, b=1, **kwargs)`.
        """
        module_name = self[name]['type']
        module_args = dict(self[name]['args'])
        assert all([k not in module_args for k in kwargs]), 'Overwriting kwargs given in config file is not allowed'
        module_args.update(kwargs)
        return partial(getattr(module, module_name), *args, **module_args)

    def __getitem__(self, name):
        """Access items like ordinary dict."""
        return self.config[name]

    @staticmethod
    def get_logger(name, verbosity=2):
        msg_verbosity = 'verbosity option {} is invalid. Valid options are {}.'.format(verbosity, LOG_LEVELS.keys())
        assert verbosity in LOG_LEVELS, msg_verbosity
        logger = logging.getLogger(name)
        logger.setLevel(LOG_LEVELS[verbosity])
        return logger

    # setting read-only attributes
    @property
    def config(self) -> dict:
        return self._config

    @property
    def save_dir(self) -> Path:
        return self._save_dir

    @property
    def log_dir(self) -> Path:
        return self._save_dir


# helper functions to update config dict with custom cli options
def _update_config(config, modification):
    if modification is None:
        return config

    for k, v in modification.items():
        if v is not None:
            _set_by_path(config, k, v)
    return config


def _get_opt_name(flags):
    for flg in flags:
        if flg.startswith('--'):
            return flg.replace('--', '')
    return flags[0].replace('--', '')


def _set_by_path(tree, keys, value):
    """Set a value in a nested object in tree by sequence of keys."""
    keys = keys.split(';')
    _get_by_path(tree, keys[:-1])[keys[-1]] = value


def _get_by_path(tree, keys):
    """Access a nested object in tree by sequence of keys."""
    return reduce(getitem, keys, tree)
