import logging
from argparse import Namespace
from os import scandir
from os.path import abspath, dirname, join
from pathlib import Path
from typing import Optional

import yaml
import torch
from torch.nn import Module


ROOT = join(dirname(dirname(dirname(abspath(__file__)))), 'results')
PREFIX = 'model_'
SUFFIX = '.pth'


def get_root(name: str, *paths: str) -> str:
    root = join(ROOT, name, *paths)
    Path(root).mkdir(parents=True, exist_ok=False)
    return root


class Logger:
    def __init__(self, root: Optional[str], level: str = 'info') -> None:
        self.root = root
        if root is None:
            logging.basicConfig(level=level.upper())
        else:
            logging.basicConfig(filename=join(root, 'log'), level=level.upper(),
                                format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                                datefmt='%Y-%m-%dT%H:%M:%S')

    def _model_path(self, index: int, prefix: str, suffix: str) -> Optional[str]:
        if self.root is None:
            return None
        return join(self.root, f'{prefix}{index}{suffix}')

    def save_obj(self, obj, name: str) -> None:
        if self.root is None:
            return
        if not name.endswith('.yaml'):
            name += '.yaml'
        with open(join(self.root, name), 'w') as file:
            yaml.dump(obj, file)

    def save_args(self, args: Namespace) -> None:
        self.save_obj(vars(args), 'arguments')

    def save_model(self, net: Module, index: int,
                   prefix: str = PREFIX, suffix: str = SUFFIX) -> None:
        if self.root is None:
            return
        torch.save(net.state_dict(), self._model_path(index, prefix, suffix))

    def load_model(self, net: Module, index: Optional[int] = None,
                   prefix: str = PREFIX, suffix: str = SUFFIX) -> None:
        if index is None:
            pre, suf = len(prefix), len(suffix)
            indices = []
            with scandir(self.root) as iterator:
                for entry in iterator:
                    if (entry.name.startswith(prefix)
                            and entry.name.endswith(suffix)
                            and entry.is_file()):
                        try:
                            indices.append(int(entry.name[pre:-suf]))
                        except ValueError:
                            pass
            index = max(indices)
        net.load_state_dict(torch.load(self._model_path(index, prefix, suffix)))
