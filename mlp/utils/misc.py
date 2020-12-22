from argparse import ArgumentParser
from datetime import datetime
import os
import random
from typing import Generator, Iterable, List, Optional, TypeVar

import numpy as np
import numpy.random as npr
import torch
from torch.utils.data import DataLoader

from .logger import Logger, get_root


DataType = TypeVar('DataType')


def add_defaults(parser: ArgumentParser, name: str, append: bool = False) -> dict:
    parser.add_argument('--log-level', default='info')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--cuda', type=int, default=0)
    group.add_argument('--no-cuda', action='store_const', dest='cuda', const=-1)
    parser.add_argument('--no-log', action='store_false', dest='log', default=True)
    parser.add_argument('--memo')
    if append:
        parser.add_argument('path')
    args = parser.parse_args()
    root: Optional[str]
    if args.log:
        if append:
            root = os.path.join(os.getcwd(), args.path)
        else:
            now = datetime.now()
            root = get_root(name, str(now.date()), str(now.time()))
            args.timestamp = now.isoformat()
    else:
        root = None
    logger = Logger(root=root, level=args.log_level)
    if not append:
        logger.save_args(args)
    if args.cuda >= 0 and torch.cuda.is_available():
        args.device = torch.device(f'cuda:{args.cuda}')
    else:
        args.device = torch.device('cpu')
    del args.cuda
    del args.log
    del args.log_level
    try:
        del args.memo
    except AttributeError:
        pass
    try:
        del args.path
    except AttributeError:
        pass
    try:
        del args.timestamp
    except AttributeError:
        pass
    args.logger = logger
    return vars(args)


def batch_sampler(dataset: Iterable[DataType],
                  batch_size: int,
                  drop_last: bool = False,
                  ) -> Generator[List[DataType], None, None]:
    batch: List[DataType] = []
    for data in dataset:
        batch.append(data)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch and not drop_last:
        yield batch


def repeat_loader(loader: DataLoader):
    while True:
        for batch in loader:
            yield batch


def arg_topk(array: np.ndarray, k: int) -> np.ndarray:
    if k == 0:
        return np.asarray([])
    return np.argpartition(array, -k)[-k:]


def arg_bottomk(array: np.ndarray, k: int) -> np.ndarray:
    if k == 0:
        return np.asarray([])
    return np.argpartition(array, k)[:k]


def set_seed(seed: int) -> None:
    random.seed(seed)
    npr.seed(seed)
    torch.manual_seed(seed)
