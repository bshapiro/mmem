from config import config
from os import makedirs
import pandas as pd
import numpy as np


def generate_output_dir():
    directory = ''
    directory += config['init'] + '/'
    directory += config['dir'] + str(config['k']) + '/'

    try:
        makedirs(directory)
    except:
        pass
        # print 'Output directory already exists.'
    return directory


def flatten_tuple_list(tuple_list):
    flat_list = []
    for a_tuple in tuple_list:
        flat_list.extend(list(a_tuple))
    return flat_list


def pair_dict(pairs):
    pair_dict = {}
    for pair in pairs:
        pair_dict[pair[0]] = pair[1]
        pair_dict[pair[1]] = pair[0]
    return pair_dict


def unpack_args(func):
    from functools import wraps
    @wraps(func)
    def wrapper(args):
        if isinstance(args, dict):
            return func(**args)
        else:
            return func(*args)
    return wrapper
