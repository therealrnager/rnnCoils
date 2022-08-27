import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os
import dill

base_direc = '/n/eddy_lab/users/rangerkuang/CC_data/'

def parse_data(end):
    global base_direc
    subdir = ['has_cc_real/to100/', 'has_cc_real/to400/', 'has_cc_real/to_max/', 'no_cc_real/to100/', 'no_cc_real/to400/', 'no_cc_real/to_max/']
    # subdir = ['has_cc_real/to100/', 'has_cc_real/to_max/']

    alphabet = {'-': 0, 'a': 1, 'b': 1, 'c': 1, 'd': 1, 'e': 1, 'f': 1, 'g': 1}

    vocab = set()
    for i in subdir:
        mid_direc = os.path.join(base_direc, i)
        direc = os.path.join(mid_direc, end)
        for filename in os.scandir(direc):
            with open(filename.path, 'r') as f:
                contents = f.readlines()
                for residue in contents[0][:-1]:
                    vocab.add(residue)
        print(i, " done")


    return vocab
vocab = parse_data('test')
print(vocab)
