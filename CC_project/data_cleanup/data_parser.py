import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os
import dill
from collections import Counter
import shutil
import random

base_direc = '/n/eddy_lab/users/rangerkuang/CC_data/'

amino_list = 'ARNDCQEGHILKMFPSTWYVOUZXB'
vocab = sorted(set(amino_list))
#print(f'{len(vocab)} unique chars')
seq_char2id = tf.keras.layers.StringLookup(vocabulary = list(vocab), mask_token=None)
seq_id2char = tf.keras.layers.StringLookup(vocabulary = seq_char2id.get_vocabulary(), invert=True, mask_token=None)
target_list = '-abcdefg'
target_char2id = tf.keras.layers.StringLookup(vocabulary = list(target_list), mask_token=None)
target_id2char = tf.keras.layers.StringLookup(vocabulary = target_char2id.get_vocabulary(), invert=True, mask_token=None)

def parse_data(end):
    global base_direc
    subdir = ['has_cc_real/to100/', 'has_cc_real/to400/', 'has_cc_real/to_max/', 'no_cc_real/to100/', 'no_cc_real/to400/', 'no_cc_real/to_max/']
    # subdir = ['has_cc_real/to100/', 'has_cc_real/to_max/']

    alphabet = {'-': 0, 'a': 1, 'b': 1, 'c': 1, 'd': 1, 'e': 1, 'f': 1, 'g': 1}
    weights = {'-': 1, 'a': 10, 'b': 10, 'c': 10, 'd': 10, 'e': 10, 'f': 10, 'g': 10}
    
    input_sequences = []
    target_sequences = []
    target_sequences_binary = []
    weights_binary = []
    for i in subdir:
        mid_direc = os.path.join(base_direc, i)
        direc = os.path.join(mid_direc, end)
        for filename in os.scandir(direc):
            with open(filename.path, 'r') as f:
                contents = f.readlines()
                seq = contents[0][:-1]
                targ = contents[2][:-2]

                X_count = seq.count("X")  
                if (X_count / len(seq) > 0.3): # seq is more than 30% X, which is kinda silly, so skip it 
                    shutil.move(filename.path, os.path.join(mid_direc, 'x_graveyard'))
                    continue
                            
                # OG input and target
                input_sequences.append(seq)
                target_sequences.append(targ)

                # binary target 
                target_seq_binary = [alphabet[ch] for ch in targ]
                target_sequences_binary.append(target_seq_binary)

                # weight stuff
                weights_b = [weights[ch] for ch in targ]
                weights_binary.append(weights_b)
        print(i, ' is done!')

    inp = tf.strings.unicode_split(input_sequences, input_encoding='UTF-8')
    target = tf.strings.unicode_split(target_sequences, input_encoding='UTF-8')
    x_ids = seq_char2id(inp)
    y_ids = target_char2id(target)

    y_ids_binary = tf.ragged.constant(target_sequences_binary)
    wb = tf.ragged.constant(weights_binary)
    return x_ids, y_ids, y_ids_binary, wb

def parse_data_x(end):
    global base_direc
    subdir = ['has_cc_real/to100/', 'has_cc_real/to400/', 'has_cc_real/to_max/', 'no_cc_real/to100/', 'no_cc_real/to400/', 'no_cc_real/to_max/']
    # subdir = ['has_cc_real/to100/', 'has_cc_real/to_max/']

    alphabet = {'-': 0, 'a': 1, 'b': 1, 'c': 1, 'd': 1, 'e': 1, 'f': 1, 'g': 1}

    input_sequences = []
    # target_sequences = []
    # target_sequences_binary = []
    for i in subdir:
        mid_direc = os.path.join(base_direc, i)
        direc = os.path.join(mid_direc, end)
        for filename in os.scandir(direc):
            with open(filename.path, 'r') as f:
                contents = f.readlines()
                input_sequences.append(contents[0][:-1])
                # target_sequences.append(contents[2][:-2])

                # target_seq_binary = [alphabet[ch] for ch in contents[2][:-2]]
                # target_sequences_binary.append(target_seq_binary)
        print(i, ' is done!')

    inp = tf.strings.unicode_split(input_sequences, input_encoding='UTF-8')
    # target = tf.strings.unicode_split(target_sequences, input_encoding='UTF-8')
    x_ids = seq_char2id(inp)
    # y_ids = target_char2id(target)

    # y_ids_binary = tf.ragged.constant(target_sequences_binary)
    return x_ids

##### TEST #####
x_test, y_test, y_test_binary, weights_test_binary = parse_data('test')
print(x_test.nrows())

pkl_direc = "/n/eddy_lab/users/rangerkuang/CC_data/unique_pkl_data/"
with open(os.path.join(pkl_direc, "x_test.pkl"), "wb+") as f:
    dill.dump(x_test, f)

with open(os.path.join(pkl_direc, "y_test.pkl"), "wb+") as f:
    dill.dump(y_test, f)

with open(os.path.join(pkl_direc, "y_test_binary.pkl"), "wb+") as f:
    dill.dump(y_test_binary, f)

with open(os.path.join(pkl_direc, "weights_test_binary_10.pkl"), "wb+") as f:
    dill.dump(weights_test_binary, f)


##### VALID #####
x_valid, y_valid, y_valid_binary, weights_valid_binary = parse_data('valid')
print(x_valid.nrows())

pkl_direc = "/n/eddy_lab/users/rangerkuang/CC_data/unique_pkl_data/"
with open(os.path.join(pkl_direc, "x_valid.pkl"), "wb+") as f:
    dill.dump(x_valid, f)

with open(os.path.join(pkl_direc, "y_valid.pkl"), "wb+") as f:
    dill.dump(y_valid, f)

with open(os.path.join(pkl_direc, "y_valid_binary.pkl"), "wb+") as f:
    dill.dump(y_valid_binary, f)

with open(os.path.join(pkl_direc, "weights_valid_binary_10.pkl"), "wb+") as f:
    dill.dump(weights_valid_binary, f)


##### train #####
x_train, y_train, y_train_binary, weights_train_binary = parse_data('train')
print(x_train.nrows())

pkl_direc = "/n/eddy_lab/users/rangerkuang/CC_data/unique_pkl_data/"
with open(os.path.join(pkl_direc, "x_train.pkl"), "wb+") as f:
    dill.dump(x_train, f)

with open(os.path.join(pkl_direc, "y_train.pkl"), "wb+") as f:
    dill.dump(y_train, f)

with open(os.path.join(pkl_direc, "y_train_binary.pkl"), "wb+") as f:
    dill.dump(y_train_binary, f)

with open(os.path.join(pkl_direc, "weights_train_binary_10.pkl"), "wb+") as f:
    dill.dump(weights_train_binary, f)


