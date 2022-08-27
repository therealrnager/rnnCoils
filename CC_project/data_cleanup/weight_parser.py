import numpy as np
import tensorflow as tf
import os
import dill

base_direc = '/n/eddy_lab/users/rangerkuang/CC_data/'

amino_list = 'ARNDCQEGHILKMFPSTWYVOUZXB'

def weight_parse(end, weight):
    global base_direc
    subdir = ['has_cc_real/to100/', 'has_cc_real/to400/', 'has_cc_real/to_max/', 'no_cc_real/to100/', 'no_cc_real/to400/', 'no_cc_real/to_max/']

    weights = {'-': 1, 'a': weight, 'b': weight, 'c': weight, 'd': weight, 'e': weight, 'f': weight, 'g': weight} # MODIFY THIS TO CHANGE WEIGHT RATIOS

    weights_binary = []
    for i in subdir:
        mid_direc = os.path.join(base_direc, i)
        direc = os.path.join(mid_direc, end)
        for filename in os.scandir(direc):
            with open(filename.path, 'r') as f:
                contents = f.readlines()

                weights_b = [weights[ch] for ch in contents[2][:-2]]
                weights_binary.append(weights_b)
               
        print(i, ' is done!')

    wb = tf.ragged.constant(weights_binary)
    return wb

pkl_direc = "/n/eddy_lab/users/rangerkuang/CC_data/unique_pkl_data/"


lst = [1]
for i in lst: 
    print(f"==========={i} weight will now be processed================")
    print("=======TEST=======")
    weights_test_binary = weight_parse('test', i)
    #print(weights_test_binary.nrows())

    with open(os.path.join(pkl_direc, f"weights_test_binary_{i}.pkl"), "wb+") as f: # w = write, b = binary, + = create if not created yet
        dill.dump(weights_test_binary, f)

    print("=======VALID=======")
    weights_valid_binary = weight_parse('valid', i)
    #print(weights_valid_binary.nrows())

    with open(os.path.join(pkl_direc, f"weights_valid_binary_{i}.pkl"), "wb+") as f:
        dill.dump(weights_valid_binary, f)

    print("=======TRAIN=======")
    weights_train_binary = weight_parse('train', i)
    #print(weights_train_binary.nrows())

    with open(os.path.join(pkl_direc, f"weights_train_binary_{i}.pkl"), "wb+") as f:
        dill.dump(weights_train_binary, f)
