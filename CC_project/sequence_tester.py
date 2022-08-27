import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os
import dill
import matplotlib.pyplot as plt


###########################
#####LOAD DATA#############
###########################

amino_list = 'ARNDCQEGHILKMFPSTWYVOUZXB'
vocab = sorted(set(amino_list))
#print(f'{len(vocab)} unique chars')
seq_char2id = tf.keras.layers.StringLookup(vocabulary = list(vocab), mask_token=None)
seq_id2char = tf.keras.layers.StringLookup(vocabulary = seq_char2id.get_vocabulary(), invert=True, mask_token=None)

pkl_direc = "/n/eddy_lab/users/rangerkuang/CC_data/unique_pkl_data/"
def get_files(ds_type, weight): 
    with open(os.path.join(pkl_direc, f"x_{ds_type}.pkl"), "rb") as f:
        object1 = dill.load(f)
    with open(os.path.join(pkl_direc, f"y_{ds_type}.pkl"), "rb") as f:
        object2 = dill.load(f)
    with open(os.path.join(pkl_direc, f"y_{ds_type}_binary.pkl"), "rb") as f:
        object3 = dill.load(f)
    with open(os.path.join(pkl_direc, f"weights_{ds_type}_binary_{weight}.pkl"), "rb") as f:
        object4 = dill.load(f)
    return object1, object2, object3, object4

def get_dataset_binary(file_type, weight):
    x, y, yb, wb = get_files(file_type, weight)
    yb = tf.expand_dims(yb,axis=2) # necessary to match with NN's output

    shuffler = np.arange(0, x.nrows(), dtype=int)
    np.random.shuffle(shuffler)

    def generator():
        for indx in range(x.nrows()):
            i = shuffler[indx]
            yield x[i,:], yb[i,:], wb[i,:]

    dataset_b = tf.data.Dataset.from_generator(
        generator, 
        # (tf.int64, tf.int64, tf.float32), 
        #output_shapes=([None], [None,None], [None]), 
        output_signature=(
            tf.TensorSpec(shape=(None,), dtype=tf.int64, name='sequences'), 
            tf.TensorSpec(shape=(None, None), dtype=tf.int64, name='targets'), 
            tf.TensorSpec(shape=(None,), dtype=tf.float32, name='weights')
            ),
        )
    

    dataset_bucketed_b = dataset_b.bucket_by_sequence_length(
        element_length_func=lambda elem, foo, foo1: tf.shape(elem)[0],
        bucket_boundaries=[100, 400, 1000],
        pad_to_bucket_boundary=False,
        bucket_batch_sizes=[128, 128, 128, 64],
        # padding_values=tf.cast(69, tf.int64),
        drop_remainder=True,
        )
    
    return dataset_bucketed_b

# dataset_bucketed_teb = get_dataset_binary("test")
# dataset_bucketed_vb = get_dataset_binary("valid")

###########################
#####SPECIFIC DATA#########
###########################

def prep_file_binary(file_loc):
    alphabet = {'-': 0, 'a': 1, 'b': 1, 'c': 1, 'd': 1, 'e': 1, 'f': 1, 'g': 1}

    amino_list = 'ARNDCQEGHILKMFPSTWYVOUZXB'
    vocab = sorted(set(amino_list))
    #print(f'{len(vocab)} unique chars')
    seq_char2id = tf.keras.layers.StringLookup(vocabulary = list(vocab), mask_token=None)

    with open(file_loc, 'r') as f:
        contents = f.readlines()
        x = np.array(contents[0][:-1])
        inp = tf.strings.unicode_split(x, input_encoding='UTF-8')
        x_ids = seq_char2id(inp)
        x_ids = x_ids.numpy()
        x_ids = np.expand_dims(x_ids, axis=0)

        y = np.array([alphabet[ch] for ch in contents[2][:-2]])
        y = np.expand_dims(y, axis=1)
        y = np.expand_dims(y, axis=0)

        return x_ids, y



###########################
#####MODEL BUILDING########
###########################
CLASS_WEIGHT = {0: 0.505, 1: 50.38} # generated using sklearn on Colab

METRICS = [
    keras.metrics.TruePositives(name='tp'),
    keras.metrics.FalsePositives(name='fp'),
    keras.metrics.TrueNegatives(name='tn'),
    keras.metrics.FalseNegatives(name='fn'), 
    keras.metrics.BinaryAccuracy(name='accuracy'),
    keras.metrics.Precision(name='precision'),
    keras.metrics.Recall(name='recall'),
    keras.metrics.AUC(name='auc'),
    keras.metrics.AUC(name='prc', curve='PR'), # precision-recall curve
]



# model = keras.Sequential()
# #model.add(layers.Input(shape=[None], dtype=tf.int64, ragged=True))
# model.add(layers.Embedding(input_dim=26, output_dim=32, mask_zero=True)) # 0 = oov token I believe, why 26 is needed
# model.add(layers.Bidirectional(layers.LSTM(64, return_sequences=True)))
# model.add(layers.Dense(1, activation='sigmoid'))

# model = keras.Sequential()
# model.add(layers.Embedding(input_dim=26, output_dim=48, mask_zero=True)) # 0 = oov token I believe, why 26 is needed
# model.add(layers.Bidirectional(layers.LSTM(96, return_sequences=True))) 
# model.add(layers.Bidirectional(layers.LSTM(32, return_sequences=True))) 
# model.add(layers.Dense(1, activation='sigmoid'))

model = keras.Sequential()
model.add(layers.Embedding(input_dim=26, output_dim=48, mask_zero=True)) # 0 = oov token I believe, why 26 is needed
model.add(layers.Bidirectional(layers.LSTM(96, return_sequences=True))) 
model.add(layers.Bidirectional(layers.LSTM(32, return_sequences=True))) 
model.add(layers.Dropout(0.25))
model.add(layers.Dense(1, activation='sigmoid'))

print(model.summary())

model.compile(
    loss=keras.losses.BinaryCrossentropy(),
    # loss = lambda yt, yp: tf.losses.BinaryCrossentropy()(yt. values, yp.values), 
    optimizer="adam",
    metrics= METRICS,
#    sample_weight_mode = "temporal",
    weighted_metrics=["accuracy"],
)


###########################
#####MODEL EVALUATION######
###########################

date = "080522_1050" # 072922_1500 = Lstm2
save_path = f"/n/eddy_lab/users/rangerkuang/CC_project/uq_checkpoints/ckp_{date}/architecture"
checkpoint_path = f"/n/eddy_lab/users/rangerkuang/CC_project/uq_checkpoints/ckp_{date}/model_{date}.ckpt"

# model = keras.models.load_model(save_path)
# print(model.summary())

model.load_weights(checkpoint_path)


FILE = "has_cc_real/to400/test/2d3e_B"
x_ids, y = prep_file_binary(f"/n/eddy_lab/users/rangerkuang/CC_data/{FILE}")


#has_cc_real/to_max/test/5tby_B , 
# x_ids = np.array([[12, 7, 4, 18, 5, 12, 1, 21, 6, 7, 1, 1, 1, 15, 24, 11, 17, 10, 18, 5, 10, 5, 17, 11, 5, 1, 16, 19, 17, 15, 6, 4, 11, 10, 10, 4, 21, 6, 21, 15, 4, 4, 10, 16, 5, 6, 21, 10, 1, 10, 9, 21, 18, 17, 5, 7, 7, 10, 21, 19, 1, 5, 19, 5, 24, 7, 10, 19, 21, 19, 21, 10, 5, 4, 16, 21, 12, 16, 16, 13, 15, 15, 10, 6, 4, 10, 9, 5, 4, 12, 1, 12, 11, 19, 6, 11, 8, 5, 15, 1, 21, 11, 24, 13, 11, 10, 4, 17, 24, 7, 18, 22, 12, 9, 24, 19, 24, 18, 7, 11, 6, 3, 21, 19, 21, 13, 15, 24, 10, 22, 11, 15, 21, 24, 19, 15, 5, 21, 21, 1, 1, 24, 17, 7, 10, 10, 17, 18, 5, 1, 15, 15, 8, 9, 6, 18, 9, 18, 4, 13, 1, 24, 16, 24, 12, 11, 19, 4, 17, 5, 13, 16, 18, 9, 11, 9, 19, 7, 5, 18, 7, 1, 7, 10, 19, 21, 13, 19, 10, 17, 21, 9, 16, 24, 6, 1, 21, 9, 1, 1, 9, 7, 4, 17, 18, 10, 10, 4, 16, 18, 15, 7, 10, 7, 19, 11, 5, 4, 16, 9, 9, 16, 1, 13, 15, 1, 11, 5, 1, 6, 7, 13, 1, 10, 19, 21, 17, 13, 4, 13, 18, 18, 17, 6, 7, 10, 6, 9, 17, 9, 8, 6, 7, 1, 19, 7, 10, 11, 1, 18, 1, 4, 9, 5, 19, 24, 11, 11, 5, 10, 18, 17, 21, 9, 6, 16, 11, 10, 1, 5, 17, 4, 24, 8, 9, 6, 24, 16, 9, 11, 18, 13, 10, 10, 15, 5, 11, 11, 4, 12, 11, 11, 9, 19, 13, 13, 15, 24, 4, 24, 1, 6, 9, 18, 16, 7, 5, 19, 19, 21, 1, 18, 9, 4, 4, 1, 5, 5, 11, 12, 1, 19, 4, 13, 1, 6, 4, 21, 11, 7, 6, 19, 18, 5, 5, 10, 13, 18, 12, 24, 10, 11, 19, 7, 1, 9, 12, 8, 6, 7, 13, 12, 10, 6, 10, 11, 10, 16, 17, 5, 5, 16, 1, 5, 15, 4, 7, 19, 5, 5, 1, 4, 10, 18, 1, 24, 11, 12, 7, 11, 13, 18, 1, 4, 11, 11, 10, 7, 11, 3, 8, 15, 17, 21, 10, 21, 7, 13, 5, 24, 21, 19, 10, 7, 16, 13, 21, 16, 16, 21, 9, 24, 1, 19, 7, 1, 11, 1, 10, 1, 21, 24, 5, 17, 12, 6, 13, 22, 12, 21, 19, 17, 9, 13, 1, 19, 11, 5, 19, 10, 16, 15, 17, 16, 24, 6, 9, 7, 21, 11, 4, 9, 1, 7, 6, 5, 9, 6, 4, 6, 13, 18, 6, 5, 16, 11, 3, 9, 13, 6, 19, 13, 5, 10, 11, 16, 16, 6, 6, 13, 8, 8, 12, 6, 21, 11, 5, 16, 5, 5, 24, 10, 10, 5, 7, 9, 5, 22, 19, 6, 9, 4, 6, 7, 12, 4, 11, 16, 1, 3, 9, 4, 11, 9, 5, 10, 15, 12, 7, 9, 12, 18, 9, 11, 5, 5, 5, 3, 12, 6, 15, 10, 1, 19, 4, 12, 19, 6, 10, 1, 10, 11, 6, 4, 13, 8, 11, 7, 10, 18, 1, 13, 6, 16, 10, 15, 17, 13, 9, 10, 7, 10, 15, 5, 1, 8, 6, 18, 11, 9, 8, 24, 1, 7, 9, 21, 4, 24, 13, 9, 9, 7, 22, 11, 16, 10, 13, 10, 4, 15, 11, 13, 5, 19, 21, 21, 7, 11, 24, 16, 10, 18, 18, 11, 10, 11, 11, 18, 19, 11, 6, 1, 13, 24, 1, 7, 1, 4, 1, 15, 9, 5, 10, 7, 10, 7, 10, 1, 10, 10, 7, 18, 18, 6, 16, 19, 21, 18, 1, 11, 8, 17, 5, 13, 11, 13, 10, 11, 12, 19, 13, 11, 17, 18, 19, 8, 15, 8, 6, 21, 17, 3, 9, 9, 15, 13, 5, 19, 10, 18, 15, 7, 21, 12, 4, 13, 15, 11, 21, 12, 8, 16, 11, 17, 3, 13, 7, 21, 11, 5, 7, 9, 17, 9, 3, 17, 10, 7, 6, 15, 13, 17, 9, 11, 24, 7, 4, 6, 17, 16, 17, 24, 17, 9, 11, 13, 15, 1, 1, 9, 15, 5, 7, 16, 6, 9, 4, 18, 17, 10, 7, 1, 5, 10, 11, 11, 18, 18, 11, 4, 9, 4, 8, 13, 16, 24, 10, 6, 7, 8, 19, 10, 21, 6, 6, 10, 1, 7, 11, 11, 7, 11, 11, 5, 5, 12, 17, 4, 5, 17, 11, 18, 17, 9, 9, 19, 17, 9, 16, 1, 16, 18, 17, 7, 21, 11, 1, 17, 12, 5, 24, 10, 10, 11, 11, 5, 17, 17, 4, 18, 11, 11, 21, 9, 16, 22, 13, 9, 17, 1, 6, 12, 7, 21, 10, 13, 22, 15, 22, 12, 10, 11, 24, 6, 10, 9, 10, 15, 11, 11, 10, 18, 1, 5, 17, 5, 10, 5, 12, 1, 18, 12, 10, 5, 5, 6, 19, 17, 11, 10, 5, 1, 11, 5, 10, 18, 5, 1, 17, 17, 10, 5, 11, 5, 5, 10, 12, 21, 18, 11, 11, 16, 5, 10, 13, 4, 11, 16, 11, 16, 21, 16, 1, 5, 16, 4, 13, 11, 1, 4, 1, 5, 5, 17, 3, 4, 16, 11, 9, 10, 13, 10, 9, 16, 11, 5, 1, 10, 21, 10, 5, 12, 13, 5, 17, 11, 5, 4, 5, 5, 5, 12, 13, 1, 5, 11, 19, 1, 10, 10, 17, 10, 11, 5, 4, 5, 3, 18, 5, 11, 10, 17, 4, 9, 4, 4, 11, 5, 11, 19, 11, 1, 10, 21, 5, 10, 5, 10, 8, 1, 19, 5, 13, 10, 21, 10, 13, 11, 19, 5, 5, 12, 1, 7, 11, 4, 5, 9, 9, 1, 10, 11, 19, 10, 5, 10, 10, 1, 11, 16, 5, 1, 8, 16, 16, 1, 11, 4, 4, 11, 16, 1, 5, 5, 4, 10, 21, 13, 19, 11, 19, 10, 1, 10, 21, 10, 11, 5, 16, 16, 21, 4, 4, 11, 5, 7, 18, 11, 5, 16, 5, 10, 10, 21, 17, 12, 4, 11, 5, 17, 1, 10, 17, 10, 11, 5, 7, 4, 11, 10, 11, 19, 16, 5, 18, 9, 12, 4, 11, 5, 13, 4, 10, 16, 16, 11, 4, 5, 17, 11, 10, 10, 10, 4, 6, 5, 11, 13, 1, 11, 13, 1, 17, 9, 5, 4, 5, 16, 1, 11, 7, 18, 16, 11, 16, 10, 10, 11, 10, 5, 11, 16, 1, 17, 9, 5, 5, 11, 5, 5, 5, 11, 5, 1, 5, 17, 19, 1, 17, 1, 10, 21, 5, 10, 11, 17, 18, 4, 11, 18, 17, 5, 11, 5, 5, 9, 18, 5, 17, 11, 5, 5, 1, 7, 7, 1, 19, 18, 21, 16, 9, 5, 12, 13, 10, 10, 17, 5, 1, 5, 6, 16, 10, 12, 17, 17, 4, 11, 5, 5, 1, 19, 11, 16, 8, 5, 1, 19, 1, 1, 1, 11, 17, 10, 10, 8, 1, 4, 18, 21, 1, 5, 11, 7, 5, 16, 9, 4, 13, 11, 16, 17, 21, 10, 16, 10, 11, 5, 10, 5, 10, 18, 5, 6, 10, 11, 5, 11, 4, 4, 21, 19, 18, 13, 12, 5, 16, 9, 9, 10, 1, 10, 1, 13, 11, 5, 10, 12, 3, 17, 19, 11, 5, 4, 16, 12, 13, 5, 8, 17, 18, 10, 1, 5, 5, 19, 16, 17, 18, 21, 13, 4, 11, 19, 18, 16, 17, 1, 10, 11, 16, 19, 5, 13, 7, 5, 11, 18, 17, 16, 11, 4, 5, 10, 5, 1, 11, 9, 18, 16, 11, 19, 17, 7, 10, 11, 19, 24, 19, 16, 16, 11, 5, 4, 11, 10, 17, 16, 11, 5, 5, 5, 21, 10, 1, 10, 13, 1, 11, 1, 8, 1, 11, 16, 18, 1, 17, 8, 4, 3, 4, 11, 11, 17, 5, 16, 24, 5, 5, 5, 19, 5, 1, 10, 1, 5, 11, 16, 17, 21, 11, 18, 10, 1, 13, 18, 5, 21, 1, 16, 22, 17, 19, 10, 24, 5, 19, 4, 1, 9, 16, 17, 19, 5, 5, 11, 5, 5, 1, 10, 10, 10, 11, 1, 16, 17, 11, 16, 5, 1, 5, 5, 1, 21, 5, 1, 21, 13, 1, 10, 3, 18, 18, 11, 5, 10, 19, 10, 8, 17, 11, 16, 13, 5, 9, 5, 4, 11, 12, 21, 4, 21, 5, 17, 18, 13, 1, 1, 1, 1, 1, 11, 4, 10, 10, 16, 17, 13, 6, 4, 10, 9, 11, 1, 5, 22, 10, 16, 10, 24, 5, 5, 18, 16, 18, 5, 11, 5, 18, 18, 16, 10, 5, 1, 17, 18, 11, 18, 19, 5, 11, 6, 10, 11, 10, 13, 1, 24, 5, 5, 18, 11, 5, 8, 11, 5, 19, 6, 10, 17, 5, 13, 10, 13, 11, 16, 5, 5, 9, 18, 4, 11, 19, 5, 16, 11, 7, 18, 18, 7, 10, 19, 9, 8, 5, 11, 5, 10, 21, 17, 10, 16, 11, 5, 1, 5, 10, 12, 5, 11, 16, 18, 1, 11, 5, 5, 1, 5, 1, 18, 11, 5, 8, 5, 5, 7, 10, 9, 11, 17, 1, 16, 11, 5, 6, 13, 16, 9, 10, 1, 5, 9, 5, 17, 10, 11, 1, 5, 10, 4, 5, 5, 12, 5, 16, 1, 10, 17, 13, 8, 11, 17, 21, 21, 4, 18, 11, 16, 19, 18, 11, 4, 1, 5, 19, 17, 18, 17, 13, 5, 1, 11, 17, 21, 10, 10, 10, 12, 5, 7, 4, 11, 13, 5, 12, 5, 9, 16, 11, 18, 8, 1, 13, 17, 12, 1, 1, 5, 1, 16, 10, 16, 21, 10, 18, 11, 16, 18, 11, 11, 10, 4, 19, 16, 9, 16, 11, 4, 4, 1, 21, 17, 1, 13, 4, 4, 11, 10, 5, 13, 9, 1, 9, 21, 5, 17, 17, 13, 13, 11, 11, 16, 1, 5, 11, 5, 5, 11, 17, 1, 21, 21, 5, 16, 19, 5, 17, 18, 17, 10, 11, 1, 5, 16, 5, 11, 9, 5, 19, 18, 5, 17, 21, 16, 11, 11, 8, 18, 16, 13, 19, 18, 11, 9, 13, 16, 10, 10, 10, 12, 4, 1, 4, 11, 18, 16, 11, 16, 19, 5, 21, 5, 5, 1, 21, 16, 5, 3, 17, 13, 1, 5, 5, 10, 1, 10, 10, 1, 9, 19, 4, 1, 1, 12, 12, 1, 5, 5, 11, 10, 10, 5, 16, 4, 19, 18, 1, 8, 11, 5, 17, 12, 10, 10, 13, 12, 5, 16, 19, 9, 10, 4, 11, 16, 8, 17, 11, 4, 5, 1, 5, 16, 9, 1, 11, 10, 7, 7, 10, 10, 16, 11, 16, 10, 11, 5, 1, 17, 21, 17, 5, 11, 5, 13, 5, 11, 5, 1, 5, 16, 10, 17, 13, 1, 5, 18, 21, 10, 7, 12, 17, 10, 18, 5, 17, 17, 9, 10, 5, 11, 19, 24, 16, 19, 5, 5, 4, 17, 10, 13, 11, 11, 17, 11, 16, 4, 11, 21, 4, 10, 11, 16, 11, 10, 21, 10, 1, 24, 10, 17, 16, 1, 5, 5, 1, 5, 5, 16, 1, 13, 19, 13, 11, 18, 10, 6, 17, 10, 21, 16, 8, 5, 11, 4, 5, 1, 5, 5, 17, 1, 4, 9, 1, 5, 18, 16, 21, 13, 10, 11, 17, 1, 10, 18, 17, 4, 9, 7, 19, 10, 7, 11, 13, 5, 5]])


# no_cc_real/to400/test/6irb_B . where Sockets got it wrong
# x_ids = np.array([[5, 15, 15, 11, 21, 6, 5, 15, 21, 19, 11, 5, 18, 11, 17, 16, 5, 10, 7, 6, 16, 5, 21, 7, 10, 10, 16, 9, 10, 5, 11, 4, 19, 11, 17, 5, 10, 8, 1, 10, 5, 17, 19, 18, 21, 16, 10, 19, 16, 13, 1, 1, 9, 4, 10, 11, 9, 10, 7, 10, 18, 10, 4, 4, 9, 17, 13, 4, 1, 13, 9, 10, 13, 18, 9, 13, 4, 16, 19, 10, 16, 22, 19, 4, 12, 9, 1, 17, 8, 17, 10, 5, 5, 22, 4, 12, 11, 17, 16, 8, 21, 16, 4, 18, 16, 4, 1, 12, 10, 1, 11, 12, 11, 19, 21, 16, 1, 1, 16, 9, 10, 16, 11, 5, 4, 17, 8, 1, 17, 4, 9, 10, 4, 11, 13, 1, 10, 16, 1, 10, 12, 18, 1, 4, 19, 1, 10, 5, 21, 16, 13, 4, 19, 11, 10, 19, 10, 13, 5, 10, 4, 17, 17, 11, 17, 5, 10, 17, 16, 13, 13, 21, 10, 17, 6, 12, 5, 5, 10, 10, 16, 9, 7, 21, 10, 16, 7, 17, 1, 12, 5, 10, 11, 10, 11, 1, 8, 18, 10, 16, 9, 5, 5, 6, 18, 19, 4, 21, 16, 10, 11]])

# no_cc_real/to400/train/6O35_A . where Sockets got it wrong
# x_ids = np.array([[7, 18, 18, 1, 5, 5, 11, 11, 17, 17, 18, 17, 5, 24, 11, 10, 10, 21, 10, 5, 5, 16, 5, 17, 10, 1, 10, 5, 6, 16, 5, 11, 11, 10, 5, 11, 18, 5, 17, 18, 5, 5, 11, 9, 17, 5, 11, 5, 5, 10, 7, 1, 1, 18, 5, 1, 5, 11, 1, 17, 12, 10, 16, 16, 8, 12, 19, 1, 24, 11, 5, 1, 16, 11, 19, 1, 22, 5, 9, 5, 18, 10, 18, 10, 9, 1, 11, 11, 5, 11, 16, 16, 13, 16, 11, 13, 11, 5, 11, 17, 8, 9]])

# to400/train/5jlh_H . Just a bunch of X's 
# x_ids = np.array([[23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23]])


# sas_6 protein - a good long-CC protein prediction
# sring = list('MSQVLFQQLVPLLVKCKDCEERRGSVRVSIELQSLSNPVHRKDLVIRLTDDTDPFFLYNLVISEEDFQSLKLQQGLLVDFLAFPQKFIDLLQQCMQEHAKETPRFLLQLLSSATLLENSPVLLNVVETNPFKHLIHLSLKLLPGNDVEIKKFLAGCLKCSKEEKLSLTRSLDDVTRQLHITQETLSEKMQELDKLRSEWASHTASLTNKHSQELTAEKEKALQTQVQCQQQHEQQKKELETLHQRNIHQLQSRLSELEAANKELTERKYKGDSTVRELKAKLAGVEEELQRAKQEVLSLRRENCTLDTECHEKEKHINQLQTKVAVLEQEIKDKDQLVLRTKEAFDTIQEQKVALEENGEKNQIQLGKLEATIKSLSAELLKANEIIKKLQGDLKTLMGKLKLKNTVTIQQEKLLAEKEEMLQKERKESQDAGQFLRAKEQEVCRLQEQLETTVQKLEESKQLLKNNEKLITWLNKELNENQLVRKQDTLGTSATPHSTSNSTIRSGLSPNLNVVDRLNYPSCGIGYPVSSALTFQNAFPHVVAAKNTSHPISGPKVHFNLQLTKPSASIDGQPGAAVNRPCSNDKENGETLGLESKYLKRREASIPLRGLSQNLLSDSDHQKDGMLGAFQLSSKPTVLPSSSSAYFPGQLPSS')
# x_ids = seq_char2id(sring)
# x_ids = np.expand_dims(x_ids, axis=0)
# y = np.zeros(654)
# y[175:472] = 1

# no_cc_real/to_max/valid/7ogt_A protein, aka. SMC1 protein, aka the misclassification protein  
# y = np.zeros(1225)
# y[173:490] = 1
# y[679:1064] = 1


# giantin protein 
# sring = list('MLSRLSGLANVVLHELSGDDDTDQNMRAPLDPELHQESDMEFNNTTQEDVQERLAYAEQLVVELKDIIRQKDVQLQQKDEALQEERKAADNKIKKLKLHAKAKLTSLNKYIEEMKAQGGTVLPTEPQSEEQLSKHDKSSTEEEMEIEKIKHKLQEKEELISTLQAQLTQAQAEQPAQSSTEMEEFVMMKQQLQEKEEFISTLQAQLSQTQAEQAAQQVVREKDARFETQVRLHEDELLQLVTQADVETEMQQKLRVLQRKLEEHEESLVGRAQVVDLLQQELTAAEQRNQILSQQLQQMEAEHNTLRNTVETEREESKILLEKMELEVAERKLSFHNLQEEMHHLLEQFEQAGQAQAELESRYSALEQKHKAEMEEKTSHILSLQKTGQELQSACDALKDQNSKLLQDKNEQAVQSAQTIQQLEDQLQQKSKEISQFLNRLPLQQHETASQTSFPDVYNEGTQAVTEENIASLQKRVVELENEKGALLLSSIELEELKAENEKLSSQITLLEAQNRTGEADREVSEISIVDIANKRSSSAEESGQDVLENTFSQKHKELSVLLLEMKEAQEEIAFLKLQLQGKRAEEADHEVLDQKEMKQMEGEGIAPIKMKVFLEDTGQDFPLMPNEESSLPAVEKEQASTEHQSRTSEEISLNDAGVELKSTKQDGDKSLSAVPDIGQCHQDELERLKSQILELELNFHKAQEIYEKNLDEKAKEISNLNQLIEEFKKNADNNSSAFTALSEERDQLLSQVKELSMVTELRAQVKQLEMNLAEAERQRRLDYESQTAHDNLLTEQIHSLSIEAKSKDVKIEVLQNELDDVQLQFSEQSTLIRSLQSQLQNKESEVLEGAERVRHISSKVEELSQALSQKELEITKMDQLLLEKKRDVETLQQTIEEKDQQVTEISFSMTEKMVQLNEEKFSLGVEIKTLKEQLNLLSRAEEAKKEQVEEDNEVSSGLKQNYDEMSPAGQISKEELQHEFDLLKKENEQRKRKLQAALINRKELLQRVSRLEEELANLKDESKKEIPLSETERGEVEEDKENKEYSEKCVTSKCQEIEIYLKQTISEKEVELQHIRKDLEEKLAAEEQFQALVKQMNQTLQDKTNQIDLLQAEISENQAIIQKLITSNTDASDGDSVALVKETVVISPPCTGSSEHWKPELEEKILALEKEKEQLQKKLQEALTSRKAILKKAQEKERHLREELKQQKDDYNRLQEQFDEQSKENENIGDQLRQLQIQVRESIDGKLPSTDQQESCSSTPGLEEPLFKATEQHHTQPVLESNLCPDWPSHSEDASALQGGTSVAQIKAQLKEIEAEKVELELKVSSTTSELTKKSEEVFQLQEQINKQGLEIESLKTVSHEAEVHAESLQQKLESSQLQIAGLEHLRELQPKLDELQKLISKKEEDVSYLSGQLSEKEAALTKIQTEIIEQEDLIKALHTQLEMQAKEHDERIKQLQVELCEMKQKPEEIGEESRAKQQIQRKLQAALISRKEALKENKSLQEELSLARGTIERLTKSLADVESQVSAQNKEKDTVLGRLALLQEERDKLITEMDRSLLENQSLSSSCESLKLALEGLTEDKEKLVKEIESLKSSKIAESTEWQEKHKELQKEYEILLQSYENVSNEAERIQHVVEAVRQEKQELYGKLRSTEANKKETEKQLQEAEQEMEEMKEKMRKFAKSKQQKILELEEENDRLRAEVHPAGDTAKECMETLLSSNASMKEELERVKMEYETLSKKFQSLMSEKDSLSEEVQDLKHQIEGNVSKQANLEATEKHDNQTNVTEEGTQSIPGETEEQDSLSMSTRPTCSESVPSAKSANPAVSKDFSSHDEINNYLQQIDQLKERIAGLEEEKQKNKEFSQTLENEKNTLLSQISTKDGELKMLQEEVTKMNLLNQQIQEELSRVTKLKETAEEEKDDLEERLMNQLAELNGSIGNYCQDVTDAQIKNELLESEMKNLKKCVSELEEEKQQLVKEKTKVESEIRKEYLEKIQGAQKEPGNKSHAKELQELLKEKQQEVKQLQKDCIRYQEKISALERTVKALEFVQTESQKDLEITKENLAQAVEHRKKAQAELASFKVLLDDTQSEAARVLADNLKLKKELQSNKESVKSQMKQKDEDLERRLEQAEEKHLKEKKNMQEKLDALRREKVHLEETIGEIQVTLNKKDKEVQQLQENLDSTVTQLAAFTKSMSSLQDDRDRVIDEAKKWERKFSDAIQSKEEEIRLKEDNCSVLKDQLRQMSIHMEELKINISRLEHDKQIWESKAQTEVQLQQKVCDTLQGENKELLSQLEETRHLYHSSQNELAKLESELKSLKDQLTDLSNSLEKCKEQKGNLEGIIRQQEADIQNSKFSYEQLETDLQASRELTSRLHEEINMKEQKIISLLSGKEEAIQVAIAELRQQHDKEIKELENLLSQEEEENIVLEEENKKAVDKTNQLMETLKTIKKENIQQKAQLDSFVKSMSSLQNDRDRIVGDYQQLEERHLSIILEKDQLIQEAAAENNKLKEEIRGLRSHMDDLNSENAKLDAELIQYREDLNQVITIKDSQQKQLLEVQLQQNKELENKYAKLEEKLKESEEANEDLRRSFNALQEEKQDLSKEIESLKVSISQLTRQVTALQEEGTLGLYHAQLKVKEEEVHRLSALFSSSQKRIAELEEELVCVQKEAAKKVGEIEDKLKKELKHLHHDAGIMRNETETAEERVAELARDLVEMEQKLLMVTKENKGLTAQIQSFGRSMSSLQNSRDHANEELDELKRKYDASLKELAQLKEQGLLNRERDALLSETAFSMNSTEENSLSHLEKLNQQLLSKDEQLLHLSSQLEDSYNQVQSFSKAMASLQNERDHLWNELEKFRKSEEGKQRSAAQPSTSPAEVQSLKKAMSSLQNDRDRLLKELKNLQQQYLQINQEITELHPLKAQLQEYQDKTKAFQIMQEELRQENLSWQHELHQLRMEKSSWEIHERRMKEQYLMAISDKDQQLSHLQNLIRELRSSSSQTQPLKVQYQRQASPETSASPDGSQNLVYETELLRTQLNDSLKEIHQKELRIQQLNSNFSQLLEEKNTLSIQLCDTSQSLRENQQHYGDLLNHCAVLEKQVQELQAGPLNIDVAPGAPQEKNGVHRKSDPEELREPQQSFSEAQQQLCNTRQEVNELRKLLEEERDQRVAAENALSVAEEQIRRLEHSEWDSSRTPIIGSCGTQEQALLIDLTSNSCRRTRSGVGWKRVLRSLCHSRTRVPLLAAIYFLMIHVLLILCFTGHL')
# x_ids = seq_char2id(sring)
# x_ids = np.expand_dims(x_ids, axis=0)
# y = np.zeros(3259)
# y[48:594] = 1
# y[677:1029] = 1
# y[1062:1245] = 1
# y[1828:3185] = 1

# has_cc_real/to400/test/2d3e_B is the tropomyosin

print(tf.strings.reduce_join(seq_id2char(x_ids), axis=-1).numpy())
print(x_ids)

result = model.predict(x_ids)


scale = np.arange(len(np.squeeze(result)))
threshold = np.full(len(np.squeeze(result)), 0.5)

plt.figure(figsize=(15, 6), dpi=200)
plt.plot(scale, np.squeeze(y), label = "True CC")
plt.plot(scale, np.squeeze(result), label = "predicted") # squeeze done to convert from 2D matrix to 1D I believe
plt.plot(scale, threshold, label = "threshold",  linestyle='dashed')
plt.legend()
plt.ylim(-0.01,1.01)
plt.xlabel('Residue Position')
plt.ylabel('Confidence of CC')
plt.title(FILE)
# plt.title("Gog-B1 \"Giantin\"")
plt.savefig('visual.png')


