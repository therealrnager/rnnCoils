import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.layers import Layer
import keras.backend as K
import os
import dill

###########################
#####LOAD DATA#############
###########################
pkl_direc = "/n/eddy_lab/users/rangerkuang/CC_data/pkl_data/"
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

def get_dataset(file_type, weight):
    x, y, yb, wb = get_files(file_type, weight)
    y = tf.expand_dims(y,axis=2) # necessary to match with NN's output

    shuffler = np.arange(0, x.nrows(), dtype=int)
    np.random.shuffle(shuffler)

    def generator():
        for indx in range(x.nrows()):
            i = shuffler[indx]
            yield x[i,:], y[i,:], wb[i,:]

    dataset = tf.data.Dataset.from_generator(
        generator, 
        output_signature=(
            tf.TensorSpec(shape=(None,), dtype=tf.int64, name='sequences'), 
            tf.TensorSpec(shape=(None, None), dtype=tf.int64, name='targets'), 
            tf.TensorSpec(shape=(None,), dtype=tf.float32, name='weights')
            ),
        )
        

    dataset_bucketed = dataset.bucket_by_sequence_length(
        element_length_func=lambda elem, foo, foo1: tf.shape(elem)[0],
        bucket_boundaries=[100, 400, 1000],
        pad_to_bucket_boundary=False,
        bucket_batch_sizes=[128, 128, 128, 64],
        # padding_values=tf.cast(69, tf.int64),
        drop_remainder=True,
        )
    
    return dataset_bucketed

dataset_bucketed_tr = get_dataset("train", 15)
dataset_bucketed_v = get_dataset("valid", 15)


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



# Add attention layer to the deep learning network
class Attention(Layer):
    def __init__(self,**kwargs):
        super(Attention,self).__init__(**kwargs)
 
    def build(self,input_shape):
        self.W=self.add_weight(name='attention_weight', shape=(input_shape[-1],1), 
                               initializer='random_normal', trainable=True)
        self.b=self.add_weight(name='attention_bias', shape=(input_shape[1],1), 
                               initializer='zeros', trainable=True)        
        super(Attention, self).build(input_shape)
 
    def call(self,x):
        # Alignment scores. Pass them through tanh function
        e = K.tanh(K.dot(x,self.W)+self.b)
        # Remove dimension of size 1
        e = K.squeeze(e, axis=-1)   
        # Compute the weights
        alpha = K.softmax(e)
        # Reshape to tensorFlow format
        alpha = K.expand_dims(alpha, axis=-1)
        # Compute the context vector
        context = x * alpha
        context = K.sum(context, axis=1)
        return context


def model_constructor():
    inp = layers.Input(shape=[None], dtype=tf.int64)
    embedding = layers.Embedding(input_dim=26, output_dim=48, mask_zero=True)(inp) # output dim 
    lstm = layers.Bidirectional(layers.LSTM(96, return_sequences=True))(embedding) # lstm layer 
    attention = Attention()(lstm)
    output = layers.Dense(1, trainable=True, activation='sigmoid')(attention)
    
    model = Model(inp, output)
    print(model.summary())

    model.compile(
        loss=keras.losses.BinaryCrossentropy(),
        # loss = lambda yt, yp: tf.losses.BinaryCrossentropy()(yt. values, yp.values), 
        optimizer="adam",
        metrics= METRICS,
        weighted_metrics=["accuracy"],
    )

    #RNN_layer = SimpleRNN(hidden_units, return_sequences=True, activation=activation)(x)
    #attention_layer = attention()(RNN_layer)
    #outputs=Dense(dense_units, trainable=True, activation=activation)(attention_layer)
    #model=Model(x,outputs)
    #model.compile(loss='mse', optimizer='adam')    
    return model  

# UPDATES FROM MK1: output_dim 32 -> 48, LSTM 64 -> 96, new LSTM 32 layer 
model = keras.Sequential()
model.add(layers.Embedding(input_dim=26, output_dim=48, mask_zero=True)) # 0 = oov token I believe, why 26 is needed
model.add(layers.Bidirectional(layers.LSTM(96, return_sequences=True))) 
model.add(layers.Bidirectional(layers.LSTM(32, return_sequences=True))) 
model.add(layers.Dense(8)) # 7 a-g, + padded 0 

print(model.summary())

model.compile(
        loss= keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        # loss = lambda yt, yp: tf.losses.BinaryCrossentropy()(yt. values, yp.values), 
        optimizer="adam",
        #metrics= METRICS,
        metrics= ["accuracy"],
        weighted_metrics=["accuracy"],
    )

#model = model_constructor()





# print(model.predict(foo_bucketed))
# print(model.evaluate(foo_bucketed))
###########################
#####MODEL TRAINING########
###########################

date = "073122_1500"
checkpoint_path = f"/n/eddy_lab/users/rangerkuang/CC_project/checkpoints/ckp_{date}/model_{date}.ckpt"

# ckp_callback = tf.keras.callbacks.ModelCheckpoint(
#     filepath=checkpoint_path,
#     save_weights_only=True,
#     monitor='val_prc',
#     mode='max',
#     save_best_only=True,
#     verbose=2
# )

# earlystopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_prc', patience=7)

history = model.fit(
    dataset_bucketed_tr.repeat(),
    validation_data=dataset_bucketed_v.repeat(), 
    #batch_size=2, 
    steps_per_epoch=10,  # about 3424 total if i did calculations correctly
    epochs=5, 
    validation_steps = 3, 
    #callbacks=[ckp_callback],
    verbose=1
)

# model.load_weights(checkpoint_path)

# save_path = f"/n/eddy_lab/users/rangerkuang/CC_project/checkpoints/ckp_{date}/architecture"

# model.save(save_path)

###########################
#####MODEL EVALUATION######
###########################
print("=====TEST RESULTS======")
# dataset_bucketed_teb = get_dataset_binary("test")

# results = model.evaluate(dataset_bucketed_teb, verbose=2)

# print("test loss, test acc: ", results)
