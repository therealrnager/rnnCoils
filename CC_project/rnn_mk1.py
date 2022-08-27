import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os
import dill


###########################
#####LOAD DATA#############
###########################
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
        bucket_batch_sizes=[64, 64, 64, 32],
        # padding_values=tf.cast(69, tf.int64),
        drop_remainder=False,
        )
    
    return dataset_bucketed_b

WEIGHT = 1
dataset_bucketed_tb = get_dataset_binary("train", WEIGHT)
dataset_bucketed_vb = get_dataset_binary("valid", WEIGHT)


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
    # tfa.metrics.F1Score(name='f1')
]


model = keras.Sequential()
#model.add(layers.Input(shape=[None], dtype=tf.int64, ragged=True))
model.add(layers.Embedding(input_dim=26, output_dim=32, mask_zero=True)) # 0 = oov token I believe, why 26 is needed
model.add(layers.Bidirectional(layers.LSTM(64, return_sequences=True)))
model.add(layers.Dense(1, activation='sigmoid'))
#model.add(layers.Reshape((-1,), input_shape=(None, None, 1))) # removes the extra dimensinos that's 1

print(model.summary())

# print(model(x_train[:1,:]))

model.compile(
    loss=keras.losses.BinaryCrossentropy(),
    # loss = lambda yt, yp: tf.losses.BinaryCrossentropy()(yt. values, yp.values), 
    optimizer="adam",
    metrics= METRICS,
#    sample_weight_mode = "temporal",
    weighted_metrics=["accuracy"],
)


# print(model.predict(foo_bucketed))
# print(model.evaluate(foo_bucketed))
###########################
#####MODEL TRAINING########
###########################

date = "080822_1800"
checkpoint_path = f"/n/eddy_lab/users/rangerkuang/CC_project/uq_checkpoints/ckp_{date}/model_{date}.ckpt"

ckp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    save_weights_only=True,
    monitor='val_prc',
    mode='max',
    save_best_only=True,
    verbose=2
)

earlystopping_callback = tf.keras.callbacks.EarlyStopping(
    monitor='val_prc', 
    patience=5,
    mode='max',
    verbose=1,
    )

history = model.fit(
    dataset_bucketed_tb.repeat(),
    validation_data=dataset_bucketed_vb.repeat(), 
    #batch_size=2, 
    steps_per_epoch=1760,  # about 1750-1760 total if i did calculations correctly
    epochs=30,
    validation_steps = 220,  # about 220 total
    callbacks=[ckp_callback, earlystopping_callback],
    verbose=2,
)

model.load_weights(checkpoint_path)

save_path = f"/n/eddy_lab/users/rangerkuang/CC_project/uq_checkpoints/ckp_{date}/architecture"

model.save(save_path)

###########################
#####MODEL EVALUATION######
###########################
print("=====TEST RESULTS======")

dataset_bucketed_teb = get_dataset_binary("test", WEIGHT)

results = model.evaluate(dataset_bucketed_teb, verbose=2)

# print("test loss, test acc: ", results)
