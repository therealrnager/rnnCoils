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
        bucket_batch_sizes=[64, 64, 64, 32],
        # padding_values=tf.cast(69, tf.int64),
        drop_remainder=False,
        )
    
    return dataset_bucketed_b

dataset_bucketed_teb = get_dataset_binary("test", 5)
dataset_bucketed_vb = get_dataset_binary("valid", 5)



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
####MODEL LOAD WEIGHTS#####
###########################

date = "080922_1600"
save_path = f"/n/eddy_lab/users/rangerkuang/CC_project/uq_checkpoints/ckp_{date}/architecture"
checkpoint_path = f"/n/eddy_lab/users/rangerkuang/CC_project/uq_checkpoints/ckp_{date}/model_{date}.ckpt"

# model = keras.models.load_model(save_path)
# print(model.summary())

model.load_weights(checkpoint_path)

#####################################
####Advanced Metrics Definitions#####
#####################################

def advanced_metrics(NUM_STEPS, HIST_LEN, file_loc):
    # NUM_STEPS = 2
    predictions = model.predict(
        dataset_bucketed_teb,
        verbose=1,
        steps=NUM_STEPS
    )
    predictions = tf.squeeze(predictions, axis = 2)


    simple_recall_num = 0 
    simple_recall_denom = 0
    simple_precision_num = 0 
    simple_precision_denom = 0


    # HIST_LEN = 30
    recall_frac = np.zeros((HIST_LEN, 2)) # each index encompasses a range of 10. i.e. first index holds ratio for CC's of length 0-10
                                        # columns are numerator denominator 
    precision_frac = np.zeros((HIST_LEN, 2)) # each index encompasses a range of 10. i.e. first index holds ratio for CC's of length 0-10
                                        # columns are numerator denominator 

    pred_iter = 0
    for foo in dataset_bucketed_teb.take(NUM_STEPS): # iterate through batches 
        sequences = foo[0].numpy()
        # print(np.shape(sequences))
        


        targets = foo[1].numpy()
        targets = np.squeeze(targets, axis=2)

        num_seqs = len(sequences) 


        for i in range(num_seqs): # iterate through each sequence in a batch
            seq = sequences[i]
            targ = targets[i]

            pred = predictions[pred_iter]
            pred = pred.numpy() # convert from eager tensor to numpy 


            pred_iter += 1


            

            

            ###########Obtain CC_lengths, an array that tracks the length of the CC region a residue is in.#########
            def get_cc_lengths(to_be_examined, is_pred):
                cc_lengths = np.zeros(len(seq))
                length = 0
                start = 0
                cc = False
                for r in range(len(seq)): # iterate through each residue in one sequence 
                    if (seq[r] == 0): # hit padded values
                        break
                    

                    if (to_be_examined[r] > 0.5): # this is a coiled coil. use 0.5 cuz even with target, 0 and 1 are on opposite sides of threshold
                        if (length == 0): # start of a coiled coil
                            start = r
                        
                        length += 1
                        cc = True
                    else: # this is NOT a coiled coil 
                        if (length > 0): # coiled coil just ended 
                            for j in range(start, r): 
                                cc_lengths[j] = length

                            # if (length > 120 and is_pred):
                            #     print("=======IS PRED==========")
                            #     print(recall_cc_lengths)
                            #     print(cc_lengths)
                            #     print(list(seq))
                            # if (length > 120 and not is_pred):
                            #     print("=======IS TARG==========")
                            #     print(cc_lengths)
                            #     print(pred)
                            #     print(list(seq))    
                        length = 0
                if (length > 0): # sequence ended with a coiled coil  (dont think this will ever happen)
                    for j in range(start, len(seq)): 
                            cc_lengths[j] = length
                cc_lengths = cc_lengths.astype('int64') # for some reason its not ints already 
                return cc_lengths
            
            recall_cc_lengths = get_cc_lengths(targ, False)
            prec_cc_lengths = get_cc_lengths(pred, True)

            num_rows, num_cols = np.shape(sequences)
            ##########################################################


            
            ###########calculate "advanced" recall###################
            cc_exists = False
            simple_predict = False

            for r in range(len(seq)): # iterate through each residue in one sequence 
                if (seq[r] == 0): # hit padded values
                    break
                if (targ[r] == 1): # there IS a coiled coil
                    cc_exists = True 
                    if (pred[r] > 0.5): # correct prediction! 
                        simple_predict = True

                        bucket_ind =  recall_cc_lengths[r] // 10
                        recall_frac[bucket_ind, 0] += 1 # num increment
                        recall_frac[bucket_ind, 1] += 1 # denom increment

                        #precision_frac[bucket_ind, 0] += 1 # num increment
                        #precision_frac[bucket_ind, 1] += 1 # denom increment

                    else: # false negative 
                        bucket_ind =  recall_cc_lengths[r] // 10
                        recall_frac[bucket_ind, 1] += 1 # denom increment

            
            if (cc_exists):
                simple_recall_denom += 1
            if (simple_predict):
                simple_recall_num += 1
            ##########################################################

            
            ###########calculate "advanced" precision###################
            cc_exists = False
            simple_predict = False

            for r in range(len(seq)): # iterate through each residue in one sequence 
                if (seq[r] == 0): # hit padded values
                    break
                if (pred[r] > 0.5): # predicted CC
                    cc_exists = True 
                    if (targ[r] == 1): # correct prediction! 
                        simple_predict = True

                        bucket_ind =  prec_cc_lengths[r] // 10

                        precision_frac[bucket_ind, 0] += 1 # num increment
                        precision_frac[bucket_ind, 1] += 1 # denom increment

                    else: # false positive 
                        bucket_ind =  prec_cc_lengths[r] // 10
                        precision_frac[bucket_ind, 1] += 1 # denom increment

            
            if (cc_exists):
                simple_precision_denom += 1
            if (simple_predict):
                simple_precision_num += 1
            ##########################################################


    def frac(num, denom):
        if (denom == 0): 
            return -0.02, False
        else:
            return num / denom, True

    sr_proportion, bleh = frac(simple_recall_num, simple_recall_denom)
    print("num sequences with 1+ true positive / num sequences with CC's")
    print(f"{simple_recall_num} / {simple_recall_denom} = {sr_proportion}")

    sp_proportion, bleh = frac(simple_precision_num, simple_precision_denom)
    print("num sequences with 1+ true positive / num sequences with predicted CC's")
    print(f"{simple_precision_num} / {simple_precision_denom} = {sp_proportion}")

    # actually calculate proportions for histogram
    recall_histogram = np.zeros(HIST_LEN)
    precision_histogram = np.zeros(HIST_LEN)
    last_nonzero = 0
    last_recall_nonzero = 0
    for i in range(HIST_LEN):
        recall_histogram[i], rec_nonzero = frac(recall_frac[i, 0], recall_frac[i, 1])
        precision_histogram[i], prec_nonzero = frac(precision_frac[i, 0], precision_frac[i, 1])
        if (rec_nonzero or prec_nonzero): # track last nonzero position 
            last_nonzero = i 
        if (rec_nonzero): 
            last_recall_nonzero = i



    recall_histogram = recall_histogram[:(last_nonzero + 1)] # HEY: change to last_nonzero if you want to include precision
    precision_histogram = precision_histogram[:(last_nonzero + 1)] # HEY: change to last_nonzero if you want to include precision


    print("histogram values: ")
    print(repr(recall_histogram))
    print(repr(precision_histogram))

    x_hist = np.arange(len(recall_histogram))
    width = 0.4

    plt.figure(figsize=(15, 6), dpi=160)

    plt.bar(x_hist - 0.2, recall_histogram,  width=width, label="recall")
    plt.bar(x_hist + 0.2, precision_histogram, width=width, label="precision")

    # labelling with text
    for i, v in enumerate(recall_histogram):
        # print(i, v, recall_frac[i, 1])
        if (int(recall_frac[i, 1]) == 0):
            continue
        plt.text(i - 0.4, v + .03, str(int(recall_frac[i, 0])), color='blue', fontsize='xx-small', fontweight='bold')
        plt.text(i - 0.4, v + .01, "/"  + str(int(recall_frac[i, 1])), color='blue', fontsize='xx-small', fontweight='bold')

    for i, v in enumerate(precision_histogram):
        # print(i, v, recall_frac[i, 1])
        if (int(precision_frac[i, 1]) == 0):
            continue
        plt.text(i, v + .03, str(int(precision_frac[i, 0])), color='orange', fontsize='xx-small', fontweight='bold')
        plt.text(i, v + .01, "/"  + str(int(precision_frac[i, 1])), color='orange', fontsize='xx-small', fontweight='bold')
    plt.xticks(np.arange(min(x_hist), max(x_hist)+1, 1.0))

    plt.legend() # ONLY IF YOU DONT HAVE PRECISION 
    plt.xlabel('CC length (buckets of 10) (value = lower bound)')
    plt.ylabel('Proportion correct')
    plt.title('Recall based on Coiled Coil Length for LSTM2 1:1')
    plt.savefig(file_loc)

def metric_visualizer(recall_histogram, precision_histogram):
    x_hist = np.arange(len(recall_histogram))# HEY: change to last_nonzero if you want to include precision
    width = 0.4

    plt.figure(figsize=(9, 6), dpi=160)

    plt.plot(x_hist - 0.2 + 0.2, recall_histogram, label="recall")
    # plt.bar(x_hist + 0.2, precision_histogram, width=width, label="precision")

    # labelling with text
    for i, v in enumerate(recall_histogram):
        # print(i, v, recall_frac[i, 1])
        if (int(recall_frac[i, 1]) == 0):
            continue
        plt.text(i - 0.4 + 0.2, v + .03, str(int(recall_frac[i, 0])), color='blue', fontsize='xx-small', fontweight='bold')
        plt.text(i - 0.4 + 0.2, v + .01, "/"  + str(int(recall_frac[i, 1])), color='blue', fontsize='xx-small', fontweight='bold')

    # for i, v in enumerate(precision_histogram):
    #     # print(i, v, recall_frac[i, 1])
    #     if (int(precision_frac[i, 1]) == 0):
    #         continue
    #     plt.text(i, v+ .03, str(int(precision_frac[i, 0])), color='orange', fontsize='xx-small', fontweight='bold')
    #     plt.text(i, v + .01, "/"  + str(int(precision_frac[i, 1])), color='orange', fontsize='xx-small', fontweight='bold')
    plt.xticks(np.arange(min(x_hist), max(x_hist)+1, 1.0))

    # plt.legend() ONLY IF YOU DONT HAVE PRECISION 
    plt.xlabel('CC length (buckets of 10) (value = lower bound)')
    plt.ylabel('Proportion correct')
    plt.title('Recall based on Coiled Coil Length')
    plt.savefig(file_loc)



###########################
#####MODEL EVALUATION######
###########################

#evaluations = model.evaluate(dataset_bucketed_teb, verbose=1)
advanced_metrics(221, 200, 'recall.png') # 221 is max 

# metric_visualizer(np.array([ 0.38676845,  0.48077969,  0.60640621,  0.68558673,  0.83360258,
#         0.97816594,  0.81025641,  0.94871795,  0.99112426, -0.02      ,
#        -0.02      ,  0.96581197, -0.02      , -0.02      , -0.02      ,
#        -0.02      ,  0.96296296]), np.array([ 0.14775673,  0.29586212,  0.3340298 ,  0.3688567 ,  0.37665104,
#         0.33581395,  0.38398018,  0.33743169,  0.33621934,  0.40396341,
#         0.27840909,  1.        ,  0.35356201,  0.40481928, -0.02      ,
#        -0.02      , -0.02      ]))
        
