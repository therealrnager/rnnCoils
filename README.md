![header ](readmeVisual.png)
_Figure: Sample prediction by rnnCoils of protein sequence B in PDB ID: 5UBT._

# rnnCoils
This directory holds (parts of) the work of undergraduate Ranger Kuang during the Harvard Systems Bio REU Summer Internship 2023. This README file will give a broad overview of where files are and what certain were meant to accomplish. Please contact me at ranger.kuang@columbia.edu if any questions arise that I can answer. 

Note that due to file size limits, this repo does not include every single file relevant to the project and thus is not meant to be a working copy of the project. However, it still showcases the majority of the files I compiled and wrote for the project.

## Data 
Data was originally collected and stored at: 
`/n/eddy_lab/users/npcarter/null-model/sequence_generation/parsed_sockets_output`

All data used for the project was stored in the `CC_data/` directory. Amongst these directory are only select directories that are actually important: 
* `has_cc_real/` and `no_cc_real/`. As these names suggest, these are the folders with files that either have coiled coil domains or not in their respective sequences. In each of these folders are additional subdirectories. 
    * The first layer of subdirectories includes `to100/`, `to400/`, and `to_max/`. These are meant to partition the files into sequences of length 1-100, 101-400, and 401 onwards.
    * The second sublayer includes `train/`, `valid/`, `test/`, and `x_graveyard/`. the first 3 are in an _approximately_ 80%:10%:10% split (reasoning in the very next sentence). `x_graveyard/` is for all files where their sequence content was more 30% or more "X". I did this parse after splitting into train, validation, test sets which is why their split is only approximately 80:10:10. 
* `unique_pkl_data/` is where I have pickled files that store the actual data in the format of TensorFlow datatype "Ragged Tensors". 
    * Due to the size of these files, they are not stored on github. Note that without these files, it is impossible to run any of the models, without re-pickling the data into the correct format.

## Data Collection and Processing
All processing of data was done by many python files stored in  `CC_project/data_cleanup/`. I've explained some of the notable files:
* `inspector.py`: Tracked data on lengths of coiled coil domains across all sequences and built a frequency list of lengths. ALSO parsed data into the sequence-length buckets described earlier
* `duplicates.py`, `dup_ranger.py`, and `dup_helper.py`: All various files that helped me delete duplicate sequence files. 
* `partitioner.py`: Was what randomly partitioned the sequence-length buckets into an 80:10:10 split for training:validation:test folders.   
* `data_parser.py`: the file that actually built the pickled files i.e. the ragged tensors that made up the training set, the validation set, and the test set. 
* `weight_parser.py`: the file that actually build the pickled files of just the weight datasets. 
* `seqlength.py`: Tracked data on lengths of sequences across all sequences and built a frequency list of lengths. 

## Models
The actual TensorFlow models (and the files that trained the models) are stored in `CC_project/` and called `rnn_mk#.py` where the `#` is a number. 
* Mark 1 was the original model using one layer of LSTM or simpleRNN.  
* Mark 2 was with 2 LSTM layers 
* mk3 and mk4 were experimental when I was trying to toy with attention networks and transformers (respectively) yet never had the time for them to be fully functioning. These files won't run. 

In each of these files, the general layout and ordering of data processing, model building, model training, and model evaluation, should be pretty clear given I wrote headers in the python files themselves. 

The only things I really changed when training different models were:
* the model build itself (obviously)
* the line `date = "..."`. This line is very important to change, as it signals where the model is to be saved. This leads into my next point --> 
## Model Saving 
Saved models are in the `CC_project/uq_checkpoints/` directory. (Old models that were trained on the dataset that had duplicate files are stored in `CC_project/checkpoints/`). Within this directory, you can see the specific date I ran that model, which is how I kept track of which model was which. 

If you ever wish to run another model, you must first do some setup to get into the right environment: 
1. run `srun -p eddy_gpu --gres=gpu:1 --mem=24000 -t 0 --pty /bin/bash` or something of this nature. What matters is that you are on eddy_gpu.
2. run `module load Anaconda3 cuda cudnn`
3. run `source activate tensorflow`
4. run `export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/n/eddy_lab/users/rangerkuang/.conda/envs/tensorflow/lib/` (this is just so that matplotlib doesn't complain wildly when evaluating the model in the future)
5. ALTERNATIVELY, you can just train the model by submitting an sbatch job; this is recommended so you don't need to keep the terminal running while the model trains. You can see `rnn_mk#.sh` for examples of how I set up my sbatch files. 


Then, you must: 
1. change the line in `rnn_mk1.py` that has `date = "..."` to a new date. I will use `"foo"` as an example. 
2. Then, inside `CC_project/uq_checkpoints/`, run 
```
mkdir foo
cd foo
vim model_foo.ckpt
``` 
Then save that new empty ckpt file.
3. Finally, if you wish to change the weights that the model's cost function will be trained on, change the line in the "LOAD DATA" section that looks like `WEIGHT = <number>`. 

NOTE: my models, when training, only saved the weights that gave the best performing model BASED ON the metric of validation PRC. PRC, in tensorflow, is tensorflow's best approximation for the area-under-the-curve of the Precision-Recall graph of your model. 

## Evaluation of the Models 
In `CC_project/`, the two main files used to evaluate models were:
* `nn_tester.py`: Used to evaluate precision and recall on a per Coiled-coil length basis. See `recall.png` and `foo.png` for examples, as this is where I often saved the plots to. To actually use this file, just ensure that:
1. The line `date = "..."` is the same as the one you wrote in the `rnn_mk#.py` file when you trained the model. 
2. the model build is actually the same as the model you want to evaluate, as the `date = ...` feature only saves weights, but not the model architecture itself. 
* `sequence_tester.py`: This is to test a model on JUST one specific sequence and print out a plot of its prediction into `visual.png`. To run this: 
1. The line `date = "..."` is the same as the one you wrote in the `rnn_mk#.py` file when you trained the model. 
2. the model build is actually the same as the model you want to evaluate, as the `date = ...` feature only saves weights, but not the model architecture itself. 
3. The line `FILE = ...` is the pathway to the file you want to analyze.  
