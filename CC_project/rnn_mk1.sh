#!/bin/bash
#SBATCH -p eddy_gpu # partition
#SBATCH --open-mode=append
#SBATCH -J unidirecLSTM_1

# used to be an batch -c 1 # number of cores? idk difference with this and -n
#SBATCH -n 1 
#SBATCH -N 1
#SBATCH -t 1-00:00 # D-HH:MM 
#SBATCH --mem=24000 # memory by MB
#SBATCH --gres=gpu:4
#### --constraint="a40"
#SBATCH --gpu-freq=high

#SBATCH -o output_unidirecLSTM_1_mk1_%j.out # output file
#SBATCH -e error_unidirecLSTM_1_mk1_%j.err # error file

#SBATCH --mail-type=END
#SBATCH --mail-user=rangerkuang@fas.harvard.edu
module load Anaconda3 cuda cudnn
source deactivate
source activate tensorflow
python3 rnn_mk1.py

