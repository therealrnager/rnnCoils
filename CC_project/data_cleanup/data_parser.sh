#!/bin/bash
#SBATCH -p eddy_gpu # partition
#SBATCH --open-mode=append

#SBATCH -c 1 # number of cores? idk difference with this and -n
#SBATCH -t 0-03:00 # D-HH:MM 
#SBATCH --mem=24000 # memory by MB
#SBATCH --gres=gpu:1
#SBATCH -o output_dataparser_unique_%j.out # output file
#SBATCH -e errors_dataparser_unique_%j.err # error file

#SBATCH --mail-type=END
#SBATCH --mail-user=rangerkuang@fas.harvard.edu

module load Anaconda3 cuda cudnn
source deactivate
source activate tensorflow
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/n/eddy_lab/users/rangerkuang/.conda/envs/tensorflow/lib/

python3 data_parser.py
