#!/bin/bash
#SBATCH -p eddy # partition
#SBATCH --open-mode=append

#SBATCH -c 1 # number of cores? idk difference with this and -n
#SBATCH -t 0-05:00 # D-HH:MM 
#SBATCH --mem=8000 # memory by MB
#SBATCH -o output_seqlength_unique_%j.out # output file
#SBATCH -e errors_seqlength_unique_%j.err # error file

#SBATCH --mail-type=END
#SBATCH --mail-user=rangerkuang@fas.harvard.edu

# python3 inspector.py
python3 seqlength.py
