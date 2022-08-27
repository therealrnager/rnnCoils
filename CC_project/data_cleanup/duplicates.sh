#!/bin/bash
#SBATCH -p eddy # partition
#SBATCH --open-mode=append
#SBATCH -J duplicates_phase3


# used to be an batch -c 1 # number of cores? idk difference with this and -n
#SBATCH -c 1
#SBATCH -n 1 
#SBATCH -N 1
#SBATCH -t 1-00:00 # D-HH:MM 
#SBATCH --mem=24000 # memory by MB


#SBATCH -o output_dup_phase3_%j.out # output file
#SBATCH -e error_dup_phase3_%j.err # error file

#SBATCH --mail-type=END
#SBATCH --mail-user=rangerkuang@fas.harvard.edu

#### PHASE 1: REMOVE ALL DUPLICATES IN EACH train/valid/test SET ####
# python3 duplicates.py -d /n/eddy_lab/users/rangerkuang/CC_data/has_cc_real/to_max/test
# python3 duplicates.py -d /n/eddy_lab/users/rangerkuang/CC_data/has_cc_real/to_max/valid
# python3 duplicates.py -d /n/eddy_lab/users/rangerkuang/CC_data/has_cc_real/to_max/train

# python3 duplicates.py -d /n/eddy_lab/users/rangerkuang/CC_data/has_cc_real/to400/test
# python3 duplicates.py -d /n/eddy_lab/users/rangerkuang/CC_data/has_cc_real/to400/valid
# python3 duplicates.py -d /n/eddy_lab/users/rangerkuang/CC_data/has_cc_real/to400/train

# python3 duplicates.py -d /n/eddy_lab/users/rangerkuang/CC_data/has_cc_real/to100/test
# python3 duplicates.py -d /n/eddy_lab/users/rangerkuang/CC_data/has_cc_real/to100/valid
# python3 duplicates.py -d /n/eddy_lab/users/rangerkuang/CC_data/has_cc_real/to100/train


# python3 duplicates.py -d /n/eddy_lab/users/rangerkuang/CC_data/no_cc_real/to_max/test
# python3 duplicates.py -d /n/eddy_lab/users/rangerkuang/CC_data/no_cc_real/to_max/valid
# python3 duplicates.py -d /n/eddy_lab/users/rangerkuang/CC_data/no_cc_real/to_max/train

# python3 duplicates.py -d /n/eddy_lab/users/rangerkuang/CC_data/no_cc_real/to400/test
# python3 duplicates.py -d /n/eddy_lab/users/rangerkuang/CC_data/no_cc_real/to400/valid
# python3 duplicates.py -d /n/eddy_lab/users/rangerkuang/CC_data/no_cc_real/to400/train

# python3 duplicates.py -d /n/eddy_lab/users/rangerkuang/CC_data/no_cc_real/to100/test
# python3 duplicates.py -d /n/eddy_lab/users/rangerkuang/CC_data/no_cc_real/to100/valid
# python3 duplicates.py -d /n/eddy_lab/users/rangerkuang/CC_data/no_cc_real/to100/train


#### PHASE 2: re-amalgamate the train, valid, test sets, then remove duplicates again

# python3 duplicates.py -d /n/eddy_lab/users/rangerkuang/CC_data/has_cc_real/to_max/tmp
# python3 duplicates.py -d /n/eddy_lab/users/rangerkuang/CC_data/has_cc_real/to400/tmp
# python3 duplicates.py -d /n/eddy_lab/users/rangerkuang/CC_data/has_cc_real/to100/tmp
# python3 duplicates.py -d /n/eddy_lab/users/rangerkuang/CC_data/no_cc_real/to_max/tmp
# python3 duplicates.py -d /n/eddy_lab/users/rangerkuang/CC_data/no_cc_real/to400/tmp
# python3 duplicates.py -d /n/eddy_lab/users/rangerkuang/CC_data/no_cc_real/to100/tmp


#### PHASE 3: Remove duplicates whre just the sequence, but not the coiled coil, are the same 

# python3 dup_ranger.py -d /n/eddy_lab/users/rangerkuang/CC_data/has_cc_real/to400/tmp
python3 dup_ranger.py -d /n/eddy_lab/users/rangerkuang/CC_data/has_cc_real/to_max/tmp