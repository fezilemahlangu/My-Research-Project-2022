#!/bin/bash

#SBATCH -p batch
#SBATCH -N 1
#SBATCH -J endextra_deepRL
#SBATCH -o my_output_file4.out
#SBATCH -e my_error_file4.err


echo "---------------------------"
echo "Job started on" `date`
## Load the python interpreter
## module load anaconda
source ~/.bashrc ##source conda 

# conda create -n my-env
conda activate my_env

conda config --env --add channels conda-forge 

# conda install -c conda-forge numpy
# conda install -c conda-forge scikit-learn
# conda install -c conda-forge tensorflow 
# conda install -c conda-forge zipfile36  # not sure it it installed 
# conda install -c conda-forge csvkit
# conda install -c conda-forge pandas 
# conda install -c conda-forge pillow 

# conda install -c conda-forge zipfile-deflate64

## Execute the python script
python DeepRL_extra.py

echo "---------------------------"
echo "Job ended on" `date`