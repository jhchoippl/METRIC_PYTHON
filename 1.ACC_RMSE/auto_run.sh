#!/bin/bash

#PBS -l nodes=1:ppn=36
#PBS -e out12.err
#PBS -o out22.log
#PBS -q pknu_ppl
#PBS -q pknu_share
#PBS -V
#PBS -j eo

cd $PBS_O_WORKDIR
source /home/jhchoi/anaconda3/etc/profile.d/conda.sh
conda activate py311
echo clear

python /data03/Glosea5/jhsim/NCL2PYTHON/METRIC_PYTHON/1.ACC_RMSE/test.py