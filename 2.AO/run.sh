#!/bin/bash
#PBS -N jhchoi_python
#PBS -q pknu_share
#PBS -r n
#PBS -l nodes=1:ppn=1
#PBS -j oe



cd /data03/Glosea5/jhsim/NCL2PYTHON/METRIC_PYTHON/2.AO
source /home/jhchoi/anaconda3/etc/profile.d/conda.sh
conda activate py311

for ss in "ON" "DJ" "FM";do
    python /data03/Glosea5/jhsim/NCL2PYTHON/METRIC_PYTHON/2.AO/001_AO.py $ss
done