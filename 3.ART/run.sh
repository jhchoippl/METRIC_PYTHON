#!/bin/bash
#PBS -N jhchoi_python
#PBS -q pknu_share
#PBS -r n
#PBS -l nodes=1:ppn=1
#PBS -j oe



cd ${PBS_O_WORKDIR}
source /home/jhchoi/anaconda3/etc/profile.d/conda.sh
conda activate py311


for ss in "ON" "DJ" "FM";do
    python ${PBS_O_WORKDIR}/001_ART.py $ss
done