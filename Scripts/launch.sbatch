#!/bin/bash
#SBATCH --time=00:20:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --job-name=python-ex
#SBATCH --mail-type=ALL
#SBATCH --partition=cuda
###### 1 Load the module
module load nvidia/cudasdk/10.1
module load intel/python/3
# 2 Activate the virtual environment
source IntesaEnv/bin/activate
# 3 Run the python script
python StronglyCVSave.py 