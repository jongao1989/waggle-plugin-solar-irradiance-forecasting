#!/bin/bash
#SBATCH --job-name=<JOB_NAME_GOES_HERE>
#SBATCH --account=<ACCOUNT_NAME_GOES_HERE>
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00

module load anaconda3
source activate <ENVIRONMENT_NAME>

python "PATH/TO/FILE/python_training.py"


#==============================================================================
# You can use this file as a template to run training on a computing cluster.
# 1) Change the job name, account, environment name, and path to file to your
#    needs
# 2) Change the number of nodes, GPUs, and time based on your needs. 1 node is
#    generally enough.
# 3) Run this file by using: sbatch bash_training.slurm
#==============================================================================