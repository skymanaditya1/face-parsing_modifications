#!/bin/bash 

#SBATCH --job-name=pert_data
#SBATCH --mem-per-cpu=1024
#SBATCH --partition long
#SBATCH --account cvit_bhaasha
#SBATCH --gres=gpu:2
#SBATCH --mincpus=20
#SBATCH --nodes=1
#SBATCH --time 4-00:00:00
#SBATCH --signal=B:HUP@600
#SBATCH -w gnode80

source /home2/aditya1/miniconda3/bin/activate stargan-v2
cd /ssd_scratch/cvit/aditya1/face-parsing.PyTorch/extras
python generate_data.py