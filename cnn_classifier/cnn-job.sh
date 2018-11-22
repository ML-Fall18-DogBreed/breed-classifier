#!/bin/bash
#SBATCH --output=simple.out #SBATCH --error=simple.err
#SBATCH --gres:gpu:1
#SBATCH -t 0:45:00

module load cudnn
module load cuda90/toolkit
module load cuda90/blas
source ~/venv/bin/activate
python classifier.py
