#!/bin/bash
#SBATCH --output=simple.out #SBATCH --error=simple.err
#SBATCH --gres=gpu:2
#SBATCH -t 0:45:00

module load cudnn
module load cuda90/toolkit
module load cuda90/blas
module load cuda90/profiler
module load cuda90/nsight
module load cuda90/fft
source ~/venv/bin/activate
python classifier.py
