#!/bin/bash

#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -p pi_raffaele # sched_mit_raffaele_gpu # 
#SBATCH --gres=gpu:1
#SBATCH --time=30:00:00
#SBATCH --mem=150GB
#SBATCH -o diag3.out
#SBATCH -e diag3.err
#SBATCH -J diag3

source /etc/profile.d/modules.sh
module load nvhpc 

export ECCO_USERNAME=ssilvestri
export ECCO_WEBDAV_PASSWORD=ZZjQeLy7oIHwvqMWvM8y

~/julia-1.12.4/bin/julia --project --check-bounds=no diagnostic_runs_3.jl
