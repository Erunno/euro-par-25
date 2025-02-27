#!/bin/bash

#SBATCH -p gpu-short
#SBATCH -A kdss
#SBATCH --cpus-per-task=32
#SBATCH --mem=64GB
#SBATCH --gres=gpu:V100
#SBATCH --time=2:00:00
#SBATCH --output=experiments-outputs/slurm-sbatch-outputs/job-%j.out

timestamp=$(date +"%Y-%m-%d--%H-%M-%S")
job_id=$SLURM_JOB_ID

stdout_file="experiments-outputs/GoL-$timestamp--$job_id.out"
stderr_file="experiments-outputs/GoL-$timestamp--$job_id.err"

tested_implementation=$1
impl_dir=$(dirname $tested_implementation)
exe=$impl_dir/bin/game_of_life

shift
program_params="$@"

make GOL_IMPL=$tested_implementation &> "$stderr_file"

if [ $? -ne 0 ]; then
    echo "Compilation failed" >& "$stderr_file"
    exit 1
fi

($exe $program_params) >"$stdout_file" 2>"$stderr_file"
