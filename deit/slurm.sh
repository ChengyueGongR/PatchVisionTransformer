#!/bin/bash

#SBATCH --job-name DeitReproduce                                        # Job name

### Logging
#SBATCH --output=logs/slurmjob_%j.out                    # Name of stdout output file (%j expands to jobId)
#SBATCH --error=logs/slurmjob_%j.err                        # Name of stderr output file (%j expands to jobId)
#SBATCH --mail-user=cygong@eee.ddd.edu  # Email of notification
#SBATCH --mail-type=END,FAIL,REQUEUE                                      

### Node info
###SBATCH --partition dgx  
#SBATCH --nodes=1                                                            # Always set to 1 when using the cluster
#SBATCH --ntasks-per-node=4                                       # Number of tasks per node (Set to the number of gpus requested)
#SBATCH --time 240:00:00                                                     # Run time (hh:mm:ss)

#SBATCH --gres=gpu:4                                                       # Number of gpus needed
#SBATCH --mem=160G                                                         # Memory requirements
#SBATCH --cpus-per-task=4                                             # Number of cpus needed per task


source ./train.sh
