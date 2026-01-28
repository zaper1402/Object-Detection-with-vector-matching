#!/bin/bash
# SLURM submission script for Mahti supercomputer
# Submit with: sbatch run_mahti.sh

#SBATCH --job-name=cuda_video_detection
#SBATCH --partition=gputest
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --gres=gpu:A100:1
#SBATCH --time=00:15:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

# Load required modules
module load gcc cuda cmake python-data pytorch git

# Set OpenCV library path for runtime
export LD_LIBRARY_PATH=$HOME/opencv-install/lib:$LD_LIBRARY_PATH

echo "Job started at $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo ""

# Build if needed
if [ ! -f video_detection_gpu ]; then
    echo "Building CUDA application..."
    make clean
    make
    
    if [ $? -ne 0 ]; then
        echo "Build failed!"
        exit 1
    fi
fi

# Run the application
echo "Running CUDA Video Object Detection..."
./video_detection_gpu

echo ""
echo "Job completed at $(date)"
