#!/bin/bash
# Build script for Mahti supercomputer
# Run this to compile the CUDA video detection application

echo "=== CUDA Video Object Detection Build Script ==="
echo ""

# Check if CUDA is available
if ! command -v nvcc &> /dev/null; then
    echo "Error: CUDA compiler (nvcc) not found."
    echo "Please run: module load cuda"
    exit 1
fi

echo "CUDA Compiler:"
nvcc --version
echo ""

# Check for custom OpenCV installation
if [ ! -d "$HOME/opencv-install" ]; then
    echo "Warning: Custom OpenCV installation not found at $HOME/opencv-install"
    echo "Please build OpenCV with CUDA support first (see README.md)"
    echo ""
else
    echo "Using custom OpenCV installation: $HOME/opencv-install"
    echo ""
fi

# Clean previous build
echo "Cleaning previous builds..."
make clean > /dev/null 2>&1

# Build
echo "Building CUDA Video Object Detection..."
make

if [ $? -eq 0 ]; then
    echo ""
    echo "=== Build Successful ==="
    echo "Executable: ./video_detection_gpu"
    echo ""
    echo "To run the application:"
    echo "  ./video_detection_gpu"
else
    echo ""
    echo "=== Build Failed ==="
    exit 1
fi
