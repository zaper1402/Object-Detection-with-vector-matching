# CUDA Video Object Detection Setup

This is a GPU-accelerated variant of the video object detection application, following the same architecture as Lab_4.

## Project Structure

```
video_detection_cuda/
├── kernel.cu        # CUDA kernels for image processing
├── main.cu          # Main application entry point
├── support.cu       # Helper functions and utilities
├── support.h        # Header file with function declarations
└── Makefile         # Build configuration
```

## GPU Accelerated Operations

### 1. **BGR to Grayscale Conversion**
   - CUDA Kernel: `bgr2grayscale_kernel()`
   - Each thread processes one pixel
   - Optimized grid/block configuration: 16×16 threads per block

### 2. **Image Resizing**
   - CUDA Kernel: `resize_kernel()`
   - Nearest neighbor interpolation
   - Efficient for preparing frames for display

### 3. **Bounding Box Drawing**
   - CUDA Kernel: `draw_polyline_kernel()`
   - GPU-accelerated visualization

## CPU Operations (can be GPU accelerated)

- **Feature Detection**: SIFT/ORB keypoint detection
- **Feature Matching**: Descriptor matching using BFMatcher
- **Homography Computation**: Perspective transform calculation

*Note: These can be accelerated using OpenCV CUDA modules (CUDA SIFT, CUDA ORB)*

## Building on Mahti Supercomputer

### Prerequisites

Load required modules:

```bash
module load cuda
module load opencv
module load gcc
```

### Build Commands

```bash
# Clean previous builds
make clean

# Build the project
make

# Run the application
./video_detection_gpu
```

## Memory Layout

Similar to Lab_4:

- **Host Memory**: Original frames, keypoints, descriptors
- **Device Memory**: Frame buffers, grayscale images
- **Constant Memory**: (Available for filter kernels in future)
- **Shared Memory**: (Used in convolution-like operations)

## Performance Optimization Techniques

1. **Coalesced Memory Access**: Sequential threads access sequential memory
2. **Shared Memory**: Reduces global memory latency (100× faster)
3. **Grid/Block Optimization**: 16×16 blocks for 2D image processing
4. **Pinned Memory**: Faster host-device transfer (not yet implemented)

## CUDA Compute Capability

Current kernel compilation targets: `-arch=sm_60`

Mahti GPU specifications:
- Volta V100 architecture (compute capability 7.0+)
- 32GB memory per GPU
- Peak memory bandwidth: 900 GB/s

Adjust `NVCCFLAGS` in Makefile if needed:
- SM 3.5: `-arch=sm_35`
- SM 5.2: `-arch=sm_52`
- SM 6.0: `-arch=sm_60` (default)
- SM 7.0: `-arch=sm_70` (recommended for Mahti)
- SM 8.0: `-arch=sm_80`

## Error Handling

The project uses `checkCudaErrors()` macro for robust CUDA error detection. All CUDA operations are checked for errors.

## Future Optimizations

1. **CUDA Streams**: Overlap computation with data transfer
2. **Pinned Memory**: Faster host-device transfers
3. **Texture Memory**: Better cache locality for image sampling
4. **OpenCV CUDA Modules**: GPU-accelerated SIFT/ORB detection
5. **Thrust Library**: Parallel algorithms for feature matching

## Debugging

Enable synchronous execution for debugging:

```bash
export CUDA_LAUNCH_BLOCKING=1
```

Profile with:

```bash
module load cuda
nvprof ./video_detection_gpu
```

## Output

The application generates:
- Output video: `../Output/main_video_detection_gpu/output_video.mp4`
- Console output with timing information
- Frame-by-frame processing progress
