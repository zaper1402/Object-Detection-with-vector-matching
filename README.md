# CUDA Video Object Detection Setup

This is a Cuda code of the video object detection using vector matching and homography computation. It use openCV feature detection and matching algorithms (SIFT/ORB) along with CUDA accelerated image processing operations.

#### Team Members: Muhammad Zahid, Sadikshya Satyal, Ayodeji Ibrahim, Ashir Kulshreshtha

#### Working Demo

![Demo](./output/output_video.mp4)

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


## Python Implementation on Mahti (Easier to run)
### **Note : Setup time for python is 5-10 min**

#### Sinteractive command ( if not in compute node )
```bash
sinteractive --time=05:00:00 --gres=gpu:a100:1,nvme:100 --partition=gpusmall --mem=32G --cpus-per-task=8 --pty bash
``` 
You can use partition 'gpusmall' also. Note : gputest has a limit of 15min max session.

#### Module load (if not loaded already)
```bash
module load gcc/10.4.0 cuda/12.6.1 cmake pytorch git
```

In root folder of this project.

#### Virtual env
```bash
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
```

#### Pip requirements
```bash
pip install -r requirements.txt
```

#### Commands to run file
```bash
python video_detection.py
```






## Building on CUDA files on Mahti Supercomputer
#### **Note: Setup time for this is 2-3 hours as lot of dependecies are need to be installed seperately from source.**

### Prerequisites

**Important:** Mahti does not have a standalone OpenCV module with CUDA support. You must build OpenCV from source with CUDA enabled.
#### Sinteractive command
```bash
sinteractive --time=05:00:00 --gres=gpu:a100:1,nvme:100 --partition=gpusmall --mem=32G --cpus-per-task=8 --pty bash
``` 

#### 1. Load Required Modules
```bash
module load gcc/10.4.0 cuda/12.6.1 cmake pytorch git
```
*Note: While pytorch includes OpenCV for Python, the C++ CUDA project requires a custom OpenCV build with CUDA support.*

#### 2. Build OpenCV with CUDA Support
```bash
cd $HOME
git clone https://github.com/opencv/opencv.git
git clone https://github.com/opencv/opencv_contrib.git
mkdir -p opencv/build && cd opencv/build
```
OpenCV repositories are located at:
- `/users/akulshre/opencv`
- `/users/akulshre/opencv_contrib`

#### 3. FFmpeg required for video I/O
```bash
mkdir -p $HOME/src $HOME/ffmpeg-build
cd $HOME/src
git clone https://git.ffmpeg.org/ffmpeg.git ffmpeg
cd ffmpeg

# Install nasm for fmpeg build success
cd $HOME/src
  wget https://www.nasm.us/pub/nasm/releasebuilds/2.15.05/nasm-2.15.05.tar.gz
  tar xzf nasm-2.15.05.tar.gz && cd nasm-2.15.05
  ./configure --prefix=$HOME/.local
  make -j2 && make install
  export PATH=$HOME/.local/bin:$PATH

  cd $HOME/src/ffmpeg
PKG_CONFIG_PATH=$HOME/ffmpeg-build/lib/pkgconfig ./configure \
  --prefix=$HOME/ffmpeg-build \
  --enable-shared --disable-static \
  --enable-pic \
  --disable-doc --disable-programs --disable-ffplay --disable-ffprobe \
  --disable-debug --disable-network \
  --enable-demuxer=mov,matroska,rawvideo \
  --enable-parser=h264,hevc \
  --enable-decoder=h264,hevc,mpeg4,mjpeg \
  --enable-avfilter --enable-swresample --enable-swscale \
  --enable-protocol=file \
  --extra-cflags="-fPIC" \
  --extra-ldflags="-L$HOME/ffmpeg-build/lib"
  make -j2
  make install


   export PKG_CONFIG_PATH=$HOME/ffmpeg-build/lib/pkgconfig:$PKG_CONFIG_PATH
   export LD_LIBRARY_PATH=$HOME/ffmpeg-build/lib:$LD_LIBRARY_PATH
   export PATH=$HOME/ffmpeg-build/bin:$PATH
```

#### CMAKE Command for OpenCV: Create build directory and configure:

```bash
cd ~/opencv/
mkdir -p build
cd build

cmake ../ \
  -D CMAKE_BUILD_TYPE=Release \
  -D CMAKE_INSTALL_PREFIX=$HOME/opencv-install \
  -D CMAKE_SKIP_INSTALL_RPATH=ON \
  -D WITH_CUDA=ON \
  -D CUDA_ARCH_BIN=80 \
  -D ENABLE_FAST_MATH=ON \
  -D CUDA_FAST_MATH=ON \
  -D CMAKE_C_COMPILER=$(which gcc) \
  -D CMAKE_CXX_COMPILER=$(which g++) \
  -D CUDA_HOST_COMPILER=$(which g++) \
  -D CUDA_NVCC_FLAGS="--use_fast_math" \
  -D BUILD_LIST=core,imgproc,highgui,videoio,features2d,calib3d,flann,cudev,cudaimgproc,cudafeatures2d,cudawarping \
  -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules \
  -D BUILD_opencv_python3=ON \
  -D PYTHON3_EXECUTABLE=$(which python3) \
  -D PYTHON3_PACKAGES_PATH="$(python3 -c 'import site; print(site.getsitepackages()[0])')" \
  -D BUILD_opencv_world=OFF \
  -D BUILD_TESTS=OFF \
  -D BUILD_PERF_TESTS=OFF \
  -D BUILD_EXAMPLES=OFF \
  -D BUILD_DOCS=OFF \
  -D WITH_QT=OFF \
  -D WITH_GTK=OFF \
  -D WITH_FFMPEG=ON \
  -D OPENCV_FFMPEG_USE_FIND_PACKAGE=OFF \
  -D FFMPEG_DIR=$HOME/ffmpeg-build \
  -D OPENCV_FFMPEG_SKIP_BUILD_CHECK=ON \
  -D BUILD_opencv_videoio_ffmpeg=ON \
  -D CMAKE_BUILD_RPATH="$HOME/ffmpeg-build/lib" \
  -D CMAKE_INSTALL_RPATH="$HOME/ffmpeg-build/lib" \
  -D CMAKE_EXE_LINKER_FLAGS="-Wl,-rpath-link,$HOME/ffmpeg-build/lib -Wl,-rpath,$HOME/ffmpeg-build/lib" \
  -D WITH_GSTREAMER=OFF \
  -D WITH_OPENGL=OFF \
  -D WITH_TBB=OFF \
  -D WITH_IPP=OFF \
  -D WITH_EIGEN=OFF

   cmake --build . -j2
   cmake --install .

   # verify install
   /users/akulshre/opencv-install/bin/opencv_version
   ls -ld $HOME/opencv-install/include/opencv4/opencv2/highgui.hpp
   ls -ld $HOME/opencv-install/lib64 | tail -n 20

   # ensure Makefile picks up your install (optional)
   export OPENCV_INSTALL_PATH=$HOME/opencv-install

   # ensure runtime linker finds libs and build project
   export LD_LIBRARY_PATH=$HOME/opencv-install/lib64:$LD_LIBRARY_PATH
   
```
This will install OpenCV to `$HOME/opencv-install`.


#### Build and install project : 
```bash
cd ~/video_detection_cuda  # adjust path if needed
   make clean
   make VERBOSE=1
```

#### 3. Run
Navigate to the video_detection_cuda directory:
```bash
cd ~/video_detection_cuda
./video_detection_gpu
```
