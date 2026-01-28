# Makefile for CUDA Video Object Detection
# Compatible with Mahti supercomputer

# Compiler settings
NVCC = nvcc
CXX = g++
CXXFLAGS = -std=c++11 -O3
# Use explicit gencode entries instead of -arch=sm_60
NVCCFLAGS = -std=c++11 -O3 \
  -gencode=arch=compute_70,code=sm_70 \
  -gencode=arch=compute_80,code=sm_80

# OpenCV settings (adjust path as needed)
# Use custom OpenCV installation with CUDA support
OPENCV_INSTALL_PATH = $(HOME)/opencv-install
OPENCV_INCLUDE = $(OPENCV_INSTALL_PATH)/include/opencv4
# prefer lib64 but fall back to lib
ifneq ("$(wildcard $(OPENCV_INSTALL_PATH)/lib64)","")
OPENCV_LIB_PATH = $(OPENCV_INSTALL_PATH)/lib64
else
OPENCV_LIB_PATH = $(OPENCV_INSTALL_PATH)/lib
endif
OPENCV_LIB = -L$(OPENCV_LIB_PATH) -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_imgcodecs -lopencv_features2d -lopencv_calib3d -lopencv_videoio -lopencv_cudaimgproc -lopencv_cudafeatures2d

# CUDA settings
CUDA_PATH = /usr/local/cuda
CUDA_INCLUDE = $(CUDA_PATH)/include
CUDA_LIB_PATH = $(CUDA_PATH)/lib64
CUDA_LIB = -lcuda -lcudart

# Directories
SRCDIR = .
OBJDIR = obj
BINDIR = .

# Source files
CUDA_SOURCES = kernel.cu support.cu
CUDA_OBJECTS = $(OBJDIR)/kernel.o $(OBJDIR)/support.o
CPP_SOURCES = main.cu
CPP_OBJECTS = $(OBJDIR)/main.o

EXECUTABLE = $(BINDIR)/video_detection_gpu

# Default target
all: $(EXECUTABLE)

# Create object directory
$(OBJDIR):
	mkdir -p $(OBJDIR)

# Compile CUDA files
$(OBJDIR)/%.o: $(SRCDIR)/%.cu | $(OBJDIR)
	$(NVCC) $(NVCCFLAGS) -I$(CUDA_INCLUDE) -I$(OPENCV_INCLUDE) -c $< -o $@

# Link executable
$(EXECUTABLE): $(CUDA_OBJECTS) $(CPP_OBJECTS)
	$(CXX) $(CXXFLAGS) -I$(CUDA_INCLUDE) -I$(OPENCV_INCLUDE) \
    -L$(CUDA_LIB_PATH) $(CUDA_OBJECTS) $(CPP_OBJECTS) \
    -o $@ $(OPENCV_LIB) -lcudart -lcuda -lpthread -lrt -ldl -lstdc++ -Wl,-rpath,$(OPENCV_LIB_PATH)

# Clean
clean:
	rm -rf $(OBJDIR) $(EXECUTABLE)

# Help
help:
	@echo "CUDA Video Object Detection Makefile"
	@echo "===================================="
	@echo "Targets:"
	@echo "  all     - Build the executable"
	@echo "  clean   - Remove build artifacts"
	@echo "  help    - Show this help message"
	@echo ""
	@echo "Usage:"
	@echo "  make                # Build everything"
	@echo "  make clean          # Clean build files"
	@echo ""
	@echo "Customize these variables for your system:"
	@echo "  OPENCV_INCLUDE - Path to OpenCV headers"
	@echo "  OPENCV_LIB     - OpenCV library flags"
	@echo "  CUDA_INCLUDE   - Path to CUDA headers"

.PHONY: all clean help
