#include "support.h"
#include <stdlib.h>
#include <stdio.h>

// Print CUDA device properties
void printDeviceProperties() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    
    if (deviceCount == 0) {
        printf("No CUDA devices found!\n");
        return;
    }
    
    for (int i = 0; i < deviceCount; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        
        printf("=== CUDA Device %d ===\n", i);
        printf("Name: %s\n", prop.name);
        printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
        printf("Total Global Memory: %lu GB\n", prop.totalGlobalMem / (1024 * 1024 * 1024));
        printf("Max Threads per Block: %d\n", prop.maxThreadsPerBlock);
        printf("Warp Size: %d\n", prop.warpSize);
    }
}

// Allocate device memory for matrix
void allocateDeviceMatrix(DeviceMatrix *mat, int width, int height, int channels) {
    mat->width = width;
    mat->height = height;
    mat->channels = channels;
    
    size_t size = (size_t)width * height * channels * sizeof(unsigned char);
    checkCudaErrors(cudaMalloc((void**)&mat->d_data, size));
    
    printf("Allocated %zu bytes on device for matrix (%dx%dx%d)\n", 
           size, width, height, channels);
}

// Free device memory
void freeDeviceMatrix(DeviceMatrix *mat) {
    if (mat->d_data != NULL) {
        checkCudaErrors(cudaFree(mat->d_data));
        mat->d_data = NULL;
    }
}

// Transfer matrix from host to device
void transferToDevice(Matrix *h_mat, DeviceMatrix *d_mat) {
    size_t size = (size_t)h_mat->width * h_mat->height * h_mat->channels * sizeof(unsigned char);
    checkCudaErrors(cudaMemcpy(d_mat->d_data, h_mat->data, size, cudaMemcpyHostToDevice));
}

// Transfer matrix from device to host
void transferFromDevice(DeviceMatrix *d_mat, Matrix *h_mat) {
    size_t size = (size_t)h_mat->width * h_mat->height * h_mat->channels * sizeof(unsigned char);
    checkCudaErrors(cudaMemcpy(h_mat->data, d_mat->d_data, size, cudaMemcpyDeviceToHost));
}
