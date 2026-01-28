#ifndef SUPPORT_H
#define SUPPORT_H

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

// Structure for storing image matrices
typedef struct {
    int width;
    int height;
    int channels;
    unsigned char *data;
} Matrix;

// Structure for device image matrices
typedef struct {
    int width;
    int height;
    int channels;
    unsigned char *d_data;
} DeviceMatrix;

// CUDA Error checking macro
#define checkCudaErrors(err) \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error in %s at line %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    }

// Kernel wrapper functions
cudaError_t gpu_bgr2grayscale(unsigned char *d_bgr, unsigned char *d_gray,
                              int width, int height);

cudaError_t gpu_resize(unsigned char *d_src, unsigned char *d_dst,
                       int src_width, int src_height,
                       int dst_width, int dst_height);

cudaError_t gpu_draw_polyline(unsigned char *d_frame, int width, int height,
                             float *d_corners, int num_corners,
                             unsigned char r, unsigned char g, unsigned char b);

// Utility functions
void printDeviceProperties();
void allocateDeviceMatrix(DeviceMatrix *mat, int width, int height, int channels);
void freeDeviceMatrix(DeviceMatrix *mat);
void transferToDevice(Matrix *h_mat, DeviceMatrix *d_mat);
void transferFromDevice(DeviceMatrix *d_mat, Matrix *h_mat);

#endif
