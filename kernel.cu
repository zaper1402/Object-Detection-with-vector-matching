#include "support.h"
#include <cuda_runtime.h>

// CUDA kernel for converting BGR to Grayscale
// Each thread processes one pixel
__global__ void bgr2grayscale_kernel(unsigned char *d_bgr, unsigned char *d_gray, 
                                     int width, int height) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (row < height && col < width) {
        int idx = row * width + col;
        int bgr_idx = idx * 3;
        
        unsigned char b = d_bgr[bgr_idx];
        unsigned char g = d_bgr[bgr_idx + 1];
        unsigned char r = d_bgr[bgr_idx + 2];
        
        // Standard grayscale conversion: Y = 0.299*R + 0.587*G + 0.114*B
        float gray_val = 0.299f * r + 0.587f * g + 0.114f * b;
        d_gray[idx] = (unsigned char)(gray_val + 0.5f);  // Proper rounding
    }
}

// CUDA kernel for image resizing using nearest neighbor interpolation
__global__ void resize_kernel(unsigned char *d_src, unsigned char *d_dst,
                             int src_width, int src_height,
                             int dst_width, int dst_height) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (row < dst_height && col < dst_width) {
        // Calculate source pixel coordinates
        int src_col = (col * src_width) / dst_width;
        int src_row = (row * src_height) / dst_height;
        
        src_col = min(src_col, src_width - 1);
        src_row = min(src_row, src_height - 1);
        
        d_dst[row * dst_width + col] = d_src[src_row * src_width + src_col];
    }
}

// CUDA kernel for drawing polylines (bounding box)
__global__ void draw_polyline_kernel(unsigned char *d_frame, int width, int height,
                                    float *d_corners, int num_corners,
                                    unsigned char r, unsigned char g, unsigned char b) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (row < height && col < width) {
        int idx = (row * width + col) * 3;
        
        // Simple line drawing using Bresenham-like approach
        // Check proximity to corners for simplified visualization
        for (int i = 0; i < num_corners; i++) {
            int cx = (int)d_corners[i * 2];
            int cy = (int)d_corners[i * 2 + 1];
            
            // Draw pixels within distance threshold of corner points
            if (abs(row - cy) < 3 && abs(col - cx) < 3) {
                d_frame[idx] = b;
                d_frame[idx + 1] = g;
                d_frame[idx + 2] = r;
                break;
            }
        }
    }
}

// Wrapper function for BGR to Grayscale conversion
cudaError_t gpu_bgr2grayscale(unsigned char *d_bgr, unsigned char *d_gray,
                              int width, int height) {
    dim3 block(16, 16);
    dim3 grid((width + 15) / 16, (height + 15) / 16);
    
    bgr2grayscale_kernel<<<grid, block>>>(d_bgr, d_gray, width, height);
    
    return cudaGetLastError();
}

// Wrapper function for image resizing
cudaError_t gpu_resize(unsigned char *d_src, unsigned char *d_dst,
                       int src_width, int src_height,
                       int dst_width, int dst_height) {
    dim3 block(16, 16);
    dim3 grid((dst_width + 15) / 16, (dst_height + 15) / 16);
    
    resize_kernel<<<grid, block>>>(d_src, d_dst, src_width, src_height, 
                                   dst_width, dst_height);
    
    return cudaGetLastError();
}

// Wrapper function for drawing polylines
cudaError_t gpu_draw_polyline(unsigned char *d_frame, int width, int height,
                             float *d_corners, int num_corners,
                             unsigned char r, unsigned char g, unsigned char b) {
    dim3 block(16, 16);
    dim3 grid((width + 15) / 16, (height + 15) / 16);
    
    draw_polyline_kernel<<<grid, block>>>(d_frame, width, height, d_corners, 
                                         num_corners, r, g, b);
    
    return cudaGetLastError();
}
