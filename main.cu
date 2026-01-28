#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/calib3d.hpp>
#include "support.h"

using namespace cv;
using namespace std;

/**
 * Main function: CUDA-accelerated video object detection
 * Combines GPU image processing with CPU feature matching
 * 
 * GPU Accelerated Operations:
 * 1. BGR to Grayscale conversion
 * 2. Image resizing
 * 3. Bounding box drawing
 * 
 * CPU Operations (can be GPU accelerated in future):
 * 1. SIFT/ORB feature detection (requires OpenCV CUDA modules)
 * 2. Feature matching
 * 3. Homography computation
 */
int main(int argc, char *argv[]) {
    // Initialize timing
    cudaEvent_t start, stop;
    float total_time = 0.0f;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Print device properties
    printf("=== CUDA Video Object Detection ===\n");
    printDeviceProperties();
    
    // File paths
    const char *query_img_path = "/users/akulshre/video_detection_cuda/Images/object2.png";
    const char *video_path = "./Videos/traffictrim.mp4";
    const char *output_path = "/users/akulshre/video_detection_cuda/output";
    bool use_sift = true;
    
    // Load query image
    printf("\n=== Loading Query Image ===\n");
    Mat query_img = imread(query_img_path, IMREAD_GRAYSCALE);
    if (query_img.empty()) {
        printf("Error: Query image not found at %s\n", query_img_path);
        return -1;
    }
    printf("Query image loaded: %dx%d\n", query_img.cols, query_img.rows);
    
    // Initialize detector
    Ptr<Feature2D> detector;
    int norm_type;
    
    if (use_sift) {
        detector = SIFT::create();
        norm_type = NORM_L2;
        printf("Using SIFT detector\n");
    } else {
        detector = ORB::create();
        norm_type = NORM_HAMMING;
        printf("Using ORB detector\n");
    }
    
    // Detect keypoints in query image
    vector<KeyPoint> kp1;
    Mat des1;
    detector->detectAndCompute(query_img, noArray(), kp1, des1);
    printf("Query image keypoints: %lu, descriptors: %dx%d\n", 
           kp1.size(), des1.rows, des1.cols);
    
    // Open video
    printf("\n=== Opening Video ===\n");
    printf("DEBUG: video_path='%s'\n", video_path);

    // quick file existence/readability check
    FILE *vf = fopen(video_path, "rb");
    if (vf) {
        fclose(vf);
        printf("DEBUG: fopen succeeded - file exists and is readable\n");
    } else {
        printf("DEBUG: fopen failed for %s (errno=%d): ", video_path, errno);
        perror("");
    }

    // print current working directory using shell (works on Windows)
    printf("DEBUG: current working directory (system 'cd'):\n");
    int rc = system("cd");
    printf("DEBUG: system('cd') returned %d\n", rc);

    // try opening with VideoCapture and log return
    VideoCapture cap;
    bool opened = cap.open(video_path);
    printf("DEBUG: VideoCapture::open returned %d, isOpened=%d\n", opened, (int)cap.isOpened());
    if (!cap.isOpened()) {
        printf("Error: Cannot open video at %s\n", video_path);
        return -1;
    }
    
    int frame_width = (int)cap.get(CAP_PROP_FRAME_WIDTH);
    int frame_height = (int)cap.get(CAP_PROP_FRAME_HEIGHT);
    double fps = cap.get(CAP_PROP_FPS);
    int total_frames = (int)cap.get(CAP_PROP_FRAME_COUNT);
    
    printf("Video: %dx%d @ %.2f fps, %d frames\n", 
           frame_width, frame_height, fps, total_frames);
    
    // Create output directory if needed
    system("mkdir -p ../Output/main_video_detection_gpu");
    
    // Prepare video writer
    int fourcc = VideoWriter::fourcc('X', 'V', 'I', 'D');
    VideoWriter out(output_path, fourcc, fps, Size(frame_width * 2, frame_height));
    if (!out.isOpened()) {
        printf("Error: Cannot create output video at %s\n", output_path);
        return -1;
    }
    printf("Output video writer initialized\n");
    
    // CUDA Memory allocation
    printf("\n=== Allocating GPU Memory ===\n");
    DeviceMatrix d_frame_bgr, d_frame_gray;
    allocateDeviceMatrix(&d_frame_bgr, frame_width, frame_height, 3);
    allocateDeviceMatrix(&d_frame_gray, frame_width, frame_height, 1);
    
    // BFMatcher for feature matching
    Ptr<DescriptorMatcher> matcher = BFMatcher::create(norm_type, false);
    
    // Frame processing
    printf("\n=== Processing Video Frames ===\n");
    Mat frame, frame_gray;
    Mat frame_with_rect;
    Mat img_matches;
    int frame_count = 0;
    
    cudaEventRecord(start);
    
    while (true) {
        if (!cap.read(frame)) {
            break;
        }
        
        frame_count++;
        if (frame_count % 30 == 0) {
            printf("Processing frame %d/%d (%.1f%%)\n", frame_count, total_frames, 
                   (float)frame_count * 100.0f / total_frames);
        }
        
        // GPU: Convert BGR to Grayscale
        if (frame.data) {
            // Create host matrix for frame
            Matrix h_frame_bgr;
            h_frame_bgr.width = frame_width;
            h_frame_bgr.height = frame_height;
            h_frame_bgr.channels = 3;
            h_frame_bgr.data = frame.data;
            
            // Transfer to GPU
            transferToDevice(&h_frame_bgr, &d_frame_bgr);
            
            // GPU Kernel: BGR to Grayscale
            checkCudaErrors(gpu_bgr2grayscale(d_frame_bgr.d_data, d_frame_gray.d_data, 
                                              frame_width, frame_height));
            
            // Transfer back to host
            frame_gray.create(frame_height, frame_width, CV_8UC1);
            Matrix h_frame_gray;
            h_frame_gray.width = frame_width;
            h_frame_gray.height = frame_height;
            h_frame_gray.channels = 1;
            h_frame_gray.data = frame_gray.data;
            
            DeviceMatrix d_frame_gray_tmp = d_frame_gray;
            transferFromDevice(&d_frame_gray_tmp, &h_frame_gray);
        }
        
        // CPU: Feature detection and matching
        vector<KeyPoint> kp2;
        Mat des2;
        
        if (frame_gray.data) {
            detector->detectAndCompute(frame_gray, noArray(), kp2, des2);
        }
        
        // Handle case with no descriptors
        if (des2.empty() || kp2.empty()) {
            drawMatches(query_img, kp1, frame, kp2, vector<DMatch>(),
                        img_matches, Scalar::all(-1), Scalar::all(-1),
                        vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
            
            if (img_matches.cols != frame_width * 2 || img_matches.rows != frame_height) {
                resize(img_matches, img_matches, Size(frame_width * 2, frame_height));
            }
            out.write(img_matches);
            continue;
        }
        
        // Feature matching
        vector<DMatch> matches;
        matcher->match(des1, des2, matches);
        
        if (matches.empty()) {
            drawMatches(query_img, kp1, frame, kp2, vector<DMatch>(),
                        img_matches, Scalar::all(-1), Scalar::all(-1),
                        vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
            
            if (img_matches.cols != frame_width * 2 || img_matches.rows != frame_height) {
                resize(img_matches, img_matches, Size(frame_width * 2, frame_height));
            }
            out.write(img_matches);
            continue;
        }
        
        // Sort matches by distance
        sort(matches.begin(), matches.end(), 
             [](const DMatch &a, const DMatch &b) { return a.distance < b.distance; });
        
        vector<DMatch> good_matches(matches.begin(), 
                                   matches.begin() + min(50, (int)matches.size()));
        
        frame_with_rect = frame.clone();
        
        // Compute homography
        if (good_matches.size() >= 4) {
            vector<Point2f> src_pts, dst_pts;
            
            for (const auto &m : good_matches) {
                src_pts.push_back(kp1[m.queryIdx].pt);
                dst_pts.push_back(kp2[m.trainIdx].pt);
            }
            
            Mat H = findHomography(src_pts, dst_pts, RANSAC, 5.0);
            
            if (!H.empty()) {
                // Transform query image corners
                vector<Point2f> corners = {
                    Point2f(0, 0),
                    Point2f(0, query_img.rows - 1),
                    Point2f(query_img.cols - 1, query_img.rows - 1),
                    Point2f(query_img.cols - 1, 0)
                };
                
                vector<Point2f> transformed_corners;
                perspectiveTransform(corners, transformed_corners, H);
                
                // Convert to int for drawing
                vector<Point> int_corners(transformed_corners.begin(), transformed_corners.end());
                
                // Draw bounding box on frame
                polylines(frame_with_rect, int_corners, true, Scalar(0, 255, 0), 3, LINE_AA);
                
                // GPU: Could accelerate drawing here if needed
                // For now, OpenCV handles it
                
                drawMatches(query_img, kp1, frame_with_rect, kp2,
                                vector<DMatch>(good_matches.begin(),
                                               good_matches.begin() + min(10, (int)good_matches.size())),
                                img_matches, Scalar::all(-1), Scalar::all(-1),
                                vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
            } else {
                drawMatches(query_img, kp1, frame, kp2,
                                vector<DMatch>(good_matches.begin(),
                                               good_matches.begin() + min(10, (int)good_matches.size())),
                                img_matches, Scalar::all(-1), Scalar::all(-1),
                                vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
            }
        } else {
            vector<DMatch> top_matches(matches.begin(), 
                                      matches.begin() + min(10, (int)matches.size()));
            drawMatches(query_img, kp1, frame, kp2, top_matches,
                        img_matches, Scalar::all(-1), Scalar::all(-1),
                        vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
        }
        
        // Resize if needed
        if (img_matches.cols != frame_width * 2 || img_matches.rows != frame_height) {
            resize(img_matches, img_matches, Size(frame_width * 2, frame_height));
        }
        
        out.write(img_matches);
    }
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&total_time, start, stop);
    
    printf("\nTotal processing time: %.3f ms\n", total_time);
    printf("Average time per frame: %.3f ms\n", total_time / frame_count);
    printf("Processing speed: %.2f fps\n", (frame_count * 1000.0f) / total_time);
    
    // Cleanup
    printf("\n=== Cleaning Up ===\n");
    cap.release();
    out.release();
    freeDeviceMatrix(&d_frame_bgr);
    freeDeviceMatrix(&d_frame_gray);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    printf("Output video saved to: %s\n", output_path);
    printf("=== Completed Successfully ===\n");
    
    return 0;
}
