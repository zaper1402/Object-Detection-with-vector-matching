import cv2
import numpy as np

def track_features_in_stream(reference_image, stream_file, result_file='output_video', enable_sift=True):
    # Read the reference image in grayscale mode
    ref_img = cv2.imread(reference_image, cv2.IMREAD_GRAYSCALE)
    if ref_img is None:
        print("Failed to load reference image from specified path!")
        return

    # Set up feature detector (SIFT or ORB based on preference)
    if enable_sift:
        feature_extractor = cv2.SIFT_create()
        distance_metric = cv2.NORM_L2
    else:
        feature_extractor = cv2.ORB_create()
        distance_metric = cv2.NORM_HAMMING

    # Extract features from the reference image
    reference_keypoints, reference_descriptors = feature_extractor.detectAndCompute(ref_img, None)

    # Open video stream for processing
    video_stream = cv2.VideoCapture(stream_file)
    if not video_stream.isOpened():
        print("Unable to access the video stream!")
        return

    # Retrieve video stream parameters
    width_pixels = int(video_stream.get(cv2.CAP_PROP_FRAME_WIDTH))
    height_pixels = int(video_stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames_per_sec = video_stream.get(cv2.CAP_PROP_FPS)

    # Configure output video writer
    codec = cv2.VideoWriter_fourcc(*'XVID')
    video_writer = cv2.VideoWriter(f'{result_file}.mp4', codec, frames_per_sec, (width_pixels * 2, height_pixels))  # Combined width for visualization

    # Create brute-force matcher with selected distance metric
    feature_matcher = cv2.BFMatcher(distance_metric, crossCheck=True)

    # Iterate through video frames
    while True:
        frame_captured, current_frame = video_stream.read()
        if not frame_captured:
            break

        # Convert current frame to grayscale
        frame_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        frame_keypoints, frame_descriptors = feature_extractor.detectAndCompute(frame_gray, None)

        # Handle frames with no detected features
        if len(frame_keypoints) == 0 or frame_descriptors is None:
            combined_view = cv2.drawMatches(ref_img, reference_keypoints, current_frame, frame_keypoints, [], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            video_writer.write(combined_view)
            continue

        # Find matching features between reference and current frame
        feature_matches = feature_matcher.match(reference_descriptors, frame_descriptors)
        if len(feature_matches) == 0:
            combined_view = cv2.drawMatches(ref_img, reference_keypoints, current_frame, frame_keypoints, [], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            video_writer.write(combined_view)
            continue

        # Order matches by quality (ascending distance)
        feature_matches = sorted(feature_matches, key=lambda x: x.distance)
        top_matches = feature_matches[:50]  # Select best 50 matches for transformation calculation

        # Calculate perspective transformation when sufficient matches exist
        if len(top_matches)>=0 and len(top_matches) >= 4:
            reference_points = np.float32([reference_keypoints[match.queryIdx].pt for match in top_matches]).reshape(-1, 1, 2)
            frame_points = np.float32([frame_keypoints[match.trainIdx].pt for match in top_matches]).reshape(-1, 1, 2)
            transformation_matrix, inlier_mask = cv2.findHomography(reference_points, frame_points, cv2.RANSAC, 5.0)

            if transformation_matrix is not None:
                # Project reference image boundaries to frame coordinates
                ref_height, ref_width = ref_img.shape
                boundary_points = np.float32([[0, 0], [0, ref_height-1], [ref_width-1, ref_height-1], [ref_width-1, 0]]).reshape(-1, 1, 2)
                projected_boundary = cv2.perspectiveTransform(boundary_points, transformation_matrix).reshape(-1, 2)
                
                # Overlay detection boundary on current frame
                annotated_frame = current_frame.copy()
                cv2.polylines(annotated_frame, [np.int32(projected_boundary)], True, (0, 255, 0), 3, cv2.LINE_AA)
                # Visualize top 10 feature matches
                combined_view = cv2.drawMatches(ref_img, reference_keypoints, annotated_frame, frame_keypoints, top_matches[:10], None,
                                              flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            else:
                combined_view = cv2.drawMatches(ref_img, reference_keypoints, current_frame, frame_keypoints, top_matches[:10], None,
                                              flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        else:
            combined_view = cv2.drawMatches(ref_img, reference_keypoints, current_frame, frame_keypoints, feature_matches[:10], None,
                                          flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        # Adjust visualization dimensions if needed
        if combined_view.shape[0] != height_pixels or combined_view.shape[1] != width_pixels * 2:
            combined_view = cv2.resize(combined_view, (width_pixels * 2, height_pixels))
        
        video_writer.write(combined_view)

    # Clean up resources
    video_stream.release()
    video_writer.release()
    print(f"Processing complete. Result saved at {result_file}.mp4")

# Configuration and execution
reference_image = "../Images/object2.png"
stream_file = "../Videos/traffictrim.mp4"
result_file = "../output/output_video"
enable_sift = True  # Use False to switch to ORB detector

track_features_in_stream(reference_image, stream_file, result_file, enable_sift)
