import cv2, os, time, math
import numpy as np
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt

def compute_optical_flow(prev_gray, curr_gray):
    flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    #print(f"DEBUG : max and min values are {np.max(magnitude)} {np.min(magnitude)}")
    return np.max(magnitude)

def compute_orb_distance(prev_frame, curr_frame, match_threshold = 40):
    # Initialize ORB detector
    orb = cv2.ORB_create()

    # Find the keypoints and descriptors with ORB
    kp1, des1 = orb.detectAndCompute(prev_frame, None)
    kp2, des2 = orb.detectAndCompute(curr_frame, None)

    # Create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Match descriptors
    orig_matches = bf.match(des1, des2)

    matches = [match for match in orig_matches if match.distance < match_threshold]

    # Sort them in the order of their distance (descriptor similarity)
    matches = sorted(matches, key=lambda x: x.distance)

    # Calculate average descriptor distance of top 10% matches
    num_matches = len(matches)  # Use 10% of matches
    if num_matches == 0:
        return 0

    max_descriptor_distance = max(match.distance for match in matches[:num_matches])

    # Calculate Euclidean distances (physical movement) for top matches
    euclidean_distances = []
    for match in matches[:num_matches]:
        # Get keypoint coordinates from both frames
        pt1 = np.array(kp1[match.queryIdx].pt)  # Coordinates in prev_frame
        pt2 = np.array(kp2[match.trainIdx].pt)  # Coordinates in curr_frame

        # Compute Euclidean distance between matched keypoints
        euclidean_distance = np.sqrt((pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2)
        #print(f"DEBUG!! euclidean_distance is {euclidean_distance} between {pt1} and {pt2}")
        euclidean_distances.append(euclidean_distance)

    # Average Euclidean distance (keypoint movement)
    max_movement_distance = np.max(euclidean_distances)

    # Normalize max descriptor distance (for 256-bit ORB descriptors)
    normalized_descriptor_distance = max_descriptor_distance / 256

    # Return both descriptor similarity and keypoint movement
    #print(f"DEBUG!! max_descriptor_distance : {max_descriptor_distance}")
    return max_movement_distance


def compute_ssim(prev_frame, curr_frame):
    return ssim(prev_frame, curr_frame, data_range=255)

def compute_pixel_diff(prev_frame, curr_frame):
    diff = cv2.absdiff(prev_frame, curr_frame)
    return np.mean(diff)

def preprocess_frame(frame, width=640, height=360):
    target_size = (width, height)
    resized_frame = cv2.resize(frame, target_size, interpolation=cv2.INTER_AREA)  # Use INTER_AREA for shrinking
    return resized_frame

def smooth_curve(data, window_size=5):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

def find_timestamp_clusters(fast_motion_timestamps, min_time_gap=5):
    clusters = []  # List to hold the clusters of timestamps
    current_cluster = []  # Temporary list to hold the current cluster

    for i, timestamp in enumerate(fast_motion_timestamps):
        # If it's the first timestamp, start a new cluster
        if i == 0:
            current_cluster.append(timestamp)
        else:
            # Check the time difference between the current and previous timestamp
            if timestamp - fast_motion_timestamps[i-1] <= min_time_gap:
                # If the difference is less than or equal to the min_time_gap, add it to the current cluster
                current_cluster.append(timestamp)
            else:
                # If the difference is greater than min_time_gap, finish the current cluster and start a new one
                clusters.append(current_cluster)
                current_cluster = [timestamp]
    
    # Add the last cluster to the clusters list
    if current_cluster:
        clusters.append(current_cluster)
    
    return clusters


def detect_fast_motion(video_path, output_dir, end_time, start_time, window_size=3, motion_threshold=0.6, step = 2):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)

    orb_scores = []
    #optical_flow_scores = []
    ssim_scores = []
    #pixel_diff_scores = []
    timestamps = []
    frame_list = []
    
    prev_frame = None
    frame_count = 0

    while cap.isOpened():
        ret, orig_frame = cap.read()
        if not ret:
            break
        #print(f"DEBUG!! frame : {frame_count} time : {frame_count/fps}")

        if height == 360 and width == 640:
            frame = orig_frame
        else:
            frame = preprocess_frame(orig_frame, width = 640, height = 360)


        if frame_count > end_time * fps:
            break

        if frame_count < start_time * fps or frame_count % step != 0:
            frame_count += 1
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if prev_frame is not None:
            #optical_flow_scores.append(compute_optical_flow(prev_frame, gray))
            orb_scores.append(compute_orb_distance(prev_frame, gray))
            ssim_scores.append(compute_ssim(prev_frame, gray))
            #pixel_diff_scores.append(compute_pixel_diff(prev_frame, gray))
            #print(f"DEBUG : time : {frame_count/fps} end_time : {end_time} start_time : {start_time}")
            timestamps.append(frame_count/fps)
        else:
            #optical_flow_scores.append(0)
            orb_scores.append(0)
            ssim_scores.append(1)
            timestamps.append(start_time)
            
        frame_list.append(frame)
        prev_frame = gray
        frame_count += 1
        
        #if frame_count % 100 == 0:
        #    print(f"Processed {frame_count} frames")
    
    cap.release()
    
    new_fps = len(timestamps)/ (max(timestamps) - min(timestamps))
    print(f"fps : {fps} frame_height : {height} frame_width : {width} New fps is {new_fps}")
    # Normalize scores by image diagonal * time between frame : https://chatgpt.com/share/66f684b9-dd4c-8010-bf9c-421c3c6ef84a

    #optical_flow_scores = np.array(optical_flow_scores) / (np.sqrt(gray.shape[0]**2 + gray.shape[1]**2) / new_fps)
    ssim_scores = (1 - np.array(ssim_scores)) * new_fps   # Invert SSIM scores
    orb_scores = (np.array(orb_scores) * new_fps)/(np.sqrt(640**2 + 360**2))

    # Smooth both SSIM and ORB scores
    smoothed_ssim_scores = smooth_curve(ssim_scores, window_size=window_size)
    smoothed_orb_scores = smooth_curve(orb_scores, window_size=window_size)

    #pixel_diff_scores = np.array(pixel_diff_scores) / np.max(pixel_diff_scores)
    
    # Combine metrics
    combined_scores = (0.3 * orb_scores) + (0.7 * ssim_scores)
    smoothed_combined_scores = (0.3 * smoothed_orb_scores) + (0.7 * smoothed_ssim_scores)

    # Adjust X-axis to reflect the center of the window used for smoothing
    adjusted_timestamps = timestamps[window_size // 2 : -(window_size // 2)]

    # Detect fast motion using sliding window
    fast_motion_timestamps = []
    fast_motion_frames = []
    fast_motion_mags = []

    #for i in range(len(combined_scores) - window_size + 1):
    #    window = combined_scores[i:i + window_size]
    #    if np.mean(window) > motion_threshold:
    #        #print(f"DEBUG!! mean : {np.mean(window)} i : {i + (start_time * fps)} i+window_size : {i+window_size + (start_time * fps)} window : {window}")
    #        #fast_motion_frames.extend(range(i + int(start_time * fps), i + window_size + int(start_time * fps)))
    #        fast_motion_mags.extend(combined_scores[i:i + window_size])
    #        fast_motion_timestamps.extend(timestamps[i:i + window_size])

    ids = []
    for i in range(len(combined_scores)):
        if combined_scores[i] > motion_threshold:
            fast_motion_mags.append(combined_scores[i])
            fast_motion_timestamps.append(timestamps[i])
            fast_motion_frames.append(frame_list[i])
            ids.append(i)

    padded_fast_motion_frames = []
    padded_fast_motion_timestamps = []
    
    if len(ids) < 5 and len(ids) > 0:
        #Padding fast_motion_frames and fast_motion_timestamps
        padded_fast_motion_frames.extend(frame_list[min(ids) - 2:min(ids)])
        padded_fast_motion_timestamps.extend(timestamps[min(ids) - 2:min(ids)])
        
        padded_fast_motion_frames.extend(fast_motion_frames)
        padded_fast_motion_timestamps.extend(fast_motion_timestamps)
        
        padded_fast_motion_frames.extend(frame_list[max(ids) + 1:max(ids) + 3])
        padded_fast_motion_timestamps.extend(timestamps[max(ids) + 1:max(ids) + 3])
        print(f"padded_fast_motion_timestamps are {padded_fast_motion_timestamps}. Length of padded_fast_motion_timestamps is {len(padded_fast_motion_frames)}")
    else:
        padded_fast_motion_frames = fast_motion_frames
        padded_fast_motion_timestamps = fast_motion_timestamps

    # Plot results
    plt.figure(figsize=(12, 6))
    plt.plot(adjusted_timestamps, smoothed_orb_scores, label='ORB Distance')
    plt.plot(adjusted_timestamps, smoothed_ssim_scores, label='Inverted SSIM')
    #plt.plot(adjusted_timestamps, optical_flow_scores, label='Optical Flow')
    plt.plot(adjusted_timestamps, smoothed_combined_scores, label='Combined Score')
    plt.axhline(y=motion_threshold, color='r', linestyle='--', label='Threshold')
    plt.xlabel('Frame')
    plt.ylabel('Normalized Score')
    plt.title('Motion Detection Metrics')
    plt.legend()
    plt.savefig(f"{output_dir}/motion_detection_plot_smoothened_{video_path.split('/')[-1].split('.')[0]}.png")

        # Plot results
    plt.figure(figsize=(12, 6))
    #plt.plot(timestamps, orb_scores, label='ORB Distance')
    plt.plot(timestamps, ssim_scores, label='Inverted SSIM')
    #plt.plot(timestamps, optical_flow_scores, label='Optical Flow')
    plt.plot(timestamps, combined_scores, label='Combined Score')
    plt.axhline(y=motion_threshold, color='r', linestyle='--', label='Threshold')
    plt.xlabel('Frame')
    plt.ylabel('Normalized Score')
    plt.title('Motion Detection Metrics')
    plt.legend()
    plt.savefig(f"{output_dir}/motion_detection_plot_raw_{video_path.split('/')[-1].split('.')[0]}.png")

    
    # Print results
    print(f"Max motion score is {np.max(combined_scores)} and mean motion score is {np.mean(combined_scores)} from {np.min(timestamps)} to {np.max(timestamps)}")
    print(f"Detected {len(fast_motion_timestamps)} frames when step = {step}.")
    try:
        print(f"fast motion between {np.min(fast_motion_timestamps)} and {np.max(fast_motion_timestamps)}")
    except:
        pass

    #for i in range(len(fast_motion_timestamps)):
    #    timestamp = fast_motion_timestamps[i]
    #    mag = fast_motion_mags[i]
    #    print(f"(Time: {timestamp:.2f}s) (Magnitude : {mag:.2f})")
    
    if len(fast_motion_timestamps) == 0:
        print("FAST MOTION NOT DETECTED!")
        return [], []
    elif len(fast_motion_timestamps) > 0.5 * len(combined_scores):
        print("More than half of the video has fast motion")
        return fast_motion_timestamps, padded_fast_motion_frames
    else:
        timestamp_clusters = find_timestamp_clusters(fast_motion_timestamps, min_time_gap = 5)
        for timestamp_cluster in timestamp_clusters:
            print(f"min time : {np.min(timestamp_cluster)} max time : {np.max(timestamp_cluster)} length : {len(timestamp_cluster)}")
        return timestamp_clusters, padded_fast_motion_frames


'''
# Open the video file
video_path = "../test_videos/"
mp4_files = [f for f in os.listdir(video_path) if f.endswith('.mp4')]
output_dir = "motion_detection_results"
os.system(f"rm -rf {output_dir}")
os.system(f"mkdir {output_dir}")
end_time = 15
start_time = 0

for mp4_file in mp4_files:
    print(f"\nAnalyzing video {mp4_file}")

    if mp4_file == "8.mp4":
        end_time = 60
        start_time = 0
    elif mp4_file == "6.mp4":
        end_time = 32
        start_time = 0
    elif mp4_file == "3.mp4":
        end_time = 6.5 #To remove last few frames that are blurry
        start_time = 0
    elif mp4_file == "2.mp4":
        end_time = 182
        start_time = 140
    else:
        end_time = 15
        start_time = 0

    #if mp4_file != "3.mp4" and mp4_file != "5.mp4" and mp4_file != "6.mp4":
    #    continue

    start = time.time()
    fast_motion_timestamps = detect_fast_motion(video_path + mp4_file, output_dir, end_time, start_time, motion_threshold = 1.5)
    end = time.time()

    print(f"Execution time for {mp4_file} : {end - start} seconds. Duration of the video was {end_time - start_time} seconds")
'''