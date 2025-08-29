import cv2 
def reverse_video(input_video_path, output_video_path): 
    # Open the video file 
    cap = cv2.VideoCapture(input_video_path) 
    # Get video properties 
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) 
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) 
    fps = int(cap.get(cv2.CAP_PROP_FPS)) 
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) 
    # Define video writer 
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Codec for MP4 
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height)) 
    frames = [] 
    # Read all frames 
    while True: 
        ret, frame = cap.read() 
        if not ret: 
            break 
        frames.append(frame) 
    cap.release() 
    # Reverse frames 
    frames.reverse() 
    # Write frames in reverse order 
    for frame in frames: 
        out.write(frame) 
    out.release() 
    print(f"Reversed video saved as: {output_video_path}") 
# Example usage 
reverse_video("input_video.mp4", "output_reversed.mp4") 
