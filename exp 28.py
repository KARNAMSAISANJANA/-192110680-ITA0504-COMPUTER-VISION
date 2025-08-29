import cv2 
def detect_vehicles(video_path): 
    # Load the pre-trained vehicle classifier 
    vehicle_cascade = cv2.CascadeClassifier('cars.xml')  # Pre-trained car cascade XML 
    # Open video file 
    cap = cv2.VideoCapture(video_path) 
    while cap.isOpened(): 
        ret, frame = cap.read() 
        if not ret: 
            break  # Stop if video ends 
        # Convert frame to grayscale 
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
        # Detect vehicles 
        vehicles = vehicle_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, 
minSize=(50, 50)) 
        # Draw bounding boxes around detected vehicles 
        for (x, y, w, h) in vehicles: 
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2) 
        # Display the frame 
        cv2.imshow("Vehicle Detection", frame) 
        # Press 'q' to exit 
        if cv2.waitKey(30) & 0xFF == ord('q'): 
            break 
    cap.release() 
    cv2.destroyAllWindows() 
# Example usage 
detect_vehicles("traffic_video.mp4") 
