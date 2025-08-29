 
import cv2 
def count_faces(image_path): 
    # Load the image 
    image = cv2.imread(image_path) 
    # Convert image to grayscale 
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
    # Load the pre-trained Haar cascade face detection model 
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml') 
    # Detect faces in the image 
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)) 
    # Count the number of faces 
    face_count = len(faces) 
    print(f"Number of faces detected: {face_count}") 
    # Draw rectangles around detected faces 
    for (x, y, w, h) in faces: 
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2) 
    # Display the image with detected faces 
    cv2.imshow("Face Detection", image) 
    cv2.waitKey(0) 
    cv2.destroyAllWindows() 
    return face_count 
# Example usage 
image_path = "face_image.jpg"  # Replace with your image path 
count_faces(image_path)
