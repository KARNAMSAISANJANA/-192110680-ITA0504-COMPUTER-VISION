import cv2 as cv
import os

def detect_watch(image_path, cascade_path):
    # Load the cascade classifier
    watch_cascade = cv.CascadeClassifier(cascade_path)
    if watch_cascade.empty():
        print(f"Error: Failed to load cascade classifier from '{cascade_path}'.")
        return

    # Read the image
    img = cv.imread(image_path)
    if img is None:
        print(f"Error: Image not found at '{image_path}'.")
        return

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Perform detection
    watches = watch_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    if len(watches) == 0:
        print("No watches detected.")
        return

    for (x, y, w, h) in watches:
        cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv.imshow("Detected Watches", img)
    cv.waitKey(0)
    cv.destroyAllWindows()

if __name__ == "__main__":
    image_path = r"C:\Users\Gayathri\Downloads\sample.jpeg"
    cascade_path = cv.data.haarcascades + "haarcascade_frontalface_default.xml"
    detect_watch(image_path, cascade_path)

