
import cv2
import numpy as np
import matplotlib.pyplot as plt

def analyze_histogram(image_path):
    # Load the image (in BGR by default)
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Unable to load image at {image_path}")
    
    # Convert BGR to RGB for accurate color display in Matplotlib
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    # Prepare to plot
    plt.figure(figsize=(14, 6))
    
    # Display the image
    plt.subplot(1, 2, 1)
    plt.imshow(img_rgb)
    plt.title("Original Image (RGB)")
    plt.axis('off')
    
    # Compute and plot histograms for each channel
    plt.subplot(1, 2, 2)
    colors = ('r', 'g', 'b')
    for i, col in enumerate(colors):
        hist = cv2.calcHist([img_rgb], [i], None, [256], [0, 256])
        plt.plot(hist, color=col, label=f"{col.upper()} channel")
    plt.title("Color Histogram")
    plt.xlabel("Pixel Intensity (0–255)")
    plt.ylabel("Frequency")
    plt.xlim([0, 256])
    plt.legend()
    
    plt.tight_layout()
    plt.show()

# Example usage:
analyze_histogram("C:\\Users\\Gayathri\\Downloads\\sample.jpeg")
