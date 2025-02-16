import cv2
import torch
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
from ultralytics import YOLO



model = YOLO("yolo11x.pt")


# Function to process the image and extract bounding boxes and labels
def process_image(image_path):
    """
    Takes an image file and performs object detection using YOLOv11.

    Args:
        image_path (str): Path to the local image file.

    Returns:
        results (list): A list of dictionaries with bounding box coordinates and labels.
    """
    # Load image
    image = Image.open(image_path)

    # Perform inference
    results = model(image)

    print(results[0].boxes)
    print(results[0].boxes.xyxy.shape)

    detections = []
    for i in range(len(results[0].boxes.xyxy)): 
        print("Boxes found here: ", len(results[0].boxes.xyxy)) 
        box = results[0].boxes.xyxy[i]
        cls = results[0].boxes.cls[i].item() 
        conf = results[0].boxes.conf[i].item()  
        label = model.names[int(cls)] 
        bbox = {
            "x1": int(box[0].item()),
            "y1": int(box[1].item()),
            "x2": int(box[2].item()),
            "y2": int(box[3].item()),
            "confidence": float(conf),
            "label": label,
        }
        detections.append(bbox)

    return detections

# Visualize results on the image
def visualize_detections(image_path, detections):
    """
    Draws bounding boxes and labels on the image and displays it.

    Args:
        image_path (str): Path to the local image file.
        detections (list): List of detection results with bounding box coordinates and labels.
    """
    # Load image
    image = cv2.imread(image_path)

    for detection in detections:
        x1, y1, x2, y2 = detection["x1"], detection["y1"], detection["x2"], detection["y2"]
        label = detection["label"]

        # Draw bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Convert BGR to RGB for displaying
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(image)
    plt.axis("off")
    plt.show()

if __name__ == "__main__":
    # Path to the local image file
    image_path = "/Users/daulettoibazar/Downloads/car.jpg"  # Replace with your image path

    if not Path(image_path).is_file():
        print(f"Image file not found: {image_path}")
    else:
        # Run object detection
        detections = process_image(image_path)

        print("Detections:")
        for detection in detections:
            print(detection)

        # Visualize detections on the image
        visualize_detections(image_path, detections)
