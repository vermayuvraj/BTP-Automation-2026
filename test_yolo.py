from ultralytics import YOLO
import cv2

# Load the YOLOv8 pre-trained model
model = YOLO('yolov8n.pt')  # 'n' stands for nano, a lightweight version

# Run inference on an example image
results = model('https://ultralytics.com/images/bus.jpg')

# Display results for each result in the list
for result in results:
    # Get the annotated image from the `plot` method
    annotated_image = result.plot()

    # Display the annotated image using OpenCV
    cv2.imshow("YOLOv8 Results", annotated_image)
    cv2.waitKey(0)  # Wait for a key press to close the window
    cv2.destroyAllWindows()
