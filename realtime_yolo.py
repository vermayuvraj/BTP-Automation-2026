from ultralytics import YOLO
import cv2

# Load the YOLOv8 pre-trained model
model = YOLO('yolov8n.pt')  # 'n' stands for nano, a lightweight version

# Initialize webcam
cap = cv2.VideoCapture(0)  # Use 0 for the default webcam

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Process webcam frames in real-time
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break

    # Run YOLOv8 inference on the frame
    results = model(frame, show=False)  # Set show=False to prevent ultralytics' default window

    # Annotate the frame with detected objects
    annotated_frame = results[0].plot()

    # Display the annotated frame
    cv2.imshow("YOLOv8 Real-Time Detection", annotated_frame)

    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
