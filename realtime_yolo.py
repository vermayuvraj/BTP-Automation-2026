import cv2

# Initialize the video capture (webcam)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Set the video resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Initialize the MultiTracker
multi_tracker = cv2.MultiTracker_create()

# Initialize a bounding box (for object tracking)
bbox = None

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break

    # Initialize tracking when the user selects the object
    if bbox is None:
        # Select the bounding box for tracking (you can use cv2.selectROI() to manually select the object)
        bbox = cv2.selectROI("Frame", frame, fromCenter=False, showCrosshair=True)
        multi_tracker.add(cv2.TrackerCSRT_create(), frame, bbox)

    # Update the trackers and get the new bounding box
    success, boxes = multi_tracker.update(frame)

    if success:
        for box in boxes:
            x, y, w, h = [int(v) for v in box]
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)  # Draw the bounding box
        cv2.putText(frame, "Tracking Object", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    else:
        cv2.putText(frame, "Lost Track", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    cv2.imshow("Tracking", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
