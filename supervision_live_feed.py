import cv2
import supervision as sv
from ultralytics import YOLO
import tkinter as tk
from PIL import Image, ImageTk
import numpy as np

# Initialize the YOLO model
model = YOLO('yolov8x.pt')

# Open the webcam
cap = cv2.VideoCapture(0)

# Check if webcam opened successfully
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Tkinter setup
window = tk.Tk()
window.title("Object Detection")

# Create a label in the window to display the video feed
label = tk.Label(window)
label.pack()

def initialize_kalman():
    kalman = cv2.KalmanFilter(4, 2)
    kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
    kalman.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
    kalman.processNoiseCov = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32) * 0.03
    return kalman

# Initialize Kalman filter
kalman = initialize_kalman()

def update_frame():
    ret, frame = cap.read()
    if ret:
        # Flip the frame horizontally
        frame = cv2.flip(frame, 1)
        
        # Run the model on the frame
        results = model(frame, classes=[0])[0]
        
        # Convert the results to Supervision Detections format
        detections = sv.Detections.from_ultralytics(results)
        
        # Priority logic to determine the largest bounding box
        largest_box_area = 0
        largest_box_index = -1
        for i, (bbox, confidence, class_name) in enumerate(zip(detections.xyxy, detections.confidence, detections.data['class_name'])):
            x1, y1, x2, y2 = bbox
            box_area = (x2 - x1) * (y2 - y1)
            if box_area > largest_box_area:
                largest_box_area = box_area
                largest_box_index = i
        
        # If a valid largest box is found, update Kalman filter
        if largest_box_index != -1:
            x1, y1, x2, y2 = detections.xyxy[largest_box_index]
            fcx, fcy = int((x1 + x2) / 2), int((y1 + y2) / 2)
            kalman.correct(np.array([[np.float32(fcx)], [np.float32(fcy)]]))
        else:
            fcx, fcy = -1, -1  # Invalid position when no bounding box is found
        
        # Predict the next position using the Kalman filter
        predicted = kalman.predict()
        predicted_fcx, predicted_fcy = int(predicted[0]), int(predicted[1])
        
        # Draw the detections on the frame
        for i, (bbox, confidence, class_name) in enumerate(zip(detections.xyxy, detections.confidence, detections.data['class_name'])):
            x1, y1, x2, y2 = bbox
            
            # Determine color and size for the crosshair based on priority
            if i == largest_box_index:
                t = 2  # thickness of crosshair
                size = 40  # size of crosshair
                crosshair_color = (0, 0, 255)  # Red for largest bounding box
            else:
                t = 1  # thickness of crosshair
                size = 20  # size of crosshair
                crosshair_color = (0, 255, 0)  # Green for other bounding boxes
            
            # Draw the bounding box
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            
            # Draw the label with confidence score if confidence is not None
            if confidence is not None:
                label_text = f"{class_name}: {confidence:.2f}"
                cv2.putText(frame, label_text, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Draw crosshair on the frame
            fcx, fcy = int((x1 + x2) / 2), int((y1 + y2) / 2)
            cv2.drawMarker(frame, (fcx, fcy), crosshair_color, markerType=cv2.MARKER_CROSS, markerSize=size, thickness=t)
            cv2.circle(frame, (fcx, fcy), radius=10, color=crosshair_color, thickness=t)
        
        # Draw the Kalman filter crosshair on the frame
        cv2.drawMarker(frame, (predicted_fcx, predicted_fcy), (255, 0, 0), markerType=cv2.MARKER_CROSS, markerSize=40, thickness=2)
        cv2.circle(frame, (predicted_fcx, predicted_fcy), radius=10, color=(255, 0, 0), thickness=2)

        # Convert the frame to an ImageTk format and update the label
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        imgtk = ImageTk.PhotoImage(image=img)
        label.imgtk = imgtk
        label.configure(image=imgtk)
    
    # Repeat after 10 milliseconds
    window.after(10, update_frame)

# Start the video loop
update_frame()
window.mainloop()

# Release resources
cap.release()
cv2.destroyAllWindows()
