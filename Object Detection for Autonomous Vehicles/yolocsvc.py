import cv2
import numpy as np
import csv
from datetime import timedelta
import time
# Load the class labels
labelsPath = "coco.names"
LABELS = open(labelsPath).read().strip().split("\n")

# Initialize colors for each class label
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")

# Paths to the YOLOv3-tiny weights and config file
weightsPath = "yolov3-tiny.weights"
configPath = "yolov3-tiny.cfg"

# Load YOLOv3-tiny model
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

# Get names of all layers, then extract only the names of the unconnected output layers
layer_names = net.getLayerNames()
unconnected_out_layers = net.getUnconnectedOutLayers().flatten()
ln = [layer_names[i - 1] for i in unconnected_out_layers]

# Load the video file
cap = cv2.VideoCapture("video.mp4")
fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

detections = []
a=[]
while True:
    s=time.time()
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break

    current_frame_number = cap.get(cv2.CAP_PROP_POS_FRAMES)
    timestamp = str(timedelta(seconds=current_frame_number/fps))

    # Get dimensions of the frame
    (H, W) = frame.shape[:2]

    # Construct a blob from the input frame and then perform a forward pass of the YOLO object detector
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layerOutputs = net.forward(ln)

    boxes = []
    confidences = []
    classIDs = []

    # Loop over each of the layer outputs
    for output in layerOutputs:
        # Loop over each of the detections
        for detection in output:
            # Extract the class ID and confidence
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            # Filter out weak predictions by checking if the detected object is a car
            if confidence > 0.5 and LABELS[classID].lower() == "car":
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

                # Save the detection with timestamp, classID, and confidence
                detections.append([timestamp, LABELS[classID], confidence])

    # Apply non-maxima suppression to suppress weak, overlapping bounding boxes
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    if len(idxs) > 0:
        for i in idxs.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            color = [int(c) for c in COLORS[classIDs[i]]]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            text = "{}: {:.2f}".format(LABELS[classIDs[i]], confidences[i])
            cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            a.append(confidences[i])
            

    # Display the resulting frame
    cv2.imshow('Frame', frame)
    
    # Press Q on keyboard to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
e = time.time()
total_time = s-e

# Release video capture object
cap.release()

print(sum(a)/len(a))
print(total_time)

# Close all OpenCV windows
cv2
