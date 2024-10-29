import cv2
import torch
from torchvision.models.detection import ssdlite320_mobilenet_v3_large  # Using SSD Lite with MobileNet v3
from torchvision.transforms import functional as F
import time

# Load a pre-trained SSD model
model = ssdlite320_mobilenet_v3_large(pretrained=True)
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Load the video
cap = cv2.VideoCapture("video.mp4")
a=[]
with torch.no_grad():
    while True:
        s=time.time()
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the image frame to a torch tensor
        image = F.to_tensor(frame).to(device).unsqueeze(0)

        # Perform inference
        predictions = model(image)

        # Process predictions
        for i, pred in enumerate(predictions):
            boxes = pred['boxes'].cpu().numpy().astype(int)
            labels = pred['labels'].cpu().numpy()
            scores = pred['scores'].cpu().numpy()

            # Filter out low confidence detections
            high_conf_indices = scores > 0.5
            boxes = boxes[high_conf_indices]
            labels = labels[high_conf_indices]
            scores = scores[high_conf_indices]

            for box, label, score in zip(boxes, labels, scores):
                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                label_text = f'{label}: {score:.2f}'  # You might want to map `label` to a human-readable class name
                cv2.putText(frame, label_text, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                a.append(score)
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
cv2.destroyAllWindows()