import cv2
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F

model = fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

# Use CUDA if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Load the video
cap = cv2.VideoCapture("video.mp4")
a=[]
with torch.no_grad():
    while True:
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
                cv2.putText(frame, f'{label}: {score:.2f}', (box[0], box[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                a.append(score)
        # Display the resulting frame
        cv2.imshow('Frame', frame)
        
        # Press Q on keyboard to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release video capture object
cap.release()
print(sum(a)/len(a))
# Close all OpenCV windows
cv2.destroyAllWindows()
