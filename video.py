import cv2
import pafy
import torch
import numpy as np

# Replace with the URL of the YouTube video
video_url = 'https://www.youtube.com/watch?v=tm2lfv3_ELc&list=PPSV'

# Load the YouTube video using pafy
video = pafy.new(video_url)
play = video.getbest(preftype="mp4")
video_capture = cv2.VideoCapture(play.url)

# Load the YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5m', pretrained=True)

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    # Perform inference on the frame
    results = model(frame)

    # Convert prediction results to NumPy arrays for indexing
    pred_people = results.pred[0][(results.pred[0][:, -1] == 0) & (results.pred[0][:, 4] > 0.3)].cpu().detach().numpy()  # Adjust confidence threshold for people
    pred_cellphones = results.pred[0][(results.pred[0][:, -1] == 67) & (results.pred[0][:, 4] > 0.3)].cpu().detach().numpy()  # Replace 67 with the correct class index for cellphones and adjust confidence threshold

    # List to store people holding phones to their ears
    people_with_phone_to_ear = []

    # Find phones close to people's faces
    for person in pred_people:
        person_bbox = person[:4]
        person_marked = False
        for cellphone in pred_cellphones:
            cellphone_bbox = cellphone[:4]
            # Check if the phone's bbox is reasonably close to the person's head area
            # You might adjust this logic based on the relative position of phone and face
            # Here, considering if the phone's X-center is within person's bounding box
            x_center = (cellphone_bbox[0] + cellphone_bbox[2]) / 2
            if person_bbox[0] < x_center < person_bbox[2]:
                people_with_phone_to_ear.append(person)  # Store people with phones near faces
                break  # Exit the loop after finding a phone near a person's head

    # Draw bounding boxes for people holding phones near faces
    for person in people_with_phone_to_ear:
        bbox = person[:4].astype(int).tolist()  # Convert bbox to list
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)  # Green rectangle

    # Display the frame with detected people holding phones near faces
    cv2.imshow('Detected People with Phones Near Faces', frame)

    # Reduce delay (waitKey) to 1 millisecond
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close any open windows
video_capture.release()
cv2.destroyAllWindows()