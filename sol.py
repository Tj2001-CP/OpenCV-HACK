# Given a CCTV camera video, detect persons and capture their images if they fall in ROI.

import json
import cv2

# Load the JSON input
with open("cam_config.json") as json_file:
    data = json.load(json_file)

# Get the video source and ROI coordinates
# Get the video source
video_source = data["siteConfig"]["cameraConfig"]["201"]["rtspServer"]

# Get the ROI coordinates
roi_coords = data["siteConfig"]["cameraConfig"]["201"]["roi"]

# Create a video capture object
cap = cv2.VideoCapture(video_source)

# Get the video dimensions
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

# Define the codec and create a video writer object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (frame_width, frame_height))

# Load the face detection classifier
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

while True:
    # Read a frame from the video
    ret, frame = cap.read()

    # Check if a frame was read successfully
    if not ret:
        break

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Use a HOG descriptor to detect persons in the frame
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    boxes, weights = hog.detectMultiScale(gray, winStride=(8, 8))

    # Draw bounding boxes around detected persons
    for (x, y, w, h) in boxes:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)

        # Check if the bounding box intersects with the ROI
        if (x > roi_coords["x1"] and x < roi_coords["x2"]) and (y > roi_coords["y1"] and y < roi_coords["y2"]):
            # Capture an image of the person
            person_image = frame[y:y+h, x:x+w]
            cv2.imwrite("person_in_roi.jpg", person_image)
        else:
            # Redact the face of the person
            face_rects = face_cascade.detectMultiScale(
                gray[y:y+h, x:x+w], scaleFactor=1.1, minNeighbors=5)
            for (fx, fy, fw, fh) in face_rects:
                face_roi = gray[y+fy:y+fy+fh, x+fx:x+fx+fw]
                face_roi = cv2.GaussianBlur(face_roi, (23, 23), 30)
                frame[y+fy:y+fy+fh, x+fx:x+fx+fw] = face_roi

    # Write the frame to the output video
    out.write(frame)

    # Display the frame
    cv2.imshow("Video", frame)

    # Exit the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# Release the video capture object and close all windows
# cap.release()
# out.release()
# cv2.destroyAllWindows()
# this code uses HOG descriptor but you can use other object detection algorithm like YOLO, etc.
