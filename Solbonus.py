# solution to bonus question:-Track pedestrians and determine where they left from the region of interest(ROI).
import json
from turtle import tracer
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

    # Initialize the list of detected faces
faces = []

# Draw bounding boxes around detected persons
for (x, y, w, h) in boxes:
    # Check if the detected person is within the ROI
    if (x, y) in roi_coords:
        # Add the face to the list of detected faces
        faces.append((x, y, w, h))
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Update the MultiTracker object with the list of detected faces
    tracer.update(frame, faces)

    # Get the list of trackers
    trackers = tracker.getObjects()

    # Iterate over the list of trackers
    for i in range(len(trackers)):
        # Get the current tracker
        (x, y, w, h) = trackers[i]

        # Check if the tracker is outside of the ROI
        if (x, y) not in roi_coords:
            # Draw a red rectangle around the tracker
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            print("Pedestrian left from the ROI at frame: ", cap.get(1))

    # Write the output frame to the video writer object
    out.write(frame)

    # Display the output frame
    cv2.imshow("Output", frame)

    # Check if the user pressed the 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Release the video capture and video writer objects
    # cap.release()
    # out.release()

    # Close all windows
    # cv2.destroyAllWindows()
