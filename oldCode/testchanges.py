import cv2
import numpy as np


# Open video stream
cap = cv2.VideoCapture(0)
first = True
while cap.isOpened():
    ret, frame = cap.read()
    if first:
        reference_image = frame
        first = False
    if not ret:
        break

    # Resize frame to match the reference image size
    frame_resized = cv2.resize(frame, (reference_image.shape[1], reference_image.shape[0]))

    # Compute Mean Squared Error (MSE) between reference and current frame
    mse = np.sum((reference_image.astype("float") - frame_resized.astype("float")) ** 2)
    mse /= float(reference_image.shape[0] * reference_image.shape[1])

    # Define a threshold for highlighting changes
    threshold = 10000

    # Highlight changes if MSE is above the threshold
    if mse > threshold:
        # Draw a rectangle around the differing region
        cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 255), 2)

    # Display the result
    cv2.imshow('Video with Changes Highlighted', frame)

    if cv2.waitKey(30) & 0xFF == 27:  # Press 'Esc' to exit
        break

cap.release()
cv2.destroyAllWindows()