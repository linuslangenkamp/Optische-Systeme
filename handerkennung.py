#https://gist.github.com/nicolalandro/533589b2ee824cd567879cfdec9002ca
from math import sqrt

import cv2
import mediapipe as mp
import time
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Load your image
image = cv2.imread('archive/asl_alphabet_test/asl_alphabet_test/W_test.jpg')  # Replace 'your_image.jpg' with the path to your image

# Convert the image to RGB
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Process the image with MediaPipe Hands
results = hands.process(image_rgb)

# Check if landmarks are detected
if results.multi_hand_landmarks:
    for hand_landmarks in results.multi_hand_landmarks:
        # Iterate through each landmark point
        for landmark_id, landmark in enumerate(hand_landmarks.landmark):
            # Get the pixel coordinates of the landmark
            height, width, _ = image.shape
            cx, cy = int(landmark.x * width), int(landmark.y * height)

            # Draw a circle on the image at the landmark position
            cv2.circle(image, (cx, cy), 2, (0, 255, 0), -1)

# Display the result
cv2.imshow('Hand Landmarks Detection', image)
cv2.waitKey(0)
cv2.destroyAllWindows()