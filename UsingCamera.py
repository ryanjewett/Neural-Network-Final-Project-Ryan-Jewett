import cv2
import numpy as np
from tensorflow.keras.models import load_model
from sense_hat import SenseHat

# Load the trained model
model = load_model("path/to/saved/model.h5")

# Define a dictionary to map predicted class indices to numbers
class_to_number = {
    0: "0", 1: "1", 2: "2", 3: "3", 4: "4",
    5: "5", 6: "6", 7: "7", 8: "8", 9: "9"
}

# Initialize the sense_hat
sense = SenseHat()

# Open the camera
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Preprocess the frame
    frame_resized = cv2.resize(frame, (64, 64))
    frame_gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
    frame_normalized = frame_gray / 255.0
    frame_expanded = np.expand_dims(frame_normalized, axis=0)
    frame_expanded = np.expand_dims(frame_expanded, axis=3)

    # Predict the number
    prediction = model.predict(frame_expanded)
    predicted_class = np.argmax(prediction)

    # Display the inferred number on sense_hat
    number = class_to_number[predicted_class]
    sense.show_letter(number)

    # Display the frame
    cv2.imshow('frame', frame)

    # Exit loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()
