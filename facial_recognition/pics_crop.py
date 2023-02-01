import cv2
import os

# Load the cascade classifier for face detection
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Specify the input and output folder paths
input_folder = "original"
output_folder = "cropped"

# Create the output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Loop through all the images in the input folder
for filename in os.listdir(input_folder):
    # Load the image
    image = cv2.imread(os.path.join(input_folder, filename))

    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale image
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5)

    # Loop through all the detected faces
    for (x, y, w, h) in faces:
        # Crop the face from the image
        cropped_face = image[y:y + h, x:x + w]

        # Save the cropped face to the output folder
        cv2.imwrite(os.path.join(output_folder, filename), cropped_face)
