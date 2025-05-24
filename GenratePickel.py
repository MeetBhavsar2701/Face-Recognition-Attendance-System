import face_recognition
import cv2
import numpy as np
import pickle
import os

# Path to the folder containing images
folder_path = "Photos"  # Replace with the path to your folder containing face images

# Path to the pickle file
pickle_file = "face_encodings.pkl"

# Load existing encodings if the file exists
if os.path.exists(pickle_file):
    with open(pickle_file, "rb") as f:
        known_face_encodings = pickle.load(f)
    print("Loaded existing face encodings.")
else:
    known_face_encodings = []
    print("No existing encodings found. Starting fresh.")

# Iterate through all images in the folder
for filename in os.listdir(folder_path):
    # Construct full file path
    file_path = os.path.join(folder_path, filename)

    # Check if the file is an image
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
        print(f"Processing {filename}...")

        # Load the image
        image = face_recognition.load_image_file(file_path)

        # Detect face locations
        face_locations = face_recognition.face_locations(image)
        print(f"Face locations in {filename}: {face_locations}")

        # Generate face encodings
        face_encodings = face_recognition.face_encodings(image, face_locations)
        print(f"Encodings in {filename}: {face_encodings}")

        # Append new encodings to the known encodings list
        if face_encodings:
            known_face_encodings.extend(face_encodings)
        else:
            print(f"No faces detected in {filename}.")

# Save updated encodings back to the pickle file
with open(pickle_file, "wb") as f:
    pickle.dump(known_face_encodings, f)
print("All face encodings have been saved to 'face_encodings.pkl'.")
