import argparse
import os
import shutil

import cv2
import dlib


# Function to check if a face exists in an image
def has_face(image_path):
    # Load the detector
    detector = dlib.get_frontal_face_detector()
    # Read the image
    img = cv2.imread(image_path)

    if img is None:
        print(f"Error reading image: {image_path}")
        return False

    # Convert image into grayscale
    gray = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2GRAY)

    # Use detector to find faces
    faces = detector(gray)

    # Return True if at least one face is detected, otherwise return False
    return len(faces) > 0


# Function to perform face detection and copying
def perform_face_detection(input_directory, output_directory):
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Iterate through images in the input directory
    for filename in os.listdir(input_directory):
        if filename.endswith((".jpg", ".jpeg", ".png")):
            image_path = os.path.join(input_directory, filename)
            output_path = os.path.join(output_directory, filename)

            # Check if a face exists in the image
            if has_face(image_path):
                # If a face exists, copy the image to the output directory
                shutil.copy(image_path, output_path)
                print(
                    f"Face detected in {filename} and copied to the output directory."
                )
            else:
                print(f"No face detected in {filename}.")

    print("Processing completed.")


if __name__ == "__main__":
    # Create argument parser
    parser = argparse.ArgumentParser(description="Face detection and copying tool")
    parser.add_argument("input_directory", help="Input directory containing images")
    parser.add_argument(
        "output_directory", help="Output directory of images with detected faces"
    )
    args = parser.parse_args()

    # Call the function to perform face detection and copying
    perform_face_detection(args.input_directory, args.output_directory)
