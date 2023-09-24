import os
import shutil
from deepface import DeepFace


def process_images(input_dir, output_dir):
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Variables to keep track of counts
    total_detected_faces = 0
    total_no_face_detected = 0

    # Function to check if a face exists in an image
    def has_face(image_path):
        try:
            # Use DeepFace to detect faces in the image
            result = DeepFace.analyze(image_path, actions=["emotion"])
            return result["emotion"] is not None
        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
            return False

    # Iterate through images in the input directory
    for entry in os.scandir(input_dir):
        if entry.is_file() and entry.name.lower().endswith((".jpg", ".jpeg", ".png")):
            image_path = entry.path
            output_path = os.path.join(output_dir, entry.name)

            # Check if a face exists in the image
            if has_face(image_path):
                # If a face exists, copy the image to the output directory
                shutil.copy(image_path, output_path)
                total_detected_faces += 1
                print(f"Face detected in {entry.name}.")
            else:
                total_no_face_detected += 1
                print(f"No face detected in {entry.name}.")

    print("Processing completed.")
    print(f"Total detected faces: {total_detected_faces}")
    print(f"Total images with no face detected: {total_no_face_detected}")
