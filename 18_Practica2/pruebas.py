import scipy.io
import os
import numpy as np

def convert_to_yolo_format(mat_file_path, output_folder, image_width, image_height, class_id):
    """
    Converts bounding box data from a .mat file to YOLO format and saves it as a .txt file.

    Parameters:
        mat_file_path (str): Path to the .mat file containing `box_coord`.
        output_folder (str): Folder to save the .txt files.
        image_width (int): Width of the image.
        image_height (int): Height of the image.
        class_id (int): Class ID to annotate in the YOLO format.
    """
    # Load the .mat file
    mat_data = scipy.io.loadmat(mat_file_path)

    # Extract the bounding box coordinates
    if 'box_coord' not in mat_data:
        raise ValueError(f"The file {mat_file_path} does not contain the variable 'box_coord'.")

    box_coords = mat_data['box_coord']  # Shape: (N, 4)

    # Normalize the bounding box coordinates to YOLO format
    yolo_data = []
    for box in box_coords:
        y_min, y_max, x_min, x_max = box

        # Calculate normalized center, width, and height
        x_center = ((x_min + x_max) / 2) / image_width
        y_center = ((y_min + y_max) / 2) / image_height
        width = (x_max - x_min) / image_width
        height = (y_max - y_min) / image_height

        # Add the class ID and normalized values
        yolo_data.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

    # Save to a .txt file
    output_filename = os.path.splitext(os.path.basename(mat_file_path))[0] + '.txt'
    output_path = os.path.join(output_folder, output_filename)

    os.makedirs(output_folder, exist_ok=True)

    with open(output_path, 'w') as f:
        f.write("\n".join(yolo_data))

    print(f"YOLO file saved: {output_path}")

# Example usage
if __name__ == "__main__":
    
    # Folder containing class folders
    input_base_folder = "path/to/class_folders"

    # Output folder for YOLO .txt files
    output_folder = "path/to/output_folder"

    # Image dimensions (replace with actual values)
    image_width = 1920
    image_height = 1080

    # Map class names to IDs
    class_mapping = {
        "buddha": 0,
        "dalmatian": 1,
        # Add more classes as needed
    }

    # Process each class folder
    for class_name, class_id in class_mapping.items():
        class_folder = os.path.join(input_base_folder, class_name)

        if not os.path.isdir(class_folder):
            print(f"Class folder not found: {class_folder}")
            continue

        # Process each .mat file in the class folder
        for file_name in os.listdir(class_folder):
            if file_name.endswith(".mat"):
                mat_file_path = os.path.join(class_folder, file_name)
                convert_to_yolo_format(mat_file_path, output_folder, image_width, image_height, class_id)
