import cv2
import numpy as np

def draw_bboxes_and_save(image_path, tracks, output_path):
    # Load the image from the provided path
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image from {image_path}")
        return

    # Iterate over each bounding box in the tracks array
    for bbox in tracks:
        # Unpack the bounding box values: x_center, y_center, width, height, class
        x_center, y_center, width, height, cls = bbox

        # Convert center format [x_center, y_center, width, height] to corner format [x_min, y_min, x_max, y_max]
        x_min = int(x_center - width / 2)
        y_min = int(y_center - height / 2)
        x_max = int(x_center + width / 2)
        y_max = int(y_center + height / 2)

        # Draw the rectangle on the image (green box with thickness 2)
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

        # Optionally, put the class number near the bounding box
        cv2.putText(image, f"{int(cls)}", (x_min, y_min - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Save the modified image to the output path
    success = cv2.imwrite(output_path, image)
    if success:
        print(f"Image saved successfully to {output_path}")
    else:
        print(f"Error: Could not save image to {output_path}")

# Example tracks bounding box array: [x_center, y_center, width, height, class]

tracks0 = np.array([
        [
            917.4364013671875,
            528.0391845703125,
            260.991455078125,
            195.35598754882812,
            1.0
        ],
        [
            698.9573974609375,
            280.2091979980469,
            136.65936279296875,
            94.82344055175781,
            1.0
        ],
        [
            490.6507873535156,
            177.2344207763672,
            187.09368896484375,
            116.19888305664062,
            1.0
        ]
    
    ])

# Replace these paths with your actual file paths
input_image_path = "/cfs/earth/scratch/kotabha1/egohands/train/frames/cards_office_H_T/frame_1549.jpg"
output_image_path = "/cfs/earth/scratch/kotabha1/output_image.jpg"

# Draw bounding boxes and save the result
draw_bboxes_and_save(input_image_path, tracks0, output_image_path)














