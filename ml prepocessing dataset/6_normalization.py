import os
import cv2
import numpy as np

input_dir = "C:\\Users\\User\\Documents\\Kuliah\\Semester 6\\ML 1\\ml_prepocessing_dataset\\6_balanced_augment_dataset"
output_dir = "C:\\Users\\User\\Documents\\Kuliah\\Semester 6\\ML 1\\ml_prepocessing_dataset\\7_normalized_dataset"

os.makedirs(output_dir, exist_ok=True)

classes = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir,d))]

for c in classes:

    class_input = os.path.join(input_dir, c)
    class_output = os.path.join(output_dir, c)

    os.makedirs(class_output, exist_ok=True)

    files = os.listdir(class_input)

    for f in files:

        path = os.path.join(class_input, f)

        img = cv2.imread(path)

        if img is None:
            continue

        img = img.astype(np.float32) / 255.0

        img_save = (img * 255).astype(np.uint8)

        save_path = os.path.join(class_output, f)

        cv2.imwrite(save_path, img_save)

print("normalization pixel done")