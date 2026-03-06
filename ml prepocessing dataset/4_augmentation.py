import cv2
import os
import numpy as np
import random

base_path = 'C:\\Users\\User\\Documents\\Kuliah\\Semester 6\\ML 1\\ml_prepocessing_dataset'
input_root = os.path.join(base_path, '4_standardized_dataset')
output_root = os.path.join(base_path, '5_augmented_dataset')

categories = [d for d in os.listdir(input_root) if os.path.isdir(os.path.join(input_root, d))]

def rotate_image(image, angle):
    h, w = image.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    return cv2.warpAffine(image, M, (w, h))

def horizontal_flip(image):
    return cv2.flip(image, 1)

def adjust_brightness(image, value):
    return cv2.convertScaleAbs(image, alpha=1, beta=value)

def zoom_crop(image, zoom_factor=1.15):
    h, w = image.shape[:2]
    nh, nw = int(h / zoom_factor), int(w / zoom_factor)
    y1, x1 = (h - nh) // 2, (w - nw) // 2
    return cv2.resize(image[y1:y1+nh, x1:x1+nw], (w, h))



report_data = [] 

for cat in categories:
    source_dir = os.path.join(input_root, cat)
    target_dir = os.path.join(output_root, cat)
    os.makedirs(target_dir, exist_ok=True)

    files = [f for f in os.listdir(source_dir) if f.lower().endswith(('.png','.jpg','.jpeg'))]
    orig_count = len(files)
    aug_count = 0

    print(f"\n> process category: {cat}")
    
    for file_name in files:
        img = cv2.imread(os.path.join(source_dir, file_name))
        if img is None: continue

        base = os.path.splitext(file_name)[0]

        # Simpan 5 variasi (Termasuk Original)
        cv2.imwrite(os.path.join(target_dir, f"{base}_ori.jpg"), img)
        cv2.imwrite(os.path.join(target_dir, f"{base}_flip.jpg"), horizontal_flip(img))
        
        ang = random.choice([random.randint(10,20), random.randint(-20,-10)])
        cv2.imwrite(os.path.join(target_dir, f"{base}_rotation.jpg"), rotate_image(img, ang))
        
        b_val = random.choice([30, -30])
        cv2.imwrite(os.path.join(target_dir, f"{base}_bright.jpg"), adjust_brightness(img, b_val))
        
        cv2.imwrite(os.path.join(target_dir, f"{base}_zoomcrop.jpg"), zoom_crop(img))

        aug_count += 5

    report_data.append({
        'class': cat,
        'original': orig_count,
        'augmented': aug_count
    })
    print(f"  done {orig_count} original -> {aug_count} augmented.")


print(f"{'category':<20} | {'original':<10} | {'augmented':<10}")
total_orig = 0
total_aug = 0

for item in report_data:
    print(f"{item['class']:<20} | {item['original']:<10} | {item['augmented']:<10}")
    total_orig += item['original']
    total_aug += item['augmented']


print(f"{'total all classes':<20} | {total_orig:<10} | {total_aug:<10}")