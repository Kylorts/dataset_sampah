import os
import random
import shutil

source_dir = "C:\\Users\\User\\Documents\\Kuliah\\Semester 6\\ML 1\\ml_prepocessing_dataset\\5_augmented_dataset"          
output_dir = "C:\\Users\\User\\Documents\\Kuliah\\Semester 6\\ML 1\\ml_prepocessing_dataset\\6_balanced_augment_dataset"   

os.makedirs(output_dir, exist_ok=True)

classes = [d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))]

class_counts = {}

for c in classes:
    class_path = os.path.join(source_dir, c)
    files = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg','.png','.jpeg'))]
    class_counts[c] = len(files)

for k,v in class_counts.items():
    print(k, ":", v)


min_count = min(class_counts.values())

for c in classes:

    class_path = os.path.join(source_dir, c)
    files = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg','.png','.jpeg'))]

    selected_files = random.sample(files, min_count)

    out_class_path = os.path.join(output_dir, c)
    os.makedirs(out_class_path, exist_ok=True)

    for f in selected_files:
        src = os.path.join(class_path, f)
        dst = os.path.join(out_class_path, f)
        shutil.copy(src, dst)


print("Balancing done.")