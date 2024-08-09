import os
from collections import Counter

def count_images(directory):
    class_counts = Counter()
    for root, dirs, files in os.walk(directory):
        for name in files:
            if name.endswith('.jpg'):
                class_name = os.path.basename(root)
                class_counts[class_name] += 1
    return class_counts

train_dir = 'dataset/train'
class_counts = count_images(train_dir)
print(class_counts)