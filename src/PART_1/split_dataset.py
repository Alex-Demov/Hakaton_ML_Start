import os
import random
import shutil

def split_dataset(images_dir, labels_dir, output_dir, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
    """Разбивает датасет на обучающую, валидационную и тестовую выборки."""

    image_files = os.listdir(images_dir)
    random.shuffle(image_files)

    train_size = int(len(image_files) * train_ratio)
    val_size = int(len(image_files) * val_ratio)

    train_files = image_files[:train_size]
    val_files = image_files[train_size:train_size + val_size]
    test_files = image_files[train_size + val_size:]

    for split, files in zip(['train', 'valid', 'test'], [train_files, val_files, test_files]):
        os.makedirs(os.path.join(output_dir, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, split, 'labels'), exist_ok=True)

        for file in files:
            image_path = os.path.join(images_dir, file)
            label_path = os.path.join(labels_dir, file.replace('.jpg', '.txt'))

            shutil.copy(image_path, os.path.join(output_dir, split, 'images', file))
            shutil.copy(label_path, os.path.join(output_dir, split, 'labels', file.replace('.jpg', '.txt')))

if __name__ == "__main__":
    images_dir = r"C:\Users\Alexey\Desktop\archive\PART_1\PART_1\images"
    labels_dir = r"C:\Users\Alexey\Desktop\archive\PART_1\PART_1\6categories"
    output_dir = r"C:\Users\Alexey\Desktop\archive\PART_1\PART_1"

    split_dataset(images_dir, labels_dir, output_dir)