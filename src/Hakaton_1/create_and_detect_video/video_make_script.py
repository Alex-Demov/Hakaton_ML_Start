import cv2
import os
import re

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

def images_to_video(image_folder, output_video, fps=30):
    images = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]
    images.sort(key=natural_keys) # Сортировка по числовому значению

    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    video = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))

    cv2.destroyAllWindows()
    video.release()

image_folder = r"C:\Users\Alexey\Desktop\create_video\images_test_365"
output_video = r"C:\Users\Alexey\Desktop\create_video\output_video2.mp4"
fps = 10  # Установите желаемую частоту кадров

images_to_video(image_folder, output_video, fps)
print(f"Видео создано: {output_video}")