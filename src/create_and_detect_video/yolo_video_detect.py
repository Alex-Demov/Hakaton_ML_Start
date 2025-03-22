import cv2
from ultralytics import YOLO

# Загрузка обученной модели YOLO
model = YOLO(r"C:\Users\Alexey\Desktop\create_video\best.pt")

# Открытие видео
video_path = r"C:\Users\Alexey\Desktop\create_video\output_video2.mp4"
cap = cv2.VideoCapture(video_path)

# Установите желаемый размер окна
desired_width = 1280  # Или другое значение, подходящее для вашего экрана
desired_height = 720 # Или другое значение, подходящее для вашего экрана

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Обнаружение объектов
    results = model(frame)

    # Отображение результатов
    annotated_frame = results[0].plot()

    # Изменение размера кадра
    resized_frame = cv2.resize(annotated_frame, (desired_width, desired_height))

    cv2.imshow("YOLO Detection", resized_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()