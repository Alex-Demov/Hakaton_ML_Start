from ultralytics import YOLO

# Загрузка модели
model = YOLO("yolov8n.pt")  # Вы можете использовать другие варианты yolov8 (s, m, l, x)

# Обучение модели
results = model.train(data="data.yaml", epochs=100, imgsz=640) # epochs - количество эпох, imgsz - размер изображений

# Экспорт модели (опционально)
model.export(format="onnx")