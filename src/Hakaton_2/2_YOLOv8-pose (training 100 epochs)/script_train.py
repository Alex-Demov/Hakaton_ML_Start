import os
import yaml
from ultralytics import YOLO
from pathlib import Path

# --- Настройки ---
# 1. Путь к файлу data.yaml (который мы создали/обновили выше)
#    Нужно убедиться, что путь правильный!
yaml_file_path = Path('C:/Users/Alexey/Desktop/razmetka/data.yaml')

# 2. Выбор базовой модели YOLOv8-pose
#    'yolov8n-pose.pt' - nano, самая быстрая, но менее точная
#    'yolov8s-pose.pt' - small
#    'yolov8m-pose.pt' - medium
#    'yolov8l-pose.pt' - large
#    'yolov8x-pose.pt' - extra large, самая точная, но медленная и требовательная
base_model_name = 'yolov8s-pose.pt' # Начинаем с 's' или 'n'

# 3. Параметры обучения
epochs = 100          # Количество эпох (может потребоваться больше/меньше)
img_size = 640        # Размер изображения для обучения
batch_size = 8        # Размер батча (зависит от памяти GPU, уменьшайте если ошибка 'CUDA out of memory')
project_name = 'fall_detection_training' # Папка для сохранения результатов
run_name = 'yolov8s_pose_100epochs'      # Название конкретного запуска

# --- Проверка и подготовка ---

# Проверяем существование data.yaml
if not yaml_file_path.is_file():
    print(f"Ошибка: Файл {yaml_file_path} не найден.")
    print("Пожалуйста, убедитесь, что путь указан верно и файл существует.")
    exit()

# Проверяем и загружаем конфигурацию датасета
try:
    with open(yaml_file_path, 'r', encoding='utf-8') as f:
        dataset_config = yaml.safe_load(f)

    # Проверяем наличие необходимых ключей и папок (базовая проверка)
    if 'path' not in dataset_config or 'train' not in dataset_config or 'val' not in dataset_config:
         raise ValueError("В data.yaml отсутствуют ключи 'path', 'train' или 'val'.")

    dataset_root = Path(dataset_config['path'])
    train_images_path = dataset_root / dataset_config['train']
    val_images_path = dataset_root / dataset_config['val']
    train_labels_path = train_images_path.parent.parent / 'labels' / train_images_path.name
    val_labels_path = val_images_path.parent.parent / 'labels' / val_images_path.name

    print(f"Корневая папка датасета: {dataset_root}")
    print(f"Обучающие изображения: {train_images_path}")
    print(f"Обучающие метки (ожидаются): {train_labels_path}")
    print(f"Валидационные изображения: {val_images_path}")
    print(f"Валидационные метки (ожидаются): {val_labels_path}")

    if not train_images_path.is_dir():
         raise FileNotFoundError(f"Папка обучающих изображений не найдена: {train_images_path}")
    if not val_images_path.is_dir():
         raise FileNotFoundError(f"Папка валидационных изображений не найдена: {val_images_path}")
    if not train_labels_path.is_dir():
         print(f"Предупреждение: Папка обучающих меток не найдена: {train_labels_path}. YOLO попытается найти их.")
    if not val_labels_path.is_dir():
         print(f"Предупреждение: Папка валидационных меток не найдена: {val_labels_path}. YOLO попытается найти их.")

    print("\nКонфигурация датасета:")
    print(yaml.dump(dataset_config))

except FileNotFoundError as e:
    print(f"Ошибка: Не найдена папка, указанная в data.yaml: {e}")
    exit()
except Exception as e:
    print(f"Ошибка при чтении или проверке {yaml_file_path}: {e}")
    exit()


# --- Обучение ---

# Загружаем предобученную модель YOLOv8-pose
print(f"\nЗагрузка базовой модели: {base_model_name}")
model = YOLO(base_model_name)

print("\nНачало обучения...")
# Запускаем обучение
results = model.train(
    data=str(yaml_file_path.resolve()), # Передаем абсолютный путь к yaml
    epochs=epochs,
    imgsz=img_size,
    batch=batch_size,
    project=project_name,
    name=run_name,
    exist_ok=True, # Не создавать новую папку, если run_name уже существует
    # Дополнительные параметры (можно раскомментировать и настроить):
    # patience=20,      # Остановить обучение, если val/loss не улучшается N эпох
    # workers=8,        # Количество потоков для загрузки данных (зависит от CPU)
    # optimizer='AdamW',# Оптимизатор (Adam, AdamW, SGD)
    # lr0=0.001,        # Начальный learning rate
    # device=0          # Указать GPU ID (0), или 'cpu'
)

print("\nОбучение завершено!")
print(f"Результаты сохранены в папке: {results.save_dir}")

# --- Экспорт модели ---

# Путь к лучшим весам (.pt), сохраненным после обучения
# Обычно это 'best.pt' в папке weights внутри директории запуска
best_pt_path = Path(results.save_dir) / 'weights' / 'best.pt'

if best_pt_path.is_file():
    print(f"\nНайдена лучшая модель: {best_pt_path}")
    print("Экспорт в ONNX...")

    # Загружаем лучшую модель
    best_model = YOLO(best_pt_path)

    # Экспортируем в ONNX
    # Формат ONNX может потребовать указания opset
    onnx_path = best_model.export(format='onnx', opset=12) # Пробуем opset 11 или 12

    print(f"Модель успешно экспортирована в ONNX: {onnx_path}")

    # Экспорт в другие форматы (опционально)
    # try:
    #     print("Экспорт в TensorRT (если установлен)...")
    #     trt_path = best_model.export(format='engine') # Требует TensorRT и CUDA
    #     print(f"Модель успешно экспортирована в TensorRT: {trt_path}")
    # except Exception as e:
    #     print(f"Не удалось экспортировать в TensorRT: {e}")

else:
    print(f"Ошибка: Файл лучшей модели {best_pt_path} не найден после обучения.")

print("\nСкрипт завершил работу.")