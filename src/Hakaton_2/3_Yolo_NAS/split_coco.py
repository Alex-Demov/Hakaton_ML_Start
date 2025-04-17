# split_coco.py
import json
import os
import random
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split # pip install scikit-learn

# --- Настройки ---
INPUT_JSON_PATH = Path('person_keypoints_default.json')
INPUT_IMAGES_DIR = Path('frames')
OUTPUT_DIR = Path('split_dataset') # Новая папка для разделенных данных
TRAIN_RATIO = 0.8 # Доля данных для обучающей выборки (остальное пойдет в val)
RANDOM_SEED = 42 # Для воспроизводимости случайного разделения
# --- Конец Настроек ---

random.seed(RANDOM_SEED)

def split_coco_dataset(json_path: Path, images_dir: Path, output_dir: Path, train_ratio: float):
    """
    Разделяет датасет в формате COCO Pose на обучающую и валидационную выборки.
    """
    print(f"Загрузка JSON: {json_path}")
    if not json_path.is_file():
        print(f"Ошибка: Файл {json_path} не найден.")
        return
    if not images_dir.is_dir():
        print(f"Ошибка: Папка с изображениями {images_dir} не найдена.")
        return

    with open(json_path, 'r', encoding='utf-8') as f:
        coco_data = json.load(f)

    images = coco_data['images']
    annotations = coco_data['annotations']
    categories = coco_data['categories']
    licenses = coco_data.get('licenses', []) # Не всегда есть
    info = coco_data.get('info', {})        # Не всегда есть

    # Разделение списка изображений
    train_images_data, val_images_data = train_test_split(
        images, train_size=train_ratio, random_state=RANDOM_SEED, shuffle=True
    )

    train_image_ids = {img['id'] for img in train_images_data}
    val_image_ids = {img['id'] for img in val_images_data}

    print(f"Всего изображений: {len(images)}")
    print(f"Изображений для обучения (train): {len(train_image_ids)}")
    print(f"Изображений для валидации (val): {len(val_image_ids)}")

    # Разделение аннотаций
    train_annotations = [anno for anno in annotations if anno['image_id'] in train_image_ids]
    val_annotations = [anno for anno in annotations if anno['image_id'] in val_image_ids]

    print(f"Всего аннотаций: {len(annotations)}")
    print(f"Аннотаций для обучения (train): {len(train_annotations)}")
    print(f"Аннотаций для валидации (val): {len(val_annotations)}")

    # --- Создание структуры папок ---
    output_dir.mkdir(parents=True, exist_ok=True)
    out_annotations_dir = output_dir / 'annotations'
    out_train_images_dir = output_dir / 'train_images'
    out_val_images_dir = output_dir / 'val_images'

    out_annotations_dir.mkdir(exist_ok=True)
    out_train_images_dir.mkdir(exist_ok=True)
    out_val_images_dir.mkdir(exist_ok=True)

    # --- Копирование изображений ---
    print("Копирование обучающих изображений...")
    for img_data in train_images_data:
        src_path = images_dir / img_data['file_name']
        dst_path = out_train_images_dir / img_data['file_name']
        if src_path.is_file():
            shutil.copy2(src_path, dst_path)
        else:
            print(f"Предупреждение: Файл изображения не найден: {src_path}")
        # Обновляем file_name в JSON - оставляем только имя файла
        img_data['file_name'] = Path(img_data['file_name']).name

    print("Копирование валидационных изображений...")
    for img_data in val_images_data:
        src_path = images_dir / img_data['file_name']
        dst_path = out_val_images_dir / img_data['file_name']
        if src_path.is_file():
            shutil.copy2(src_path, dst_path)
        else:
            print(f"Предупреждение: Файл изображения не найден: {src_path}")
        # Обновляем file_name в JSON
        img_data['file_name'] = Path(img_data['file_name']).name

    # --- Создание новых JSON файлов ---
    train_coco_data = {
        'licenses': licenses,
        'info': info,
        'categories': categories,
        'images': train_images_data,
        'annotations': train_annotations
    }
    val_coco_data = {
        'licenses': licenses,
        'info': info,
        'categories': categories,
        'images': val_images_data,
        'annotations': val_annotations
    }

    train_json_path = out_annotations_dir / 'train.json'
    val_json_path = out_annotations_dir / 'val.json'

    print(f"Сохранение {train_json_path}")
    with open(train_json_path, 'w', encoding='utf-8') as f:
        json.dump(train_coco_data, f, indent=4, ensure_ascii=False)

    print(f"Сохранение {val_json_path}")
    with open(val_json_path, 'w', encoding='utf-8') as f:
        json.dump(val_coco_data, f, indent=4, ensure_ascii=False)

    print("\nРазделение датасета завершено.")
    print(f"Результаты сохранены в папке: {output_dir.resolve()}")
    print("Структура:")
    print(f"  ├── {out_annotations_dir.name}/")
    print(f"  │   ├── train.json")
    print(f"  │   └── val.json")
    print(f"  ├── {out_train_images_dir.name}/")
    print(f"  │   └── *.jpg")
    print(f"  └── {out_val_images_dir.name}/")
    print(f"  │   └── *.jpg")

if __name__ == "__main__":
    if not INPUT_JSON_PATH.is_file():
         print(f"Ошибка: Входной JSON файл не найден: {INPUT_JSON_PATH}")
    elif not INPUT_IMAGES_DIR.is_dir():
         print(f"Ошибка: Входная папка с изображениями не найдена: {INPUT_IMAGES_DIR}")
    else:
        split_coco_dataset(INPUT_JSON_PATH, INPUT_IMAGES_DIR, OUTPUT_DIR, TRAIN_RATIO)