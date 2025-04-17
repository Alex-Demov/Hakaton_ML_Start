# train_nas_pose_local.py
import os
import torch
from super_gradients.training import Trainer, models
from super_gradients.training.dataloaders.dataloaders import coco_pose_estimation_train, coco_pose_estimation_val
from super_gradients.training.losses import PoseEstimationLoss
from super_gradients.training.metrics import PoseEstimationMetrics
from super_gradients.training.utils.distributed_training_utils import setup_device
from pathlib import Path # Добавлено для удобной работы с путями

# --- Настройки ---

# 1. Пути к РАЗДЕЛЕННОМУ датасету (результат работы split_coco.py)

base_data_dir = Path('/content/drive/MyDrive/train_yolo/split_dataset') # Корневая папка разделенного датасета

train_images_dir = base_data_dir / 'train_images'
train_anno_file = base_data_dir / 'annotations/train.json'

val_images_dir = base_data_dir / 'val_images'
val_anno_file = base_data_dir / 'annotations/val.json'

# 2. Имена классов (из JSON)
#    ID категорий в JSON: standing person=1, fall person=19
#    SuperGradients обычно сам разбирается с ID, если указать правильные имена и их количество.
class_names = ['standing person', 'fall person'] # Нужно убедиться, что они точно соответствуют именам в JSON
num_classes = len(class_names) # Количество КЛАССОВ ОБЪЕКТОВ = 2

# 3. Параметры модели и обучения
model_name = 'yolo_nas_pose_s'  # Выбор модели: _n, _s, _m, _l
pretrained_weights = 'coco_pose' # Загрузка весов, предобученных на COCO Pose

experiment_name = 'yolo_nas_pose_s_fall_local' # Имя папки для результатов этого эксперимента
# Папка для сохранения всех чекпоинтов/логов (создаеттся, если нужно)
ckpt_root_dir = Path('/content/drive/MyDrive/train_yolo/nas_pose_checkpoints')

# Параметры обучения (настройка под ПК и задачу)
max_epochs = 10 # Количество эпох
batch_size = 4   # !!! Стоит умень, если мало видеопамяти (VRAM) или используется CPU
num_workers = 2  # Количество потоков для загрузки данных (зависит от CPU, 0 или 2 для начала)

initial_lr = 1e-4 # Начальная скорость обучения
optimizer = "AdamW"
lr_mode = "cosine" # Тип планировщика learning rate
# Веса для компонентов лосса (можно оставить по умолчанию для начала)
loss_weights = {"heatmap": 1.0, "reg": 1.0, "oks": 0.1}

# Параметры датасета COCO Pose (стандартные, должны подходить, т.к. экспорт из CVAT)
num_joints = 17
edge_links = [[0, 1], [0, 2], [1, 3], [2, 4], [5, 6], [5, 7], [7, 9], [6, 8], [8, 10],
              [5, 11], [6, 12], [11, 12], [11, 13], [13, 15], [12, 14], [14, 16]]
joint_weights = [1.] * num_joints
sigmas = [0.026, 0.025, 0.025, 0.035, 0.035, 0.079, 0.079, 0.072, 0.072, 0.062, 0.062, 0.107, 0.107, 0.087, 0.087, 0.089, 0.089] # Сигмы для OKS

# --- Настройка среды ---
# Автоматически выберет GPU, если доступен и настроен PyTorch+CUDA, иначе CPU
# -1 означает использовать все доступные GPU. Можно указать [0] для первого GPU.
setup_device(num_gpus=-1)

# Создаем папку для чекпоинтов, если ее нет
ckpt_root_dir.mkdir(parents=True, exist_ok=True)

# --- Проверка путей ---
if not train_images_dir.is_dir(): raise FileNotFoundError(f"Папка train_images не найдена: {train_images_dir}")
if not val_images_dir.is_dir(): raise FileNotFoundError(f"Папка val_images не найдена: {val_images_dir}")
if not train_anno_file.is_file(): raise FileNotFoundError(f"Файл train.json не найден: {train_anno_file}")
if not val_anno_file.is_file(): raise FileNotFoundError(f"Файл val.json не найден: {val_anno_file}")

# --- Загрузка данных ---
print("Настройка загрузчиков данных...")
train_dataloader = coco_pose_estimation_train(
    dataset_params={
        'data_dir': base_data_dir, # Не используется напрямую, но может быть нужен для некоторых transforms
        'images_dir': str(train_images_dir), # Путь к картинкам
        'json_file': str(train_anno_file),   # Путь к JSON
        'num_joints': num_joints,
        'edge_links': edge_links,
        'joint_weights': joint_weights,
        'sigmas': sigmas,
        'remove_empty_annotations': False,
        'transforms_sigmas': sigmas
    },
    dataloader_params={
        'batch_size': batch_size,
        'num_workers': num_workers,
        'pin_memory': True # Ускоряет передачу данных на GPU (если используется)
    }
)

val_dataloader = coco_pose_estimation_val(
     dataset_params={
        'data_dir': base_data_dir,
        'images_dir': str(val_images_dir),
        'json_file': str(val_anno_file),
        'num_joints': num_joints,
        'edge_links': edge_links,
        'joint_weights': joint_weights,
        'sigmas': sigmas,
        'remove_empty_annotations': False,
        'transforms_sigmas': sigmas
    },
    dataloader_params={
        'batch_size': batch_size * 2, # Можно увеличить батч для валидации
        'num_workers': num_workers,
        'pin_memory': True
    }
)
print("Загрузчики данных настроены.")

# --- Настройка модели ---
print(f"Загрузка модели: {model_name}...")
# num_classes = количество КЛАССОВ объектов ('standing', 'fall')
model = models.get(
    model_name,
    num_classes=num_classes,
    pose_estimation_backend_output_channels=num_joints, # Убеждаемся, что кол-во точек = 17
    pretrained_weights=pretrained_weights
)
print("Модель загружена.")

# --- Настройка параметров обучения ---
training_params = {
    "max_epochs": max_epochs,
    "initial_lr": initial_lr,
    "loss": PoseEstimationLoss(oks_sigmas=sigmas, num_joints=num_joints, weights=loss_weights),
    "optimizer": optimizer,
    "optimizer_params": {"weight_decay": 0.0001},
    "lr_mode": lr_mode,
    "cosine_final_lr_ratio": 0.01,
    "lr_warmup_epochs": 5,
    "train_metrics_list": [PoseEstimationMetrics(post_prediction_callback=model.get_post_prediction_callback(pose_confidence_threshold=0.5), # Добавлен порог уверенности
                                                  num_joints=num_joints, oks_sigmas=sigmas)],
    "valid_metrics_list": [PoseEstimationMetrics(post_prediction_callback=model.get_post_prediction_callback(pose_confidence_threshold=0.5), # Добавлен порог уверенности
                                                  num_joints=num_joints, oks_sigmas=sigmas)],
    "metric_to_watch": "AP",
    "greater_metric_to_watch_is_better": True,
    "average_best_models": True,
    "mixed_precision": torch.cuda.is_available(), # Включаем MP, только если есть CUDA
    "loss_logging_items_names": ["heatmap_loss", "reg_loss", "oks_loss", "loss"],
    "valid_dataset_cropping_padding_factor": 0.0,
    "save_ckpt_epoch_list": [max_epochs] # Сохранять чекпоинт только последней эпохи (экономит место)
                                          # Можно указать [50, 75, 100] для промежуточных
}

# --- Инициализация тренера ---
trainer = Trainer(experiment_name=experiment_name, ckpt_root_dir=str(ckpt_root_dir))

# --- Запуск обучения ---
print("Начало обучения YOLO-NAS-Pose...")
trainer.train(
    model=model,
    training_params=training_params,
    train_loader=train_dataloader,
    valid_loader=val_dataloader
)
print("Обучение завершено!")

# --- Экспорт в ONNX (Пример) ---
print("\nЭкспорт лучшей модели в ONNX...")
try:
    # Путь к лучшему чекпоинту (обычно усредненная модель)
    best_model_path = ckpt_root_dir / experiment_name / "average_model.pth"
    if not best_model_path.is_file():
         best_model_path = ckpt_root_dir / experiment_name / "ckpt_best.pth" # Запасной вариант

    if best_model_path.is_file():
        print(f"Загрузка лучшей модели из: {best_model_path}")
        # Загружаем модель для конвертации
        best_model = models.get(model_name,
                                num_classes=num_classes,
                                pose_estimation_backend_output_channels=num_joints,
                                checkpoint_path=str(best_model_path))

        # Подготовка к конвертации
        # Используется размер, с которым будет делатся инференс, или стандартный (например, 640x640)
        input_h, input_w = 640, 640
        input_shape = (1, 3, input_h, input_w) # (Batch, Channels, Height, Width)
        onnx_export_path = ckpt_root_dir / experiment_name / f"{experiment_name}_best_{input_h}x{input_w}.onnx"

        best_model.eval() # Перевод модели в режим оценки
        # Используйте input_size=(H, W)
        best_model.prep_model_for_conversion(input_size=(input_h, input_w))

        # Создаем фиктивный входной тензор
        # Перемещаем на то же устройство, где модель (CPU или GPU)
        device = next(best_model.parameters()).device
        dummy_input = torch.randn(input_shape).to(device)

        torch.onnx.export(
            best_model,
            dummy_input,
            str(onnx_export_path),
            verbose=False,
            opset_version=12, # Рекомендуется 11 или 12
            input_names=['input'],
            output_names=['output'], # Уточните реальные имена выходов для NAS-Pose, если нужно
            # dynamic_axes={'input': {0: 'batch_size', 2: 'height', 3: 'width'}, # Если нужен динамический размер
            #               'output': {0: 'batch_size'}}
        )
        print(f"Модель успешно экспортирована в ONNX: {onnx_export_path}")

    else:
        print(f"Ошибка: Не найден лучший чекпоинт '{best_model_path}'")

except Exception as e:
    print(f"Ошибка при экспорте в ONNX: {e}")
    import traceback
    traceback.print_exc() # Печать детальной ошибки

print("\nСкрипт завершил работу.")