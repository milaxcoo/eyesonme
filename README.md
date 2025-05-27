# YOLOv5 Object Detection for Autonomous Vehicle Navigation

## Описание
Проект реализует обучение и оценку модели YOLOv5 для обнаружения объектов (автомобили, пешеходы, знаки и др.) на дорожных сценах с использованием датасета.

## Структура проекта
- `train.py` — скрипт запуска обучения модели.
- `evaluate.py` — скрипт оценки модели на валидационном наборе.
- `data/bdd100k_yolo/` — подготовленные данные для обучения и валидации (не добавлены в репозиторий).
- `runs/` — папка с результатами и сохранёнными весами.

## Требования
- Python 3.10 или 3.11
- PyTorch с поддержкой CUDA 11.8 (если есть GPU)
- Установить зависимости:
  ```bash
  pip install -r yolov5/requirements.txt
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

## Установка YOLOv5
Клонируйте репозиторий YOLOv5 отдельно (если еще не сделали):
- `git clone https://github.com/ultralytics/yolov5.git`

## Запуск обучения
- `cd yolov5`
- `python train.py --data ../data/bdd100k_yolo/bdd100k.yaml --weights yolov5s.pt --img 640 --batch 16 --epochs 50 --name bdd100k_yolov5_exp`

## Запуск оценки
- `python evaluate.py --data_dir data/bdd100k_yolo --val_json data/bdd100k/labels/bdd100k_labels_images_val.json --weights runs/train/bdd100k_yolov5_exp/weights/best.pt`

## Игнорируемые файлы
Данные и модели не включены в репозиторий. См. .gitignore для деталей.
