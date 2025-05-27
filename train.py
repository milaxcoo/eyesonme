import subprocess
import os
import sys
import argparse

def train_yolov5(data_yaml, weights, cfg, epochs, batch_size, img_size, project_name):
    """
    Вызывает обучение YOLOv5 через командную строку с заданными параметрами.
    """
    # Формируем команду для запуска обучения с помощью скрипта YOLOv5
    cmd = [
        "python",
        sys.executable,
        "yolov5/train.py",            # путь к train.py (предполагается, что репозиторий yolov5 установлен)
        "--data", data_yaml,          # YAML-файл с описанием датасета (путь до bdd100k.yaml из подготовки)
        "--weights", weights,         # начальные веса (yolov5s.pt по умолчанию)
        "--cfg", cfg,                 # конфиг архитектуры модели (если хотим изменить сеть, можно указать другой yaml)
        "--img", str(img_size),       # размер изображения при обучении (например, 640 пикселей)
        "--batch", str(batch_size),
        "--epochs", str(epochs),
        "--name", project_name        # имя эксперимента (папки с результатами внутри runs/train)
    ]
    # Дополнительно: можно указать другие гиперпараметры, например --hyp для своего файла гиперпараметров.
    print("Запуск обучения YOLOv5:")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Обучение модели YOLOv5 на датасете BDD100K")
    parser.add_argument("--data", type=str, default="data/bdd100k_yolo/bdd100k.yaml",
                        help="Путь к YAML-файлу с настройками датасета")
    parser.add_argument("--weights", type=str, default="yolov5s.pt",
                        help="Начальный вес модели (путь к .pt файлу, по умолчанию yolov5s.pt предобученный)")
    parser.add_argument("--cfg", type=str, default="", 
                        help="Файл конфигурации модели (YOLOv5 *.yaml, не обязателен, можно оставить пустым для авто)")
    parser.add_argument("--epochs", type=int, default=50, help="Количество эпох обучения")
    parser.add_argument("--batch_size", type=int, default=16, help="Размер batch (число изображений на итерацию)")
    parser.add_argument("--img_size", type=int, default=640, help="Размер изображения для обучения (pixels)")
    parser.add_argument("--name", type=str, default="bdd100k_yolov5_exp",
                        help="Имя эксперимента (папка в runs/train/)")
    args = parser.parse_args()

    # Если пользователь не указал cfg, уберем этот аргумент из команды (YOLOv5 возьмет стандартный для выбранных весов)
    cfg_path = args.cfg if args.cfg else "models/yolov5s.yaml"
    train_yolov5(data_yaml=args.data,
                 weights=args.weights,
                 cfg=cfg_path,
                 epochs=args.epochs,
                 batch_size=args.batch_size,
                 img_size=args.img_size,
                 project_name=args.name)
