import os
import json
import shutil
from tqdm import tqdm

# Список классов BDD100K для задачи детекции (10 классов)
CLASSES = [
    "person",
    "rider",
    "car",
    "truck",
    "bus",
    "train",
    "motorcycle",
    "bicycle",
    "traffic light",
    "traffic sign"
]

def convert_annotation(bdd_annotation, images_dir, labels_dir):
    """
    Конвертирует аннотацию одного изображения из формата BDD100K (JSON) в YOLO-формат.
    Создает .txt файл с bounding box для данного изображения.
    """
    file_name = bdd_annotation["name"]  # имя файла изображения, например "abcd123.jpg"
    # Путь к исходному изображению
    img_path = os.path.join(images_dir, file_name)
    # Загрузка изображения для получения размеров (ширина, высота)
    try:
        import cv2
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Image {img_path} not found or unable to open.")
        img_h, img_w = img.shape[0], img.shape[1]
    except ImportError:
        # Если OpenCV недоступен, можно использовать альтернативу
        from PIL import Image
        img = Image.open(img_path)
        img_w, img_h = img.size

    # Открываем файл для записи меток в YOLO-формате
    label_file_path = os.path.join(labels_dir, file_name.replace('.jpg', '.txt'))
    with open(label_file_path, 'w') as lf:
        # Если у изображения нет объектов, создается пустой файл (YOLOv5 ожидает пустой .txt, если объектов нет)
        if "labels" in bdd_annotation and bdd_annotation["labels"]:
            for obj in bdd_annotation["labels"]:
                category = obj.get("category")
                # Пропускаем объекты, не относящиеся к 10 классам (на случай лишних аннотаций)
                if category not in CLASSES:
                    continue
                cls_id = CLASSES.index(category)  # индекс класса (0-9)
                # Координаты прямоугольника в исходном JSON
                box2d = obj.get("box2d")
                if box2d is None:
                    continue  # пропустить, если нет box2d (например, для полигонов др. задач)
                x1, y1 = box2d["x1"], box2d["y1"]
                x2, y2 = box2d["x2"], box2d["y2"]
                # Рассчитываем центр и размеры для YOLO (нормализованные 0..1)
                w = x2 - x1
                h = y2 - y1
                x_center = x1 + w / 2.0
                y_center = y1 + h / 2.0
                # Нормализация
                x_center /= img_w
                y_center /= img_h
                w /= img_w
                h /= img_h
                # Убеждаемся, что значения находятся в диапазоне [0,1]
                if x_center < 0: x_center = 0
                if y_center < 0: y_center = 0
                if w < 0: w = 0
                if h < 0: h = 0
                if x_center > 1: x_center = 1
                if y_center > 1: y_center = 1
                if w > 1: w = 1
                if h > 1: h = 1
                # Записываем строку: класс и нормализованные координаты
                lf.write(f"{cls_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")

def prepare_dataset(source_dir, output_dir):
    """
    Выполняет конвертацию аннотаций BDD100K (train и val) в YOLOv5-формат.
    source_dir: директория с исходными данными BDD100K (содержит папки images и labels).
    output_dir: директория, куда будут сохранены изображения и метки в формате YOLO.
    """
    # Пути к исходным файлам
    train_json = os.path.join(source_dir, "labels", "bdd100k_labels_images_train.json")
    val_json = os.path.join(source_dir, "labels", "bdd100k_labels_images_val.json")
    train_images_dir = os.path.join(source_dir, "images", "100k", "train")
    val_images_dir = os.path.join(source_dir, "images", "100k", "val")

    # Создаем структуру папок под YOLOv5
    for split in ["train", "val"]:
        os.makedirs(os.path.join(output_dir, "images", split), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "labels", split), exist_ok=True)

    # Конвертация обучающей выборки
    print("Конвертация TRAIN выборки...")
    with open(train_json, 'r') as f:
        train_data = json.load(f)
    for item in tqdm(train_data, desc="Processing train images"):
        # Копируем изображение в новую структуру
        src_img_path = os.path.join(train_images_dir, item["name"])
        dst_img_path = os.path.join(output_dir, "images", "train", item["name"])
        if not os.path.isfile(dst_img_path):
            shutil.copy2(src_img_path, dst_img_path)
        # Конвертируем аннотации и сохраняем .txt
        convert_annotation(item, train_images_dir, os.path.join(output_dir, "labels", "train"))

    # Конвертация валидационной выборки
    print("Конвертация VAL выборки...")
    with open(val_json, 'r') as f:
        val_data = json.load(f)
    for item in tqdm(val_data, desc="Processing val images"):
        src_img_path = os.path.join(val_images_dir, item["name"])
        dst_img_path = os.path.join(output_dir, "images", "val", item["name"])
        if not os.path.isfile(dst_img_path):
            shutil.copy2(src_img_path, dst_img_path)
        convert_annotation(item, val_images_dir, os.path.join(output_dir, "labels", "val"))

    # Создаем файл конфигурации датасета для YOLOv5 (например, data/bdd100k.yaml)
    dataset_yaml_path = os.path.join(output_dir, "bdd100k.yaml")
    with open(dataset_yaml_path, 'w') as f:
        # Абсолютные пути к папкам с изображениями (или можно оставить относительные)
        train_images_path = os.path.abspath(os.path.join(output_dir, "images", "train"))
        val_images_path = os.path.abspath(os.path.join(output_dir, "images", "val"))
        f.write(f"path: {os.path.abspath(output_dir)}\n")  # корневая папка датасета
        f.write(f"train: {train_images_path}\n")
        f.write(f"val: {val_images_path}\n")
        f.write(f"nc: {len(CLASSES)}\n")
        # Список имен классов
        names_str = ", ".join([f"'{name}'" for name in CLASSES])
        f.write(f"names: [{names_str}]\n")
    print(f"Готово! YOLOv5-формат датасета сохранен в папке {output_dir}")
    print(f"Файл конфигурации датасета: {dataset_yaml_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Конвертация BDD100K аннотаций в YOLOv5 формат")
    parser.add_argument("--source_dir", type=str, default="data/bdd100k",
                        help="Путь к исходной директории BDD100K (с папками images и labels)")
    parser.add_argument("--output_dir", type=str, default="data/bdd100k_yolo",
                        help="Папка для сохранения сконвертированных данных для YOLOv5")
    args = parser.parse_args()
    prepare_dataset(args.source_dir, args.output_dir)
