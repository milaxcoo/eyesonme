import os
import argparse
import matplotlib.pyplot as plt
import csv
import cv2
import random
import torch
import json

# Имена классов (для подписей на изображениях)
CLASSES = [
    "person", "rider", "car", "truck", "bus",
    "train", "motorcycle", "bicycle", "traffic light", "traffic sign"
]

def plot_training_curves(results_csv, output_dir):
    """
    Строит графики метрик обучения/валидации на основе файла results.csv, сохраненного YOLOv5.
    """
    epochs = []
    precision = []
    recall = []
    mAP50 = []
    mAP5095 = []
    train_loss = []
    val_loss = []
    # Читаем CSV (пропустим заголовок)
    with open(results_csv, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            if len(row) > 0:
                # В зависимости от версии YOLOv5 формат CSV может меняться;
                # Здесь предполагается: epoch, train_box_loss, train_obj_loss, train_cls_loss, val_box_loss, val_obj_loss, val_cls_loss, precision, recall, mAP50, mAP50-95
                epoch = int(row[0])
                t_box_loss = float(row[1]); t_obj_loss = float(row[2]); t_cls_loss = float(row[3])
                v_box_loss = float(row[4]); v_obj_loss = float(row[5]); v_cls_loss = float(row[6])
                p = float(row[7]); r = float(row[8]); m50 = float(row[9]); m5095 = float(row[10])
                epochs.append(epoch)
                precision.append(p); recall.append(r)
                mAP50.append(m50); mAP5095.append(m5095)
                train_loss.append(t_box_loss + t_obj_loss + t_cls_loss)
                val_loss.append(v_box_loss + v_obj_loss + v_cls_loss)
    # График Precision/Recall/mAP
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, precision, label="Precision")
    plt.plot(epochs, recall, label="Recall")
    plt.plot(epochs, mAP50, label="mAP@0.5")
    # Можно также добавить mAP@0.5:0.95
    plt.plot(epochs, mAP5095, label="mAP@0.5:0.95")
    plt.xlabel("Epoch")
    plt.ylabel("Metric value")
    plt.title("Precision, Recall, mAP over epochs")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "training_metrics.png"))
    plt.close()

    # График функции потерь (train/val)
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, train_loss, label="Train loss")
    plt.plot(epochs, val_loss, label="Val loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend(loc="upper right")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "training_loss.png"))
    plt.close()

def visualize_detections(model, image_paths, output_dir):
    """
    Применяет модель к заданным изображениям и сохраняет их с нарисованными bounding box.
    """
    # Задаем цвета для классов (генерируем случайно, фиксируя seed для повторяемости)
    random.seed(0)
    class_colors = {cls: (random.randint(0,255), random.randint(0,255), random.randint(0,255)) for cls in range(len(CLASSES))}
    for img_path in image_paths:
        img = cv2.imread(img_path)
        if img is None:
            print(f"Изображение {img_path} не найдено.")
            continue
        # Выполняем детекцию
        results = model(img_path)
        detections = results.xyxy[0].cpu().numpy()  # форма (N,6): [x1, y1, x2, y2, conf, cls]
        # Рисуем обнаруженные bounding box на изображении
        for (x1, y1, x2, y2, conf, cls_id) in detections:
            cls_id = int(cls_id)
            color = class_colors.get(cls_id, (0, 255, 0))
            # Рисуем прямоугольник
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness=2)
            label = f"{CLASSES[cls_id]} {conf:.2f}"
            # Вывод метки класса над прямоугольником
            cv2.putText(img, label, (int(x1), int(y1) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness=2)
        # Сохраняем результат рядом с исходным именем файла
        img_name = os.path.basename(img_path)
        out_path = os.path.join(output_dir, f"det_{img_name}")
        cv2.imwrite(out_path, img)
        print(f"Сохранено изображение с детекциями: {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Визуализация результатов обучения и работы модели YOLOv5")
    parser.add_argument("--results_csv", type=str, default="runs/train/bdd100k_yolov5_exp/results.csv",
                        help="Путь к CSV-файлу с результатами обучения (results.csv)")
    parser.add_argument("--weights", type=str, default="runs/train/bdd100k_yolov5_exp/weights/best.pt",
                        help="Вес модели (.pt файл) для визуализации детекций")
    parser.add_argument("--sample_images", type=str, nargs='+', default=None,
                        help="Список путей к изображениям для детекции. Если не указано, выберется по одному примеру: день, ночь, дождь.")
    parser.add_argument("--output_dir", type=str, default="visualizations",
                        help="Директория для сохранения результатов визуализации")
    parser.add_argument("--val_json", type=str, default="data/bdd100k/labels/bdd100k_labels_images_val.json",
                        help="Путь к JSON аннотациям валидации (для выбора примеров по условиям)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Строим графики обучения
    plot_training_curves(args.results_csv, args.output_dir)
    print(f"Графики метрик обучения сохранены в папку {args.output_dir}")

    # Загружаем модель для детекций
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=args.weights, force_reload=False)
    model.conf = 0.25  # порог уверенности

    # Определяем изображения для примера детекции
    sample_images = args.sample_images
    if sample_images is None:
        # Если не указаны, выбираем случайные примеры из категорий день/ночь/дождь
        with open(args.val_json, 'r') as f:
            val_data = json.load(f)
        day_imgs = [item["name"] for item in val_data if item.get("attributes", {}).get("timeofday", "").lower() == "daytime"]
        night_imgs = [item["name"] for item in val_data if item.get("attributes", {}).get("timeofday", "").lower() == "night"]
        rain_imgs = [item["name"] for item in val_data if item.get("attributes", {}).get("weather", "").lower() == "rainy"]
        sample_images = []
        if day_imgs:
            sample_images.append(os.path.join("data/bdd100k_yolo/images/val", random.choice(day_imgs)))
        if night_imgs:
            sample_images.append(os.path.join("data/bdd100k_yolo/images/val", random.choice(night_imgs)))
        if rain_imgs:
            sample_images.append(os.path.join("data/bdd100k_yolo/images/val", random.choice(rain_imgs)))
        # Если какой-то список пуст, просто не берем изображение для этой категории

    # Выполняем детекцию и сохраняем изображения с bounding boxes
    visualize_detections(model, sample_images, args.output_dir)
