import os
import json
import argparse
from tqdm import tqdm
import numpy as np
import cv2
import torch

# Классы (те же, что и в data_preparation.py)
CLASSES = [
    "person", "rider", "car", "truck", "bus",
    "train", "motorcycle", "bicycle", "traffic light", "traffic sign"
]

def load_model(weights_path):
    """Загружает YOLOv5 модель с заданными весами."""
    # Используем torch.hub для загрузки модели YOLOv5 (необходимо наличие ultralytics/yolov5)
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights_path, force_reload=False)  # source='local' по умолчанию
    model.conf = 0.25  # порог уверенности для предсказаний (можно настроить)
    return model

def compute_metrics_for_subset(image_list, model, labels_dir):
    """
    Вычисляет precision, recall, mAP@0.5 для заданного списка изображений.
    image_list: список имен файлов изображений (без пути, .jpg)
    model: загруженная модель YOLOv5
    labels_dir: путь к директории с YOLO-метками (*.txt) для этих изображений
    """
    # Хранилища для вычисления AP
    all_preds = {cls: [] for cls in range(len(CLASSES))}  # preds_by_class: cls -> list of (conf, matched_flag)
    gt_counts = {cls: 0 for cls in range(len(CLASSES))}    # количество GT боксов каждого класса

    # Проходим по всем изображениям, получаем предсказания и сравниваем с GT
    for img_name in tqdm(image_list, desc="Eval images"):
        img_path = os.path.join(val_images_dir, img_name)
        # Загружаем изображение
        img = cv2.imread(img_path)
        if img is None:
            continue  # на случай, если изображение не найдено
        height, width = img.shape[0], img.shape[1]
        # Загружаем предсказания модели для этого изображения
        results = model(img_path)  # можно передать путь
        pred_tensor = results.xyxy[0]  # предсказания для первого (единственного) изображения
        pred_boxes = pred_tensor.cpu().numpy()  # numpy массив формы (N,6): [x1, y1, x2, y2, conf, cls]
        # Загружаем аннотацию (истинные боксы) из YOLO-текстового файла
        label_file = os.path.join(labels_dir, img_name.replace('.jpg', '.txt'))
        gt_boxes = []
        if os.path.exists(label_file):
            with open(label_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls_id = int(parts[0])
                        x_center, y_center, w, h = map(float, parts[1:5])
                        # Конвертируем обратно из YOLO в координаты изображения
                        x_center *= width
                        y_center *= height
                        w *= width
                        h *= height
                        x1 = x_center - w/2
                        y1 = y_center - h/2
                        x2 = x_center + w/2
                        y2 = y_center + h/2
                        gt_boxes.append((cls_id, x1, y1, x2, y2))
        else:
            # Если файла нет, значит на изображении нет объектов (gt_boxes остается пустым)
            gt_boxes = []
        # Обновляем счетчик GT для каждого класса
        for (cls_id, x1, y1, x2, y2) in gt_boxes:
            gt_counts[cls_id] += 1

        # Для удобства проверок создадим флаг matched для GT (чтобы не засчитать несколько TP для одного GT)
        gt_matched = [False] * len(gt_boxes)
        # Обрабатываем каждое предсказание
        for pred in pred_boxes:
            x1_p, y1_p, x2_p, y2_p, conf, cls_pred = pred
            cls_pred = int(cls_pred)
            # Находим соответствующий GT для этого предсказания (с наибольшим IoU)
            best_iou = 0.0
            best_gt_idx = -1
            for gt_idx, (cls_gt, x1_g, y1_g, x2_g, y2_g) in enumerate(gt_boxes):
                if cls_gt != cls_pred or gt_matched[gt_idx]:
                    continue  # класс не совпадает или GT уже сопоставлен с другим предсказанием
                # вычисляем IoU (Intersection over Union)
                inter_x1 = max(x1_p, x1_g)
                inter_y1 = max(y1_p, y1_g)
                inter_x2 = min(x2_p, x2_g)
                inter_y2 = min(y2_p, y2_g)
                if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
                    iou = 0.0
                else:
                    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
                    pred_area = (x2_p - x1_p) * (y2_p - y1_p)
                    gt_area = (x2_g - x1_g) * (y2_g - y1_g)
                    union_area = pred_area + gt_area - inter_area
                    iou = inter_area / union_area if union_area > 0 else 0.0
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
            # Проверяем, считается ли это предсказание True Positive или False Positive
            if best_gt_idx != -1 and best_iou >= 0.5:
                # Совпадение найдено
                gt_matched[best_gt_idx] = True
                tp_flag = 1
                fp_flag = 0
            else:
                tp_flag = 0
                fp_flag = 1
            # Сохраняем предсказание (для AP рассчитываем позже)
            all_preds[cls_pred].append((conf, tp_flag, fp_flag))
        # Любые ненайденные GT (gt_matched == False) будут учитываться как FN через gt_counts и TP count

    # Теперь считаем метрики по накопленным данным:
    sum_TP = 0
    sum_FP = 0
    sum_FN = 0
    precisions = {}
    recalls = {}
    APs = {}

    # Считаем precision/recall и AP для каждого класса
    for cls_id in range(len(CLASSES)):
        preds = all_preds[cls_id]
        total_gt = gt_counts[cls_id]
        if total_gt == 0:
            # Если в датасете не было этого класса, пропускаем (или считаем AP=0)
            APs[cls_id] = 0.0
            continue
        # Сортируем предсказания этого класса по убыванию уверенности
        preds.sort(key=lambda x: x[0], reverse=True)
        tp_cum = 0
        fp_cum = 0
        precisions_cls = []
        recalls_cls = []
        # Идем по предсказаниям
        for conf, tp_flag, fp_flag in preds:
            tp_cum += tp_flag
            fp_cum += fp_flag
            # текущая точность и полнота
            prec = tp_cum / (tp_cum + fp_cum) if (tp_cum + fp_cum) > 0 else 0.0
            rec = tp_cum / total_gt if total_gt > 0 else 0.0
            precisions_cls.append(prec)
            recalls_cls.append(rec)
        # Вычисляем AP (интеграл кривой precision-recall) методом 11 точек или непрерывным методом
        if recalls_cls:
            # Преобразуем в numpy для удобства
            mrec = np.concatenate(([0.0], np.array(recalls_cls), [1.0]))
            mpre = np.concatenate(([0.0], np.array(precisions_cls), [0.0]))
            # Применяем монотонную огибающую для precision
            for i in range(len(mpre) - 2, -1, -1):
                mpre[i] = max(mpre[i], mpre[i+1])
            # Суммируем площади под P-R кривой на отрезках, где recall меняется
            idxs = np.where(mrec[1:] != mrec[:-1])[0]
            ap = 0.0
            for i in idxs:
                ap += (mrec[i+1] - mrec[i]) * mpre[i+1]
            APs[cls_id] = ap
        else:
            APs[cls_id] = 0.0

        # Накопим для общего precision/recall (всех классов вместе)
        sum_TP += tp_cum
        sum_FP += fp_cum
        sum_FN += (total_gt - tp_cum)  # непросчитанные GT как FN для этого класса

    # Общие precision и recall по всем классам (micro-averaged)
    micro_precision = sum_TP / (sum_TP + sum_FP) if (sum_TP + sum_FP) > 0 else 0.0
    micro_recall = sum_TP / (sum_TP + sum_FN) if (sum_TP + sum_FN) > 0 else 0.0
    # Средняя AP по классам (мacro-averaged mAP@0.5)
    mAP = np.mean(list(APs.values()))
    return micro_precision, micro_recall, mAP

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Оценка модели YOLOv5 на BDD100K (условия день/ночь/дождь)")
    parser.add_argument("--data_dir", type=str, default="data/bdd100k_yolo",
                        help="Путь к папке с подготовленным датасетом (в формате YOLO, содержит images/val и labels/val)")
    parser.add_argument("--val_json", type=str, default="data/bdd100k/labels/bdd100k_labels_images_val.json",
                        help="Путь к исходному JSON-файлу с аннотациями валидации (для извлечения условий съемки)")
    parser.add_argument("--weights", type=str, default="runs/train/bdd100k_yolov5_exp/weights/best.pt",
                        help="Путь к файлу весов модели (best.pt) для оценки")
    args = parser.parse_args()

    data_dir = args.data_dir
    val_json_path = args.val_json
    weights_path = args.weights

    # Загружаем атрибуты изображений из JSON (время суток, погода)
    with open(val_json_path, 'r') as f:
        val_data = json.load(f)
    # Группируем изображения по условиям
    day_images = []
    night_images = []
    rain_images = []
    for item in val_data:
        # Определяем имя файла и атрибуты
        img_name = item["name"]
        attrs = item.get("attributes", {})
        timeofday = attrs.get("timeofday", "").lower()  # 'daytime', 'night', 'dawn/dusk', ...
        weather = attrs.get("weather", "").lower()      # 'clear', 'rainy', 'snowy', ...
        if timeofday == "daytime":
            day_images.append(img_name)
        elif timeofday == "night":
            night_images.append(img_name)
        if weather == "rainy":
            rain_images.append(img_name)

    # Пути к изображениям и меткам
    val_images_dir = os.path.join(data_dir, "images", "val")
    val_labels_dir = os.path.join(data_dir, "labels", "val")

    # Загружаем модель
    model = load_model(weights_path)

    # Оцениваем для каждой группы
    if day_images:
        precision_day, recall_day, mAP_day = compute_metrics_for_subset(day_images, model, val_labels_dir)
    else:
        precision_day = recall_day = mAP_day = 0.0
    if night_images:
        precision_night, recall_night, mAP_night = compute_metrics_for_subset(night_images, model, val_labels_dir)
    else:
        precision_night = recall_night = mAP_night = 0.0
    if rain_images:
        precision_rain, recall_rain, mAP_rain = compute_metrics_for_subset(rain_images, model, val_labels_dir)
    else:
        precision_rain = recall_rain = mAP_rain = 0.0

    # Вывод результатов
    print("Результаты оценки модели YOLOv5 на BDD100K:")
    print(f"Всего изображений валидации: {len(val_data)}")
    print(f"Дневные (timeofday=daytime): {len(day_images)} | Precision={precision_day:.4f}, Recall={recall_day:.4f}, mAP@0.5={mAP_day:.4f}")
    print(f"Ночные  (timeofday=night):   {len(night_images)} | Precision={precision_night:.4f}, Recall={recall_night:.4f}, mAP@0.5={mAP_night:.4f}")
    print(f"Дождь   (weather=rainy):     {len(rain_images)} | Precision={precision_rain:.4f}, Recall={recall_rain:.4f}, mAP@0.5={mAP_rain:.4f}")
