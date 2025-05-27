import json, glob, os, pathlib
train_dir = r"data\bdd100k\images\100k\train"
json_path = r"data\bdd100k\labels\bdd100k_labels_images_train.json"
j = json.load(open(json_path))
have = {pathlib.Path(p).name for p in glob.glob(train_dir+r'\*.jpg')}
need = {item["name"] for item in j}
missing = sorted(need - have)
print("Отсутствует файлов:", len(missing))
open("missing.txt", "w").write("\n".join(missing))
