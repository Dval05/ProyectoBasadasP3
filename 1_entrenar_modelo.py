# -*- coding: utf-8 -*-
"""
🧠 1. Entrenamiento de YOLOv11 para Detección de Tumores
"""

import os
import zipfile
import shutil
from sklearn.model_selection import train_test_split
from ultralytics import YOLO

print("Librerías importadas ✅")

current_dir = os.getcwd()
zip_filename = os.path.join(current_dir, 'tumor.zip')
extract_path = os.path.join(current_dir, 'dataset')

if not os.path.exists(extract_path):
    if os.path.exists(zip_filename):
        print(f"Extrayendo dataset desde {zip_filename}...")
        with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
        print("Dataset extraído en:", extract_path)
    else:
        print(f"❌ No se encontró {zip_filename}.")
        exit(1)
else:
    print("✅ El dataset ya estaba extraído.")

dataset_directory = os.path.join(extract_path, 'Brain Tumor labeled dataset')
if not os.path.exists(dataset_directory):
    dataset_directory = extract_path 

class_labels = {'pituitary': 0, 'meningioma': 1, 'glioma': 2, 'notumor': 3}
class_names = [k for k, v in sorted(class_labels.items(), key=lambda item: item[1])]

def _process_class(class_name, dataset_directory, val_ratio, random_seed, dirs):
    train_dir, train_lbl, val_dir, val_lbl = dirs
    class_path = os.path.join(dataset_directory, class_name)
    if not os.path.exists(class_path): return
    image_files = [f for f in os.listdir(class_path) if f.endswith('.jpg')]
    if len(image_files) < 2: return
    train_imgs, val_imgs = train_test_split(image_files, test_size=val_ratio, random_state=random_seed)
    for split_imgs, img_dst, lbl_dst in [(train_imgs, train_dir, train_lbl), (val_imgs, val_dir, val_lbl)]:
        for img_name in split_imgs:
            src_img = os.path.join(class_path, img_name)
            src_lbl = os.path.join(class_path, img_name.replace('.jpg', '.txt'))
            if os.path.exists(src_lbl):
                shutil.copy(src_img, os.path.join(img_dst, img_name))
                shutil.copy(src_lbl, os.path.join(lbl_dst, img_name.replace('.jpg', '.txt')))

def custom_train_val_split(dataset_directory, class_names, val_ratio=0.2, random_seed=42):
    train_dir = os.path.join(dataset_directory, 'train', 'images')
    train_lbl = os.path.join(dataset_directory, 'train', 'labels')
    val_dir = os.path.join(dataset_directory, 'val', 'images')
    val_lbl = os.path.join(dataset_directory, 'val', 'labels')
    if os.path.exists(train_dir):
        print("✅ División train/val ya existe.")
        return train_dir, val_dir
    for d in [train_dir, train_lbl, val_dir, val_lbl]:
        os.makedirs(d, exist_ok=True)
    dirs = (train_dir, train_lbl, val_dir, val_lbl)
    for class_name in class_names:
        _process_class(class_name, dataset_directory, val_ratio, random_seed, dirs)
    print("División train/val completada ✅")
    return train_dir, val_dir

train_dir, val_dir = custom_train_val_split(dataset_directory, class_names)

yaml_content = f"""train: {os.path.abspath(train_dir).replace('\\\\', '/')}
val: {os.path.abspath(val_dir).replace('\\\\', '/')}

nc: {len(class_names)}
names: {class_names}
"""
yaml_path = os.path.join(dataset_directory, 'brain_tumor.yaml')
with open(yaml_path, 'w') as f:
    f.write(yaml_content)
print(f"Archivo data.yaml creado en: {yaml_path}")

if __name__ == '__main__':
    print("\n🚀 Iniciando entrenamiento con YOLOv11...")
    model = YOLO('yolo11n.pt')
    results = model.train(
        data=yaml_path,
        epochs=50,
        imgsz=640,
        batch=16,
        name='yolov11_btd',
        device='0',
        workers=4
    )
    print("\n" + "="*50)
    print("¡Entrenamiento Finalizado!")
    print(f"El modelo entrenado se guardó en: {model.trainer.save_dir}\\weights\\best.pt")
    print("Ahora puedes cerrar esto y ejecutar el script '2_probar_modelo.py' para predecir imágenes.")
    print("="*50)
