# -*- coding: utf-8 -*-
"""
🧠 Detección de Tumores Cerebrales con YOLOv11 (Versión Local)

Este script ha sido adaptado de Google Colab para ejecutarse localmente.
Aprovechará tu tarjeta gráfica (RTX 4070) si tienes CUDA configurado con PyTorch.
"""

import os
import zipfile
import random
import secrets
import shutil
import tkinter as tk
from tkinter import filedialog

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2

from sklearn.model_selection import train_test_split
from ultralytics import YOLO

print("Librerías importadas ✅")

# 1. Rutas locales
current_dir = os.getcwd()
zip_filename = os.path.join(current_dir, 'tumor.zip')
extract_path = os.path.join(current_dir, 'dataset')

# 2. Descomprimir el dataset (si no está descomprimido ya)
if not os.path.exists(extract_path):
    if os.path.exists(zip_filename):
        print(f"Extrayendo dataset desde {zip_filename}...")
        with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
        print("Dataset extraído en:", extract_path)
    else:
        print(f"❌ No se encontró el archivo {zip_filename}. Por favor, asegúrate de tenerlo en la carpeta.")
        exit(1)
else:
    print("✅ El dataset ya estaba extraído.")

# 3. Definir la ruta base y las clases
dataset_directory = os.path.join(extract_path, 'Brain Tumor labeled dataset')
if not os.path.exists(dataset_directory):
    # Por si se extrajo de manera distinta
    dataset_directory = extract_path 

class_labels = {'pituitary': 0, 'meningioma': 1, 'glioma': 2, 'notumor': 3}
# Ordenamos las clases explícitamente por su ID para que coincidan 0, 1, 2, 3
class_names = [k for k, v in sorted(class_labels.items(), key=lambda item: item[1])]

print("Clases:", class_names)

# 4. Dividir en train / val
def _process_class(class_name, dataset_directory, val_ratio, random_seed, dirs):
    train_dir, train_lbl, val_dir, val_lbl = dirs
    class_path = os.path.join(dataset_directory, class_name)
    if not os.path.exists(class_path):
        return
    image_files = [f for f in os.listdir(class_path) if f.endswith('.jpg')]
    if len(image_files) < 2:
        return

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
    print("Train:", len(os.listdir(train_dir)), "imágenes")
    print("Val:", len(os.listdir(val_dir)), "imágenes")
    return train_dir, val_dir

train_dir, val_dir = custom_train_val_split(dataset_directory, class_names, val_ratio=0.2, random_seed=42)

# 5. Crear el archivo data.yaml para YOLO con RUTAS ABSOLUTAS LOCALES
yaml_content = f"""train: {os.path.abspath(train_dir).replace('\\\\', '/')}
val: {os.path.abspath(val_dir).replace('\\\\', '/')}

nc: {len(class_names)}
names: {class_names}
"""

yaml_path = os.path.join(dataset_directory, 'brain_tumor.yaml')
with open(yaml_path, 'w') as f:
    f.write(yaml_content)

print(f"Archivo data.yaml creado en: {yaml_path}")

# 6. Entrenar el modelo YOLOv11
print("\\n🚀 Iniciando entrenamiento con YOLOv11...")

if __name__ == '__main__':
    # YOLO requiere que la inicialización se haga bajo __main__ en Windows para multiprocessing.
    model = YOLO('yolo11n.pt')

    #entrenamos utilizando device=0 (GPU).
    results = model.train(
        data=yaml_path,
        epochs=50,
        imgsz=640,
        batch=16,
        name='yolov11_btd',
        device='0',
        workers=4
    )

    # 7. Métricas y Función de predicción
    run_dir = model.trainer.save_dir
    best_weights = os.path.join(run_dir, 'weights', 'best.pt')
    trained_model = YOLO(best_weights)

    def predict_and_show(image_path, conf_threshold=0.3):
        result = trained_model.predict(image_path, imgsz=640, conf=conf_threshold, verbose=False)[0]
        img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)

        plt.figure(figsize=(6, 6))
        plt.imshow(img)
        ax = plt.gca()

        if len(result.boxes) == 0:
            plt.title("No se detectó ningún tumor", fontsize=14, color='green')
        else:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                label = f"{result.names[cls_id].capitalize()} ({conf*100:.1f}%)"

                rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                          linewidth=2, edgecolor='lime', facecolor='none')
                ax.add_patch(rect)
                ax.text(x1, max(y1 - 8, 0), label, color='black', fontsize=11,
                        bbox={'facecolor': 'lime', 'alpha': 0.7, 'pad': 2})

        plt.axis('off')
        plt.show()
        return result

    def predict_random_image():
        all_images = [f for f in os.listdir(val_dir) if f.lower().endswith('.jpg')]
        chosen = secrets.choice(all_images)
        image_path = os.path.join(val_dir, chosen)
        
        print(f"\\n📄 Imagen aleatoria: {chosen}")
        result = predict_and_show(image_path)
        if len(result.boxes) > 0:
            best_box = max(result.boxes, key=lambda b: float(b.conf[0]))
            pred_class = result.names[int(best_box.cls[0])]
            pred_conf = float(best_box.conf[0]) * 100
            print(f"🔍 Predicción: {pred_class.capitalize()} ({pred_conf:.1f}% confianza)")
        else:
            print("🔍 El modelo no detectó ningún tumor.")

    def predict_uploaded_image():
        # Usar tkinter para abrir ventana de selección de archivos
        root = tk.Tk()
        root.withdraw() # Ocultar ventana principal
        root.attributes('-topmost', True) # Poner al frente
        
        print("\\nAbriendo ventana para seleccionar imagen...")
        file_path = filedialog.askopenfilename(
            title="Selecciona una imagen MRI",
            filetypes=[("Image files", "*.jpg *.jpeg *.png")]
        )
        
        if file_path:
            print(f"Imagen seleccionada: {file_path}")
            result = predict_and_show(file_path)
            if len(result.boxes) > 0:
                best_box = max(result.boxes, key=lambda b: float(b.conf[0]))
                pred_class = result.names[int(best_box.cls[0])]
                pred_conf = float(best_box.conf[0]) * 100
                print(f"🔍 Predicción: {pred_class.capitalize()} ({pred_conf:.1f}% confianza)")
            else:
                print("🔍 El modelo no detectó ningún tumor en la imagen.")
        else:
            print("❌ No se seleccionó ninguna imagen.")

    # 8. Menú Interactivo en consola
    print("\\n" + "="*50)
    print("¡Entrenamiento Finalizado! Modelo guardado en:", best_weights)
    print("="*50)
    
    while True:
        print("\\n¿Qué deseas hacer ahora?")
        print("1. Probar con una imagen aleatoria del dataset de validación")
        print("2. Subir (seleccionar) tu propia imagen desde la computadora")
        print("3. Salir")
        
        opcion = input("Elige una opción (1/2/3): ")
        if opcion == '1':
            predict_random_image()
        elif opcion == '2':
            predict_uploaded_image()
        elif opcion == '3':
            print("¡Hasta luego!")
            break
        else:
            print("Opción no válida. Intenta de nuevo.")
