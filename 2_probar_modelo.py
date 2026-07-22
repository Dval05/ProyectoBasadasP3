# -*- coding: utf-8 -*-
"""
🧠 2. Detección de Tumores (Inferencia / Pruebas)
"""

import os
import secrets
import tkinter as tk
from tkinter import filedialog
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from ultralytics import YOLO

# 1. Encontrar el modelo más reciente entrenado
runs_dir = os.path.join(os.getcwd(), 'runs', 'detect')
if not os.path.exists(runs_dir):
    print("❌ No se encontró la carpeta 'runs/detect'. Asegúrate de haber ejecutado '1_entrenar_modelo.py' primero.")
    exit(1)

# Buscar la carpeta yolov11_btd más reciente
subdirs = [os.path.join(runs_dir, d) for d in os.listdir(runs_dir) if d.startswith('yolov11_btd') and os.path.isdir(os.path.join(runs_dir, d))]
if not subdirs:
    print("❌ No se encontró ninguna carpeta de entrenamiento 'yolov11_btd'.")
    exit(1)

latest_run = max(subdirs, key=os.path.getmtime)
best_weights = os.path.join(latest_run, 'weights', 'best.pt')

if not os.path.exists(best_weights):
    print(f"❌ No se encontró el archivo de pesos: {best_weights}")
    exit(1)

print(f"Cargando modelo desde: {best_weights}")
trained_model = YOLO(best_weights)

# 2. Rutas para probar imágenes aleatorias
val_dir = os.path.join(os.getcwd(), 'dataset', 'Brain Tumor labeled dataset', 'val', 'images')
if not os.path.exists(val_dir):
    val_dir = os.path.join(os.getcwd(), 'dataset', 'val', 'images')

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
    if not os.path.exists(val_dir):
        print("❌ No se encontró la carpeta de validación para elegir una imagen.")
        return
    all_images = [f for f in os.listdir(val_dir) if f.lower().endswith('.jpg')]
    if not all_images:
        print("❌ No hay imágenes en la carpeta de validación.")
        return
    chosen = secrets.choice(all_images)
    image_path = os.path.join(val_dir, chosen)
    
    print(f"\n📄 Imagen aleatoria: {chosen}")
    result = predict_and_show(image_path)
    if len(result.boxes) > 0:
        best_box = max(result.boxes, key=lambda b: float(b.conf[0]))
        pred_class = result.names[int(best_box.cls[0])]
        pred_conf = float(best_box.conf[0]) * 100
        print(f"🔍 Predicción: {pred_class.capitalize()} ({pred_conf:.1f}% confianza)")
    else:
        print("🔍 El modelo no detectó ningún tumor.")

def predict_uploaded_image():
    root = tk.Tk()
    root.withdraw() 
    root.attributes('-topmost', True) 
    
    print("\nAbriendo ventana para seleccionar imagen...")
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

if __name__ == '__main__':
    print("\n" + "="*50)
    print("🎯 MODO INFERENCIA - YOLOv11")
    print("="*50)
    
    while True:
        print("\n¿Qué deseas hacer ahora?")
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
