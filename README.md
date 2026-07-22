# 🧠 Detección y Clasificación de Tumores Cerebrales con YOLOv11

Este proyecto es un sistema avanzado de **Visión por Computador (Computer Vision)** diseñado para el análisis de imágenes médicas (*Medical Imaging*). Utiliza modelos de *Deep Learning* para detectar y clasificar automáticamente distintos tipos de tumores cerebrales a partir de Resonancias Magnéticas (MRI).

## 🚀 Sobre el Proyecto

El objetivo principal de este proyecto es asistir en el diagnóstico médico temprano mediante Inteligencia Artificial. A diferencia de un modelo tradicional de clasificación que solo te dice "qué hay" en la imagen, este sistema utiliza **Detección de Objetos (Object Detection)**. 

Esto significa que la red neuronal no solo clasifica el tipo de tumor, sino que **localiza espacialmente la anomalía** dibujando una caja delimitadora (*bounding box*) exactamente donde se encuentra el tumor dentro del cerebro.

### Clases Identificadas
El modelo es capaz de distinguir entre las siguientes condiciones:
1. **Glioma:** Tumor originado en las células gliales del cerebro.
2. **Meningioma:** Tumor que se forma en las meninges (membranas protectoras).
3. **Pituitary (Adenoma Pituitario):** Tumor en la glándula pituitaria.
4. **Notumor (Sano):** Cerebros sin anomalías tumorales detectadas.

## 🤖 ¿Por qué es un proyecto de Visión por Computador?

Este proyecto aplica los conceptos fundamentales de la Visión Artificial moderna:
* **Arquitectura CNN:** Utiliza la arquitectura de Redes Neuronales Convolucionales (CNN) de **YOLOv11** para extraer características complejas (bordes, texturas, formas) directamente de los píxeles de las resonancias magnéticas.
* **Detección Localizada:** Resuelve un problema de regresión de coordenadas matemáticas para predecir las coordenadas `(x, y)` y el tamaño `(ancho, alto)` del tumor en tiempo real.
* **Inferencia de Alto Rendimiento:** Aprovecha la aceleración por hardware (NVIDIA CUDA / GPUs) para procesar las predicciones matriciales casi instantáneamente.

## 🛠️ Tecnologías y Librerías Utilizadas

* **Python 3.12**: Lenguaje base del proyecto.
* **Ultralytics (YOLOv11)**: Arquitectura principal de *Deep Learning* para el entrenamiento y la predicción.
* **PyTorch (con soporte CUDA)**: Motor de tensores utilizado por YOLO para entrenar en la tarjeta gráfica (GPU).
* **OpenCV & Matplotlib**: Para el procesamiento de matrices de imágenes y visualización gráfica.
* **Scikit-learn**: Para la estructuración y partición matemática del dataset de validación.

## 📖 Instrucciones de Instalación y Uso

Para ejecutar el código localmente, instalar las dependencias (PyTorch, Ultralytics) y hacer uso del menú interactivo, por favor dirígete a la guía técnica detallada:

👉 **[Ver INSTRUCCIONES.md](./INSTRUCCIONES.md)**

Allí encontrarás el paso a paso sobre cómo crear el Entorno Virtual, aprovechar el rendimiento y usar las interfaces de predicción.
