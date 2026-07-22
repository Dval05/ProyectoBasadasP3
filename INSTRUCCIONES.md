# Guía de Instalación y Ejecución Local (YOLOv11 - Detección de Tumores)

Este documento detalla los pasos necesarios para preparar tu entorno de Windows, instalar las librerías aprovechando tu tarjeta gráfica NVIDIA RTX 4070, y cómo ejecutar los scripts de entrenamiento y predicción.

---

## 1. Requisitos Previos e Instalación

Es altamente recomendable utilizar un Entorno Virtual (`venv`) para no mezclar las librerías de este proyecto con las globales de tu sistema. Además, para que el modelo entrene rápido y utilice correctamente la GPU, debes instalar una versión específica de PyTorch compatible con CUDA (requiere **Python 3.12** o inferior).

### Paso 1.1: Crear y activar un Entorno Virtual (venv)
Abre una terminal (PowerShell o CMD) en la carpeta de tu proyecto (`Proyecto`) y ejecuta lo siguiente para crear el entorno con Python 3.12 (si instalaste esta versión):

```bash
py -3.12 -m venv venv
```

Luego, actívalo con:
* **En PowerShell:** `.\venv\Scripts\Activate.ps1`
* **En Símbolo del Sistema (CMD):** `.\venv\Scripts\activate.bat`

*(Sabrás que está activado porque verás un `(venv)` al inicio de la línea de tu consola).*

### Paso 1.2: Instalar PyTorch con soporte para CUDA
Con el entorno virtual activado, ejecuta el siguiente comando. Esto descargará los motores que permitirán a Python usar tu RTX 4070.

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Paso 1.3: Instalar el resto de dependencias
A continuación, instala Ultralytics (el creador de YOLO) y las librerías necesarias para procesar imágenes y gráficas:

```bash
pip install ultralytics opencv-python matplotlib scikit-learn
```

---

## 2. Archivos del Proyecto

Tienes dos formas de ejecutar este proyecto, dependiendo de tu preferencia. Ambas hacen exactamente lo mismo.

### Opción A: El Script Unificado
* **`brain_tumor_yolov11.py`**: Este archivo hace absolutamente todo de una sola vez. Descomprime las imágenes, configura los archivos, entrena el modelo por 50 épocas y, al terminar, abre el menú interactivo para que pruebes las imágenes.

### Opción B: Los Scripts Separados (Recomendado para pruebas rápidas)
* **`1_entrenar_modelo.py`**: Solo entrena el modelo y guarda los resultados. Lo usas una vez.
* **`2_probar_modelo.py`**: Solo abre el menú interactivo para probar imágenes. Lo usas cada vez que quieras predecir un tumor sin tener que esperar a que el modelo se vuelva a entrenar.

---

## 3. Detalles de la Configuración del Entrenamiento

* **Modelo base:** `yolo11n.pt` (La versión "Nano" de YOLOv11. Es la más ligera y rápida).
* **Épocas (Epochs):** `50` (El modelo pasará por todo el dataset 50 veces para aprender).
* **Tamaño de imagen (imgsz):** `640` (Redimensiona las imágenes a 640x640 píxeles).
* **Lote (Batch):** `16` (Envía las imágenes a la tarjeta gráfica en grupos de 16 para optimizar la memoria de video).
* **Trabajadores (Workers):** `4` (Número de subprocesos paralelos que se encargan de cargar y preprocesar las imágenes antes de enviarlas a entrenar. Ayuda a acelerar el proceso evitando que la GPU se quede esperando datos).
* **Dispositivo (Device):** `0` (Fuerza estrictamente a YOLO a usar la tarjeta gráfica NVIDIA).

> [!TIP]
> **¿Qué pasa si no tengo Tarjeta Gráfica NVIDIA (Laptops o MacBooks)?**
> El código está configurado por defecto para tarjetas NVIDIA (`device='0'`). Si lo ejecutas en una computadora sin tarjeta gráfica compatible, dará error.
> Para solucionarlo, debes editar el script (`1_entrenar_modelo.py` o `brain_tumor_yolov11.py`) y cambiar el parámetro `device` en la función `model.train()`:
> * **En Laptops Windows/Linux sin gráfica:** Usa `device='cpu'`
> * **En MacBooks modernas (chips Apple Silicon M1/M2/M3):** Usa `device='mps'`
> * **En MacBooks antiguas (procesadores Intel):** Usa `device='cpu'`
> 
> *Advertencia sobre el tiempo:* Entrenar el modelo usando `cpu` será considerablemente más lento que con GPU (podría tomar días en lugar de minutos/horas). Si vas a usar CPU, es recomendable reducir el número de `epochs` o utilizar alternativas en la nube como Google Colab o Kaggle.

> [!NOTE]
> **Corrección de Etiquetas del Dataset (Bug Fix):**
> En el código original de Kaggle, el orden de las clases parecía ser alfabético. Sin embargo, los archivos `.txt` internos del dataset tenían otro orden (`0`: Pituitary, `1`: Meningioma, `2`: Glioma, `3`: Notumor). Esto causaba que el modelo predijera clases incorrectas. **Este error ya está corregido en todos los scripts de este proyecto** ordenando el diccionario explícitamente.
> 
> **Filtrado Inteligente de Datos:** Adicionalmente, el script está programado para procesar el 100% de tus imágenes, pero descartará de manera automática aquellas fotos (`.jpg`) que no tengan su archivo de etiqueta correspondiente (`.txt`). Esto garantiza que la red neuronal aproveche toda la data posible, pero aprenda solo de datos limpios.

---

## 4. Uso del Menú Interactivo

Al finalizar el entrenamiento (o al abrir `2_probar_modelo.py`), la consola te mostrará el siguiente menú:

```text
==================================================
¡Entrenamiento Finalizado! Modelo guardado en: ...
==================================================

¿Qué deseas hacer ahora?
1. Probar con una imagen aleatoria del dataset de validación
2. Subir (seleccionar) tu propia imagen desde la computadora
3. Salir
Elige una opción (1/2/3):
```

* **Opción 1:** El sistema escogerá al azar una resonancia de la carpeta de validación y mostrará en pantalla la detección con su nivel de confianza (porcentaje).
* **Opción 2:** Se abrirá una **ventana del explorador de archivos de Windows**. Busca cualquier imagen de un cerebro (MRI) que hayas descargado, selecciónala, y Python te mostrará si el modelo pudo encontrar un tumor allí.
* **Opción 3:** El programa se cerrará de forma segura.
