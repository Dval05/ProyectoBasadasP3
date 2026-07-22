# 🧠 Explicación General del Código del Proyecto

Este documento explica de forma clara y general cómo funciona el código fuente de este proyecto de detección de tumores cerebrales utilizando Inteligencia Artificial (YOLOv11).

---

## 1. Estructura de los Archivos `.py`

El proyecto está diseñado para ser flexible, por lo que la lógica se dividió para que puedas ejecutar todo de golpe o paso a paso. Existen tres scripts principales:

*   **`brain_tumor_yolov11.py` (El Todo-en-Uno):** Este es el archivo principal que hace *absolutamente todo*. Descomprime los datos, divide las imágenes, configura el entorno, entrena a la Inteligencia Artificial por 50 épocas y, finalmente, abre el menú interactivo para probar imágenes.
*   **`1_entrenar_modelo.py` (Solo Entrenamiento):** Contiene únicamente la primera mitad de la lógica. Su único propósito es preparar los datos, entrenar la red neuronal con tu Tarjeta Gráfica y guardar el "cerebro" resultante (`best.pt`). Lo usas una sola vez.
*   **`2_probar_modelo.py` (Solo Predicción):** Contiene la segunda mitad lógica. Asume que el modelo ya fue entrenado. Simplemente carga los pesos (`best.pt`) y despliega el menú interactivo para que puedas subir radiografías en cualquier momento sin tener que volver a esperar horas de entrenamiento.

---

## 2. La Lógica Paso a Paso (Cómo funciona por dentro)

Sin importar cuál de los scripts ejecutes, la lógica sigue un orden secuencial estricto. Aquí te explicamos qué hace el código por detrás:

### Fase A: Preparación de Datos
1.  **Extracción del Dataset:** El código busca un archivo `tumor.zip` en la carpeta. Si lo encuentra, lo descomprime automáticamente creando una carpeta llamada `dataset`. Si ya existe, se salta este paso para ahorrar tiempo.
2.  **Definición de las Clases:** Le enseña a la IA que existen 4 categorías enumeradas: `0: Pituitary`, `1: Meningioma`, `2: Glioma` y `3: Notumor` (Cerebro sano). *El orden numérico es crítico para evitar confusiones.*
3.  **División (Train / Validation):** De todas las fotos extraídas, separa aleatoriamente el **80%** para "Entrenamiento" (imágenes que la IA estudiará) y el **20%** para "Validación" (imágenes que se le ocultan a la IA para hacerle un examen sorpresa más adelante).
4.  **Generación del `data.yaml`:** YOLO necesita saber las rutas exactas de tu computadora (ej. `D:\Proyecto\dataset\...`). El script genera este archivo automáticamente para que YOLO no se pierda buscando las imágenes.

### Fase B: Entrenamiento del Modelo
5.  **El "Cerebro" Base:** Se descarga `yolo11n.pt` (la versión Nano). Es un modelo básico que ya sabe detectar objetos comunes, pero no sabe nada de medicina.
6.  **Configuración de Hardware y Aprendizaje:** 
    *   Se le ordena entrenar por `50 épocas` (repetir el estudio del dataset 50 veces).
    *   Se utiliza `batch=16` (agrupa 16 imágenes a la vez).
    *   Se usan `workers=4` (hilos paralelos cargando fotos para agilizar).
    *   **El motor:** Usa `device='0'` para forzar a la computadora a utilizar todo el poder de tu tarjeta gráfica NVIDIA en lugar del procesador central.

### Fase C: Predicción e Interfaz de Usuario
7.  **Dibujo de Resultados (Matplotlib & OpenCV):** Una vez entrenado, se crea una función matemática. Al pasarle una foto de un cerebro, la IA escupe coordenadas `(X, Y)`. El código usa estas coordenadas para dibujar un recuadro verde brillante alrededor del tumor y estamparle el porcentaje de seguridad (ej. *Glioma 98.4%*).
8.  **Menú Interactivo en Consola:** Es un bucle que no termina (`while True`) y espera tus órdenes:
    *   *Opción 1:* Saca una foto al azar de las de "Validación" y te muestra si la IA acierta.
    *   *Opción 2:* Abre la clásica ventanita de Windows (`tkinter`) para que explores tus carpetas, elijas un archivo `.jpg` y se lo envíes a la IA para analizarlo.
    *   *Opción 3:* Cierra el programa de forma limpia.
