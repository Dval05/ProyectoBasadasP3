# 🧠 Explicación del Código: `brain_tumor_yolov11.py`

Este documento explica paso a paso cómo funciona el script principal que se encarga de entrenar el modelo de Inteligencia Artificial (YOLOv11) para detectar tumores cerebrales.

El código está dividido en 8 pasos lógicos principales. Aquí te explico qué hace cada uno de ellos:

---

## 1. Importación de Librerías
Al principio del código, se importan las "herramientas" que Python va a necesitar:
*   **`os`, `zipfile`, `shutil`**: Sirven para manejar archivos, mover carpetas y descomprimir el `.zip` del dataset.
*   **`tkinter`, `filedialog`**: Se usan para crear la ventanita de Windows que te permite buscar y seleccionar una imagen manualmente.
*   **`cv2` (OpenCV), `matplotlib`, `numpy`**: Herramientas matemáticas y gráficas para abrir las imágenes y dibujar los recuadros de colores sobre ellas.
*   **`sklearn`**: Usado para dividir matemáticamente el dataset en "Entrenamiento" y "Validación".
*   **`ultralytics`**: Es la librería oficial que contiene a **YOLO**, el cerebro de la Inteligencia Artificial.

---

## 2. Extracción del Dataset
El código busca un archivo llamado `tumor.zip` en la misma carpeta donde estás ejecutando el proyecto.
*   Si existe, lo descomprime automáticamente en una nueva carpeta llamada `dataset`.
*   Si ya estaba descomprimido previamente (porque ya corriste el código antes), se salta este paso para ahorrar tiempo.

---

## 3. Definición de las Clases (Etiquetas)
Se define un "diccionario" con las 4 categorías que la Inteligencia Artificial debe aprender a reconocer:
1.  `pituitary` (Tumor pituitario)
2.  `meningioma` (Tumor meningioma)
3.  `glioma` (Tumor glioma)
4.  `notumor` (Cerebro sano sin tumor)

Se les asigna un número (del 0 al 3) y se ordenan explícitamente. Esto es crítico porque si el orden de las palabras no coincide con el número que YOLO espera, la IA empezará a confundir los tumores.

---

## 4. División de Datos (Train y Validation)
Para que una IA aprenda correctamente, no se le pueden dar todas las imágenes para estudiar. El código toma todas las fotos y las divide en dos grupos (usando la función `custom_train_val_split` y `sklearn`):
*   **Train (80% de las imágenes):** Son las fotos y etiquetas (coordenadas) que el modelo usará para estudiar y aprender a identificar patrones.
*   **Val (20% de las imágenes):** Son las fotos que se esconden durante el estudio. Luego se usan como un "examen sorpresa" para comprobar si el modelo de verdad aprendió o si solo memorizó las primeras fotos.

---

## 5. Creación del Archivo Configuración (`data.yaml`)
YOLO es muy estricto y necesita un archivo de configuración `.yaml` para saber dónde están guardadas las fotos.
El código detecta las *rutas absolutas locales* de tu computadora (por ejemplo `D:\ESPE - ISOW\...`) y genera este archivo llamado `brain_tumor.yaml` de forma automática, indicando cuántas clases existen y cuáles son sus nombres.

---

## 6. El Entrenamiento (El Corazón del Código)
Esta parte siempre debe estar debajo de `if __name__ == '__main__':` en Windows para evitar que la computadora colapse al usar múltiples procesadores.
```python
model = YOLO('yolo11n.pt')
results = model.train( ... )
```
Aquí ocurren 3 cosas:
1.  Descarga un "cerebro" pre-entrenado básico llamado `yolo11n.pt` (la versión Nano).
2.  Le pasa el archivo `.yaml` que creamos.
3.  Comienza el entrenamiento usando la Tarjeta Gráfica (`device='0'`), tomando lotes de 16 fotos a la vez (`batch=16`), usando 4 procesos paralelos (`workers=4`), y repitiendo el estudio 50 veces (`epochs=50`).

---

## 7. Función de Predicción y Visualización (`predict_and_show`)
Una vez que el modelo termina de entrenar, ya sabe detectar tumores.
Se crea una función matemática que:
1.  Recibe la ruta de una foto cualquiera.
2.  Le pide a YOLO que la analice (`trained_model.predict`).
3.  Usa `matplotlib` para mostrar la foto en la pantalla y dibujar un recuadro verde brillante (`lime`) justo donde YOLO encontró las coordenadas del tumor, incluyendo el porcentaje de confianza (ej. "Meningioma 95.2%").

---

## 8. El Menú Interactivo en Consola
Es un bucle infinito (`while True`) que mantiene el programa vivo en la terminal y te da 3 opciones:
*   **Opción 1 (`predict_random_image`):** Va a la carpeta de validación (las fotos del "examen sorpresa"), elige una foto al azar, se la manda a la función de predicción y te la muestra en pantalla.
*   **Opción 2 (`predict_uploaded_image`):** Llama a `tkinter` para abrir una ventanita clásica de Windows donde puedes buscar cualquier imagen en tu computadora, seleccionarla y hacer que la IA la analice.
*   **Opción 3:** Rompe el bucle con un `break` y finaliza el programa de forma segura.
