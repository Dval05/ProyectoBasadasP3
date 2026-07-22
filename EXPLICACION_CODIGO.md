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
3.  **Filtrado y División (Train / Validation):** El código utiliza el **100% de las imágenes válidas** del dataset. Al procesar las carpetas, hace una verificación de seguridad muy importante: si encuentra una foto (`.jpg`) que no tiene su archivo de texto correspondiente con las coordenadas (`.txt`), la ignora automáticamente para no arruinar el entrenamiento. Todo el resto de la data (el 100% válido) se divide aleatoriamente: el **80%** se guarda en `train` (imágenes que la IA estudiará) y el **20%** se guarda en `val` (imágenes que se le ocultan a la IA para hacerle un examen sorpresa más adelante).
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

---

## 3. Datos Técnicos Adicionales (Importante para Reportes)

Si necesitas documentar tu proyecto para la universidad o explicarlo en una presentación, ten en cuenta estos 3 detalles técnicos que ocurren "tras bambalinas" en el código:

### 1. La Carpeta de Resultados (`runs/`)
Cuando YOLO termina de entrenar, genera automáticamente una carpeta llamada `runs/detect/yolov11_btd/`.
Esta carpeta es invaluable porque contiene **evidencia matemática** de que el modelo aprendió. Allí encontrarás imágenes de prueba y gráficas estadísticas de rendimiento (como la *Matriz de Confusión*, la curva *F1-Confidence*, y las gráficas de *Pérdida/Loss*). Estas gráficas son perfectas y necesarias para adjuntar en reportes académicos.

### 2. El Archivo de Pesos Final (`best.pt`)
Al inicio descargamos el modelo vacío `yolo11n.pt`, pero el objetivo real de todo este entrenamiento es generar un nuevo archivo llamado `best.pt` (el cual se guarda dentro de la carpeta `runs/detect/.../weights/`).
Este archivo es tu **"Cerebro Congelado"**. Contiene todo el conocimiento médico que la IA adquirió tras estudiar las fotos durante las 50 épocas. Si en el futuro quieres crear una página web o una app móvil para detectar tumores, no necesitas volver a llevarte el código de entrenamiento ni la enorme carpeta de imágenes; solo necesitas llevarte ese pequeño archivo `best.pt`.

### 3. El Umbral de Confianza (`conf_threshold = 0.3`)
En la función que dibuja el recuadro verde de la predicción, existe un parámetro oculto llamado `conf_threshold=0.3`.
Esto significa que has programado a la IA para ser cautelosa: si la red neuronal encuentra una "mancha" en el cerebro pero está **menos del 30% segura** de que es un tumor, simplemente no la marcará. Esto es crucial para evitar *Falsos Positivos* (decirle a una persona sana que tiene un tumor) por culpa de cualquier ruido o mancha normal en la radiografía.
