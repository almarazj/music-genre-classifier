# 🏆 Machine Learning Challenge: Audio Specialization
### Tabla de contenidos
- [🌱 Introducción](#🌱-introducción)
- [ℹ️ Consideraciones generales del desarrollo](#ℹ️-consideraciones-generales-del-desarrollo)
- [📨 Entrega del challenge](#📨-entrega-del-challenge)
- [🧩 Preprocesamiento](#🧩-preprocesamiento)
- [🤖 Diseño del modelo](#🤖-diseño-del-modelo)
- [🏋️‍♂️ Entrenamiento](#🏋️‍♂️-entrenamiento)
- [🪄 Inferencia](#🪄-inferencia)
- [📚 Bibliografía](#📚-bibliografía)

---
# 🌱 Introducción
El siguiente challenge tiene como finalidad implementar un clasificador simple de géneros musicales para introducir la lógica, arquitectura, estructura y metodologías implementadas en el desarrollo de proyectos de Machine Learning.

Se evaluará el desarrollo en las siguientes categorías: 
- Prolijidad y legibilidad del código.
- Criterio en el diseño de funciones y/o clases.
- Lectura e implementación criteriosa de la documentación.
- Comunicación con el resto del equipo ante dudas y/o inconvenientes.
- Uso mínimo de versionado de código.

---

# ℹ️ Consideraciones generales del desarrollo

- Cada persona que lleve a cabo el challenge, debe **desarrollar todo su código en una nueva rama del repositorio**, la cual se debe llamar `challenge/usuario-de-github` con su usuario de GitHub pertinente.

- El código desarrollado debe estar en idioma **ingles**, lo cual abarca:
    - Nombres de variables
    - Nombres de funciones/clases
    - Comentarios en el documento
    - Documentación (docstrings)

- Cada función, clase o método debe ser correctamente documentado a través de sus **docstrings** en **formato numpy**.
- Para poder replicar el entorno de desarrollo, se debe **generar un archivo** `requirements.txt` con las librerías utilizadas a lo largo del challenge, como sus versiones.

---

# 📨 Entrega del challenge 

- La entrega final de challenge será mediante un **pull request** hacia la rama `main` del repositorio. En el mismo deberán incluir **capturas de pantalla** con el log de las ultimas epochs del entrenamiento, exhibiendo las métricas, asi como los prints en consola con la inferencia.

---
# 🧩 Preprocesamiento
El preprocesamiento del dataset se debe realizar dentro del script `challenge/preprocessing.py` con los siguientes requerimientos:
### **Requerimientos**
- En el preprocesamiento se debe poder splitear el audio en `n` segmentos, de forma que aumente la cantidad de muestras en el dataset.
- Para no procesar los audios como series temporales, se deben transformar los audios en **espectrogramas de Mel** (*Consejo: Utilizar funcionalidades de la librería Librosa*)
- El dataset deberá ser procesado por género, y **cada espectrograma debe ser guardado en un** `.npz` (NumPy Compressed file), es decir, un archivo `.npz` por espectrograma. Investigar que funciones de NumPy permiten guardar un array de forma comprimida.
    - Para que el Dataset y DataLoader funcionen correctamente (los cuales están dentro de `challenge/data_managment/dataset.py` y ya están codeados), el nombre de la variable dentro de la función de guardado se debe llamar `spectrogram`
- Los archivos deben estár organizados por género dentro de la carpeta  `genres_mel_npz/`, la cual se aloja dentro de `dataset/`.
- **La estructura de carpetas y archivos debe quedar así**:
    ```python
    dataset/
        ├── genres_original/
        ├── images_original/
        ├── features_3_sec.csv
        ├── features_30_sec.csv
        └── genres_mel_npz/
                ├── blues
                |    ├── blues.00000-0.npz
                |    ├── blues.00000-1.npz
                |    ├── blues.00000-2.npz
                |    ├── blues.00000-3.npz
                |    ├── blues.00000-4.npz
                |    └── ...
                ├── rock
                └── ...
    ```

### **Recomendación**
- Recordar que las transformaciones aplicadas al dataset original se deben poder aplicar tambien en la etapa de inferencia del modelo, es decir, si mi modelo está entrenado con espectrogramas de Mel, la inferencia la debo hacer con espectrogramas de Mel tambien. Por lo tanto, si quiero inferir un audio representado por una serie temporal (archivo .wav, .mp3, etc), se le debe aplicar el mismo preprocesamiento que se le aplicó al conjunto de entrenamiento.

---

# 🤖 Diseño del modelo
Se solicita diseñar una **red neuronal convolucional** con 3 capaz convolutivas, seguido de una capa "aplanadora" (flatten) y un clasificador. La misma debe ser desarrollada dentro del script `challenge/models/cnn.py`

### **Requerimientos**

- El modelo debe ser construido utilizando **métodos de PyTorch**. La arquitectura de la misma debe ser replicada del siguiente fragmento de código, donde se define la arquitectura utilizando métodos de **Keras**

- El objetivo principal es **practicar y agilizar la lectura de documentación**, tanto de TensorFlow como de PyTorch, para analizar las similitudes y diferencias entre estas dos librerias.

    ```python
    # 1st conv layer
    model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(...)))
    model.add(keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same'))
    model.add(keras.layers.BatchNormalization())

    # 2nd conv layer
    model.add(keras.layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same'))
    model.add(keras.layers.BatchNormalization())

    # 3rd conv layer
    model.add(keras.layers.Conv2D(32, (2, 2), activation='relu'))
    model.add(keras.layers.MaxPooling2D((2, 2), strides=(2, 2), padding='same'))
    model.add(keras.layers.BatchNormalization())

    # flatten output and feed it into dense layer
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.Dropout(0.3))

    # output layer
    model.add(keras.layers.Dense(10, activation='softmax'))
    ```

---

# 🏋️‍♂️ Entrenamiento
El entrenamiento debe ser desarrollado dentro del script `challenge/training.py`.

### **Requerimientos**

- Las **métricas** para evaluar el rendimiento del modelo serán la pérdida (`loss`) y la presición (`accuracy`).

- Dentro de este script, se debe definir el loop de entrenamiento, el loop de validación, el cálculo de métricas por batch, y el **guardado del modelo entrenado**.

- A su vez, para calcular la pérdida por batch se debe utilizar la entropía cruzada (`CrossEntropyLoss`), y la optimización debe ser implementada con el algoritmo de `Adam`. 

- Con respecto a los **hiperparámetros del modelo**, para poder comparar resultados, se precisa:

    - Epochs = 300
    - Learning rate = 1e-3


---

# 🪄 Inferencia

La inferencia se debe realizar dentro del script `challenge/inference.py`

### **Requerimientos**

- En primer lugar, se debe cargar el **modelo guardado** en la carpeta `results/`
- A la hora de exhibir los resultados, se debe imprimir en consola el género predicho.

---

# 📚 Bibliografía
[Understanding the Mel Spectrogram - Medium](https://medium.com/analytics-vidhya/understanding-the-mel-spectrogram-fca2afa2ce53)

[Feature extraction - Librosa](https://librosa.org/doc/main/feature.html)

[Training a classifier - PyTorch Tutorials](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html)

[Learning PyTorch for Deep Learning in a day. Literally - Youtube](https://www.youtube.com/watch?v=Z_ikDlimN6A&t=37814s&ab_channel=DanielBourke)


