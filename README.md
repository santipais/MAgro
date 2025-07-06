## Introduction

# Toolbox MMSegmentation
## ⚙️ Pasos de instalación

### ⚠️ Nota

Estos pasos fueron los utilizados para instalar la toolbox **MMSegmentation** en el siguiente entorno.  
Ante cualquier duda o incompatibilidad, se recomienda revisar la guía oficial de instalación de MMSegmentation:

👉 https://mmsegmentation.readthedocs.io/en/latest/get_started.html

---

### 🖥️ Entorno utilizado

- **Sistema operativo:** Windows 10/11 usando **WSL 2.0**
- **Distribución:** Ubuntu en WSL
- **GPU:** NVIDIA GeForce RTX 3060
- **CUDA Toolkit:** 11.5 (descargar aquí: https://developer.nvidia.com/cuda-11-5-0-download-archive?target_os=Linux&target_arch=x86_64&Distribution=WSL-Ubuntu&target_version=2.0&target_type=deb_local)

---

### 🔍 Verificación de GPU

Antes de instalar, verificar que WSL puede acceder a la GPU correctamente:

```bash
nvidia-smi
```

---

### 🐍 Instalación de Conda

Instalar Miniconda o Anaconda desde la documentación oficial:  
https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html

---

### 🧪 Crear y activar un entorno Conda

```bash
conda create --name MAgro python=3.11 -y
conda activate MAgro
```

---

### 🔧 Instalación de PyTorch y dependencias principales

```bash
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118
pip install -U openmim
mim install mmengine
mim install "mmcv==2.1.0"
```

---

### 📦 Clonar el repositorio y configurar el entorno

```bash
git clone git@github.com:santipais/MAgro.git
cd MAgro
pip install -v -e .
```

---

### ✅ Instalación de dependencias adicionales

Estas ya deberian estar instaladas, pero por las dudas:

```bash
pip install ftfy scipy regex
```

⚠️ **Importante**: Es posible que algunas versiones de dependencias fallen si no se fuerza la instalación de `numpy`.

```bash
pip install numpy==1.25 --force-reinstall
```

---

### 🚀 Probar que todo funciona

Descargar checpoint y pegarlo en `checkpoints/`.

Correr:

```bash
python inference.py
```

---

## 🏋️‍♂️ Entrenamiento del modelo

### ⚠️ Nota previa

Antes de comenzar, si ocurre algún error relacionado al path de los módulos, ejecutar:

```bash
export PYTHONPATH=$(pwd):$PYTHONPATH
```

Y acordate de estar en el entorno de conda que instalamos!

---

### 📁 Preparar los datos

1. Descargar las imágenes y etiquetas correspondientes.
2. Colocar:
   - Las imágenes en: `data/images`
   - Las etiquetas en: `data/labels`

3. Ejecutar el siguiente comando para dividir el dataset:

```bash
python tools/dataset_converters/magro.py --test <porcentaje_test> --val <porcentaje_val> --seed <semilla>
```

Por ejemplo:

```bash
python tools/dataset_converters/magro.py --test 0.1 --val 0.2 --seed 42
```

Por default se tiene un valor de test 0.2, val 0.15 (por lo que, 0.65 de train) y semilla 42.

Esto generará las carpetas:

```
data/MAgro/images/
data/MAgro/annotatios/
```

Cada una con sus correspondientes subcarpetas 
```
train/
val/
test/
```

---

### 📥 Descargar checkpoint preentrenado

1. Descargar el checkpoint deseado del cual partir para realizar el fine-tuning (por ejemplo, un modelo `SegFormer`). Más adelante comentaremos algunos checkpoints obtenidos y de los cual se recomendamos partir.
2. Guardarlo en el directorio que prefieras.

⚠️ Muy importante: editar el archivo `configs/MAgro/segformer_mit-b5_MAgro.py`  
Ir a la **línea 30** y reemplazar la ruta del checkpoint con la ruta local donde lo guardaste.

---

### 🚀 Entrenar el modelo

Ejecutar el siguiente comando:

```bash
python tools/train.py configs/MAgro/segformer_mit-b5_MAgro.py
```

El entrenamiento creara el archivo `work_dirs/segformer_mit-b5_MAgro/` donde se almacenara todos los pesos y logs.

---

### ℹ️ Detalles adicionales

El archivo de configuración `configs/MAgro/segformer_mit-b5_MAgro.py` está configurado para:

- Entrenar durante **40,000 iteraciones**
- Guardar los pesos del modelo cada **5,000 iteraciones**

Estos valores se pueden modificar dentro del mismo archivo de configuración según tus necesidades.

---

## 👀 Inferencia y Test del modelo.

## 🧪 Test

Para testear el modelo con un peso obtenido, simplemente ejecutar:

```bash
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE}
```
Donde CONFIG_FILE probablemente será `configs/MAgro/segformer_mit-b5_MAgro.py` y CHEKPOINT_FILE será del estilo `work_dirs/segformer_mit-b5_MAgro/iter_5000.pth`

## Inferencia

Se crearon dos archivos, `inference.py` y `inference_dir.py` para inferar una imagen individual o un directorio de images respectivamente. En ambos casos se va a tener que modificar los archivos para utilizar el archivo config del modelo deseado, los pesos deseados y donde se quieren guardar los resultados. Las lineas a modificar para esto estan marcadas con un comment `# Modificar` al final

---

# 🏷️ Proceso de etiquetado

## 1. Creación del entorno para etiquetar

El proceso de etiquetado se realizó fuera de WSL, directamente en Windows, creando un nuevo entorno virtual `conda` y luego instalando la herramienta [Labelme](https://github.com/wkentaro/labelme):

```bash
conda create --name etiquetado python=3.11 -y
conda activate etiquetado
pip install labelme
```

---

## 2. Asistencia por notebooks

Durante el desarrollo, se utilizaron dos notebooks de Google Colab:

- **[RemapOfClasses](colabs/RemapOfClasses.ipynb):** Se convierte las máscaras de los resultados de inferir con el modelo `san-vit-l14_coco-stuff164k-640x640` , a las 5 clases deseadas por nosotros.
- **[GrayScaleToLabelMe](colabs/GrayScaleToLableMe.ipynb):** convertía esas máscaras grises a formato `JSON` con polígonos, compatibles con `Labelme`.
- **[LabelMeToGrayScaleAndImageVis](colabs/LabelMeToGrayScaleAndImageVis.ipynb):** una vez finalizado el proceso de etiquetado con `Labelme`, se utilizó otro notebook para convertir las anotaciones en formato de polígonos (`JSON`) al formato compatible con MMSegmentation (máscaras en escala de grises).

---

## 3. Edición final con Labelme

Una vez obtenidas las predicciones en formato `JSON` por el segundo notebook, se realizaron ajustes y retoques manuales ejecutando Labelme con el siguiente comando:

```bash
labelme {directorio_imagenes} --output {directorio_de_etiquetas}
```

Esto permitió generar las anotaciones finales en formato `Labelme`, que luego al pasar por el tecer notebook, se tienen las etiquetas listas para ser utilizadas por la toolbox MMSegmentation.

---



# 📊 Resultados de Segmentación y Pesos

A continuación se muestran los resultados y pesos obtenidos.
Todos fueron entrenados mediante el mismo archivo de configuración, es decir con el mismo modelo, pero se cambio el dataset usado.
Cada tabla incluye las métricas por clase (IoU y Accuracy), así como métricas globales como mIoU y aAcc.

---

## 🔷 Modelo 1: Dataset formado exclusivamente por imagenes nuestras.

|   Class    |  IoU  |  Acc  |
|------------|-------|-------|
|    road    | 96.46 | 99.01 |
| vegetation | 93.11 | 96.58 |
|    sky     | 95.54 | 97.25 |
|  obstacle  | 47.66 | 53.92 |
|   others   |  8.24 |  8.62 |

> **aAcc**: 96.94 &nbsp;&nbsp;&nbsp; **mIoU**: 68.20 &nbsp;&nbsp;&nbsp; **mAcc**: 71.08 &nbsp;&nbsp;&nbsp; **data_time**: 0.0079 &nbsp;&nbsp;&nbsp; **time**: 0.1589

[Descargar peso]()

---

## 🔷 Modelo 2: Dataset formado por todas nuestras imagenes y las obtenidas con RELLIS-3D


|   Class    |  IoU  |  Acc  |
|------------|-------|-------|
|    road    | 87.59 | 95.31 |
| vegetation | 86.01 | 91.59 |
|    sky     | 96.23 | 97.55 |
|  obstacle  | 64.58 | 73.08 |
|   others   | 80.33 | 84.71 |

> **aAcc**: 93.95 &nbsp;&nbsp;&nbsp; **mIoU**: 82.95 &nbsp;&nbsp;&nbsp; **mAcc**: 88.45 &nbsp;&nbsp;&nbsp; **data_time**: 0.0032 &nbsp;&nbsp;&nbsp; **time**: 0.0989

[Descargar peso]()

---

## 🔷 Modelo 3: Dataset equilibrado. (Todas nuestras imagenes y una selección de 200 de RELLIS-3d)

En este modelo obtuvimos los mejores resultados. Son los pesos que recomendamos seguir la linea.


|   Class    |  IoU  |  Acc  |
|------------|-------|-------|
|    road    | 90.32 | 97.85 |
| vegetation | 92.26 | 95.51 |
|    sky     | 95.32 | 97.65 |
|  obstacle  | 64.95 | 68.15 |
|   others   | 83.22 | 89.86 |

> **aAcc**: 94.8 &nbsp;&nbsp;&nbsp; **mIoU**: 85.21 &nbsp;&nbsp;&nbsp; **mAcc**: 89.80 &nbsp;&nbsp;&nbsp; **data_time**: 0.0040 &nbsp;&nbsp;&nbsp; **time**: 0.1190

[Descargar peso]()

---

# 📖 Citación

Este proyecto usa [MMSegmentation](https://github.com/open-mmlab/mmsegmentation). Si lo usas, para citarlo:

```bibtex
@misc{mmseg2020,
    title={{MMSegmentation}: OpenMMLab Semantic Segmentation Toolbox and Benchmark},
    author={MMSegmentation Contributors},
    howpublished = {\url{https://github.com/open-mmlab/mmsegmentation}},
    year={2020}
}
