## Introduction
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
git clone <URL-de-nuestro-repositorio>
cd <nombre-del-repo>
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


## 📖 Citación

Este proyecto usa [MMSegmentation](https://github.com/open-mmlab/mmsegmentation). Si lo usas, para citarlo:

```bibtex
@misc{mmseg2020,
    title={{MMSegmentation}: OpenMMLab Semantic Segmentation Toolbox and Benchmark},
    author={MMSegmentation Contributors},
    howpublished = {\url{https://github.com/open-mmlab/mmsegmentation}},
    year={2020}
}
