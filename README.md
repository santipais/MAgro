## Introduction
## ⚙️ Pasos de instalación

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


## 📖 Citación

Este proyecto usa [MMSegmentation](https://github.com/open-mmlab/mmsegmentation). Si lo usas, para citarlo:

```bibtex
@misc{mmseg2020,
    title={{MMSegmentation}: OpenMMLab Semantic Segmentation Toolbox and Benchmark},
    author={MMSegmentation Contributors},
    howpublished = {\url{https://github.com/open-mmlab/mmsegmentation}},
    year={2020}
}
