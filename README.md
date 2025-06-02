![Python](https://img.shields.io/badge/python-3.10-blue)
![Issues](https://img.shields.io/github/issues/zxcuser1/ImageClassifier)
![Stars](https://img.shields.io/github/stars/zxcuser1/ImageClassifier?style=social)

# 🖼️ ImageClassifier

Простой, но гибкий классификатор изображений на Python с использованием OpenCV, scikit-learn и TensorFlow.  
Проект адаптирован для запуска в Docker-контейнере с поддержкой GPU (CUDA 11.8 + cuDNN 8.2 + Python 3.10).

---

## 🚀 Возможности

- 📂 Автоматическая загрузка изображений из папок
- 🔧 Предобработка и нормализация данных
- 🖥️ Работа как на CPU, так и с GPU через Docker

---

## 🐳 Запуск через Docker (с поддержкой GPU)

> Убедитесь, что у вас установлен [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/)

### 🔧 Сборка вручную

```bash
docker build -t image-classifier .
docker run --gpus all -it image-classifier
```

### 🔧 С Docker Compose
```bash
docker compose up --build
```
### 📁 Структура проекта
```text
ImageClassifier/
├──models/                  # Папка с обученными моделями
├── main.py                 # Точка входа
├── ImageClassifier.py      # Основная логика классификации
├── ImageProcessor.py       # Предобработка изображений
├── ImageManager.py         # Управление файлами и загрузкой
├── requirements.txt        # Python зависимости
├── Dockerfile              # Docker-образ
├── docker-compose.yml      # Запуск через Docker Compose
└── README.md               # Описание проекта

```

### 📥 Установка вручную
```bash
# Убедитесь, что установлен Python 3.10
pip install -r requirements.txt
python main.py
```

### 💻 Требования
Python 3.10

CUDA 11.8 + cuDNN 8.2

Docker и Docker Compose
---

### 📦 Ресурсы

**🔗 Датасеты:**

- [Датасет для обучения модели фильтрации](https://disk.yandex.ru/d/LcbO6vxpqack1Q)
- [Датасет для обучения модели сегментации](https://disk.yandex.ru/d/XK-0a8AdjDimmg)
- [Датасет для обучения модели классификации кривых](https://disk.yandex.ru/d/bQp4jT2FmFi8vg)

**📥 Модели**

- [Все обученные модели](https://disk.yandex.ru/d/5-DdgwBtTGN8uA)
---

### 📜 Лицензия

Проект распространяется под лицензией MIT.
Автор: @zxcuser1


### ❤️ Благодарности
OpenCV

TensorFlow

scikit-learn

NVIDIA CUDA
