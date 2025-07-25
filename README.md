# YOLO Детектор людей

Проект для детекции людей на видео и изображениях с использованием оптимизированной модели YOLOv12 и ONNX Runtime.

## Оглавление

- [Описание](#описание-предложенного-решения)
- [Эксперименты](#эксперименты)
- [Оптимизации](#оптимизации)
- [Пользовательский интерфейс](#пользовательский-интерфейс)
- [Установка](#установка)
- [Использование](#использование)
  - [Инференс](#инференс)
  - [Обучение](#обучение)
- [Результаты](#результаты)
- [Ссылки на датасеты и веса моделей](#ссылки-на-датасеты-и-веса-моделей)

## Описание предложенного решения

В качестве модели для решения задачи детекции людей я выбрал семейство моделей YOLOv12. Выбор связан с тем, что это очень популярные модели с высоким качеством и хорошей скоростью. YOLO модели очень просто адаптировать под down-stream задачи. В своем решении я хотел сделать акцент не только на высоком качестве детекции (IoU, mAP), но и на высоком FPS модели в условиях запуска на CPU (так как в ТЗ был указан и macOS). Поэтому я использовал только маленькие версии моделей (n и s) в своих экспериментах, так как такие модели могут работать хорошо даже на CPU. В качестве вторичной оптимизации runtime я конвертировал модели в ONNX формате, это позволило не только ускорить инференс на CPU, но и избавиться от таких тяжелых пакетов как PyTorch и Ultralytics в зависимостях. Это позволит существенно облегчить образ контейнера (при дальнейшем развитии приложения).

## Эксперименты

В качестве валидации для моих решений, помимо субъективной оценки по качеству детекции на предложенном видео, я использовал Validation subset датасета CrowdHuman.

* **YOLOv12n без дообучения**: без дообучения модель достаточно плохо справилась с тестовым видео, поэтому было принято решение дообучить ее
* **Обучение только на CrowdHuman Train subset**: дообучение на этом датасете существенно улучшило метрики на валидационной выборке (table. 1) и субъективное качество на тестовом видео. Модель стала предсказывать больше людей в кадре и ее уверенность стала значительно выше, поэтому bboxes стали меньше "моргать" от кадра к кадру.
* **Обучение YOLOv12s на CrowdHuman**: в качестве эксперимента я решил проверить scaling laws для этой задачи. Как оказалось (table.1), взятие модели большего размера не сказалось существенно на качестве, но сильно ударило по runtime (увеличив его практически вдвое)
* **Сбор данных**: я собрал 4 датасета для детекции людей: CrowdHuman, TinyPerson, CityPerson и кастомный датасет с Roboflow. На этих данных я обучил YOLOv12n и собрал новый валидационный бенчмарк, состоящий из валидационных сетов всех этих датасетов.

## Оптимизации

Для задачи детекции очень важно, чтобы инференс был максимально быстрым и на CPU. Именно по этой причине я выбрал самую маленькую модель из семейства YOLOv12, так как по результатам сравнения с более старшими моделями, я посчитал, что трейдофф рантайма и качества модели играет не в пользу больших моделей, так как изменения в качестве остаются минорными, даже при условии 2.5 кратного увеличения количества параметров и FLOPS. Также для ускорения инференса на CPU и избавления от PyTorch и Ultralytics из зависимостей я конвертировал модель в ONNX. С batch-size=16 YOLOv12n запускается 19 FPS на серверном процессоре XEON.

## Пользовательский интерфейс

Согласно ТЗ реализован CLI для взаимодействия с программой. Чтобы избежать избыточное количество аргументов, добавил конфиги для визуала (цвет bbox, шрифт и т.д.) и модели.


# Возможные улучшения и направления развития

На основании проведённых экспериментов я пришёл к выводу, что в данной задаче рациональнее сосредоточиться на сборе новых и более разнообразных данных, а не на увеличении размера модели.

## 1. Расширение датасетов

Я предлагаю:
- **Дополнить существующие данные** путём парсинга открытых датасетов с платформ вроде Roboflow
- **Добавить специализированные датасеты**, такие как **TinyPerson**, где люди находятся на большом расстоянии от камеры. Это улучшит детекцию мелких объектов в кадре

## 2. Оптимизация производительности

Ключевой аспект — **скорость работы модели (RTF)**. Для ускорения инференса я предлагаю:
- **Заменить CLI-решение на Triton Inference Server**, что даст заметный прирост производительности
- **Перенести вычисления на GPU** и конвертировать модель в **TensorRT** для максимальной оптимизации

## 3. Выбор модели в зависимости от задачи

- **Если важна скорость (стриминг, реальное время)** → использовать **лёгкие модели** (YOLO12n, YOLO12s) с Triton на CPU.
- **Если приоритет — точность, а не ресурсы** → взять **крупную модель (YOLOv12m/l)**, ускорить её через TensorRT и развернуть на Triton на GPU.

Таким образом, подход можно гибко адаптировать под требования проекта: **улучшение данных для качества или оптимизация инференса для скорости**.

## Установка

### Для инференса

```bash
# Клонировать репозиторий
git clone https://github.com/yourusername/human_detection_tc.git
cd human_detection_tc

# Установить FFmpeg (необходим для обработки видео)
# Ubuntu/Debian
sudo apt update && sudo apt install ffmpeg -y

# CentOS/RHEL
# sudo yum install epel-release -y
# sudo yum install ffmpeg ffmpeg-devel -y

# macOS (с Homebrew)
# brew install ffmpeg

# Windows (с Chocolatey)
# choco install ffmpeg

# Установить зависимости для инференса
pip install -r requirements_inference.txt
```

### Для обучения

```bash
# Клонировать репозиторий
git clone https://github.com/yourusername/human_detection_tc.git
cd human_detection_tc


# Установить зависимости для обучения
pip install -r requirements_train.txt
```

## Использование

### Инференс

Финальные веса модели доступны по ссылке в секции [Ссылки на датасеты и веса моделей](#ссылки-на-датасеты-и-веса-моделей). 

Запуск инференса на видео с использованием модели ONNX:

```bash
python main.py \
    --input path/to/video.mp4 \
    --output output.mp4 \
    --model weights/yolo12n/onnx/model.onnx \
    --conf 0.5 \
    --iou 0.8 \
    --device cpu \
    --batch_size 16
```

#### Аргументы командной строки

| Аргумент        | Описание                                              | Значение по умолчанию        |
|-----------------|-------------------------------------------------------|-----------------------------|
| --input         | Путь к входному видео                                 | (обязательный аргумент)     |
| --output        | Путь для сохранения выходного видео                   | (обязательный аргумент)     |
| --model         | Путь к ONNX модели                                    | (обязательный аргумент)     |
| --model-config  | Путь к конфигурационному файлу модели                | config/model_config.yaml    |
| --batch_size    | Размер батча для инференса                            | 1                           |
| --conf          | Порог уверенности для детекций                        | 0.5                         |
| --iou           | Порог IoU для NMS                                     | 0.7                         |
| --device        | Устройство для инференса ('cpu' или 'cuda')           | cpu                         |
| --img-size      | Размер входного изображения                           | 640                         |
| --vis-config    | Путь к конфигурационному файлу визуализации           | config/visual_config.yaml   |
| --preview       | Показать окно предпросмотра во время обработки        | False                       |
| --frame-step    | Обрабатывать каждый N-ый кадр (1 = все кадры)         | 1                           |
| --codec         | Кодек FourCC для выходного видео                      | mp4v                        |

### Обучение

Для обучения модели на датасетах (требуется установка зависимостей для обучения):

```bash
python tools/train.py \
    --model_name weights/yolo12n.pt \
    --data_path data/fused_dataset/data.yaml \
    --epochs 100 \
    --img_size 640 \
    --batch_size 16 \
    --device 0
```

#### Аргументы командной строки для обучения

| Аргумент        | Описание                                              | Значение по умолчанию        |
|-----------------|-------------------------------------------------------|-----------------------------|
| --model_name    | Путь к файлу модели для fine-tuning                   | yolov12n.pt                 |
| --data_path     | Путь к YAML файлу с описанием данных                  | data.yaml                   |
| --epochs        | Количество эпох обучения                              | 100                         |
| --img_size      | Размер изображений для обучения                       | 640                         |
| --batch_size    | Размер батча                                          | 16                          |
| --device        | Устройство для обучения ('cpu', '0', '0,1,2,3')       | 0                           |
| --workers       | Количество воркеров для загрузки данных               | 4                           |
| --lr0           | Начальная скорость обучения                           | 0.01                        |

### Экспорт модели в ONNX

После обучения модель можно экспортировать в формат ONNX:

```bash
python tools/export_onnx.py \
    --model_path tools/runs/detect/train/weights/best.pt \
    --model_name yolo12n_finetune \
    --opset 12 \
    --simplify
```

#### Аргументы для экспорта в ONNX

| Аргумент        | Описание                                              | Значение по умолчанию        |
|-----------------|-------------------------------------------------------|-----------------------------|
| --model_path    | Путь к PyTorch модели (.pt)                           | None                         |
| --model_name    | Имя модели (для сохранения)                           | (обязательный аргумент)      |
| --opset         | Версия ONNX opset                                     | 12                          |
| --simplify      | Упростить ONNX модель                                 | True                        |
| --dynamic       | Использовать динамические оси                          | True                        |
| --batch         | Размер батча для инференса                            | 8                           |
| --device        | Устройство для экспорта ('cpu', 'cuda')               | cpu                         |

## Результаты

### Таблица 1. Результаты оценки на валидационном наборе CrowdHuman

| Модель | Precision | Recall | mAP50 | mAP50-95 |
|--------|-----------|--------|-------|----------|
| YOLOv12n (базовая) | 0.647 | 0.416 | 0.487 | 0.241 |
| YOLOv12s (базовая) | 0.662 | 0.447 | 0.517 | 0.260 |
| YOLOv12n (дообучение на CrowdHuman) | 0.838 | 0.677 | 0.791 | 0.483 |
| YOLOv12n (дообучение на 4 датасетах) | 0.832 | 0.674 | 0.785 | 0.475 |
| YOLOv12s (дообучение на CrowdHuman) | 0.852 | 0.715 | 0.820 | 0.515 |

### Таблица 2. Результаты оценки на комбинированном валидационном наборе (CrowdHuman + TinyPerson + CityPerson + кастомный датасет)

| Модель | Precision | Recall | mAP50 | mAP50-95 |
|--------|-----------|--------|-------|----------|
| YOLOv12n (базовая) | 0.646 | 0.408 | 0.473 | 0.237 |
| YOLOv12s (базовая) | 0.666 | 0.443 | 0.509 | 0.260 |
| YOLOv12n (дообучение на CrowdHuman) | 0.813 | 0.630 | 0.727 | 0.433 |
| YOLOv12n (дообучение на 4 датасетах) | 0.821 | 0.651 | 0.755 | 0.456 |
| YOLOv12s (дообучение на CrowdHuman) | 0.830 | 0.664 | 0.757 | 0.463 |

### Демонстрационное видео

[Демонстрация детекции людей](assets/final_video.mp4)

## Файлы конфигурации

Проект использует конфигурационные файлы в формате YAML для настройки параметров модели и визуализации:

### model_config.yaml

Содержит настройки модели, включая имена классов для отображения.

### visual_config.yaml

Содержит настройки визуализации, такие как цвета для отображения рамок, размеры шрифтов и т.д.

## Ссылки на датасеты и веса моделей

### Датасеты

- [CrowdHuman](https://www.crowdhuman.org/) - датасет с плотными сценами с большим количеством людей
- [TinyPerson](https://github.com/ucas-vg/TinyBenchmark) - датасет с мелкими объектами людей, вид сверху
- [CityPersons](https://github.com/cvgroup-njust/CityPersons) - датасет с пешеходами в городской среде
- [Roboflow Human Dataset](https://universe.roboflow.com/roboflow-100/human-detection-serie) - кастомный датасет с разметкой людей

### Модели

- [Финальная модель в моем решение YOLOv12n дообученная на 4 датасетах](https://drive.google.com/file/d/1vqjeipKZ2_SYzxGiBbe1AtidqprkqsQ7/view?usp=sharing) - дообученная модель на 4 датасетах

- [YOLOv12n (базовая)](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov12n.pt) - базовая модель YOLOv12n
- [YOLOv12s (базовая)](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov12s.pt) - базовая модель YOLOv12s
- [YOLOv12n дообученная на CrowdHuman](https://drive.google.com/file/d/1XSfAJnYSn0oyzXUayNbQcxl6ny_vn6IL/view?usp=sharing) - дообученная модель на CrowdHuman
- [YOLOv12n дообученная на 4 датасетах](https://drive.google.com/file/d/1vqjeipKZ2_SYzxGiBbe1AtidqprkqsQ7/view?usp=sharing) - дообученная модель на 4 датасетах
- [YOLOv12s дообученная на CrowdHuman](https://drive.google.com/file/d/13anuoJ3IgVWI7rqVqAOgOUq7HWXv1Lju/view?usp=sharing) - дообученная модель YOLOv12s

Все модели доступны в формате ONNX (.onnx) для инференса.
