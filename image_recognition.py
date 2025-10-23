import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import datasets, models, transforms
import numpy as np
import matplotlib.pyplot as plt
import time
import os
from PIL import Image, ImageOps
from tempfile import TemporaryDirectory
from tqdm import tqdm


class CustomImageDataset(Dataset):
    """Кастомный датасет для загрузки изображений из папок, организованных по классам.

    Атрибуты:
        img_dir (str): Путь к директории с изображениями
        image_paths (list): Список путей к изображениям
        labels (list): Список меток классов
        transform (callable, optional): Преобразования для применения к изображениям
        class_names (list): Список имен классов
        class_to_idx (dict): Словарь для преобразования имени класса в индекс
    """

    def __init__(self, img_dir, transform=None):
        """Инициализация датасета.

        Args:
            img_dir (str): Путь к директории с изображениями
            transform (callable, optional): Преобразования для применения к изображениям
        """
        self.img_dir = img_dir
        self.image_paths = []
        self.labels = []
        self.transform = transform

        # Проверка существования директории
        if not os.path.isdir(self.img_dir):
            raise FileNotFoundError(f"Directory not found: {self.img_dir}")

        # Сбор информации об изображениях и классах
        for class_name in os.listdir(self.img_dir):
            class_dir = os.path.join(self.img_dir, class_name)
            if os.path.isdir(class_dir):
                for filename in os.listdir(class_dir):
                    if filename.endswith(('.jpg', '.jpeg', '.png')):
                        self.image_paths.append(os.path.join(class_dir, filename))
                        self.labels.append(class_name)

        if not self.image_paths:
            raise FileNotFoundError(f"No images found in {self.img_dir}")

        self.class_names = sorted(list(set(self.labels)))
        self.class_to_idx = {class_name: i for i, class_name in enumerate(self.class_names)}

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        """Загрузка и преобразование изображения по индексу.

        Args:
            idx (int): Индекс изображения

        Returns:
            tuple: (image, label) где label - индекс класса
        """
        img_path = self.image_paths[idx]
        try:
            with Image.open(img_path) as img:
                image = img.convert('RGB')
                if self.transform:
                    image = self.transform(image)
                label = self.class_to_idx[self.labels[idx]]
                return image, label
        except (IOError, OSError) as e:
            print(f"Error loading image {img_path}: {e}")
            return None, 0


def create_data_loaders(train_dir, val_dir, batch_size=4):
    """Создает загрузчики данных для обучения и валидации.

    Args:
        train_dir (str): Путь к тренировочным данным
        val_dir (str): Путь к валидационным данным
        batch_size (int): Размер батча

    Returns:
        tuple: (dataloaders, dataset_sizes, class_names)
    """
    # Преобразования для данных
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(),
            # transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    # Создание датасетов
    image_datasets = {
        'train': CustomImageDataset(train_dir, data_transforms['train']),
        'val': CustomImageDataset(val_dir, data_transforms['val'])
    }

    # Создание загрузчиков данных
    dataloaders = {
        x: DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=0)
        for x in ['train', 'val']
    }

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].class_names

    return dataloaders, dataset_sizes, class_names


def train_model(model, criterion, optimizer, scheduler, dataloaders, dataset_sizes, num_epochs=25):
    """Функция для обучения модели.

    Args:
        model: Модель для обучения
        criterion: Функция потерь
        optimizer: Оптимизатор
        scheduler: Планировщик скорости обучения
        dataloaders: Словарь с загрузчиками данных
        dataset_sizes: Словарь с размерами датасетов
        num_epochs (int): Количество эпох обучения

    Returns:
        model: Обученная модель
    """
    # Засекаем время начала обучения
    since = time.time()

    # Создаем временную директорию для сохранения лучших весов модели
    with TemporaryDirectory() as tempdir:
        # Путь к файлу с лучшими параметрами модели
        best_model_params_path = os.path.join(tempdir, 'best_model_params.pt')
        # Сохраняем начальные веса модели
        torch.save(model.state_dict(), best_model_params_path)
        # Инициализируем лучшую точность
        best_acc = 0.0

        # Цикл по эпохам
        for epoch in range(num_epochs):
            print(f'Epoch {epoch}/{num_epochs - 1}')
            print('-' * 10)

            # Каждую эпоху имеет две фазы: обучение и валидация
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()  # Устанавливаем модель в режим обучения
                    # (включает вычисление градиентов и такие слои как Dropout/BatchNorm)
                else:
                    model.eval()  # Устанавливаем модель в режим оценки
                    # (отключает вычисление градиентов и фиксирует поведение Dropout/BatchNorm)

                # Инициализируем метрики
                running_loss = 0.0  # Накопитель потерь
                running_corrects = 0  # Накопитель правильных предсказаний

                # Итерация по батчам данных
                for inputs, labels in dataloaders[phase]:
                    # Переносим данные на устройство (GPU/CPU)
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # Обнуляем градиенты перед каждым батчем
                    optimizer.zero_grad()

                    # Включаем вычисление градиентов только в фазе обучения
                    with torch.set_grad_enabled(phase == 'train'):
                        # Прямой проход (forward pass)
                        outputs = model(inputs)
                        # Получаем предсказанные классы (индекс с максимальной вероятностью)
                        _, preds = torch.max(outputs, 1)
                        # Вычисляем значение функции потерь
                        loss = criterion(outputs, labels)

                        # Обратный проход (backward pass) и оптимизация только в фазе обучения
                        if phase == 'train':
                            loss.backward()  # Вычисление градиентов
                            optimizer.step()  # Обновление весов

                    # Обновляем метрики
                    running_loss += loss.item() * inputs.size(0)  # Суммируем потери с учетом веса
                    running_corrects += torch.sum(preds == labels.data)  # Считаем правильные предсказания

                # После каждой фазы (train/val) обновляем планировщик обучения
                if phase == 'train':
                    scheduler.step()

                # Вычисляем средние метрики по эпохе
                epoch_loss = running_loss / dataset_sizes[phase]  # Средние потери
                epoch_acc = running_corrects.double() / dataset_sizes[phase]  # Точность

                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

                # Сохраняем модель, если достигнута лучшая точность на валидации
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    torch.save(model.state_dict(), best_model_params_path)

            print()  # Пустая строка между эпохами

        # Выводим общее время обучения
        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best val Acc: {best_acc:4f}')

        # Загружаем веса лучшей модели
        model.load_state_dict(torch.load(best_model_params_path))

    return model


def test_and_visualize_model(model, dataloaders, class_names, num_images=6):
    """Визуализирует предсказания модели на валидационных данных.

    Показывает сетку изображений с подписями предсказанных классов.
    Если модель предсказывает неверно, это легко заметить визуально.

    Args:
        model (nn.Module): Обученная модель PyTorch
        dataloaders (dict): Словарь DataLoader'ов (используется 'val')
        class_names (list): Список названий классов в порядке их индексов
        num_images (int, optional): Количество изображений для отображения. По умолчанию 6.
    """

    # Сохраняем исходный режим модели (train/eval)
    was_training = model.training

    # Переводим модель в режим оценки (отключаем dropout/batchnorm)
    model.eval()

    # Счетчик показанных изображений
    images_so_far = 0

    # Создаем новую фигуру matplotlib
    fig = plt.figure()

    # Отключаем вычисление градиентов для экономии памяти
    with torch.no_grad():
        # Итерируем по валидационным данным
        for inputs, labels in dataloaders['val']:
            # Переносим данные на нужное устройство (GPU/CPU)
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Прямой проход - получаем предсказания модели
            outputs = model(inputs)

            # Получаем индексы предсказанных классов
            _, preds = torch.max(outputs, 1)

            # Обрабатываем каждый элемент в батче
            for j in range(inputs.size()[0]):
                # Увеличиваем счетчик показанных изображений
                images_so_far += 1

                # Создаем subplot для текущего изображения
                # (сетка num_images//2 строк, 2 столбца)
                ax = plt.subplot(num_images // 2, 2, images_so_far)

                # Отключаем оси для лучшей визуализации
                ax.axis('off')

                # Устанавливаем заголовок с предсказанным классом
                ax.set_title(f'predicted: {class_names[preds[j]]}')

                # Отображаем изображение
                imshow(inputs.cpu().data[j])

                # Если достигли нужного количества изображений
                if images_so_far == num_images:
                    # Возвращаем модель в исходный режим
                    model.train(mode=was_training)
                    # Выходим из функции
                    return

        # Возвращаем модель в исходный режим (на случай, если не показали все num_images)
        model.train(mode=was_training)


def imshow(inp, title=None):
    """Отображение изображения из тензора.

    Args:
        inp: Тензор изображения
        title (str, optional): Заголовок изображения
    """
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)


def add_reflected_copies(dataset_dir):
    """
    Добавляет отраженные копии изображений в исходные папки
    :param dataset_dir: Путь к папке с данными (train или val)
    """
    for class_name in os.listdir(dataset_dir):
        class_dir = os.path.join(dataset_dir, class_name)

        if not os.path.isdir(class_dir):
            continue

        # Получаем список ТОЛЬКО оригинальных изображений (исключаем уже созданные копии)
        original_images = [f for f in os.listdir(class_dir)
                           if f.lower().endswith(('.png', '.jpg', '.jpeg'))
                           and not f.startswith(('mirror_', 'flip_'))]

        for img_name in tqdm(original_images, desc=f'Обработка {class_name}'):
            img_path = os.path.join(class_dir, img_name)

            try:
                with Image.open(img_path) as img:
                    # Горизонтальное отражение
                    mirrored = ImageOps.mirror(img)
                    mirrored.save(os.path.join(class_dir, f"mirror_h_{img_name}"))

                    # Вертикальное отражение (опционально)
                    flipped = ImageOps.flip(img)
                    flipped.save(os.path.join(class_dir, f"flip_v_{img_name}"))

            except Exception as e:
                print(f"Ошибка при обработке {img_path}: {e}")


def main():
    """Основная функция для выполнения обучения."""
    cudnn.benchmark = True  # Включение оптимизаций для CUDA
    plt.ion()  # Включение интерактивного режима для matplotlib

    # Установка устройства (GPU или CPU)
    global device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device")

    # Пути к данным (замените на свои)
    train_dir = r'C:\Users\user\Desktop\NeuralNetwork\train_data\train'
    val_dir = r'C:\Users\user\Desktop\NeuralNetwork\train_data\train'

    # add_reflected_copies(train_dir)

    # Проверка путей
    if not os.path.isdir(train_dir):
        raise FileNotFoundError(f"Training directory not found: {train_dir}")
    if not os.path.isdir(val_dir):
        raise FileNotFoundError(f"Validation directory not found: {val_dir}")

    # Создание загрузчиков данных
    dataloaders, dataset_sizes, class_names = create_data_loaders(train_dir, val_dir, batch_size=4)

    # Визуализация примеров данных
    inputs, classes = next(iter(dataloaders['train']))
    out = torchvision.utils.make_grid(inputs)
    imshow(out, title=[class_names[x] for x in classes])

    # Загрузка предобученной модели
    model_ft = models.resnet18(weights='IMAGENET1K_V1')
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, len(class_names))
    model_ft = model_ft.to(device)

    # Настройка обучения
    criterion = nn.CrossEntropyLoss()
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    # Обучение модели
    model_ft = train_model(
        model_ft, criterion, optimizer_ft, exp_lr_scheduler,
        dataloaders, dataset_sizes, num_epochs=25
    )

    # Визуализация результатов
    test_and_visualize_model(model_ft, dataloaders, class_names)

    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()