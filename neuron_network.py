import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


class FriendshipPredictor(nn.Module):
    # Метод __init__ определяет компоненты (слои) модели
    def __init__(self, input_layer, hidden_1_layer, hidden_2_layer, output_layer):
        super(FriendshipPredictor, self).__init__()

        # 1. Первый линейный слой: Входы -> Первый скрытый слой
        self.fc1 = nn.Linear(input_layer, hidden_1_layer)

        # 2. Второй линейный слой: Первый скрытый слой -> Второй скрытый слой
        self.fc2 = nn.Linear(hidden_1_layer, hidden_2_layer)

        # 3. Выходной слой: Второй скрытый слой -> Выходы (классы)
        self.fc3 = nn.Linear(hidden_2_layer, output_layer)

    # Метод forward определяет, как данные проходят через слои
    def forward(self, x):
        # x - это входной тензор с 10 признаками пары

        # Прямой проход: Слой 1 -> ReLU
        x = self.fc1(x)
        x = F.relu(x)  # F.relu() или nn.ReLU()

        # Прямой проход: Слой 2 -> ReLU
        x = self.fc2(x)
        x = F.relu(x)

        # Прямой проход: Выходной слой
        # Здесь мы не применяем Softmax, так как CrossEntropyLoss сделает это за нас
        x = self.fc3(x)

        return x


def train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs, train_size, val_size):
    """
    Основная функция для тренировки и валидации модели по эпохам.

    :param model: Экземпляр модели PyTorch.
    :param criterion: Функция потерь (nn.CrossEntropyLoss).
    :param optimizer: Оптимизатор (optim.Adam).
    :param train_loader: DataLoader для обучающего набора.
    :param val_loader: DataLoader для валидационного набора.
    :param num_epochs: Количество эпох.
    :param train_size: Общий размер обучающего набора (для расчета среднего loss).
    :param val_size: Общий размер валидационного набора.
    """
    for epoch in range(num_epochs):

        # ----------------------------------------
        # I. ФАЗА ОБУЧЕНИЯ (TRAINING PHASE)
        # ----------------------------------------

        # Устанавливаем модель в режим обучения (активирует Dropout/BatchNorm, если есть)
        model.train()

        running_train_loss = 0.0

        for batch_idx, (data, targets) in enumerate(train_loader):
            # 1. Обнуляем градиенты с предыдущего шага
            optimizer.zero_grad()

            # 2. Прямой проход (Forward)
            outputs = model(data)

            # 3. Расчет потерь
            loss = criterion(outputs, targets)

            # 4. Обратный проход (Backward)
            loss.backward()

            # 5. Обновление весов
            optimizer.step()

            running_train_loss += loss.item() * data.size(0)

        # Расчет средней потери за эпоху
        epoch_train_loss = running_train_loss / train_size

        # ----------------------------------------
        # II. ФАЗА ВАЛИДАЦИИ (VALIDATION PHASE)
        # ----------------------------------------

        # Устанавливаем модель в режим оценки (отключает Dropout/BatchNorm)
        model.eval()

        running_val_loss = 0.0
        correct_predictions = 0

        # torch.no_grad() отключает расчет градиентов для экономии памяти и ускорения
        with torch.no_grad():
            for data, targets in val_loader:
                # 1. Прямой проход
                outputs = model(data)

                # 2. Расчет потерь
                loss = criterion(outputs, targets)
                running_val_loss += loss.item() * data.size(0)

                # Расчет точности (Accuracy)
                # outputs.argmax(1) берет индекс нейрона с максимальным значением (наше предсказание класса)
                _, predicted_classes = outputs.max(1)
                correct_predictions += (predicted_classes == targets).sum().item()

        # Расчет метрик за эпоху
        epoch_val_loss = running_val_loss / val_size
        epoch_val_accuracy = correct_predictions / val_size

        # ----------------------------------------
        # III. ВЫВОД РЕЗУЛЬТАТОВ
        # ----------------------------------------
        print(f"Эпоха {epoch + 1}/{num_epochs}:")
        print(f"  Обучение Loss: {epoch_train_loss:.4f}")
        print(f"  Валидация Loss: {epoch_val_loss:.4f}, Accuracy: {epoch_val_accuracy:.4f}")

if __name__ == '__main__':
    # Устанавливаем размерности
    input_layer = 10 # 10 признаков сходства (например, 5 категориальных + 5 числовых)
    hidden_1_layers = 64
    hidden_2_layers = 32
    output_layer = 2 # 2 класса: 0 (Не дружат) или 1 (Дружит)

    batch_size = 64
    num_epochs = 25

    # --- Имитация данных (ваши реальные данные будут здесь) ---
    # X_train_t: обучающие признаки, y_train_t: обучающие метки (0 или 1)
    # X_val_t: валидационные признаки, y_val_t: валидационные метки
    # 10 - количество признаков (INPUT_SIZE)
    #TODO: написать класс с данными
    train_size = 1000
    val_size = 200

    X_train_t = torch.randn(1000, 10)
    y_train_t = torch.randint(0, 2, (1000,))

    X_val_t = torch.randn(200, 10)
    y_val_t = torch.randint(0, 2, (200,))

    # Создание DataLoader
    train_dataset = TensorDataset(X_train_t, y_train_t)
    val_dataset = TensorDataset(X_val_t, y_val_t)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Создание экземпляра модели
    model = FriendshipPredictor(input_layer, hidden_1_layers, hidden_2_layers, output_layer)
    print(model)

    # 1. Функция потерь (Loss Function)
    # CrossEntropyLoss идеально подходит для бинарной и многоклассовой классификации
    criterion = nn.CrossEntropyLoss()

    # 2. Оптимизатор (Optimizer)
    # Optimizer отвечает за обновление весов (W и b) на основе градиентов
    # Мы передаем ему все обучаемые параметры модели (model.parameters())
    optimizer = optim.Adam(model.parameters(), lr=0.001)  # lr - learning rate (скорость обучения)

    # --- ЗАПУСК ОБУЧЕНИЯ ---
    train_model(
        model,
        criterion,
        optimizer,
        train_loader,
        val_loader,
        num_epochs,
        train_size=train_size,  # Передаем размер обучающего набора
        val_size=val_size  # Передаем размер валидационного набора
    )
