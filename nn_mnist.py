import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class my_nn(nn.Module):
    def __init__(self, n_in_neu, n_mid_neu, n_out_neu):
        super(my_nn, self).__init__()
        self.layer1 = nn.Linear(n_in_neu, n_mid_neu)
        self.layer2 = nn.Linear(n_mid_neu, n_out_neu)

    def forward(self, in_neu):

        mid_neu = self.layer1.forward(in_neu)
        mid_neu = F.relu(mid_neu)
        out_neu = self.layer2.forward(mid_neu)
        out_neu = F.relu(out_neu)

        return out_neu

if __name__ == '__main__':
    in_train= torch.Tensor([(-1, -1, -1), (-1, -1, 1), (-1, 1, -1), (1, -1, -1), (1, 1, -1), (1, 1, 1), (-1, 1, 1), (1,-1, 1)])
    right_ans = torch.FloatTensor([-1, -1, -1, -1, 1, 1, 1, 1])
    lr = 0.01  # шаг обучения
    N = len(right_ans)

    model = my_nn(3,2,1)
    print(model)

    # weight_decay: с какой скоростью будет уменьшаться шаг
    optimizer = optim.SGD(model.parameters(), lr, weight_decay=0.05)
    criterion = nn.MSELoss()

    # --- Тренировка ---
    model.train()

    for _ in range(1000):
        k = np.random.randint(0, N-1)
        optimizer.zero_grad()
        out_neu = model(in_train[k])
        out_neu = out_neu.squeeze()
        loss = criterion(out_neu, right_ans[k])

        # --- Обратный проход ---
        loss.backward()

        # --- Оптимизатор выполняет процесс спуска ---
        optimizer.step()

    # --- Проверка ---
    model.eval()

    for in_neu, answ in zip(in_train, right_ans):
        out_neu = model(in_neu)
        print(f'Вход: {in_neu.data} | Ожидаемый выход: {answ.data} | Полученный выход: {out_neu.data}')
