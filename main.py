import numpy as np
import torch
import torch.optim as optim
from hypnettorch.mnets import MLP
from torch.nn import MSELoss, BCELoss
from torch.utils.data import Dataset
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader
import pickle
from hypnettorch.hnets import HMLP

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class MyDataset(Dataset):
    def calculate_target(self, x0, x1):
        if x0 ** 2 + x1 ** 2 <= 10 ** 2:
            return 1
        else:
            return 0

    def __init__(self):
        self.data = [[i, j] for i in range(20) for j in range(20)]
        self.y = [self.calculate_target(d[0], d[1]) for d in self.data]
        self.scaler = MinMaxScaler()
        self.scaler.fit(self.data)
        self.data = self.scaler.transform(self.data)

    def __getitem__(self, idx):

        return torch.FloatTensor([self.data[idx][0], self.data[idx][1]]), torch.FloatTensor([self.y[idx]])

    def __len__(self):
        return len(self.y)


LR = 0.0001
dataset = MyDataset()

# mnet = MLP(n_in=2, n_out=1, hidden_layers=(20, 20), out_fn=torch.sigmoid).to(device)
#
# optimizer = optim.Adam(mnet.parameters(), lr=LR)
#
# n_epochs = 20
#
# dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
# it = 0
# for epoch in range(n_epochs):
#     for i, data in tqdm(enumerate(dataloader)):
#         X, y = data
#
#         X = X.to(device)
#         y = y.to(device)
#
#         optimizer.zero_grad()
#         with torch.set_grad_enabled(True):
#             outputs = mnet.forward(X)
#             criterion = BCELoss()
#             loss = criterion(outputs[0], y)
#
#             loss.backward()
#             optimizer.step()
#             it += 1

mnet = MLP(n_in=2, n_out=1, hidden_layers=(5, 5, 5), no_weights=True, out_fn=torch.sigmoid).to(device)

# print(outputs[0])

hnet = HMLP(mnet.param_shapes, uncond_in_size=0, cond_in_size=8,
            layers=[100, 100], num_cond_embs=2).to(device)

hnet.apply_hyperfan_init(mnet=mnet)

n_epochs = 100
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
criterion = BCELoss()
optimizer = optim.Adam(hnet.internal_params, lr=LR)
for epoch in range(n_epochs):
    for i, data in tqdm(enumerate(dataloader)):
        optimizer.zero_grad()

        X, y = data

        condition = int(np.random.choice([0, 1]))
        if condition == 1:
            y = (y + 1) % 2

        X = X.to(device)
        y = y.to(device)
        # with torch.set_grad_enabled(True):
        Weights = hnet(cond_id=condition)
        outputs = mnet.forward(X, weights=Weights)
        loss = criterion(outputs[0], y)
        print(f'loss: {loss}')
        loss.backward()
        optimizer.step()

# points = [[i, j] for i in range(20) for j in range(20)]
# x_ = [point[0] for point in points]
# y_ = [point[1] for point in points]
#
# predictions = [point[1] for point in dataset]
#
# colors = []
# for pred in predictions:
#     if pred > 0.5:
#         colors.append('green')
#     else:
#         colors.append('red')
#
# import matplotlib.pyplot as plt
#
# plt.scatter(x_, y_, color=colors)

Weights = hnet(cond_id=0) #try cond_id = 1

points = [[i, j] for i in range(20) for j in range(20)]
predictions = mnet.forward(
    torch.FloatTensor(dataset.scaler.transform(np.array([12, 1]).reshape(1, -1))).flatten().to(device), weights=Weights)

predictions = []

for point in points:
    predictions.append(mnet.forward(
        torch.FloatTensor(dataset.scaler.transform(np.array(point).reshape(1, -1))).flatten().to(device),
        weights=Weights)[0])

colors = []
for pred in predictions:
    if pred > 0.5:
        colors.append('green')
    else:
        colors.append('red')

x_ = [point[0] for point in points]
y_ = [point[1] for point in points]

import matplotlib.pyplot as plt

plt.scatter(x_, y_, color=colors)
