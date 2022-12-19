from pickletools import optimize
from data_loader.test import LinearRegression
from normalization.test import TestNorm
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as opt
from tqdm import tqdm

normmer = TestNorm(norm_path='qq')
dataset = LinearRegression(data_path='qq',
                           normmer=normmer,
                           is_aug=False)
data_loader = DataLoader(dataset=dataset,
                         shuffle=True,
                         batch_size=16,
                         num_workers=4)

model = nn.Linear(1, 1, bias=False)

criterion = nn.MSELoss()
optimizer = opt.Adam(model.parameters(), lr=1.e-3)

epochs = [x for x in range(100)]
for epoch in tqdm(epochs):
    for xs, ys in data_loader:
        optimizer.zero_grad()
        # print(xs.size(), ys.size())

        preds = model(xs)

        loss = criterion(preds, ys)

        loss.backward()

        optimizer.step()

        print(ys[0], preds[0], xs[0])
