import random
import numpy as np
import torch
from torch import nn
from torch import optim


# NN为kx+b形式，不能拟合幂函数，如果有的话需要计算x ** y再作为输入
# y = 3.0 * a + 1.5 + 5.2 * b
def curve(a, b):
    return 3.0 * a + 1.5 + 5.2 * b


data = np.array([])
for i in range(100):
    a = random.random()
    b = random.random()
    data = np.append(data, [a, b, curve(a, b)])
data = data.reshape(-1, 3)


class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 16)
        self.fc2 = nn.Linear(16, 1)

    def forward(self, input):
        output = self.fc1(input)
        output = self.fc2(output)
        return output


batchSize = 64
learningRate = 0.01
net = Network()
opt = optim.SGD(params=net.parameters(), lr=learningRate)
lossFunction = nn.MSELoss()

for epoch in range(1000):
    samples = torch.Tensor(data[np.random.choice(data.shape[0], batchSize), :])
    xSamples = samples[:, :-1]
    # print(xSamples)
    ySamples = samples[:, -1]
    # print(ySamples)
    yPredict = net(xSamples).view(-1)
    # print(yPredict)
    loss = lossFunction(yPredict, ySamples)
    print("epoch:%d loss:%.10f" % (epoch, loss.item()))
    loss.backward()
    opt.step()
    opt.zero_grad()

data = np.array([0.1, 0.1])
y = curve(data[0], data[1])
_y = net(torch.Tensor(data)).item()
print("\nテスト:\n目標:%f 予測:%f" % (y, _y))
