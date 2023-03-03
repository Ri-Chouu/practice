import torch
from torch import nn
from torch import optim
import numpy as np


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.fc1 = nn.Linear(2, 16)
        self.fc2 = nn.Linear(16, 3)

    def forward(self, input):
        output = self.fc1(input)
        output = self.fc2(output)
        return output


class DQN:
    def __init__(self):
        self.predictNet = Network()
        self.targetNet = Network()
        self.learningRate = 0.01
        self.batchSize = 128
        self.opt = optim.SGD(params=self.predictNet.parameters(), lr=self.learningRate)
        self.lossFunction = nn.MSELoss()
        self.replaySize = 2000
        self.replayMemory = np.zeros((self.replaySize, 6))  # size X (p, v, r, a, _p, _v)
        self.replayIndex = 0  # 经验回放池索引
        self.trainCount = 0 # 训练次数
        self.updateRate = 10  # 更新targetNet频率

    def store(self, p, v, r, a, _p, _v):
        self.replayMemory[self.replayIndex] = [p, v, r, a, _p, _v]
        if self.replayIndex + 1 >= self.replaySize:
            self.replayIndex = (self.replayIndex + 1) % self.replaySize
            self.train()
            self.trainCount += 1
            if self.trainCount % self.updateRate == 0:
                self.targetNet = self.targetNet.load_state_dict(self.predictNet.state_dict())

    def train(self):
        pass
