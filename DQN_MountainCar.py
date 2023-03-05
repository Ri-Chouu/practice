import torch
from torch import nn
from torch import optim
import numpy as np
import gymnasium as gym
import random


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 3)
        )

    def forward(self, input):
        return self.fc(input)


class DQN:
    def __init__(self):
        self.predictNet = Network()
        self.targetNet = Network()
        self.learningRate = 0.001
        self.gamma = 0.9
        self.batchSize = 128
        self.opt = optim.Adam(params=self.predictNet.parameters(), lr=self.learningRate)
        self.lossFunction = nn.MSELoss()
        self.replaySize = 2000
        self.replayMemory = np.zeros((self.replaySize, 6))  # size X (p, v, r, a, _p, _v)
        self.replayIndex = 0  # 经验回放池索引
        self.trainCount = 0  # 训练次数
        self.updateRate = 20  # 更新targetNet频率

    def store(self, p, v, r, a, _p, _v):
        self.replayMemory[self.replayIndex % self.replaySize] = [p, v, r, a, _p, _v]
        self.replayIndex += 1
        if self.replayIndex >= self.replaySize:
            self.train()
            self.trainCount += 1
            # print("Train:", self.trainCount)
            # 更新目标网络
            if self.trainCount % self.updateRate == 0:
                self.targetNet.load_state_dict(self.predictNet.state_dict())

    def train(self):
        Samples = torch.Tensor(self.replayMemory[np.random.choice(self.replayMemory.shape[0], self.batchSize), :])
        sSamples = Samples[:, :2]
        rSamples = Samples[:, 2]
        aSamples = Samples[:, 3]
        _sSamples = Samples[:, -2:]
        predict = self.predictNet(sSamples)
        target = self.targetNet(_sSamples).detach()
        predictQ = predict[torch.arange(0, self.batchSize), aSamples.long()]
        targetQ = rSamples + self.gamma * torch.max(target, 1)[0]
        loss = self.lossFunction(predictQ, targetQ)
        loss.backward()
        self.opt.step()
        self.opt.zero_grad()
        # print("\nTrain:", self.trainCount, "loss:%.10f" % loss.item())

    def getAction(self, s):
        input = torch.from_numpy(s)
        return np.argmax(self.predictNet(input).detach().numpy())


dqn = DQN()

step = 0  # 单次回合数
stopTraining = 0
"""  训练  """
episode = 0
env = gym.make("MountainCar-v0")
while True:
    if step < 200:
        stopTraining += 1
        if stopTraining == 10:
            print("訓練完了、テストを実行する")
            break
    else:
        stopTraining = 0
    observation, _ = env.reset()
    s = observation
    step = 0
    # 限制回合数
    while step <= 400:
        epsilon = max(0.9 - 0.8 * episode / 1000, 0.1)
        # print(epsilon)
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            action = dqn.getAction(s)
        observation, reward, done, _, _ = env.step(action)
        _s = observation
        dqn.store(s[0], s[1], reward, action, _s[0], _s[1])
        s = _s
        env.render()
        # print("Step:%d Position:%f Velocity:%f Reward:%f Done:%d" % (step, observation[0], observation[1], reward,
        # done))
        if done:
            reward = 1
            dqn.store(s[0], s[1], reward, action, _s[0], _s[1])
            break
        step += 1
    episode += 1
    print("Episode:%d Step:%d" % (episode, step - 1))

"""  测试  """

env = gym.make("MountainCar-v0", render_mode="human")
while True:
    observation, _ = env.reset()
    step = 0
    while step <= 200:
        s = observation
        action = dqn.getAction(s)
        observation, reward, done, _, _ = env.step(action)
        env.render()
        # print("Step:%d Position:%f Velocity:%f Reward:%f Done:%d" % (step, observation[0], observation[1], reward, done))
        if done:
            break
        step += 1
