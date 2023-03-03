import random
import gymnasium as gym
import numpy as np

min_position = -1.2
max_position = 0.6
min_velocity = -0.07
max_velocity = 0.07

size = 30  # 字典大小 size*size


# 转换为字典索引
def transform(s):
    p = size * (s[0] - min_position) / (max_position - min_position)
    v = size * (s[1] - min_velocity) / (max_position - min_velocity)
    return int(p), int(v)


class QLearning:
    def __init__(self):
        self.QTable = {}
        for i in range(size):
            for j in range(size):
                self.QTable[(i, j)] = [0, 0, 0]
        self.alpha = 0.1
        self.gamma = 0.9

    # 训练
    def train(self, s, r, a, _s):
        s = transform(s)
        _s = transform(_s)
        self.QTable[s][a] = self.QTable[s][a] + self.alpha * (
                r + self.gamma * np.max(self.QTable[_s]) - self.QTable[s][a])

    # 获得最佳动作
    def getAction(self, s):
        s = transform(s)
        return np.argmax(self.QTable[s])


Q = QLearning()

episode = 2000  # 训练次数
step = 0  # 单次回合数

"""  训练  """

env = gym.make("MountainCar-v0")
for i in range(1, episode + 1):
    print("Episode:%d/%d Step:%d" % (i, episode, step - 1))
    observation, _ = env.reset()
    s = observation
    step = 0
    # 限制回合数
    while step <= 400:
        epsilon = 0.9 - 0.8 * i / episode
        # print(epsilon)
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            action = Q.getAction(s)
        observation, reward, done, _, _ = env.step(action)
        _s = observation
        Q.train(s, reward, action, _s)
        s = _s
        env.render()
        # print("Step:%d Position:%f Velocity:%f Reward:%f Done:%d" % (step, observation[0], observation[1], reward, done))
        if done:
            reward = 1
            Q.train(s, action, reward, _s)
            break
        step += 1

for q in Q.QTable:
    print(q, Q.QTable[q])

"""  测试  """

env = gym.make("MountainCar-v0", render_mode="human")
while True:
    observation, _ = env.reset()
    step = 0
    while step <= 200:
        s = observation
        action = Q.getAction(s)
        observation, reward, done, _, _ = env.step(action)
        env.render()
        # print("Step:%d Position:%f Velocity:%f Reward:%f Done:%d" % (step, observation[0], observation[1], reward, done))
        if done:
            break
        step += 1
