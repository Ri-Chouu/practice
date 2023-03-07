import random

import numpy as np

direction = [[-1, -1], [-1, 0], [-1, 1], [0, 1], [1, 1], [1, 0], [0, -1], [1, -1]]


class Game:
    def __init__(self):
        self.currentPlayer = None
        self.player1 = 1
        self.player2 = -1

    def getLegalActions(self, board):
        legalActions = []
        for i in range(8):
            for j in range(8):
                if board[i][j] == 0 and self.confirmLegality(board, [i, j]):
                    legalActions.append([i, j])
        return np.array(legalActions)

    def confirmLegality(self, board, action):
        for d in direction:
            l = 1
            hasWhite = False
            while True:
                # 超出边界
                if action[0] + l * d[0] < 0 or action[1] + l * d[1] < 0 or action[0] + l * d[0] >= 8 or action[
                    1] + l * d[1] >= 8:
                    break
                # 断开
                if board[action[0] + l * d[0]][action[1] + l * d[1]] == 0:
                    break
                if board[action[0] + l * d[0]][action[1] + l * d[1]] == 1:
                    # 两个黑子夹了白子
                    if hasWhite:
                        return True
                    # 两个黑子相连
                    else:
                        break
                # 夹住白子
                if board[action[0] + l * d[0]][action[1] + l * d[1]] == -1:
                    hasWhite = True
                    l += 1
        return False

    def reset(self):
        self.currentPlayer = None
        board = np.zeros((8, 8))
        board[3][3] = 1
        board[4][4] = 1
        board[3][4] = -1
        board[4][3] = -1
        return board

    def switchPlayer(self):
        if not self.currentPlayer:
            self.currentPlayer = self.player1
        else:
            self.currentPlayer = [self.player1, self.player2][self.currentPlayer == self.player1]

    def judge(self, board):
        black = np.sum(board == 1)
        white = np.sum(board == -1)
        if black > white:
            return 1
        elif black < white:
            return -1
        else:
            return 0

    def step(self, board, action):
        # print("************")
        # print(board, action)
        board[action[0]][action[1]] = 1
        for d in direction:
            white = []
            l = 1
            hasWhite = False
            while True:
                # 超出边界
                if action[0] + l * d[0] < 0 or action[1] + l * d[1] < 0 or action[0] + l * d[0] >= 8 or action[
                    1] + l * d[1] >= 8:
                    break
                # 断开
                if board[action[0] + l * d[0]][action[1] + l * d[1]] == 0:
                    break
                if board[action[0] + l * d[0]][action[1] + l * d[1]] == 1:
                    # 两个黑子夹了白子
                    if hasWhite:
                        for w in white:
                            board[w[0]][w[1]] = 1
                        break
                    # 两个黑子相连
                    else:
                        break
                # 夹住白子
                if board[action[0] + l * d[0]][action[1] + l * d[1]] == -1:
                    hasWhite = True
                    white.append([action[0] + l * d[0], action[1] + l * d[1]])
                    l += 1
        # 如果都无法下子或者已经下满则结束
        if (not len(self.getLegalActions(board * -1)) and not len(self.getLegalActions(board))) or np.sum(
                board == 0) == 0:
            done = True
        else:
            done = False
        # print(board)
        # print("************")
        return board, done


g = Game()
episode = 1000
blackWin = 0
whiteWin = 0
draw = 0
for e in range(1, episode + 1):
    b = g.reset()
    g.switchPlayer()
    # print("board:\n", b)
    while True:
        action = g.getLegalActions(b * g.currentPlayer)
        # 无子可下，换人
        if not len(action):
            g.switchPlayer()
            continue
        b, done = g.step(b * g.currentPlayer, random.choice(action))
        b *= g.currentPlayer  # 转换为正常盘面
        r = 0
        # print("current player:", g.currentPlayer)
        # print("board:\n", b)
        g.switchPlayer()
        if done:
            # print(np.sum(b==1),np.sum(b==-1))
            r = g.judge(b)
            print("episode%d reward:%d" % (e, r))
            if r == 1:
                blackWin += 1
            elif r == -1:
                whiteWin += 1
            else:
                draw += 1
            break
print("%d試合の中に先手%d勝、後手%d勝、引き分け%d" % (episode, blackWin, whiteWin, draw))
