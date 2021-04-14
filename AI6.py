from copy import deepcopy

import numpy as np
import random
import time

from numpy import argmax

COLOR_BLACK = -1
COLOR_WHITE = 1
COLOR_NONE = 0
random.seed(0)
row = [0, -1, -1, -1, 0, 1, 1, 1]
col = [-1, -1, 0, 1, 1, 1, 0, -1]
INF = 100000005


class AI(object):
    weighting = [[500, -50, 20, 10, 10, 20, -50, 500],
                 [-50, -100, 1, 1, 1, 1, -100, -50],
                 [10, 1, 3, 2, 2, 3, 1, 10],
                 [5, 1, 2, 1, 1, 2, 10, 50],
                 [5, 1, 2, 1, 1, 2, 10, 50],
                 [10, 1, 3, 2, 2, 3, 1, 10],
                 [-50, -100, 1, 1, 1, 1, -100, -50],
                 [500, -50, 20, 10, 10, 20, -50, 500]]

    def __init__(self, chessboard_size, color, time_out):
        self.chessboard_size = chessboard_size
        self.color = color
        self.time_out = time_out
        self.candidate_list = []
        self.board = []
        self.DPT = 0

    def getDepth(self, chessboard):
        cnt = 0
        for i in range(8):
            for j in range(8):
                if chessboard[i][j] != COLOR_NONE:
                    cnt = cnt + 1
        if cnt < 10:
            return 5
        elif 10 <= cnt < 20:
            return 4
        elif 20 <= cnt < 45:
            return 3
        elif 45 <= cnt < 50:
            return 3
        elif 50 <= cnt < 53:
            return 4
        elif 53 <= cnt < 58:
            return 5
        return 6

    def go(self, chessboard):
        self.start_time = time.time()
        self.candidate_list.clear()
        self.board.clear()
        self.candidate_list = self.find_move(chessboard, self.color)
        self.DPT = self.getDepth(chessboard)
        self.sort(self.candidate_list)
        score, best = self.minmax(chessboard, self.color, -INF, INF, self.DPT)
        if len(best) > 0:
            self.candidate_list.append(best)
        # print(self.candidate_list)

    def check(self, chessboard, i, j, color):
        x = 0
        y = 0
        for k in range(8):
            if 8 > i + row[k] >= 0 and 8 > j + col[k] >= 0:
                x = i + row[k]
                y = j + col[k]
                if chessboard[x][y] == -color:
                    if self.nextCheck(chessboard, x, y, k, color):
                        return True
        return False

    def nextCheck(self, chessboard, i, j, k, color):
        if 8 > i + row[k] >= 0 and 8 > j + col[k] >= 0:
            xx = i + row[k]
            yy = j + col[k]
            if chessboard[xx][yy] == -color:
                if self.nextCheck(chessboard, xx, yy, k, color):
                    return True
            if chessboard[xx][yy] == color:
                return True
        return False

    def find_move(self, chessboard, color):
        available_move = []
        for i in range(8):
            for j in range(8):
                if chessboard[i][j] != COLOR_NONE:
                    continue
                if self.check(chessboard, i, j, color):
                    available_move.append((i, j))
        return available_move

    def update(self, board, move, color):
        new_board = deepcopy(board)
        new_board[move[0]][move[1]] = color
        for i in range(8):
            temp_change = []
            flag = 0
            x = move[0]
            y = move[1]
            while 8 > x + row[i] >= 0 and 8 > y + col[i] >= 0:
                x = x + row[i]
                y = y + col[i]
                if new_board[x][y] == -color:
                    temp_change.append((x, y))
                    continue
                if new_board[x][y] == color:
                    flag = 1
                    break
                break
            if flag == 1:
                for obj in temp_change:
                    new_board[obj[0]][obj[1]] = color
        return new_board

    def sort(self, avilable_move):
        for i in range(len(avilable_move)):
            for j in range(i + 1, len(avilable_move)):
                if self.weighting[avilable_move[i][0]][avilable_move[i][1]] > self.weighting[avilable_move[j][0]][
                    avilable_move[j][1]]:
                    temp = avilable_move[i]
                    avilable_move[i] = avilable_move[j]
                    avilable_move[j] = temp

    def minmax(self, board, color, alpha, beta, depth):
        best_move = ()
        avb_move = self.find_move(board, color)
        if len(avb_move) == 0:
            if len(self.find_move(board, -color)) == 0:
                my_cnt = 0
                op_cnt = 0
                for i in range(8):
                    for j in range(8):
                        if board[i][j] == color:
                            my_cnt += 1
                        if board[i][j] == -color:
                            op_cnt += 1
                if my_cnt > op_cnt:
                    return INF, best_move
                else:
                    return -INF, best_move
            return -self.minmax(board, -color, -beta, -alpha, depth)[0], best_move
        if depth == 0 or time.time() - self.start_time > self.time_out - 0.4:
            return self.getWei(board, color), best_move
        for move in avb_move:
            new_board = self.update(board, move, color)
            val = -self.minmax(new_board, -color, -beta, -alpha, depth - 1)[0]
            if val > alpha:
                alpha = val
                best_move = move
                if beta < alpha:
                    return val, best_move
        return alpha, best_move

    def getStable(self, board, color):
        sb = [[False, False, False, False, False, False, False, False],
              [False, False, False, False, False, False, False, False],
              [False, False, False, False, False, False, False, False],
              [False, False, False, False, False, False, False, False],
              [False, False, False, False, False, False, False, False],
              [False, False, False, False, False, False, False, False],
              [False, False, False, False, False, False, False, False],
              [False, False, False, False, False, False, False, False]]
        # 左上
        len = 8
        for i in range(8):
            for j in range(len):
                if board[i][j] == color:
                    sb[i][j] = True
                else:
                    len = j
                    break
        # 右上
        len = -1
        for i in range(8):
            for j in range(7, len, -1):
                if board[i][j] == color:
                    sb[i][j] = True
                else:
                    len = j
                    break
        # 左下
        len = 8
        for i in range(7, -1, -1):
            for j in range(len):
                if board[i][j] == color:
                    sb[i][j] = True
                else:
                    len = j
                    break
        # 右下
        len = -1
        for i in range(7, -1, -1):
            for j in range(7, len, -1):
                if board[i][j] == color:
                    sb[i][j] = True
                else:
                    len = j
                    break
        # 左上
        len = 8
        for i in range(8):
            for j in range(len):
                if board[j][i] == color:
                    sb[j][i] = True
                else:
                    len = j
                    break
        # 右上
        len = -1
        for i in range(8):
            for j in range(7, len, -1):
                if board[j][i] == color:
                    sb[j][i] = True
                else:
                    len = j
                    break
        # 左下
        len = 8
        for i in range(7, -1, -1):
            for j in range(len):
                if board[j][i] == color:
                    sb[j][i] = True
                else:
                    len = j
                    break
        # 右下
        len = -1
        for i in range(7, -1, -1):
            for j in range(7, len, -1):
                if board[j][i] == color:
                    sb[j][i] = True
                else:
                    len = j
                    break
        cnt = 0
        for i in range(8):
            for j in range(8):
                if sb[i][j] is True:
                    cnt += 1
        return cnt

    def getPot(self, board, color):
        value = 0
        for i in range(8):
            for j in range(8):
                if board[i][j] == color:
                    for k in range(8):
                        if 0 <= i + row[k] < 8 and 0 <= j + col[k] < 8:
                            if board[i + row[k]][j + col[k]] == 0:
                                value -= 1
                elif board[i][j] == -color:
                    for k in range(8):
                        if 0 <= i + row[k] < 8 and 0 <= j + col[k] < 8:
                            if board[i + row[k]][j + col[k]] == 0:
                                value += 1
        return value

    def getDiff(self, board, color):
        sumC = 0
        sumO = 0
        for i in range(8):
            for j in range(8):
                if board[i][j] == color:
                    sumC += 1
                if board[i][j] == -color:
                    sumO += 1
        return sumC - sumO

    def getWei(self, board, color):
        self_cnt = 0
        op_cnt = 0
        cnt = 0
        for i in range(8):
            for j in range(8):
                if board[i][j] != COLOR_NONE:
                    cnt = cnt + 1
                if board[i][j] == color:
                    self_cnt = self_cnt + 1
                if board[i][j] == -color:
                    op_cnt = op_cnt + 1
        board_weight = 0
        if cnt == 64 and self_cnt > op_cnt:
            return INF
        if cnt == 64 and self_cnt < op_cnt:
            return -INF
        for i in range(8):
            for j in range(8):
                if board[i][j] == color:
                    board_weight += self.weighting[i][j]
                elif board[i][j] == -color:
                    board_weight -= self.weighting[i][j]
        choose_weight = len(self.find_move(board, color))
        op_choose_weight = len(self.find_move(board, -color))
        stable_weight = self.getStable(board, color)
        potential_weight = self.getPot(board, color)
        if cnt < 20:
            total = 100 * board_weight + (choose_weight - op_choose_weight) + 10 * potential_weight
            return total
        elif 20 <= cnt < 40:
            total = 0.8 * board_weight + 40 * (
                        choose_weight - op_choose_weight) + 80 * stable_weight + 15 * potential_weight
            return total
        elif 40 <= cnt:
            total = board_weight + 30 * (choose_weight - op_choose_weight) + 100 * stable_weight + 15 * potential_weight
        return total

# if __name__ == '__main__':
#     board = [[0, 0, 0, 0, 0, 0, 0, 0],
#              [0, 0, -1, 1, 0, 1, 0, 0],
#              [0, 0, 0, -1, 1, 1, 0, 0],
#              [0, 0, 0, 1, -1, 1, 0, 0],
#              [0, 0, -1, -1, -1, -1, -1, 0],
#              [0, -1, -1, 0, 0, 1, 0, 0],
#              [0, 0, 0, 0, 0, 0, 0, 0],
#              [0, 0, 0, 0, 0, 0, 0, 0]]
#     ai = AI(board, 1, 100)
#     ai.go(board)
