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
                 [-50, -200, 1, 1, 1, 1, -200, -50],
                 [10, 1, 3, 2, 2, 3, 1, 10],
                 [5, 1, 2, 1, 1, 2, 10, 50],
                 [5, 1, 2, 1, 1, 2, 10, 50],
                 [10, 1, 3, 2, 2, 3, 1, 10],
                 [-50, -200, 1, 1, 1, 1, -200, -50],
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
            return 5
        elif 20 <= cnt < 45:
            return 4
        elif 45 <= cnt < 50:
            return 5
        elif 50 <= cnt < 53:
            return 6
        return 7

    def go(self, chessboard):
        self.candidate_list.clear()
        self.board.clear()
        self.candidate_list = self.find_move(chessboard, self.color)
        self.DPT = self.getDepth(chessboard)
        # best_move = None
        # best_score = -INF
        # for move in self.candidate_list:
        #     if best_score < self.weighting[move[0]][move[1]]:
        #         best_move = move
        #         best_score = self.weighting[move[0]][move[1]]
        # if best_move is not None:
        #     self.candidate_list.append(best_move)
        # self.sort(self.candidate_list)
        if len(self.candidate_list) > 1:
            self.minmax(chessboard, self.color, -INF, INF, self.DPT, True)

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
                if self.weighting[avilable_move[i][0]][avilable_move[i][1]] < self.weighting[avilable_move[j][0]][
                    avilable_move[j][1]]:
                    temp = avilable_move[i]
                    avilable_move[i] = avilable_move[j]
                    avilable_move[j] = temp

    def minmax(self, board, color, alpha, beta, depth, flag):
        available_move = self.find_move(board, color)
        # self.sort(available_move)
        if depth == 0:
            return self.getWei(board, flag)
        if len(available_move) == 0:
            return self.minmax(board, -color, alpha, beta, depth - 1, not flag)
        # self.sort(available_move)
        if flag:  # flag means max
            for move in available_move:
                new_board = self.update(board, move, color)
                v = self.minmax(new_board, -color, alpha, beta, depth - 1, False)
                if v > alpha:
                    alpha = v
                    if depth == self.DPT:
                        self.candidate_list.append(move)
                if alpha >= beta:
                    return beta
        else:
            for move in available_move:
                new_board = self.update(board, move, color)
                v = self.minmax(new_board, -color, alpha, beta, depth - 1, True)
                if v < beta:
                    beta = v
                if alpha >= beta:
                    return alpha
        if flag:
            return alpha
        return beta

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

    def getWei(self, board, flag):
        color = self.color
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
        if len(self.find_move(board, color)) == 0 and len(self.find_move(board, -color)) == 0:
            if self_cnt > op_cnt:
                return INF
            elif self_cnt < op_cnt:
                return -INF
        board_weight = 0
        for i in range(8):
            for j in range(8):
                if board[i][j] == color:
                    board_weight += self.weighting[i][j]
                elif board[i][j] == -color:
                    board_weight -= self.weighting[i][j]
        choose_weight = len(self.find_move(board, color))
        stable_weight = self.getStable(board, -color)
        # diff_weight = self.getDiff(board, self.color)
        # c = 0
        # if board[0][1] == self.color:
        #     c += 1
        # if board[0][6] == self.color:
        #     c += 1
        # if board[1][0] == self.color:
        #     c += 1
        # if board[1][7] == self.color:
        #     c += 1
        # if board[7][1] == self.color:
        #     c += 1
        # if board[6][0] == self.color:
        #     c += 1
        # if board[7][6] == self.color:
        #     c += 1
        # if board[6][7] == self.color:
        #     c += 1
        # if board[1][1] == self.color:
        #     c += 2
        # if board[1][6] == self.color:
        #     c += 2
        # if board[6][1] == self.color:
        #     c += 2
        # if board[6][6] == self.color:
        #     c += 2
        # punish = c * board[1][1]
        potential_weight = self.getPot(board, color)
        if cnt < 30:
            total = 50 * board_weight + choose_weight + 10 * choose_weight
            return total
        elif 30 <= cnt < 40:
            total = board_weight + 35 * choose_weight + 80 * stable_weight + 10 * potential_weight
            return total
        elif 40 <= cnt:
            total = 2 * board_weight + 15 * choose_weight + 80 * stable_weight + 15 * potential_weight
            return total
