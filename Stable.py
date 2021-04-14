import copy
import numpy as np
import time

dr = [0, 1, 1, 1, 0, -1, -1, -1]
dc = [-1, -1, 0, 1, 1, 1, 0, -1]

weighting = np.array([[500, -100, 10, 5, 5, 10, -100, 500],
                      [-100, -180, 1, 1, 1, 1, -180, -100],
                      [10, 1, 3, 2, 2, 3, 1, 10],
                      [5, 1, 2, 1, 1, 2, 1, 5],
                      [5, 1, 2, 1, 1, 2, 1, 5],
                      [10, 1, 3, 2, 2, 3, 1, 10],
                      [-100, -180, 1, 1, 1, 1, -180, -100],
                      [500, -100, 10, 5, 5, 10, -100, 500]])


class AI(object):

    def __init__(self, chessboard_size, color, time_out):
        self.chessboard_size = chessboard_size
        self.color = color
        self.time_out = time_out
        self.candidate_list = []

    def go(self, chessboard):

        self.candidate_list.clear()
        self.start_time = time.time()

        if self.cal_num(chessboard) >= 57:
            best_move = self.alpha_beta_pruning_end(self.color, chessboard, -10000, 10000, 7)[1]
        else:
            best_move = self.alpha_beta_pruning(self.color, chessboard, -10000, 10000, 4)[1]
        self.candidate_list = self.find_available_move(self.color, chessboard)
        if len(best_move) > 0:
            self.candidate_list.append(best_move)

    def alpha_beta_pruning(self, color, chessboard, alpha, beta, tree_depth):
        best_move = ()

        available_move = self.find_available_move(color, chessboard)
        len_available_move = len(available_move)

        if len_available_move == 0:
            if len(self.find_available_move(-color, chessboard)) == 0:
                value = np.sum(chessboard)
                if color == -1:
                    value = -value
                value += 10000 if value > 0 else -10000
                return value, best_move
            return -self.alpha_beta_pruning(-color, chessboard, -beta, -alpha, tree_depth)[0], best_move

        if tree_depth == 0 or time.time() - self.start_time > self.time_out - 0.4:
            return self.calculate_value(chessboard, len_available_move, color), best_move

        for i in range(len(available_move)):
            if weighting[available_move[i][0]][available_move[i][1]] == 500:
                return 1000, available_move[i]
            new_chessboard = self.apply_move(chessboard, color, available_move[i])
            value = -self.alpha_beta_pruning(-color, new_chessboard, -beta, -alpha, tree_depth - 1)[0]
            if value > alpha:
                alpha = value
                best_move = available_move[i]
                if beta < alpha:
                    return value, best_move
        return alpha, best_move

    def cal_num(self, chessboard):
        cnt = 0
        for i in range(8):
            for j in range(8):
                if chessboard[i][j] != 0:
                    cnt += 1
        return cnt

    def calculate_value(self, chessboard, len_available_move, color):
        value = np.sum(chessboard * weighting)
        if color == -1:
            value = -value
        len_reverse_move = len(self.find_available_move(-color, chessboard))
        if self.cal_num(chessboard) < 20:
            return value + (len_available_move - len_reverse_move) * 20 + self.eval_potential(chessboard, color) * 10
        else:
            return value + (len_available_move - len_reverse_move) * 35 + \
                   self.eval_potential(chessboard, color) * 20 + self.eval_stable(chessboard, color) * 100

    def find_available_move(self, color, chessboard):
        available_move = []
        for col in range(8):
            for row in range(8):
                if chessboard[row][col] == 0:
                    for i in range(8):
                        if 8 > row + dr[i] >= 0 and 8 > col + dc[i] >= 0:
                            r = row + dr[i]
                            c = col + dc[i]
                            if chessboard[r][c] == -color:
                                if self.next_search(r, c, i, chessboard, row, col, color, available_move):
                                    break
        return available_move

    def next_search(self, r, c, i, chessboard, row, col, color, available_move):
        if 8 > r + dr[i] >= 0 and 8 > c + dc[i] >= 0:
            r = r + dr[i]
            c = c + dc[i]
            if chessboard[r][c] == -color:
                return self.next_search(r, c, i, chessboard, row, col, color, available_move)
            elif chessboard[r][c] == color:
                available_move.append((row, col))
                return True
            else:
                return False

    def apply_move(self, chessboard, color, available_move):
        new_chessboard = copy.deepcopy(chessboard)
        x = available_move[0]
        y = available_move[1]
        new_chessboard[x][y] = color

        for d in range(8):
            i = x + dr[d]
            j = y + dc[d]
            while 0 <= i < 8 and 0 <= j < 8 and new_chessboard[i][j] == -color:
                i += dr[d]
                j += dc[d]
            if 0 <= i < 8 and 0 <= j < 8 and new_chessboard[i][j] == color:
                while True:
                    i -= dr[d]
                    j -= dc[d]
                    if i == x and j == y:
                        break
                    new_chessboard[i][j] = color
        return new_chessboard

    def alpha_beta_pruning_end(self, color, chessboard, alpha, beta, tree_depth):
        best_move = ()
        available_move = self.find_available_move(color, chessboard)

        len_available_move = len(available_move)
        if len_available_move == 0:
            if len(self.find_available_move(-color, chessboard)) == 0:
                value = np.sum(chessboard)
                if color == -1:
                    value = -value
                value += 1000 if value > 0 else -1000
                return value, best_move
            return -self.alpha_beta_pruning(-color, chessboard, -beta, -alpha, tree_depth)[0], best_move

        for i in range(len(available_move)):
            new_chessboard = self.apply_move(chessboard, color, available_move[i])
            value = -self.alpha_beta_pruning_end(-color, new_chessboard, -beta, -alpha, tree_depth - 1)[0]
            if value > alpha:
                alpha = value
                best_move = available_move[i]
                if beta < alpha:
                    return value, best_move
        return alpha, best_move

    def eval_potential(self, chessboard, color):
        count_black = 0
        count_white = 0
        for col in range(8):
            for row in range(8):
                if chessboard[row][col] == -1:
                    for i in range(8):
                        if 8 > row + dr[i] >= 0 and 8 > col + dc[i] >= 0:
                            r = row + dr[i]
                            c = col + dc[i]
                            if chessboard[r][c] == 0:
                                count_black += 1
                elif chessboard[row][col] == 1:
                    for i in range(8):
                        if 8 > row + dr[i] >= 0 and 8 > col + dc[i] >= 0:
                            r = row + dr[i]
                            c = col + dc[i]
                            if chessboard[r][c] == 0:
                                count_white += 1
        if color == 1:
            return count_black - count_white
        return count_white - count_black

    def eval_stable(self, chessboard, color):
        new_chessboard = np.zeros((8, 8))
        # 左上
        if chessboard[0][0] != 0:
            now_color = chessboard[0][0]
            len = 8
            for i in range(8):
                for j in range(len):
                    if chessboard[i][j] == now_color:
                        new_chessboard[i][j] = now_color
                    else:
                        len = j
                        break

        # 右上
        if chessboard[0][7] != 0:
            now_color = chessboard[0][7]
            len = -1
            for i in range(8):
                for j in range(7, len, -1):
                    if chessboard[i][j] == now_color:
                        new_chessboard[i][j] = now_color
                    else:
                        len = j
                        break
        # 左下
        if chessboard[7][0] != 0:
            now_color = chessboard[7][0]
            len = 8
            for i in range(7, -1, -1):
                for j in range(len):
                    if chessboard[i][j] == now_color:
                        new_chessboard[i][j] = now_color
                    else:
                        len = j
                        break
        # 右下
        if chessboard[7][7] != 0:
            now_color = chessboard[7][7]
            len = -1
            for i in range(7, -1, -1):
                for j in range(7, len, -1):
                    if chessboard[i][j] == now_color:
                        new_chessboard[i][j] = now_color
                    else:
                        len = j
                        break

        value = np.sum(new_chessboard)
        if color == -1:
            value = -value
        return value


if __name__ == "__main__":
    chessboard = [[-1, -1, -1, -1, -1, -1, -1, 1],
                  [-1, 1, 1, 1, 1, 1, 1, 1],
                  [-1, 1, 1, 1, -1, 1, -1, 1],
                  [-1, -1, 1, 1, 1, -1, -1, 1],
                  [-1, -1, -1, 1, 1, 1, -1, 1],
                  [-1, -1, 1, -1, -1, -1, 1, 1],
                  [-1, 1, 1, 1, -1, -1, 0, 1],
                  [0, -1, -1, -1, -1, -1, 0, 0]]
    ai = AI(chessboard, -1, 100)

    ai.go(np.array(chessboard))
