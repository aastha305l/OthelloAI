from copy import deepcopy
import time

def current_milli_time():
     return int(round(time.time() * 1000))

SPACE = ' '
BLACK = 'B'
WHITE = 'W'
NEG_INF = -1000000000000
POS_INF = 1000000000000

class Boardstate:
    '''A class with board, next_turn, is_game_over, and everything that
    we need to know about current state'''

    DEBUG_MODE = False

    def __init__(self, n, board=None, next_turn=None):
        self.n = n
        if board == None:
            self.board = [[SPACE] * n for i in range(n)]
            self.add_middle_pieces()
            self.num_pieces = 4 # board's initialized with 4 pieces
        else:
            self.board = board
            self.num_pieces = 0
            for r in board:
                for p in r:
                    if p != ' ': self.num_pieces += 1
        self.next_turn = BLACK if next_turn == None else next_turn
        self.is_game_over = False
        self.winner = None
        self.last_turn_passed = False

    def add_middle_pieces(self):
        mid_x = (self.n - 1) // 2
        mid_y = mid_x
        self.board[mid_x][mid_y] = BLACK
        self.board[mid_x][mid_y + 1] = WHITE
        self.board[mid_x + 1][mid_y] = WHITE
        self.board[mid_x + 1][mid_y + 1] = BLACK

    def get_piece_count(self):
        b = 0
        w = 0
        for r in self.board:
            for c in r:
                if c == BLACK: b += 1
                elif c == WHITE: w += 1
        return b, w

    def update_winner(self):
        b, w = self.get_piece_count()
        if b > w:
            self.winner = BLACK
        elif w > b:
            self.winner = WHITE
        else:
            self.winner = SPACE # draw
        return

    def make_move(self, move):
        if move == (-1, -1):
            self.next_turn = BLACK if self.next_turn == WHITE else WHITE
            if self.last_turn_passed:
                self.is_game_over = True
                self.update_winner()
            else:
                self.last_turn_passed = True
        else:
            self.board[move[0]][move[1]] = self.next_turn
            self.next_turn = BLACK if self.next_turn == WHITE else WHITE
            self.num_pieces += 1
            self.last_turn_passed = False
            if self.num_pieces == self.n * self.n:
                self.is_game_over = True
                self.update_winner()

    def __str__(self):
        s = ""
        if Boardstate.DEBUG_MODE:
            s += "n: " + str(self.n) + '\n'
            s += "next_turn: "
            if self.next_turn == BLACK:
                s += "Black"
            else:
                s += "White"
            s += '\n'
            s += "is_game_over: " + str(self.is_game_over) + '\n'
            s += "winner: "
            if self.winner == BLACK:
                s += "Black"
            elif self.winner == WHITE:
                s += "White"
            else:
                s += "None"
            s += '\n'
        s += ' '
        for i in range(self.n):
            s += ' ' + str(i)
        s += '\n'
        s += ' '
        for _ in range(self.n):
            s += '--'
        s += '-\n'
        i = 0
        for r in self.board:
            s += str(i)
            i += 1
            for c in r:
                s += '|'
                if c == BLACK:
                    s += 'B'
                elif c == WHITE:
                    s += 'W'
                else:
                    s += ' '
            s += '|'
            s += '\n'
            s += ' '
            for _ in range(self.n):
                s += '--'
            s += '-\n'
        return s


def copy_b_state(b_state):
    new_b_state = Boardstate(b_state.n)
    new_b_state.next_turn = b_state.next_turn
    new_b_state.is_game_over = b_state.is_game_over
    new_b_state.winner = b_state.winner
    new_b_state.board = deepcopy(b_state.board)
    new_b_state.num_pieces = b_state.num_pieces
    new_b_state.last_turn_passed = b_state.last_turn_passed
    return new_b_state


def coin_count_value(b, w): # positive is good for black, negative good for white
    return 100 * (b - w)/(b + w)

def mobility_value(b_state): # positive is good for black, negative good for white
    b_mobility = 0
    w_mobility = 0
    for i in range(b_state.n):
        for j in range(b_state.n):
            c_b_state = copy_b_state(b_state)
            c_b_state.next_turn = 'B'
            c_b_state, is_valid = get_next_state(c_b_state, (i, j))
            if is_valid: b_mobility += 1
            c_b_state = copy_b_state(b_state)
            c_b_state.next_turn = 'W'
            c_b_state, is_valid = get_next_state(c_b_state, (i, j))
            if is_valid: w_mobility += 1
    if b_mobility + w_mobility == 0: return 0
    return 100 * (b_mobility - w_mobility) / (b_mobility + w_mobility)

def corner_value(b_state): # positive is good for black, negative good for white
    b_corner = 0
    w_corner = 0
    n = b_state.n
    four_corners = [(0, 0), (0, n - 1), (n - 1, 0), (n - 1, n - 1)]
    for corner in four_corners:
        r = corner[0]
        c = corner[1]
        if b_state.board[r][c] == 'B': b_corner += 1
        if b_state.board[r][c] == 'W': w_corner += 1
    if b_corner + w_corner == 0: return 0
    return 100 * (b_corner - w_corner) / (b_corner + w_corner)

def stability_value(b_state):
    b_stability = 0
    w_stability = 0
    stability = []
    if b_state.n != 6 and b_state.n != 8 and b_state.n != 4:
        return 0
    if b_state.n == 4:
        stability = [[4, -3, -3, 4],
                [-3, -4, -4, -3],
            [-3, -4, -4, -3],
        [4, -3, -3, 4]]
    if b_state.n == 6:
        stability = [[4, -3, 2, 2, -3, 4],
        [-3, -4, -1,-1,-4,-3],
        [2, -1, 1, 1, -1, 2],
        [2, -1, 1, 1, -1, 2],
        [-3, 4, -1, -1, -4, -3],
        [4, -3, 2, 2, -3, 4]]
    if b_state.n == 8:
        stability = [[4, -3, 2, 2, 2, 2, -3, 4],
        [-3, -4, -1, -1, -1, -1, -4, -3],
        [2, -1, 1, 0, 0, 1, -1, 2],
        [2, -1, 0, 1, 1, 0, -1, 2],
        [2, -1, 0, 1, 1, 0, -1, 2],
        [2, -1, 1, 0, 0, 1, -1, 2],
        [-3, -4, -1, -1, -1, -1, -4, -3],
        [4, -3, 2, 2, 2, 2, -3, 4]]
    for i in range(b_state.n):
        for j in range(b_state.n):
            if b_state.board[i][j] == 'B': b_stability += stability[i][j]
            if b_state.board[i][j] == 'W': w_stability += stability[i][j]

    if b_stability + w_stability == 0: return 0
    return 100 * ( b_stability - w_stability) / (b_stability + w_stability)

def heuristics(b_state):
    b, w = b_state.get_piece_count()
    total_percent = (b + w) / (b_state.n * b_state.n)
    coin_count_v = coin_count_value(b, w)
    mobility = mobility_value(b_state)
    corner_v = corner_value(b_state)
    stability = stability_value(b_state)
    v = 0

    if total_percent < 0.4:  # starting
        v = 40 * mobility + 40 * stability + 10 * coin_count_v +  10 * corner_v
    elif total_percent < 0.8: # mid-game
        v = 40 * stability + 40 * corner_v + 10 * mobility + 10 * coin_count_v
    else:  # end-game
        v = 80 * coin_count_v + 20 * stability
    return v

def minimax(b_state, depth_left, alpha, beta, maximizer):
    if depth_left == 0 or b_state.is_game_over:
        return heuristics(b_state), None
    if b_state.next_turn == maximizer:
        maximum = NEG_INF
        arg_max = None
        move_found = False
        for r in range(b_state.n):
            for c in range(b_state.n):
                new_state, is_valid = get_next_state(b_state, (r, c))
                if is_valid:
                    move_found = True
                    score, move = minimax(new_state, depth_left - 1, alpha, beta, maximizer)
                    alpha = max(alpha, score)
                    if score > maximum:
                        maximum = score
                        arg_max = (r, c)
                    if beta <= alpha:
                        break
        if not move_found:
            new_state, is_valid = get_next_state(b_state, (-1, -1))
            score, move = minimax(new_state, depth_left - 1, alpha, beta, maximizer)
            return score, (-1, -1)
        return maximum, arg_max
    else:
        minimum = POS_INF
        arg_min = None
        move_found = False
        for r in range(b_state.n):
            for c in range(b_state.n):
                new_state, is_valid = get_next_state(b_state, (r, c))
                if is_valid:
                    move_found = True
                    score, move = minimax(new_state, depth_left - 1, alpha, beta, maximizer)
                    beta = min(score, beta)
                    if score < minimum:
                        minimum = score
                        arg_min = (r, c)
                    if beta <= alpha:
                        break
        if not move_found:
            new_state, is_valid = get_next_state(b_state, (-1, -1))
            score, move = minimax(new_state, depth_left - 1, alpha, beta, maximizer)
            return score, (-1, -1)
        return minimum, arg_min

def moves_left(n, board_state):
    space_count = 0
    for r in range(0, n):
        for c in range(0, n):
            if board_state[r][c] == ' ': space_count += 1
    return space_count // 2

def get_move(board_size, board_state, turn, time_left, opponent_time_left):
    start_time = current_milli_time()
    maximizer = BLACK
    max_depth = 5
    best_score = NEG_INF if turn == BLACK else POS_INF
    best_move = None
    move_left = moves_left(board_size, board_state)
    average_move_time = time_left / move_left
    print (average_move_time)
    last_cycle_time = 1
    b_state = Boardstate(board_size, board_state, turn)
    while current_milli_time() - start_time + (last_cycle_time + 4) < average_move_time:
        c_b_state = copy_b_state(b_state)
        b_score, b_move = minimax(c_b_state, max_depth, NEG_INF, POS_INF, maximizer)
        if turn == maximizer:
            if b_score > best_score:
                best_score = b_score
                best_move = b_move
        else:
            if b_score < best_score:
                best_score = b_score
                best_move = b_move
        max_depth += 1
        last_cycle_time = current_milli_time() - start_time

    if best_move == (-1, -1): best_move = None
    return best_move

def outofbounds(n, r, c):
    return r < 0 or c < 0 or r >= n or c >=n

def is_capture(b_state, next_move, direction):
    next_r = next_move[0] + direction[0]
    next_c = next_move[1] + direction[1]
    if outofbounds(b_state.n, next_r, next_c) or b_state.board[next_r][next_c] == b_state.next_turn \
       or b_state.board[next_r][next_c] == SPACE:
        return False
    next_r = next_r + direction[0]
    next_c = next_c + direction[1]
    while not outofbounds(b_state.n, next_r, next_c):
        if b_state.board[next_r][next_c] == b_state.next_turn:
            return True
        if b_state.board[next_r][next_c] == SPACE:
            return False
        next_r = next_r + direction[0]
        next_c = next_c + direction[1]
    return False

def capture(b_state, next_move, direction):
    next_r = next_move[0] + direction[0]
    next_c = next_move[1] + direction[1]
    while b_state.board[next_r][next_c] != b_state.next_turn:
        b_state.board[next_r][next_c] = b_state.next_turn
        next_r = next_r + direction[0]
        next_c = next_c + direction[1]
    return

def get_next_state(board_state, next_move):
    b_state = copy_b_state(board_state)
    if next_move == (-1, -1):
        b_state.make_move(next_move)
        return b_state, True

    if b_state.board[next_move[0]][next_move[1]] != SPACE:
        return b_state, False
    d = [-1, 0, 1]
    captured = False
    for dx in d:
        for dy in d:
            if dx != 0 or dy != 0:
                if is_capture(b_state, next_move, (dx, dy)):
                    captured = True
                    capture(b_state, next_move, (dx, dy))
    if capture:
         b_state.make_move(next_move)
    return b_state, captured




def test():
    board = [[" ", " ", " ", " ", " ", " "],
    [" ", " ", " ", " ", " ", " "],
    [" ", " ", "W", "B", " ", " "],
    [" ", " ", "B", "W", " ", " "],
    [" ", " ", " ", " ", " ", " "],
    [" ", " ", " ", " ", " ", " "]]
    board_size = 6
    time_left = 150000
    opponent_time_left = 100000
    print (get_move(board_size, board, 'B', time_left, opponent_time_left))

test()
