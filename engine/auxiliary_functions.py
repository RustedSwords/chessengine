import numpy as np
from chess import Board

def board_to_matrix(board: Board):
    # 8x8 grid representing the chessboard layout.
    # 12 channels for each distinct chess piece type.
    # 13th channel for indicating legal moves (possible destination squares).
    # Optionally, a 14th channel to highlight squares where moves can originate.
    matrix = np.zeros((13, 8, 8))
    piece_map = board.piece_map()

    # Fill the first 12 channels of the 8x8 board with the positions of each piece type.
    for square, piece in piece_map.items():
        row, col = divmod(square, 8)
        piece_type = piece.piece_type - 1
        piece_color = 0 if piece.color else 6
        matrix[piece_type + piece_color, row, col] = 1

    # Fill the 13th channel of the 8x8 board with legal move locations.
    legal_moves = board.legal_moves
    for move in legal_moves:
        to_square = move.to_square
        row_to, col_to = divmod(to_square, 8)
        matrix[12, row_to, col_to] = 1

    return matrix

def create_input_for_nn(games):
    x = []
    y = []
    for game in games:
        board = game.board()
        for move in game.mainline_moves():
            x.append(board_to_matrix(board))
            y.append(move.uci())
            board.push(move)
            return np.array(x, dtype = np.float32), np.array(y)
        
def encode_moves(moves):
    move_to_int = {move: idx for idx, move in enumerate(set(moves))}
    return np.array([move_to_int[move] for move in moves], dtype = np.float32), move_to_int