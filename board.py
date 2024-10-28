import chess

def initialize_board(): 
    return chess.Board()

def make_move(board, move):
    chess_move = chess.Move.from_uci(move)
    if chess_move in board.legal_moves:
        board.push(chess_move)
        return True
    return False

def undo_move(board):
    if board.move_stack:
        board.pop()

def print_board(board):
    return