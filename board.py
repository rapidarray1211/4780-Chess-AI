import chess

def initialize_board(): 
    return chess.Board()

def make_move(board, move):
    if move in board.legal_moves:
        board.push(move)
        return True
    return False

def undo_move(board):
    if board.move_stack:
        board.pop()

def print_board(board):
    return