import chess
import chess.pgn

board = chess.Board()

print(board.legal_moves)

print(board)

pgn = open("data/lichess_db_standard_rated_2014-07.pgn")

first_game = chess.pgn.read_game(pgn)

board = first_game.board()
for move in first_game.mainline_moves():
    board.push(move)
    print(board.legal_moves)
    print(board)

print(board)