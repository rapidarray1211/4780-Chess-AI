import chess.pgn

def load_pgn_file(filepath):
    #Returns list of games
    games = []
    with open(filepath) as pgn_file:
        while True:
            game = chess.pgn.read_game(pgn_file)
            if game is None:
                break
            games.append(game)
    return games

def get_move_sequence(game):
    return [move.uci() for move in game.mainline_moves()]