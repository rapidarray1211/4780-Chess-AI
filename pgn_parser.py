import chess.pgn

def load_pgn_file(filepath, numGames):
    #Returns list of games
    games = []
    with open(filepath) as pgn_file:
        while numGames > 0:
            game = chess.pgn.read_game(pgn_file)
            if game is None:
                break
            games.append(game)
            numGames-=1
    return games

def get_move_sequence(game):
    return [move.uci() for move in game.mainline_moves()]