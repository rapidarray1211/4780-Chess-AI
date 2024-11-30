import chess.pgn
import chess
import chess.pgn
import pandas as pd
import io

def load_pgn_file(filepath, numGames, minELO=0):
    games = []
    with open(filepath, buffering=1048576) as pgn_file:
        while numGames > 0:
            game = chess.pgn.read_game(pgn_file)
            if game is None:
                break
            elo1 = game.headers.get("WhiteElo")
            elo2 = game.headers.get("BlackElo")
            type = game.headers.get("Event")
            result = game.headers.get("Result")

            try:
                elo1 = int(elo1)
                elo2 = int(elo2)
                type = str(type)

            except ValueError:
                continue

            try:
                result = str(result)
            except ValueError:
                result = ""
            
            if minELO <= 0 or (elo1 >= minELO and elo2 >= minELO) and type.find("Classical"):
                games.append((game, result))
                numGames -= 1
                if numGames % 100 == 0:
                    print(f"[PARSE] 100 Games Parsed, {numGames} left")

    return games

def read_pgns_from_csv(filename: str, chunksize: int, numgames: int, skipAmount: int):
    cols_to_keep = ["pgn"]
    df_iter = pd.read_csv(filename, chunksize=chunksize, usecols=cols_to_keep)
    games = []
    for df_chunk in df_iter:
        if numgames == 0:
            break

        if skipAmount > 0:
            skipAmount -= chunksize
            continue
        
        numgames -= chunksize
        
        for game in df_chunk['pgn']:
            pgn_io = io.StringIO(game)
            game = chess.pgn.read_game(pgn_io)
            result = game.headers.get("Result")
            try:
                result = str(result)
            except ValueError:
                result = ""
            games.append((game, result))

        print(f"[PARSE] {chunksize} Games Parsed, {numgames} left")
    return games


def get_move_sequence(game):
    return [move.uci() for move in game.mainline_moves()]


