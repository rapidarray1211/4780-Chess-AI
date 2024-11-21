import chess
import chess.pgn
import chess.engine
from typing import List, Tuple

STOCKFISH_PATH = "stockfish-windows-x86-64-avx2.exe"
stockfish_depth = 15
level = 10
engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)

def evaluate_move_stockfish(board: chess.Board):
    engine.configure({"Skill level": level})
    print(f"Evaluating games at Stockfish level {level}...\n")
    evaluation = engine.analyse(board, chess.engine.Limit(depth=10))
    score = evaluation["score"].relative

    if score.is_mate():
        print(f"Stockfish Level {level}: Mate in {score.mate()}")
    else:
        print(f"Stockfish Level {level}: Evaluation {score.score() / 100:.2f} (centipawns)\n")
    
def close_engine():
    engine.quit()
