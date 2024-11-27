import chess
import chess.pgn
import chess.engine
from typing import List, Tuple

STOCKFISH_PATH = "data/stockfish/stockfish-windows-x86-64-avx2.exe"
stockfish_depth = 15
level = 10
engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)

def evaluate_move_stockfish(board: chess.Board):
    engine.configure({"Skill level": level})
    print(f"Evaluating games at Stockfish level {level}...\n")
    evaluation = engine.analyse(board, chess.engine.Limit(depth=10))
    score = evaluation["score"].relative

    move_recommend = generate_move(board)

    if score.is_mate():
        print(f"Stockfish Level {level}: Mate in {score.mate()}")
        print(f"Recommended move: {move_recommend}")
    else:
        print(f"Stockfish Level {level}: Evaluation {score.score() / 100:.2f} (centipawns)\n")
        print(f"Recommended move: {move_recommend}")

    
def evaluate_game_stockfish(move_sequence):
    # Rather than evaluate a single move, this attempts to evaluate a move sequence
    board = chess.Board()
    accuracy_scores = []
    
    with chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH) as engine:
        for move_str in move_sequence:
            move = chess.Move.from_uci(move_str)
            if move in board.legal_moves:
                board.push(move) 
                
                evaluation = engine.analyse(board, chess.engine.Limit(depth=stockfish_depth))
                best_move = engine.play(board, chess.engine.Limit(time=0.5)) 
                # If increase time, it increases accuracy but decreases speed (obviously)
                
                eval_score = evaluation['score'].relative.score(mate_score=10000)
                
                best_move_evaluation = engine.analyse(board, chess.engine.Limit(depth=stockfish_depth))
                best_move_score = best_move_evaluation['score'].relative.score(mate_score=10000)
                # We compare the centipawn loss between the best move, and the move given to get a delta. Average the delta for average centipawn loss
                
                print(f"Move: {move_str} -> Eval: {eval_score}, Best Move: {best_move_score}")
                
                accuracy_score = abs(eval_score - best_move_score)
                accuracy_scores.append(accuracy_score)
            else:
                print(f"Invalid move: {move_str}")
        
        if accuracy_scores:
            average_accuracy = sum(accuracy_scores) / len(accuracy_scores)
            print(f"Average centipawn loss: {average_accuracy:.2f}")
        else:
            print("No valid moves to evaluate.")

def generate_move(board: chess.Board): 
    result = engine.play(board, chess.engine.Limit(time=2.0))
    print(result.move)
    return result.move

def close_engine():
    engine.quit()
