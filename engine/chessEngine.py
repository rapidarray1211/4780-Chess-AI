from keras import models
from keras._tf_keras.keras.utils import to_categorical
from typing import Tuple
from engine.PeSTO import *
import concurrent.futures
import time
from neural_net import *
import chess.polyglot
from typing import Dict
from threading import Lock

#############################################
# MOVE ENGINE
#############################################



class ChessEngine:
    model: models.Sequential
    depth: int
    numMoveChecks: int
    tp_finds: int

    def __init__(self, modelPath: str, depth: int, moves: int):
        self.model = models.load_model(modelPath, safe_mode=False)
        self.depth = depth
        self.numMoveChecks = moves
        self.tp_finds = 0

    def get_legal_move_probabilities(self, board: chess.Board):

        encoded_board = encode_board(board)
        encoded_board = np.expand_dims(encoded_board, axis=-1)
        encoded_board = np.expand_dims(encoded_board, axis=0)

        move_probabilities = self.model.predict(encoded_board, verbose = 1)[0]
        legal_moves_probabilities = {
            move: move_probabilities[encode_move(move)] for move in board.legal_moves
        }

        sorted_moves = sorted(
            legal_moves_probabilities.items(),
            key=lambda item: item[1],
            reverse=True
        )
        
        return sorted_moves[:self.numMoveChecks]
    
    def evaluate_move(self, current_score: int, move: chess.Move, board: chess.Board, probability: float):
        board.push(move)
        alpha = float('-inf')
        beta = float('inf')
        value = 0

        if board.turn == chess.WHITE:
            value, _ = self.alphaBetaMax(board, alpha, beta, self.depth)
        else:
            value, _ = self.alphaBetaMin(board, alpha, beta, self.depth)

        board.pop()
        change = abs(current_score) - abs(value)
        return change, value, move, probability

    def predict_best_move(self, board: chess.Board, networkOnly: bool):
        start_time = time.time()

        sorted_moves = self.get_legal_move_probabilities(board)
        if(networkOnly): # Skips Static Analysis
            return sorted_moves[0][0] if len(sorted_moves) > 0 else None

        best_value = 0
        current_score = self.eval(board)
        best_move = sorted_moves[0][0]
        turn = "White" if board.turn == chess.WHITE else "Black"

        val_diffs = []
        vals = []
        moves = []
        probs = []

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = []
            for move, probability in sorted_moves:
                futures.append(executor.submit(self.evaluate_move, current_score, move, board.copy(), probability))

            for future in concurrent.futures.as_completed(futures):
                vdif, value, gmove, prob = future.result()
                val_diffs.append(vdif)
                vals.append(value)
                moves.append(gmove)
                probs.append(prob)

        val_diffs_norm = (val_diffs - np.min(val_diffs)) / (np.max(val_diffs) - np.min(val_diffs))

        for vdif, value, gmove, prob, static_change in zip(val_diffs_norm, vals, moves, probs, val_diffs):
            wvalue = (vdif * 0.4) + (prob * 0.6)
            if(wvalue > best_value):
                best_move = gmove
                best_value = wvalue
            print(f"[{turn}] Trying Move: {gmove} ModelProb[{prob}] StaticVal[{value}] WeightedVal[{wvalue}] CurrStaticVal[{current_score}] StaticChange[{static_change}]")

        elapsed_time = time.time() - start_time
        print(f"Move Calculation Time: {elapsed_time:.4f}s Transposition Finds: {self.tp_finds}")
        self.tp_finds = 0
        return best_move


    def eval(self, board: chess.Board) -> int:

        if board.is_checkmate():
            return -99999 if board.turn else 99999
        if board.is_stalemate() or board.is_insufficient_material():
            return 0

        eval_score = static_eval(board, WHITE if board.turn else BLACK)
        
        return eval_score
    

    # Credits: https://www.chessprogramming.org/Alpha-Beta

    def alphaBetaMax(self, board: chess.Board, alpha: int, beta: int, depth: int) -> Tuple[int, chess.Move]:
        if depth == 0 or board.is_game_over():
            return self.eval(board), None
        
        best_move = None
        bestValue = float('-inf')
        moves = list(board.legal_moves)
        ordered_moves = []
        other = []

        for move in moves:
            ordered_moves.append(move) if board.is_capture(move) else other.append(move)
        ordered_moves.extend(other)
        
        for move in ordered_moves:
            board.push(move)
            score, _ = self.alphaBetaMin(board, alpha, beta, depth - 1)
            board.pop()
            
            if score > bestValue:
                bestValue = score
                best_move = move
                alpha = max(alpha, bestValue)
                
            if beta <= alpha:
                break
                
        return bestValue, best_move

    def alphaBetaMin(self, board: chess.Board, alpha: int, beta: int, depth: int) -> Tuple[int, chess.Move]:
        if depth == 0 or board.is_game_over():
            return self.eval(board), None
        
        best_move = None
        bestValue = float('inf')
        moves = list(board.legal_moves)
        ordered_moves = []
        other = []
        for move in moves:
            ordered_moves.append(move) if board.is_capture(move) else other.append(move)
        ordered_moves.extend(other)
        
        for move in ordered_moves:
            board.push(move)
            score, _ = self.alphaBetaMax(board, alpha, beta, depth - 1)
            board.pop()
            
            if score < bestValue:
                bestValue = score
                best_move = move
                beta = min(beta, bestValue)
                
            if beta <= alpha:
                break
                
        return bestValue, best_move
    
