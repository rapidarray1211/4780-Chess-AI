import matplotlib.pyplot as plt
from pgn_parser import load_pgn_file, get_move_sequence
from board import initialize_board, make_move
from stockfish import evaluate_move_stockfish, close_engine, evaluate_game_stockfish
from engine.chessEngine import ChessEngine
import numpy as np

engine = ChessEngine("models/movepredictorV2_24.keras", 1, 3)

def play_game_pgn(pgn_file):
    # Load games from the PGN file
    print(f"Loading games from {pgn_file}.")
    games = load_pgn_file(pgn_file, 100)
    if games:
        print(f"Loaded {len(games)} games from {pgn_file}.")
        
        move_sequence = get_move_sequence(games[5])
        print("Move sequence (UCI):", move_sequence)
        
        board = initialize_board()
        for move in move_sequence:
            print(move)
            make_move(board, move)
        evaluate_game_stockfish(move_sequence)
        close_engine()

def play_ai_vs_ai(num_games=100):
    all_accuracy_scores = []

    for game_index in range(num_games):
        print(f"Playing game {game_index + 1}/{num_games}")
        board = initialize_board()
        move_sequence = []
        while not board.is_game_over():
            move = engine.predict_best_move(board, True)
            move_sequence.append(move.uci())
            make_move(board, move.uci())
        
        accuracy_scores = evaluate_game_stockfish(move_sequence)
        all_accuracy_scores.append(accuracy_scores)
    
    return all_accuracy_scores

def visualize_accuracy_scores(all_accuracy_scores):
    # Find the maximum number of moves in any game
    max_moves = max(len(scores) for scores in all_accuracy_scores)
    
    # Initialize a list to hold the sum of centipawn losses for each move number
    summed_scores = np.zeros(max_moves)
    move_counts = np.zeros(max_moves)
    
    # Sum the centipawn losses for each move number across all games
    for game_scores in all_accuracy_scores:
        for i, score in enumerate(game_scores):
            if score <= 8000:  # Filter out scores greater than 8000
                capped_score = min(score, 1000)  # Cap the score at 1000
                summed_scores[i] += capped_score
                move_counts[i] += 1
    
    # Calculate the average centipawn loss for each move number
    average_scores = summed_scores / move_counts
    
    # Plot the average centipawn loss per move
    plt.plot(average_scores)
    plt.xlabel('Move Number')
    plt.ylabel('Average Centipawn Loss')
    plt.title('Average Centipawn Loss per Move (Capped at 1000)')
    plt.show()

if __name__ == "__main__":
    num_games = 10
    all_accuracy_scores = play_ai_vs_ai(num_games)
    visualize_accuracy_scores(all_accuracy_scores)
    close_engine()
