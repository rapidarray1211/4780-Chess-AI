from pgn_parser import load_pgn_file, get_move_sequence
from board import initialize_board, make_move
from stockfish import evaluate_move_stockfish, close_engine

def play_game_pgn(pgn_file):
    # Load games from the PGN file
    print(f"Loading games from {pgn_file}.")
    games = load_pgn_file(pgn_file, 100)
    if games:
        print(f"Loaded {len(games)} games from {pgn_file}.")
        
        move_sequence = get_move_sequence(games[0])
        print("Move sequence (UCI):", move_sequence)
        
        board = initialize_board()
        for move in move_sequence:
            print(move)
            make_move(board, move)
            evaluate_move_stockfish(board)
        close_engine()
    

if __name__ == "__main__":
    play_game_pgn("data/lichess_dataset_1.pgn")
