import signal, os, sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from keras import models, layers, regularizers, callbacks
from keras._tf_keras.keras.utils import to_categorical
from pgn_parser import *
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed


def get_board_and_moves(game):
    board = game.board()
    board_and_moves = []
    for move in game.mainline_moves():
        board_state = board.copy() 
        board.push(move)
        board_and_moves.append((board_state, move))
    return board_and_moves

def encode_board(board: chess.Board):
    encoded = np.zeros((8, 8, 17), dtype=np.float32)
    
    piece_to_channel = {
        (chess.PAWN, chess.WHITE): 0,
        (chess.KNIGHT, chess.WHITE): 1,
        (chess.BISHOP, chess.WHITE): 2,
        (chess.ROOK, chess.WHITE): 3,
        (chess.QUEEN, chess.WHITE): 4,
        (chess.KING, chess.WHITE): 5,
        (chess.PAWN, chess.BLACK): 6,
        (chess.KNIGHT, chess.BLACK): 7,
        (chess.BISHOP, chess.BLACK): 8,
        (chess.ROOK, chess.BLACK): 9,
        (chess.QUEEN, chess.BLACK): 10,
        (chess.KING, chess.BLACK): 11
    }
    WHITEATTACK = 12
    BLACKATTACK = 13
    PINNED = 14
    LEGALDEST = 15
    MOBILITY = 16
    

    mobility_map = np.zeros(64)

    for square in chess.SQUARES:
        row = 7 - (square // 8) 
        col = square % 8 
       
        piece = board.piece_at(square)
        if piece is not None:
            channel = piece_to_channel[(piece.piece_type, piece.color)]
            encoded[row, col, channel] = 1.0

        if board.is_attacked_by(chess.WHITE, square):
            encoded[row, col, WHITEATTACK] = 1.0
        if board.is_attacked_by(chess.BLACK, square):
            encoded[row, col, BLACKATTACK] = 1.0

        if board.is_pinned(board.turn, square):
            encoded[row, col, PINNED] = 1.0

        
        for move in board.legal_moves:
            mobility_map[move.from_square] += 1

            to_square = move.to_square
            r2 = 7 - (to_square // 8)
            c2 = to_square % 8
            encoded[r2, c2, LEGALDEST] = 1.0
    
    mobility_map = (np.array(mobility_map) - np.min(mobility_map)) / (np.max(mobility_map) - np.min(mobility_map))

    for i in range(len(mobility_map)):
        row = 7 - (i // 8)
        col = i % 8
        encoded[row, col, MOBILITY] = mobility_map[i]
            
    return encoded


def encode_move(move):
    start_square = move.from_square
    end_square = move.to_square
    return start_square * 64 + end_square

def process_single_game(game):
    game_data, result = game
    
    weighting_scheme = {
        "1-0": {chess.WHITE: 1.2, chess.BLACK: 0.8},  
        "0-1": {chess.WHITE: 0.8, chess.BLACK: 1.2},    
        "1/2-1/2": {chess.WHITE: 1.0, chess.BLACK: 1.0}  
    }
    
    board_and_moves = get_board_and_moves(game_data)
    total_moves = len(board_and_moves)
    base_weights = weighting_scheme.get(result, {chess.WHITE: 1.0, chess.BLACK: 1.0})
    
    X, y, weights = [], [], []
    
    for move_num, (board, move) in enumerate(board_and_moves, 1):
        move_factor = (
            (1.0 + (move_num / total_moves) * 0.5)  
            * base_weights[board.turn]      
        )
        
        X.append(encode_board(board))
        y.append(encode_move(move))
        weights.append(move_factor)
    
    return X, y, weights

def prepare_dataset(games, parallelization_factor: int = 1):
    print(f"[NEURAL] - Preparing Dataset with {len(games)} games")
    
    X, y, weights_all = [], [], []
    processed_games = 0

    with ProcessPoolExecutor(max_workers=parallelization_factor) as executor:
        future_to_game = {
            executor.submit(process_single_game, game): game 
            for game in games
        }
        
        for future in as_completed(future_to_game):
            try:
                X_sub, y_sub, game_weights = future.result()
                X.extend(X_sub)
                y.extend(y_sub)
                weights_all.extend(game_weights)
                
                processed_games += 1
                if processed_games % 100 == 0:
                    print(f"[NEURAL] - Processed {processed_games}/{len(games)} games")
            
            except Exception as exc:
                print(f'Game processing generated an exception: {exc}')
    
    print(f"[NEURAL] - Total games processed: {processed_games}")
    
    X = np.array(X)
    y = np.array(y)
    weights = np.array(weights_all)
    y = to_categorical(y, num_classes=64 * 64)
    
    return X, y, weights


def build_chess_cnn() -> models.Model:
    print("[NEURAL] - Creating Model")

    board_input = layers.Input(shape=(8, 8, 17), name='board_input')
    
    piece_channels = layers.Lambda(lambda x: x[..., :12], name='piece_channels')(board_input)
    info_channels = layers.Lambda(lambda x: x[..., 12:], name='info_channels')(board_input)

    piece_conv = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(piece_channels)
    piece_conv = layers.BatchNormalization()(piece_conv)
    residual = piece_conv 
    piece_conv = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(piece_conv)
    piece_conv = layers.BatchNormalization()(piece_conv)
    piece_conv = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(piece_conv)
    piece_conv = layers.BatchNormalization()(piece_conv)
    piece_conv = layers.Add()([piece_conv, residual]) 
    piece_conv = layers.Conv2D(128, (1, 1), activation='relu')(piece_conv)
    piece_conv = layers.BatchNormalization()(piece_conv)
    piece_conv = layers.SpatialDropout2D(0.1)(piece_conv)

    info_conv = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(info_channels)
    info_conv = layers.BatchNormalization()(info_conv)
    info_conv2 = layers.GlobalAveragePooling2D()(info_conv)
    info_conv2 = layers.Dense(8, activation='relu')(info_conv2)
    info_conv2 = layers.Dense(32, activation='relu')(info_conv2)
    info_conv2 = layers.Reshape((1, 1, 32))(info_conv2)
    info_conv = layers.Multiply()([info_conv, info_conv2])
    info_conv = layers.SpatialDropout2D(0.1)(info_conv)

    combined = layers.Concatenate()([piece_conv, info_conv])

    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(combined)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling2D()(x)
    
    dense1 = layers.Dense(512, activation='relu')(x)
    dense1 = layers.BatchNormalization()(dense1)
    dense1 = layers.Dropout(0.1)(dense1)
    dense2 = layers.Dense(512, activation='relu')(dense1)
    dense2 = layers.BatchNormalization()(dense2)
    dense2 = layers.Add()([dense1, dense2]) 
    
    outputs = layers.Dense(4096, activation='softmax')(dense2)
    model = models.Model(inputs=board_input, outputs=outputs, name='chess_cnn')
    
    return model


def train_model(model: models.Model, X, y, weights):
    print(f"[NEURAL] - Training Model")
    print("Shape of X:", X.shape)
    print("Shape of y:", y.shape)
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


    reduce_learning_rate = callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.2,
        patience=10,
        verbose=1,
        mode="auto",
        min_delta=0.0001,
        cooldown=2,
        min_lr=0.0001
    )

    model.fit(
        X, y,
        sample_weight=weights,
        batch_size=32,
        epochs=50,
        verbose=1,
        validation_split=0.2,
        callbacks=[reduce_learning_rate]
    )
    
    print(f"[NEURAL] - Saving Model")


def training(modelToLoad = ""):
    minELO = 2500
    numGames = 1000
    keepTraining = True
    iter = 0

    def kill_training(sig, frame):
        print("\n[NEURAL] - Training STOPPED")
        sys.exit(0)

    def stop_training(sig, frame):
        nonlocal keepTraining
        if keepTraining:
            keepTraining = False
            print("\n[NEURAL] - Training will stop after current iteration. Press Ctrl+C Again to Exit")
            signal.signal(signal.SIGINT, kill_training)
        else:
            print("\n[NEURAL] - Exiting...")
            sys.exit(0)

    signal.signal(signal.SIGINT, stop_training)
    model = None
    if modelToLoad == "":
        model = build_chess_cnn()
    else:
        model = models.load_model(modelToLoad, safe_mode=False)

    try:
        while keepTraining:
            iter += 1
            print(f"\n[NEURAL] - Training Iteration {iter} - {numGames*iter} total games processed")
            
            games = read_pgns_from_csv("./data/GM_games_dataset.csv", numGames/5, numGames, numGames*iter)
            X, y, weights = prepare_dataset(games, 10)
            
            train_model(model, X, y, weights)
            model.save(f"models/movepredictorV2_{iter}.keras")
    
    except KeyboardInterrupt:
        print("\n[NEURAL] - Training interrupted")
    finally:
        model.save(f"models/temp_model_{iter}.keras")


if __name__ == "__main__":
    training()
