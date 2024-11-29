import pygame
import chess
import math
from stockfish import generate_move
import random
from engine.chessEngine import ChessEngine

SQUARE_SIZE = 100

SCREENX = 800
SCREENY = 800

#https://blog.devgenius.io/simple-interactive-chess-gui-in-python-c6d6569f7b6c

engine = ChessEngine("models/movepredictorV2_24.keras", 1, 3)

def highlight_king_square(scrn, outcome, BOARD):
    king_square = None
    if outcome.winner == chess.WHITE:
        king_square = BOARD.king(chess.BLACK)
    else:
        king_square = BOARD.king(chess.WHITE)

    if king_square is not None:
        row, col = divmod(king_square, 8)
        flipped_row = 7 - row 
        square_x = col * 100
        square_y = flipped_row * 100
        pygame.draw.rect(scrn, (255, 215, 0), pygame.Rect(square_x, square_y, 100, 100))
        
def display_game_over(scrn, outcome, BOARD):
    font = pygame.font.Font(None, 74)
    
    if outcome.winner is not None:
        winner = "White" if outcome.winner == chess.WHITE else "Black"
        text = font.render(f"Checkmate! {winner} wins!", True, (255, 0, 0)) 
    else:
        text = font.render("Game Over - Draw!", True, (255, 0, 0))
        
    
    scrn.blit(text, (150, 300)) 
    pygame.display.flip() 
    
def promotion_choice():
    # Display choices for promotion (e.g., through Pygame or console)
    print("Select a piece to promote your pawn to:")
    print("1. Queen")
    print("2. Rook")
    print("3. Bishop")
    print("4. Knight")
    
    choice = input("Enter your choice (1-4): ")

    if choice == "1":
        return chess.QUEEN
    elif choice == "2":
        return chess.ROOK
    elif choice == "3":
        return chess.BISHOP
    elif choice == "4":
        return chess.KNIGHT
    else:
        print("Invalid choice, defaulting to Queen.")
        return chess.QUEEN

def draw_piece(scrn, piece_image, row, col, square_size, piece_scale=0.8):
    scaled_size = (int(square_size * piece_scale), int(square_size * piece_scale))
    piece_image = pygame.transform.scale(piece_image, scaled_size)
    
    square_x = col * square_size
    square_y = row * square_size

    piece_x = square_x + (square_size - scaled_size[0]) // 2
    piece_y = square_y + (square_size - scaled_size[1]) // 2

    scrn.blit(piece_image, (piece_x, piece_y))
    
def get_legal_moves(board, clicked_square):
    all_moves = list(board.legal_moves)

    piece_moves = [move.to_square for move in all_moves if move.from_square == clicked_square]

    return piece_moves
    
def highlight_moves(scrn, board, moves, selected_square):
    transparency = 128

    overlay = pygame.Surface((SQUARE_SIZE, SQUARE_SIZE), pygame.SRCALPHA)
    overlay.fill((*HIGHLIGHT_COLOUR, transparency)) 

    if selected_square is not None:
        row, col = divmod(selected_square, 8)
        flipped_row = 7 - row
        scrn.blit(overlay, (col * SQUARE_SIZE, flipped_row * SQUARE_SIZE))


    circle_overlay = pygame.Surface((SQUARE_SIZE, SQUARE_SIZE), pygame.SRCALPHA)
    
    circle_overlay.fill((0, 0, 0, 0))
    
    for move in moves:
        row, col = divmod(move, 8)
        
        circle_overlay.fill((0, 0, 0, 0))
        
        center_x = SQUARE_SIZE // 2
        center_y = SQUARE_SIZE // 2

        pygame.draw.circle(
            circle_overlay,
            (*GREY, transparency), 
            (center_x, center_y),  
            SQUARE_SIZE // 6    
        )
        
        flipped_row = 7 - row
          
        scrn.blit(circle_overlay, (col * SQUARE_SIZE, flipped_row * SQUARE_SIZE))

def draw_board(scrn):

   
    COLUMN_LABEL_ALIGNX = SQUARE_SIZE * 0.05
    COLUMN_LABEL_ALIGNY = SQUARE_SIZE * 0.05
    
    ROW_LABEL_ALIGNX = -SQUARE_SIZE * 0.15
    ROW_LABEL_ALIGNY = SCREENY - 0.3 * SQUARE_SIZE
    
    for i in range(8):
        for j in range(8):
            rect = pygame.Rect(j * SQUARE_SIZE, i * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE)
            if (i + j) % 2 == 0:
                pygame.draw.rect(scrn, LIGHT_SQUARE_COLOUR, rect)
            else:
                pygame.draw.rect(scrn, DARK_SQUARE_COLOUR, rect)

            if j == 0:
                row_number = 8 - i 
                row_text = font.render(
                    str(row_number),
                    True,
                    DARK_SQUARE_COLOUR if (i + j) % 2 == 0 else LIGHT_SQUARE_COLOUR
                )
                scrn.blit(row_text, (COLUMN_LABEL_ALIGNX, i * SQUARE_SIZE + COLUMN_LABEL_ALIGNY)) 

            if i == 7:
                col_letter = chr(97 + j) 
                col_text = font.render(
                    col_letter,
                    True,
                    DARK_SQUARE_COLOUR if (i + j) % 2 == 0 else LIGHT_SQUARE_COLOUR
                )
                scrn.blit(col_text, ((j + 1) * SQUARE_SIZE + ROW_LABEL_ALIGNX, ROW_LABEL_ALIGNY))


def draw_pieces(scrn, board):
    for i in range(64):
        piece = board.piece_at(i)
        if piece is None:
            continue

        row = i // 8 
        col = i % 8  
        flipped_row = 7 - row 

        draw_piece(scrn, pieces[str(piece)], flipped_row, col, 100, piece_scale=0.9)


def main(BOARD):

    scrn.fill(BLACK)
    pygame.display.set_caption('Chess')
    
    selected_piece = None
    selected_square = None
    moves = []
    
    human_player = chess.WHITE if random.choice([True, False]) else chess.BLACK
    ai_player = chess.BLACK if human_player == chess.WHITE else chess.WHITE

    Running = True
    while Running:
        outcome = BOARD.outcome()

        draw_board(scrn)
        if outcome is not None:
            highlight_king_square(scrn, outcome, BOARD)
        draw_pieces(scrn, BOARD)
        highlight_moves(scrn, BOARD, moves, selected_square)
        pygame.display.flip()
        
        if outcome is not None:
            display_game_over(scrn, outcome, BOARD)
            pygame.time.wait(3000) 
            break
        
        if BOARD.turn == ai_player:
            ai_move = engine.predict_best_move(BOARD, True)
            BOARD.push(ai_move)
            continue
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                Running = False

            if event.type == pygame.MOUSEBUTTONDOWN:
                pos = pygame.mouse.get_pos()

                square = (math.floor(pos[0] / 100), math.floor(pos[1] / 100))
                clicked_square = (7 - square[1]) * 8 + square[0]
                
                if selected_square is None:
                    piece = BOARD.piece_at(clicked_square)
                    if piece and piece.color == human_player: 
                        selected_piece = piece
                        selected_square = clicked_square
                        moves = get_legal_moves(BOARD, selected_square)
                else:
                    if clicked_square in moves:
                        move = chess.Move(selected_square, clicked_square)
                        
                        if BOARD.piece_at(selected_square).piece_type == chess.PAWN:
                            if (BOARD.turn == chess.WHITE and clicked_square // 8 == 7) or \
                               (BOARD.turn == chess.BLACK and clicked_square // 8 == 0):
                                promotion_piece = chess.QUEEN 
                                promotion_piece = promotion_choice()
                                
                                move.promotion = promotion_piece 
    
                        if move in BOARD.legal_moves: 
                            BOARD.push(move)
                        
                        selected_square = None
                        selected_piece = None
                        moves = []
                    else:
                        clicked_square = None
                        selected_square = None
                        selected_piece = None
                        moves = []
                        
    pygame.quit()

scrn = pygame.display.set_mode((SCREENX, SCREENY))
pygame.init()

WHITE = (255, 255, 255)
GREY = (128, 128, 128)
YELLOW = (204, 204, 0)
BLUE = (50, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 128, 0)
LIGHT_SQUARE_COLOUR = (222, 184, 135)
DARK_SQUARE_COLOUR = (139, 69, 19)
HIGHLIGHT_COLOUR = (200, 200, 180) 

font = pygame.font.SysFont(None, 36)

board = chess.Board()

pieces = {'p': pygame.image.load('data/assets/black-pawn.png').convert_alpha(),
          'n': pygame.image.load('data/assets/black-knight.png').convert_alpha(),
          'b': pygame.image.load('data/assets/black-bishop.png').convert_alpha(),
          'r': pygame.image.load('data/assets/black-rook.png').convert_alpha(),
          'q': pygame.image.load('data/assets/black-queen.png').convert_alpha(),
          'k': pygame.image.load('data/assets/black-king.png').convert_alpha(),
          'P': pygame.image.load('data/assets/white-pawn.png').convert_alpha(),
          'N': pygame.image.load('data/assets/white-knight.png').convert_alpha(),
          'B': pygame.image.load('data/assets/white-bishop.png').convert_alpha(),
          'R': pygame.image.load('data/assets/white-rook.png').convert_alpha(),
          'Q': pygame.image.load('data/assets/white-queen.png').convert_alpha(),
          'K': pygame.image.load('data/assets/white-king.png').convert_alpha(),
          
          }
          
main(board)
