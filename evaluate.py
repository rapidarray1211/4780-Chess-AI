import chess
import chess.engine
from engine.chessEngine import ChessEngine

#This program evaluates our chess engine by playing against stockfish.
#it plays against stockfish at 5 skill levels (1,5,10,15,20).

engine_name = "models/movepredictorV2_34.keras"
fname = "data/model34.txt"

def playGame(stock,engine):
    player = 0
    n_moves = 0
    board = chess.Board()
    while not board.is_game_over():
        if player == 1:
            result = stock.play(board, chess.engine.Limit(time=0.01))
            board.push(result.move)
            player = 0
            n_moves += 1
        else :
            result = engine.predict_best_move(board,True)
            board.push(result)
            player = 1
    
    return (n_moves,player)
    



stock1 = chess.engine.SimpleEngine.popen_uci("data/stockfish/stockfish-windows-x86-64-avx2.exe")
stock1.configure({"Skill Level": 1})
stock2 = chess.engine.SimpleEngine.popen_uci("data/stockfish/stockfish-windows-x86-64-avx2.exe")
stock2.configure({"Skill Level": 5})
stock3 = chess.engine.SimpleEngine.popen_uci("data/stockfish/stockfish-windows-x86-64-avx2.exe")
stock3.configure({"Skill Level": 10})
stock4 = chess.engine.SimpleEngine.popen_uci("data/stockfish/stockfish-windows-x86-64-avx2.exe")
stock4.configure({"Skill Level": 15})
stock5 = chess.engine.SimpleEngine.popen_uci("data/stockfish/stockfish-windows-x86-64-avx2.exe")
stock5.configure({"Skill Level": 20})

engine = ChessEngine(engine_name, 1, 3)
max_iter = 50
iter = 0
n_wins = [0,0,0,0,0]

f = open(fname,"w")
f.write("iter   Level 01   Level 05   Level 10   Level 15   Level 20\n")

while (iter < max_iter):
    n_moves = [0,0,0,0,0]
    
    res = playGame(stock1,engine)
    n_moves[0] = res[0]
    if (res[1] == 1):
        n_wins[0] += 1

    res = playGame(stock2,engine)
    n_moves[1] = res[0]
    if (res[1] == 1):
        n_wins[1] += 1

    res = playGame(stock3,engine)
    n_moves[2] = res[0]
    if (res[1] == 1):
        n_wins[2] += 1

    res = playGame(stock4,engine)
    n_moves[3] = res[0]
    if (res[1] == 1):
        n_wins[3] += 1

    res = playGame(stock5,engine)
    n_moves[4] = res[0]
    if (res[1] == 1):
        n_wins[4] += 1
    
    iter += 1
    
    txt = "{0:3d}:   {1:8d}   {2:8d}   {3:8d}   {4:8d}   {5:8d}\n"
    f.write(txt.format(iter,n_moves[0],n_moves[1],n_moves[2],n_moves[3],n_moves[4]))

txt = "win:   {0:8d}   {1:8d}   {2:8d}   {3:8d}   {4:8d}\n"
f.write(txt.format(n_wins[0],n_wins[1],n_wins[2],n_wins[3],n_wins[4]))
f.close()

stock1.quit()
stock2.quit()
stock3.quit()
stock4.quit()
stock5.quit()
