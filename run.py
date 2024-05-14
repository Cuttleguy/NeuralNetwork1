from cnn import Net, makeBoard
import chess
import torch
if torch.cuda.is_available():

    device = torch.device("cuda:0")
    print("CUDA")
else:
    device = torch.device("cpu")
board=chess.Board()
net=Net().to(device)
net.load_state_dict(torch.load("cnn1.pth"))

def getInput(board: chess.Board):
    moveToMake = input("Your Move: ")
    move=chess.Move.from_uci(moveToMake)
    if not board.is_legal(move):
        print("Not Legal")
        return getInput(board)
    else:
        board.push(move)
while not board.is_checkmate():
    if board.turn:
        getInput(board)
    else:
        moves=[]
        evals=[]
        for move in board.legal_moves:
            moves.append(move)
            copy=board.copy()

            copy.push(move)
            boardInfo=makeBoard(copy).to(device)
            net_out=net(boardInfo)
            evals.append(net_out.item())




