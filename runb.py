from cnn import Net,makeBoard
import torch
from torch import nn
from torch.nn import functional as F
import chess
if torch.cuda.is_available():
    device=torch.device("cuda:0")
else:
    device=torch.device("cpu")
net=Net().to(device)
net.load_state_dict(torch.load("PetImages/cnn1.pth", map_location=device))
board=chess.Board()

for move in board.legal_moves:
    copy=board.copy()
    copy.push(move)
    boardInfo=makeBoard(copy.fen())
    net(boardInfo)