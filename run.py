import chess
import torch
from torch import nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(777, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 64)
        self.fc5 = nn.Linear(64, 64)
        self.fc6 = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = self.fc6(x)
        return x


def makeBoard(boardData):
    KBoard = []
    QBoard = []
    NBoard = []
    Rboard = []
    Bboard = []
    Pboard = []
    kBoard = []
    qBoard = []
    nBoard = []
    rboard = []
    bboard = []
    pboard = []
    turnBoards = [KBoard, QBoard, NBoard, Rboard, Bboard, Pboard, kBoard, qBoard, nBoard, rboard, bboard, pboard]
    charToBoard = {"K": KBoard, "Q": QBoard, "N": NBoard, "R": Rboard, "B": Bboard, "P": Pboard, "k": kBoard,
                   "q": qBoard, "n": nBoard, "r": rboard, "b": bboard, "p": pboard}
    boardData = boardData.replace("/", "")
    boardInfo = boardData.split(" ")
    posData = boardInfo[0]
    for char in posData:

        if char in ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]:
            spaces = int(char)
            for i in range(spaces):
                for board in turnBoards:
                    board.append(0)
        elif char in charToBoard.keys():
            boardToAdd = charToBoard[char]
            boardToAdd.append(1)
            for board in turnBoards:
                if board != boardToAdd:
                    board.append(0)
    boardBits = []
    for board in turnBoards:
        boardBits.extend(board)
        # print(len(board))
    if boardInfo[1] == "b":
        boardBits.append(0)
    if boardInfo[1] == "w":
        boardBits.append(1)
    charToInt = {"a": 1, "b": 2, "c": 3, "d": 4, "e": 5, "f": 6, "g": 7, "h": 8}
    castlingBits = [0, 0, 0, 0]
    if "K" in boardInfo[2]:
        castlingBits[0] = 1
    if "Q" in boardInfo[2]:
        castlingBits[1] = 1
    if ("k" in boardInfo[2]):
        castlingBits[2] = 1
    if ("q" in boardInfo[2]):
        castlingBits[3] = 1
    boardBits.extend(castlingBits)
    if (boardInfo[3] == "-"):
        boardBits.append(0)
        boardBits.append(0)
    else:
        boardBits.append(charToInt[boardInfo[3][0]])
        boardBits.append(int(boardInfo[3][1]))

    boardBits.append(int(boardInfo[4]))
    boardBits.append(int(boardInfo[5]))
    # print(len(boardBits))
    return boardBits


if torch.cuda.is_available():

    device = torch.device("cuda:0")
    print("CUDA")
else:
    device = torch.device("cpu")
board = chess.Board()
net = Net().to(device)
net.load_state_dict(torch.load("cnn1.pth", device))
print(net)


def getInput(board: chess.Board):
    try:
        moveToMake = input("Your Move: ")
        move = chess.Move.from_uci(moveToMake)
        if not board.is_legal(move):
            print("Not Legal")
            return getInput(board)
        else:
            board.push(move)
    except chess.InvalidMoveError as e:
        print(e)
        getInput(board)


def forsee(board: chess.Board, oldWeight: float):
    totalEval = 0
    weight = oldWeight / 1.5
    if weight < 1 / 1000:
        return 0
    moves = []
    evals = []
    for move in board.legal_moves:
        copy = board.copy()
        copy.push(move)
        cureval = net(torch.Tensor(makeBoard(copy.fen())).to(device))
        evals.append(cureval)
        moves.append(move)
        totalEval += cureval * weight

    minPos = evals.index(min(evals))
    moveb = moves[minPos]
    copy=board.copy()
    copy.push(moveb)
    return totalEval+forsee(board,weight)

while not board.is_checkmate() or board.is_stalemate():
    if board.turn:
        getInput(board)
    else:
        moves = []
        evals = []
        for move in board.legal_moves:
            moves.append(move)
            copy = board.copy()

            copy.push(move)
            if(copy.is_checkmate()):
                board.push(move)
                continue

            boardInfo = torch.Tensor(makeBoard(copy.fen())).to(device)
            net_out = net(boardInfo)
            net_out += forsee(board, 1)
            evals.append(net_out.item())
        minPos = evals.index(min(evals))
        moveb = moves[minPos]
        board.push(moveb)
        print(moveb.uci())
