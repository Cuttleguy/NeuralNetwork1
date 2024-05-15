import csv
from tqdm import tqdm
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.optim as optim
import statistics
import math

if torch.cuda.is_available():

    device = torch.device("cuda:0")
    print("CUDA")
else:
    device = torch.device("cpu")


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
    if ("K" in boardInfo[2]):
        castlingBits[0] = 1
    if ("Q" in boardInfo[2]):
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


REBUILD_DATA = False


def convertToPair(chessData):
    boardData = chessData[0]
    print("Converting")
    return [makeBoard(boardData), chessData[1]]


if (REBUILD_DATA):
    # print(makeBoard("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w - - 0 0"))
    data = []
    evals = []
    with open("chessData.csv", "r") as f:
        csvFile = csv.reader(f)
        lineInt = 0
        for line in tqdm(csvFile):
            if (lineInt > 0):
                data.append(line[0])
                evals.append(int(line[1].replace("#", "")))
            lineInt += 1
            # print(line)

    np.save("chess_data.npy", data)
    np.save("eval_data.npy", evals)
X = np.load("chess_data.npy", allow_pickle=True)

y = torch.from_numpy(np.load("eval_data.npy"))
VAL_PCT = 0.1
val_size = int(len(X) * VAL_PCT)
print(val_size)
train_X = X[:-val_size]
train_y = y[:-val_size]
test_X = X[-val_size:]
test_y = y[-val_size:]
net = Net().to(device)
print(net)
optimizer = optim.Adam(net.parameters(), lr=0.001)
loss_function = nn.MSELoss()


def train(net):
    BATCH_SIZE = 100
    EPOCHS = 3
    for epoch in range(EPOCHS):

        for i in tqdm(range(0, len(train_X), BATCH_SIZE)):
            # print(i,i+BATCH_SIZE)
            batch_Xraw = train_X[i:i + BATCH_SIZE]
            batch_Xlist = list(map(makeBoard, batch_Xraw))
            # for batch in batch_Xraw:
            #     batch_Xlist.append(makeBoard(batch))
            # for batch in batch_Xlist:
            #     for bit in batch:
            #         print(type(bit))
            batch_X = torch.Tensor(batch_Xlist).to(device)
            batch_y = train_y[i:i + BATCH_SIZE].to(device).to(torch.float32)
            optimizer.zero_grad()
            outputs = net(batch_X)
            loss = loss_function(outputs, batch_y.view([len(batch_X), 1]))
            loss.backward()

            optimizer.step()
        print(epoch)
        print(loss)
        torch.save(net.state_dict(), "cnn1.pth")


#
#
print(len(test_X))


def test(net):
    correct = 0
    total = 0
    differences=[]
    with torch.no_grad():
        for i in tqdm(range(len(test_X))):
            real_class = test_y[i]
            # print(test_X[i])
            batch = torch.Tensor(makeBoard(test_X[i])).to(device)

            net_out = net(batch)
            # predicted_class=torch.argmax(net_out)

            # Test Acurracy 1
            if abs(net_out - real_class).item() <= 50:
                correct += 1
            # else:
            #
            #     print(abs(net_out - real_class).item())
            differences.append(abs(net_out-real_class.item()))
            total += 1
    print(round(correct / total, 3))
    print(statistics.fmean(differences))
old_dict = net.state_dict()
net.load_state_dict(torch.load("cnn1.pth"))
# train(net)


test(net)
