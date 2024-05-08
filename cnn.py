import csv
from tqdm import tqdm

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
    boardData=boardData.replace("/", "")
    boardInfo = boardData.split(" ")
    posData=boardInfo[0]
    for char in posData:
        print(char)
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
    for board in turnBoards:
        print(len(board))
    return turnBoards

# print(makeBoard("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w - - 0 0"))
data=[]

with open("chessData.csv","r") as f:
    csvFile=csv.reader(f)
    for line in tqdm(csvFile):
        data.append(line)
        # print(line)
boards=[]



for chessData in data:
    boardData = chessData[0]
    boards.append(makeBoard(boardData))

