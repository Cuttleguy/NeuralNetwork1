import numpy as np
import string


class Board:
    def __init__(self, fen: str):
        self.fen = fen
        self.boardArray = [['']]
        self.Kcastle = True
        self.kcastle = True
        self.Qcastle = True
        self.qcastle = True
        self.player = True  # True for black
        self.genBoard()

    def genBoard(self):
        currentPieces = []
        boardInfo = self.fen.split(" ")
        for char in boardInfo[0]:
            if char.isnumeric():
                for i in range(int(char)):
                    currentPieces.append("-")
            elif char == "/":
                self.boardArray.append(currentPieces)
                currentPieces = []
            else:
                currentPieces.append(char)
        self.player = boardInfo[1] == "b"

    def validate(self, move: list[str], castle, enpassant):
        if (move[0] == move[1]):
            return False
        square1 = [string.ascii_lowercase.index(move[0][0]), int(move[0][1]) - 1]
        square2 = [string.ascii_lowercase.index(move[1][0]), int(move[1][1]) - 1]
        piece1 = self.boardArray[square1[0]][square1[1]]
        piece2 = self.boardArray[square2[0]][square2[1]]
        trajectory = np.subtract(square1, square2)
        piece1White = piece1.upper() == piece1
        piece2White = piece2.upper() == piece2
        piece1 = piece1.lower()
        piece2 = piece2.lower()
        trajectoryWorks = True
        if piece1 == "-":
            return False
        elif piece1 == "r":
            trajectoryWorks = trajectory[0] == 0 ^ trajectory[1] == 0
        elif piece1 == "b":
            trajectoryWorks = trajectory[0] == trajectory[1]
        elif piece1 == "q":
            trajectoryWorks = trajectory[0] == 0 ^ trajectory[1] == 0 or trajectory[0] == trajectory[1]
        elif piece1 == "n":
            validTrajs = [[1, 2], [-1, 2], [1, -2], [-1, -2], [2, 1], [2, -1], [-2, 1], [-2, -1]]
            trajectoryWorks = trajectory in validTrajs
        elif piece1 == "k":
            trajectoryWorks = trajectory[0] in [0, 1, -1] and trajectory[1] in [0, 1, -1]
        elif piece1 == "p":
            cToDir = {True: -1, False: 1}
            if trajectory == [1, 0]:
                trajectoryWorks = True
            elif piece2 != "-" and trajectory in [[1, 1], [1, -1]]:
                trajectoryWorks = True
            elif trajectory == [cToDir[self.player], 1] or trajectory == [cToDir[self.player], -1]:
                enpassant = True


board = Board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
enpassant=False
castle=False
board.validate(["b1", "f4"],enpassant=enpassant,castle=castle)
