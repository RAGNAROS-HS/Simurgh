import io
import chess.pgn
import chess
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def load_pgn(file_path):
    games = []
    with open(file_path, encoding="utf-8") as pgn_file:
        for i in range(1000):
            game = chess.pgn.read_game(pgn_file)
            if game is None:
                break

            games.append({
                "result":   game.headers.get("Result"),
                "white_elo": game.headers.get("WhiteElo"),
                "black_elo": game.headers.get("BlackElo"),
                "opening":  game.headers.get("Opening"),
                "num_moves": game.end().ply() // 2,
                "moves":    game.mainline_moves().__str__(),
            })
    return games


def fen_from_pgn_string(pgn_string, ply=20):
    game = chess.pgn.read_game(io.StringIO(pgn_string))
    if game is None:
        return None
    
    board = game.board()
    for move in game.mainline_moves():
        board.push(move)
        if board.ply() == ply:
            return board.fen()
    return None



if __name__ == "__main__":
    DATASET_PATH = r"E:\lichess_db_standard_rated_2025-05.pgn"
    df = pd.DataFrame(load_pgn(DATASET_PATH))
    df.to_csv("games.csv", index=False)

    df["white_elo"] = pd.to_numeric(df["white_elo"], errors="coerce")
    df["black_elo"] = pd.to_numeric(df["black_elo"], errors="coerce")
    print(df[["white_elo", "black_elo"]].isna().sum())  #checking for null values

    df["avg_elo"] = ((df["white_elo"] + df["black_elo"]) / 2).astype(int)
    sns.histplot(data=df["avg_elo"])
    plt.show()
    plt.savefig("avg_elo.png")

    sns.histplot(data=df["num_moves"])
    plt.show()
    plt.savefig("num_moves.png")

    df["fen_ply20"] = df["moves"].apply(lambda pgn: fen_from_pgn_string(pgn, ply=20))
    df.to_csv("fens.csv", index=False)



