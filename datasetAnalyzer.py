import io
import chess.pgn
import chess
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

PIECE_TO_CHANNEL = {
    (chess.PAWN,   chess.WHITE): 0,
    (chess.KNIGHT, chess.WHITE): 1,
    (chess.BISHOP, chess.WHITE): 2,
    (chess.ROOK,   chess.WHITE): 3,
    (chess.QUEEN,  chess.WHITE): 4,
    (chess.KING,   chess.WHITE): 5,
    (chess.PAWN,   chess.BLACK): 6,
    (chess.KNIGHT, chess.BLACK): 7,
    (chess.BISHOP, chess.BLACK): 8,
    (chess.ROOK,   chess.BLACK): 9,
    (chess.QUEEN,  chess.BLACK): 10,
    (chess.KING,   chess.BLACK): 11,
}

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


def pgn_to_bitboard(pgn_string: str, ply: int = 20, max_turns: int = 200) -> np.ndarray | None:
    game = chess.pgn.read_game(io.StringIO(pgn_string))

    if game is None:
        return None

    board = game.board()
    for move in game.mainline_moves():
        board.push(move)
        if board.ply() == ply:
            planes = np.zeros((13, 8, 8), dtype=np.float32)

            for sq, piece in board.piece_map().items():
                rank, file = divmod(sq, 8)
                planes[PIECE_TO_CHANNEL[(piece.piece_type, piece.color)], rank, file] = 1.0


            planes[12] = board.ply() / (2 * max_turns)

            return planes

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
    plt.savefig("avg_elo.png")

    sns.histplot(data=df["num_moves"])
    plt.savefig("num_moves.png")

    df2 = df["moves"].apply(lambda pgn: pgn_to_bitboard(pgn, ply=20)).to_frame(name="bitboard_ply20")
    df2.to_csv("bitboards.csv", index=False)



