import io
import chess.pgn
import chess
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import random

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


def pgn_to_bitboard(pgn_string: str, max_turns: int = 200) -> np.ndarray | None:
    game = chess.pgn.read_game(io.StringIO(pgn_string))

    if game is None:
        return None

    total_ply = game.end().ply()
    max_ply = min(total_ply, max_turns)

    if max_ply < 0:
        return None

    random_ply = random.randint(0, max_ply)

    board = game.board()
    moves_list = list(game.mainline_moves())

    if random_ply == 0:
        num_pieces = len(board.piece_map())
        target_ply = 0
    else:
        for i, move in enumerate(moves_list):
            board.push(move)
            if board.ply() == random_ply:
                num_pieces = len(board.piece_map())
                if num_pieces <= 7:
                    target_ply = max(0, random_ply - 20)
                else:
                    target_ply = random_ply
                break


    if target_ply < board.ply():
        for _ in range(board.ply() - target_ply):
            board.pop()
    elif target_ply > board.ply():
        # Should not happen, since target_ply <= random_ply <= max_ply
        pass


    planes = np.zeros((13, 8, 8), dtype=np.float32)

    for sq, piece in board.piece_map().items():
        rank, file = divmod(sq, 8)
        planes[PIECE_TO_CHANNEL[(piece.piece_type, piece.color)], rank, file] = 1.0

    planes[12] = board.ply() / (2 * max_turns)

    return planes


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

    df2 = df["moves"].apply(lambda pgn: pgn_to_bitboard(pgn)).to_frame(name="bitboard_random")
    df2.to_csv("bitboards.csv", index=False)

    # Add piece count metric
    df2["piece_count"] = df2["bitboard_random"].apply(lambda x: np.sum(x[0:12]) if x is not None else None)
    print(f"Piece count stats:\n{df2['piece_count'].describe()}")

    # Plot piece count distribution
    sns.histplot(data=df2["piece_count"].dropna())
    plt.savefig("piece_count.png")



