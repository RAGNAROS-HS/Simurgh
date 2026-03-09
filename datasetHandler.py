import io
import chess.pgn
import chess
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import random
import os

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

    # Advance to the randomly selected ply
    for move in moves_list[:random_ply]:
        board.push(move)
    
    # Walk back if piece count is too low (minimum 8 pieces)
    while len(board.piece_map()) < 8 and board.ply() > 0:
        board.pop()


    planes = np.zeros((13, 8, 8), dtype=np.float32)

    for sq, piece in board.piece_map().items():
        rank, file = divmod(sq, 8)
        planes[PIECE_TO_CHANNEL[(piece.piece_type, piece.color)], rank, file] = 1.0

    planes[12] = board.ply() / (2 * max_turns)

    return planes


def generate_unreachable_bitboard(bitboard: np.ndarray, max_turns: int = 200) -> np.ndarray | None:
    """
    Generates unreachable board states via perturbations.
    Currently implements:
    - Turn number perturbation (Inflation/Truncation)
    
    Future implementations will include:
    - Pawn swaps/teleports
    - Piece mobility violations
    """
    if bitboard is None:
        return None
    
    unreachable_bitboard = bitboard.copy()
    
    # Perturb turn number (Plane 12)
    current_ply = unreachable_bitboard[12, 0, 0] * (2 * max_turns)
    
    perturbation_type = random.choice(["inflation", "truncation"])
    
    if perturbation_type == "inflation":
        # Add 50 to 100 extra plies
        inflated_ply = current_ply + random.randint(50, 100)
        unreachable_bitboard[12] = inflated_ply / (2 * max_turns)
    else:
        # Set to a very low ply (e.g., 0 to 2) regardless of piece count
        # (Assuming the original logic ensures at least 8 pieces)
        truncated_ply = random.randint(0, 2)
        unreachable_bitboard[12] = truncated_ply / (2 * max_turns)

    return unreachable_bitboard

if __name__ == "__main__":
    PLOT_DIR = "plots"
    DATASET_DIR = "datasets"
    os.makedirs(PLOT_DIR, exist_ok=True)
    os.makedirs(DATASET_DIR, exist_ok=True)

    DATASET_PATH = r"E:\lichess_db_standard_rated_2025-05.pgn"
    df = pd.DataFrame(load_pgn(DATASET_PATH))
    df.to_csv(os.path.join(DATASET_DIR, "games.csv"), index=False)

    df["white_elo"] = pd.to_numeric(df["white_elo"], errors="coerce")
    df["black_elo"] = pd.to_numeric(df["black_elo"], errors="coerce")
    print(df[["white_elo", "black_elo"]].isna().sum())  #checking for null values

    df["avg_elo"] = ((df["white_elo"] + df["black_elo"]) / 2).astype(int)
    sns.histplot(data=df["avg_elo"])
    plt.savefig(os.path.join(PLOT_DIR, "avg_elo_combined.png"))
    plt.close()

    sns.histplot(data=df["num_moves"])
    plt.savefig(os.path.join(PLOT_DIR, "num_moves_combined.png"))
    plt.close()

    # Generate reachable bitboards from PGN
    df["bitboard"] = df["moves"].apply(lambda pgn: pgn_to_bitboard(pgn))
    df_with_bitboards = df.dropna(subset=["bitboard"])
    
    # Split into two halves: 50% for actual reachable samples, 50% for generating unreachable
    from sklearn.model_selection import train_test_split
    df_reachable_source, df_unreachable_source = train_test_split(
        df_with_bitboards, 
        test_size=0.5, 
        random_state=42
    )
    
    # Create reachable samples
    reachable_df = df_reachable_source[["bitboard", "avg_elo", "num_moves"]].copy()
    reachable_df["is_reachable"] = 1
    
    # Generate unreachable samples by perturbing the second half
    unreachable_df = df_unreachable_source[["bitboard", "avg_elo", "num_moves"]].copy()
    
    # Track plies for visualization
    def get_ply(bitboard):
        return bitboard[12, 0, 0] * (2 * 200) if bitboard is not None else None

    unreachable_df["ply_before"] = unreachable_df["bitboard"].apply(get_ply)
    unreachable_df["bitboard"] = unreachable_df["bitboard"].apply(lambda x: generate_unreachable_bitboard(x, max_turns=200))
    unreachable_df["ply_after"] = unreachable_df["bitboard"].apply(get_ply)
    unreachable_df["is_reachable"] = 0
    
    # Plot turn perturbation distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(unreachable_df["ply_before"], color="blue", label="Original", kde=True, alpha=0.5)
    sns.histplot(unreachable_df["ply_after"], color="red", label="Perturbed", kde=True, alpha=0.5)
    plt.title("Turn Number Perturbation Distribution (Before vs After)")
    plt.xlabel("Ply Count")
    plt.ylabel("Frequency")
    plt.legend()
    plt.savefig(os.path.join(PLOT_DIR, "ply_perturbation_comparison.png"))
    plt.close()
    
    # Combine into a single balanced dataset
    combined_df = pd.concat([reachable_df, unreachable_df]).dropna(subset=["bitboard"])
    
    # Perform 80/10/10 Split (Stratified)
    # First split: 80% train, 20% temp (for val and test)
    train_df, temp_df = train_test_split(
        combined_df, 
        test_size=0.2, 
        random_state=42, 
        stratify=combined_df["is_reachable"]
    )
    
    # Second split: Split temp (20% of total) into 50/50 for val and test (each 10% of total)
    val_df, test_df = train_test_split(
        temp_df, 
        test_size=0.5, 
        random_state=42, 
        stratify=temp_df["is_reachable"]
    )
    
    # Save the splits using pickle (since bitboards are numpy arrays)
    train_df.to_pickle(os.path.join(DATASET_DIR, "train_dataset.pkl"))
    val_df.to_pickle(os.path.join(DATASET_DIR, "val_dataset.pkl"))
    test_df.to_pickle(os.path.join(DATASET_DIR, "test_dataset.pkl"))
    
    print(f"Dataset split complete (Distinct Origins):")
    print(f"Total reachable:   {len(reachable_df)}")
    print(f"Total unreachable: {len(unreachable_df)}")
    print(f"Train set: {len(train_df)} rows ({train_df['is_reachable'].mean()*100:.1f}% reachable)")
    print(f"Val set:   {len(val_df)} rows ({val_df['is_reachable'].mean()*100:.1f}% reachable)")
    print(f"Test set:  {len(test_df)} rows ({test_df['is_reachable'].mean()*100:.1f}% reachable)")

    # Add piece count metric (using combined_df for the plot)
    combined_df["piece_count"] = combined_df["bitboard"].apply(lambda x: np.sum(x[0:12]) if x is not None else None)
    print(f"\nPiece count stats (all samples):\n{combined_df['piece_count'].describe()}")

    # Plot piece count distribution
    sns.histplot(data=combined_df, x="piece_count", hue="is_reachable", multiple="stack")
    plt.title("Piece Count Distribution (Reachable vs Unreachable)")
    plt.savefig(os.path.join(PLOT_DIR, "piece_count_distribution.png"))
    plt.close()

    # Separate plots for reachable and unreachable
    sns.histplot(data=combined_df[combined_df["is_reachable"] == 1], x="piece_count", color="green")
    plt.title("Piece Count Distribution (Reachable)")
    plt.savefig(os.path.join(PLOT_DIR, "piece_count_reachable.png"))
    plt.close()

    sns.histplot(data=combined_df[combined_df["is_reachable"] == 0], x="piece_count", color="red")
    plt.title("Piece Count Distribution (Unreachable)")
    plt.savefig(os.path.join(PLOT_DIR, "piece_count_unreachable.png"))
    plt.close()

    # Separate plots for avg_elo
    sns.histplot(data=combined_df[combined_df["is_reachable"] == 1], x="avg_elo", color="green")
    plt.title("Average ELO Distribution (Reachable)")
    plt.savefig(os.path.join(PLOT_DIR, "avg_elo_reachable.png"))
    plt.close()

    sns.histplot(data=combined_df[combined_df["is_reachable"] == 0], x="avg_elo", color="red")
    plt.title("Average ELO Distribution (Unreachable)")
    plt.savefig(os.path.join(PLOT_DIR, "avg_elo_unreachable.png"))
    plt.close()

    # Separate plots for num_moves
    sns.histplot(data=combined_df[combined_df["is_reachable"] == 1], x="num_moves", color="green")
    plt.title("Number of Moves Distribution (Reachable)")
    plt.savefig(os.path.join(PLOT_DIR, "num_moves_reachable.png"))
    plt.close()

    sns.histplot(data=combined_df[combined_df["is_reachable"] == 0], x="num_moves", color="red")
    plt.title("Number of Moves Distribution (Unreachable)")
    plt.savefig(os.path.join(PLOT_DIR, "num_moves_unreachable.png"))
    plt.close()



