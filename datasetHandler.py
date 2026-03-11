import io
import chess.pgn
import chess
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import random
import os
import seaborn as sns

sns.set_theme(style="whitegrid", palette="muted")

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


def perturb_turn_number(bitboard: np.ndarray, max_turns: int = 200) -> str:
    """
    Perturbs the turn number (Plane 12) of a bitboard via inflation or truncation.
    """
    current_ply = bitboard[12, 0, 0] * (2 * max_turns)
    
    perturbation_type = random.choice(["inflation", "truncation"])
    
    if perturbation_type == "inflation":
        # Add 50 to 100 extra plies
        inflated_ply = current_ply + random.randint(50, 100)
        bitboard[12] = inflated_ply / (2 * max_turns)
    else:
        # Truncate the ply count significantly (e.g., between 0 and half the current ply)
        # This creates a more natural distribution and ensures it's always smaller than current_ply
        if current_ply > 1:
            truncated_ply = random.randint(0, int(current_ply) // 2)
        else:
            truncated_ply = 0
        bitboard[12] = truncated_ply / (2 * max_turns)

    return f"turn_number_{perturbation_type}"


def perturb_pawn_structure(bitboard: np.ndarray) -> str:
    """
    Perturbs the pawn structure to create an unreachable state regardless of turn number.
    Implements:
    - Rank violation (pawns on rank 1 or 8)
    - Excess pawns (9 pawns of one color)
    - Capture contradiction (stacking pawns on one file such that it requires more captures than missing opponent pieces)
    """
    color = random.choice(["white", "black"])
    pawn_channel = 0 if color == "white" else 6
    opponent_start = 6 if color == "white" else 0
    opponent_end = 12 if color == "white" else 6
    
    pawn_coords = np.argwhere(bitboard[pawn_channel] == 1)
    
    # Calculate opponent's missing pieces
    opponent_pieces = np.sum(bitboard[opponent_start:opponent_end])
    missing_pieces = 16 - opponent_pieces
    
    p_types = ["rank_violation", "excess_pawns"]
    if len(pawn_coords) - 1 > missing_pieces and len(pawn_coords) > 0:
        p_types.append("capture_contradiction")
        
    p_type = random.choice(p_types)
    
    if p_type == "rank_violation" and len(pawn_coords) > 0:
        idx = random.randint(0, len(pawn_coords) - 1)
        r, c = pawn_coords[idx]
        bitboard[pawn_channel, r, c] = 0
        
        target_rank = random.choice([0, 7])
        for ch in range(12):
            bitboard[ch, target_rank, c] = 0
        bitboard[pawn_channel, target_rank, c] = 1
        
    elif p_type == "excess_pawns":
        pawns_needed = 9 - len(pawn_coords)
        if pawns_needed > 0:
            empty_sqs = np.argwhere(np.sum(bitboard[:12], axis=0) == 0)
            valid_empty = [sq for sq in empty_sqs if sq[0] not in [0, 7]]
            
            if len(valid_empty) >= pawns_needed:
                valid_empty_list = [tuple(sq) for sq in valid_empty]
                chosen = random.sample(valid_empty_list, pawns_needed)
                for r, c in chosen:
                    bitboard[pawn_channel, r, c] = 1
            elif len(pawn_coords) > 0:
                # Fallback to rank violation if no room
                p_type = "excess_pawns_fallback_rank_violation"
                r, c = pawn_coords[0]
                bitboard[pawn_channel, r, c] = 0
                for ch in range(12):
                    bitboard[ch, 0, c] = 0
                bitboard[pawn_channel, 0, c] = 1
                
    elif p_type == "capture_contradiction":
        for r, c in pawn_coords:
            bitboard[pawn_channel, r, c] = 0
        
        target_file = random.randint(0, 7)
        max_pawns_to_place = min(len(pawn_coords), 6) # Can't place more than 6 pawns on one file (ranks 1-6)
        ranks = random.sample([1, 2, 3, 4, 5, 6], max_pawns_to_place)
        for r in ranks:
            # Clear target squares
            for ch in range(12):
                bitboard[ch, r, target_file] = 0
            bitboard[pawn_channel, r, target_file] = 1

    return f"pawn_structure_{p_type}"


def perturb_piece_mobility(bitboard: np.ndarray, max_turns: int = 200) -> str:
    """
    Perturbs piece mobility and placement to create an unreachable state.
    Implements:
    - Impossible promotion (extra higher pieces but 8 pawns still present)
    - Bishops on same color (2 bishops on same color complex while having 8 pawns)
    - Illegal check (the player who just moved is left in check)
    """
    p_types = ["impossible_promotion", "bishops_on_same_color", "illegal_check"]
    p_type = random.choice(p_types)
    
    if p_type == "impossible_promotion":
        color = random.choice(["white", "black"])
        pawn_ch = 0 if color == "white" else 6
        queen_ch = 4 if color == "white" else 10
        
        # Ensure 8 pawns
        pawn_coords = np.argwhere(bitboard[pawn_ch] == 1)
        pawns_needed = 8 - len(pawn_coords)
        if pawns_needed > 0:
            empty_sqs = np.argwhere(np.sum(bitboard[:12], axis=0) == 0)
            valid_empty = [sq for sq in empty_sqs if sq[0] not in [0, 7]]
            if len(valid_empty) >= pawns_needed:
                valid_empty_list = [tuple(sq) for sq in valid_empty]
                chosen = random.sample(valid_empty_list, pawns_needed)
                for r, c in chosen:
                    bitboard[pawn_ch, r, c] = 1
                    
        # Add an extra queen to force promotion contradiction
        queens_coords = np.argwhere(bitboard[queen_ch] == 1)
        empty_sqs = np.argwhere(np.sum(bitboard[:12], axis=0) == 0)
        queens_to_add = max(1, 2 - len(queens_coords))
        if len(empty_sqs) >= queens_to_add:
            empty_sqs_list = [tuple(sq) for sq in empty_sqs]
            chosen = random.sample(empty_sqs_list, queens_to_add)
            for r, c in chosen:
                bitboard[queen_ch, r, c] = 1

        return "mobility_impossible_promotion"
        
    elif p_type == "bishops_on_same_color":
        color = random.choice(["white", "black"])
        pawn_ch = 0 if color == "white" else 6
        bishop_ch = 2 if color == "white" else 8
        
        # Ensure 8 pawns to rule out underpromotion
        pawn_coords = np.argwhere(bitboard[pawn_ch] == 1)
        pawns_needed = 8 - len(pawn_coords)
        if pawns_needed > 0:
            empty_sqs = np.argwhere(np.sum(bitboard[:12], axis=0) == 0)
            valid_empty = [sq for sq in empty_sqs if sq[0] not in [0, 7]]
            if len(valid_empty) >= pawns_needed:
                valid_empty_list = [tuple(sq) for sq in valid_empty]
                chosen = random.sample(valid_empty_list, pawns_needed)
                for r, c in chosen:
                    bitboard[pawn_ch, r, c] = 1
                    
        # Put 2 bishops on same color
        for r in range(8):
            for c in range(8):
                bitboard[bishop_ch, r, c] = 0 # Clear existing bishops
        
        empty_sqs = np.argwhere(np.sum(bitboard[:12], axis=0) == 0)
        empty_sqs_list = [tuple(sq) for sq in empty_sqs]
        light_sqs = [sq for sq in empty_sqs_list if (sq[0] + sq[1]) % 2 == 1]
        dark_sqs = [sq for sq in empty_sqs_list if (sq[0] + sq[1]) % 2 == 0]
        
        chosen_color_sqs = light_sqs if random.random() < 0.5 else dark_sqs
        if len(chosen_color_sqs) >= 2:
            b1, b2 = random.sample(chosen_color_sqs, 2)
            bitboard[bishop_ch, b1[0], b1[1]] = 1
            bitboard[bishop_ch, b2[0], b2[1]] = 1
            return "mobility_bishops_same_color"
        else:
            return "mobility_bishops_same_color_failed"
            
    elif p_type == "illegal_check":
        ply = int(round(bitboard[12, 0, 0] * (2 * max_turns)))
        just_moved_color = "black" if ply % 2 == 0 else "white"
        jm_king_ch = 11 if just_moved_color == "black" else 5
        opp_knight_ch = 1 if just_moved_color == "black" else 7
        
        king_pos = np.argwhere(bitboard[jm_king_ch] == 1)
        if len(king_pos) > 0:
            kr, kc = king_pos[0]
            knight_moves = [
                (kr+2, kc+1), (kr+2, kc-1), (kr-2, kc+1), (kr-2, kc-1),
                (kr+1, kc+2), (kr+1, kc-2), (kr-1, kc+2), (kr-1, kc-2)
            ]
            valid_km = [(r, c) for r, c in knight_moves if 0 <= r < 8 and 0 <= c < 8]
            if len(valid_km) > 0:
                random.shuffle(valid_km)
                for r, c in valid_km:
                    # Do not overwrite a king
                    if bitboard[5, r, c] == 0 and bitboard[11, r, c] == 0:
                        for ch in range(12): bitboard[ch, r, c] = 0
                        bitboard[opp_knight_ch, r, c] = 1
                        return "mobility_illegal_check"
        return "mobility_illegal_check_failed"

    return "mobility_unknown"


def generate_unreachable_bitboard(bitboard: np.ndarray, max_turns: int = 200) -> tuple[np.ndarray | None, str | None]:
    """
    Generates unreachable board states via perturbations.
    Currently implements:
    - Turn number perturbation (Inflation/Truncation)
    - Pawn structural perturbations (Rank violations, Excess pieces, Capture contradictions)
    - Piece mobility perturbations (Promotion contradiction, Bishops on same color, Illegal check)
    """
    if bitboard is None:
        return None, None
    
    unreachable_bitboard = bitboard.copy()
    
    perturb_category = random.choice(["turn", "pawn", "mobility"])
    
    if perturb_category == "turn":
        perturb_type = perturb_turn_number(unreachable_bitboard, max_turns)
    elif perturb_category == "pawn":
        perturb_type = perturb_pawn_structure(unreachable_bitboard)
    else:
        perturb_type = perturb_piece_mobility(unreachable_bitboard, max_turns)

    return unreachable_bitboard, perturb_type

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

    plt.figure(figsize=(10, 6))
    sns.histplot(data=df["avg_elo"], kde=True)
    plt.title("Distribution of Average ELO")
    plt.xlabel("Average ELO")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "avg_elo_combined.png"))
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.histplot(data=df["num_moves"], kde=True)
    plt.title("Distribution of Number of Moves")
    plt.xlabel("Number of Moves")
    plt.ylabel("Frequency")
    plt.tight_layout()
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
    reachable_df["perturbation_type"] = "none"
    
    # Generate unreachable samples by perturbing the second half
    unreachable_df = df_unreachable_source[["bitboard", "avg_elo", "num_moves"]].copy()
    
    # Track plies for visualization
    def get_ply(bitboard):
        return bitboard[12, 0, 0] * (2 * 200) if bitboard is not None else None

    unreachable_df["ply_before"] = unreachable_df["bitboard"].apply(get_ply)
    
    unreachable_df_results = unreachable_df["bitboard"].apply(lambda x: generate_unreachable_bitboard(x, max_turns=200))
    unreachable_df["bitboard"] = unreachable_df_results.apply(lambda x: x[0] if isinstance(x, tuple) else None)
    unreachable_df["perturbation_type"] = unreachable_df_results.apply(lambda x: x[1] if isinstance(x, tuple) else None)
    
    unreachable_df["ply_after"] = unreachable_df["bitboard"].apply(get_ply)
    unreachable_df["is_reachable"] = 0
    
    # Plot turn perturbation distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(data=unreachable_df, x="ply_before", color="blue", label="Original", kde=True, alpha=0.5)
    sns.histplot(data=unreachable_df, x="ply_after", color="red", label="Perturbed", kde=True, alpha=0.5)
    plt.title("Turn Number Perturbation Distribution (Before vs After)")
    plt.xlabel("Ply Count")
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "ply_perturbation_comparison.png"))
    plt.close()
    # Plot ply distribution for turn number perturbations only
    for ptype in ["turn_number_inflation", "turn_number_truncation"]:
        subset = unreachable_df[unreachable_df["perturbation_type"] == ptype]
        # Only plot if there's data to avoid errors
        if len(subset) == 0:
            continue
            
        plt.figure(figsize=(10, 6))
        sns.histplot(data=subset, x="ply_before", color="blue", label="Original", kde=True, alpha=0.5)
        sns.histplot(data=subset, x="ply_after", color="red", label="Perturbed", kde=True, alpha=0.5)
        plt.title(f"Ply Distribution Before vs After ({ptype})")
        plt.xlabel("Ply Count")
        plt.ylabel("Frequency")
        plt.legend()
        plt.tight_layout()
        # Clean up ptype for filename just in case
        safe_ptype = str(ptype).replace(" ", "_").replace("/", "_").lower()
        plt.savefig(os.path.join(PLOT_DIR, f"ply_perturbation_comparison_{safe_ptype}.png"))
        plt.close()

    # Plot histogram of perturbation types
    plt.figure(figsize=(12, 6))
    sns.countplot(data=unreachable_df, y="perturbation_type", order=unreachable_df["perturbation_type"].value_counts().index, color="skyblue")
    plt.title("Number of Instances per Perturbation Type")
    plt.xlabel("Count")
    plt.ylabel("Perturbation Type")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "perturbation_type_distribution.png"))
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
    plt.figure(figsize=(10, 6))
    sns.histplot(data=combined_df, x="piece_count", hue="is_reachable", multiple="stack", palette="Set1")
    plt.title("Piece Count Distribution (Reachable vs Unreachable)")
    plt.xlabel("Piece Count")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "piece_count_distribution.png"))
    plt.close()

    # Separate plots for reachable and unreachable
    plt.figure(figsize=(10, 6))
    sns.histplot(data=combined_df[combined_df["is_reachable"] == 1], x="piece_count", color="green", kde=True)
    plt.title("Piece Count Distribution (Reachable)")
    plt.xlabel("Piece Count")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "piece_count_reachable.png"))
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.histplot(data=combined_df[combined_df["is_reachable"] == 0], x="piece_count", color="red", kde=True)
    plt.title("Piece Count Distribution (Unreachable)")
    plt.xlabel("Piece Count")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "piece_count_unreachable.png"))
    plt.close()

    # Separate plots for avg_elo
    plt.figure(figsize=(10, 6))
    sns.histplot(data=combined_df[combined_df["is_reachable"] == 1], x="avg_elo", color="green", kde=True)
    plt.title("Average ELO Distribution (Reachable)")
    plt.xlabel("Average ELO")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "avg_elo_reachable.png"))
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.histplot(data=combined_df[combined_df["is_reachable"] == 0], x="avg_elo", color="red", kde=True)
    plt.title("Average ELO Distribution (Unreachable)")
    plt.xlabel("Average ELO")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "avg_elo_unreachable.png"))
    plt.close()

    # Separate plots for num_moves
    plt.figure(figsize=(10, 6))
    sns.histplot(data=combined_df[combined_df["is_reachable"] == 1], x="num_moves", color="green", kde=True)
    plt.title("Number of Moves Distribution (Reachable)")
    plt.xlabel("Number of Moves")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "num_moves_reachable.png"))
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.histplot(data=combined_df[combined_df["is_reachable"] == 0], x="num_moves", color="red", kde=True)
    plt.title("Number of Moves Distribution (Unreachable)")
    plt.xlabel("Number of Moves")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "num_moves_unreachable.png"))
    plt.close()



