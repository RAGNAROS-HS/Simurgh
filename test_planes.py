import numpy as np
import chess
from datasetAnalyzer import pgn_to_bitboard, PIECE_TO_CHANNEL

def test_planes():
    # A simple PGN for testing (Ruy Lopez opening)
    pgn = "1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 6. Re1 b5 7. Bb3 d6 8. c3 O-O 9. h3 Na5 10. Bc2 c5"
    
    # Generate bitboard for ply 20 (10 full moves)
    planes = pgn_to_bitboard(pgn, ply=20)
    
    if planes is None:
        print("Error: pgn_to_bitboard returned None")
        return

    print(f"Planes shape: {planes.shape}")
    assert planes.shape == (13, 8, 8), f"Expected shape (13, 8, 8), got {planes.shape}"
    
    # Check if piece channels are correctly populated (0-11)
    # At ply 20, white has moved 10 times, black has moved 10 times.
    # The board should have 32 pieces unless some were captured (none in this PGN).
    piece_count = np.sum(planes[:12])
    print(f"Piece count in channels 0-11: {piece_count}")
    assert piece_count == 32, f"Expected 32 pieces, found {piece_count}"
    
    # Check move number normalization (channel 12)
    # ply 20 / (2 * 200 max turns) = 0.05
    move_norm = planes[12, 0, 0] # It should be the same across the 8x8 grid
    print(f"Normalized move number at index 12: {move_norm}")
    assert abs(move_norm - 20/400) < 1e-6, f"Expected 0.05, got {move_norm}"
    
    print("Verification successful!")

if __name__ == "__main__":
    test_planes()
