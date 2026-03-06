# Implemented So Far

## Project Setup
- Repository initialized with basic structure
- README.md created with project overview, problem statement, and related work

## Dataset Handling
- `datasetAnalyzer.py`: Script to load PGN games from Lichess database
- Extracts game metadata: result, ELO ratings, opening, number of moves, moves string
- Converts PGN games to bitboard representations at ply 20 (10 full moves)
- Generates CSV files: `games.csv` (game metadata) and `bitboards.csv` (bitboard data)
- Includes data visualization for ELO distributions and game lengths

## Testing
- `test_planes.py`: Unit test for bitboard conversion functionality
- Verifies correct shape (13x8x8 planes), piece count (32 pieces), and move number normalization

## Pending Implementation
- Neural network architecture (`neuralNetwork.py`)
- Reachable position generator (`reachableGenerator.py`)
- Unreachable position generator (`unreachableGenerator.py`)
- Training pipeline
- Model evaluation and classification system