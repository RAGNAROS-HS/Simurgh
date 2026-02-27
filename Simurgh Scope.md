# Simurgh: Chess Position Possibility Classification – Data Scope Proposal

## Overview

This report proposes a concrete, tractable data scope and synthetic data generation plan for a project that classifies chess positions as **possible vs. impossible** (reachable vs. unreachable from the standard starting position, given legal play and a turn number). It focuses on how to construct training and evaluation datasets, what impossibility patterns to target, how large each subset should be, and whether to include endgames.


## Core Modeling Choice

The classification task can be framed as:

- Input: FEN (or equivalent board encoding) + side-to-move + fullmove number (and possibly halfmove clock/castling/EP flags).
- Output: Binary label – `reachable` (possible) vs. `unreachable` (impossible).

Because exact reachability is PSPACE-complete in general, the model is explicitly an **approximator** trained on curated examples rather than a solver.[^1]


## Source of Possible Positions

A practical primary source of reachable positions is the Hugging Face dataset `lapp0/100M_random_chess_boards`.[^2]

Key points for scope:

- The dataset is already large (100M boards) and uses random-play generation, so it provides diverse legal positions without being biased to only human-like play.[^3][^2]
- For a first iteration, sampling **1–5M positions** is already plenty for training; beyond that, model capacity and training time, not raw data volume, will likely be the bottleneck.
- You can downsample by constraints (e.g., limit by move number ranges or piece counts) to target specific subdomains.

Recommended first-pass sampling:

- Total: ~2M reachable positions.
- Sub-stratify by fullmove number (or ply):
  - Early game: fullmove 1–15 (say 25%).
  - Midgame: 16–40 (50%).
  - Late/endgame: 41+ (25%).

This gives you coverage for impossibility type (2) “state beyond turn number possibility” without needing synthetic turn-number labels.


## High-Level Synthetic Strategy

Synthetic “impossible” positions are needed because unreachable positions are extremely rare among random or natural-play positions, except for trivial illegality patterns.[^4]

Design goals:

- Control difficulty from **obvious impossibilities** (good for calibration) to **subtle but real unreachable patterns**.
- Ensure that synthetic positions are **legal according to a move generator** so the model does not just learn “illegal = impossible.”
- Make each impossibility type a parametric generator, so you can dial frequency and difficulty.

Overall class balance:

- For an initial model: aim for **50–50 reachable vs. unreachable** in training.
- For evaluation, also create:
  - A **balanced synthetic test set** (easy–hard unreachable cases vs. reachable controls).
  - A **natural test set** drawn from real games, where unreachable is extremely rare (near-all reachable); this lets you measure false positives in realistic conditions.

Suggested totals for v1:

- Training: 1.5M reachable + 1.5M unreachable (3M total).
- Validation: 150k reachable + 150k unreachable.
- Test (synthetic): 150k reachable + 150k unreachable.
- Test (natural only): 300k reachable from real games (e.g., Lichess HF data).[^5]


## Impossibility Taxonomy for Synthetic Data

This section outlines concrete generators for each of your three envisioned impossibility types and adds a few low-hanging categories from retrograde analysis discussions.[^6][^7][^4]

### Category A – Structural Pawn Impossibilities

Goal: Positions that are **piecewise legal** but whose pawn structure cannot be obtained from the start position via legal moves.

Typical motifs from retrograde analysis and programming discussions:[^7][^6][^4]

- **Too many pawns on one file** without sufficient captures on neighbor files (e.g., three white pawns on the a-file without enough missing black pieces to justify the captures).
- **Pawns “behind” enemy pawns** in ways that cannot be explained by captures (e.g., two opposing pawns on the same file, in the wrong ordering, with no missing pieces on adjacent files to justify side-stepping captures).[^4]
- **Pawns on first rank** (non-promoted) or on eighth rank without being promoted.
- **Impossible doubled passed pawns** whose capture histories would require mutually inconsistent previous piece counts.

Practical generators (in order of increasing subtlety):

- A1: Take a legal position, randomly move one or two pawns backwards by one or two ranks (keeping them on the board and not creating immediate obvious illegalities like a pawn on rank 1 or 8). Then run a legality check to ensure the resulting diagram is still “structurally legal” (kings, checks, etc.). These are simple “time-reversal” pawn impossible positions.
- A2: Start from a legal position and **duplicate a pawn** onto an adjacent file while deleting some random enemy minor piece, arranged to keep both kings safe. This creates pawn-count or capture-budget inconsistencies.
- A3: Construct small local pawn configurations from hand-designed templates known to be impossible (e.g., three white pawns on a2, a3, b2 with no black captures to justify the extra pawn structure), then embed them into otherwise legal random positions.[^4]

Dataset sizing for Category A:

- Around **30–40% of unreachable synthetic examples** should be pawn-structure based.
- Within that, mix roughly:
  - 40% A1 (easy, transparent).
  - 40% A2 (medium subtlety).
  - 20% A3 (hard, more crafted templates).

### Category B – Turn-Number / Tempo Impossibilities

Goal: Positions that are legal but cannot be reached **within N moves** (fullmoves) from the initial position.

This is where you leverage the fullmove number as an additional input. For a given board, you can often put a **too-small fullmove number label** to create impossibility.

Practical scheme:

- Generate a large corpus of positions by **forward random play** with a legal engine (e.g., python-chess stock random moves) and record the ply at which each position was first seen.
- For each recorded position with “first-seen ply” equal to p (or fullmove f), you know it is reachable in f moves.
- To synthesize impossible examples:
  - Sample such positions and assign them **smaller fullmove numbers** than the minimum required, e.g., `f_impossible = max(1, floor(0.5 f))`.
  - Keep the board itself unchanged but modify the fullmove number field in the FEN.

This creates training data where the only signal is the mismatch between **complexity of the position** (material spread, pawn structure, piece development) and the claimed move number.[^7]

Dataset sizing for Category B:

- Around **30–40% of unreachable samples**.
- Choose several relative thresholds (e.g., 50%, 33%, 25% of minimal required moves) so that the model learns a graded intuition for complexity vs. time.

### Category C – Piece-Mobility History Impossibilities

Goal: Positions where some piece’s path from the starting position cannot exist given blocking pieces or missing captures.

Typical motifs:

- A bishop on a given color complex where all necessary “gate” squares on its path from the starting square are permanently blocked by unmoved pawns, and no captures can have removed them because they are still present.
- Rooks or queens trapped behind their own untouched pawn wall with no open files, yet located on squares that require passing through that wall.
- Knights or bishops in corners that could only have arrived by paths crossing illegally through pieces that are still on starting squares.

Practical generators:

- C1: Take a legal midgame position, **freeze some back-rank or pawn structure as “unmoved”** (e.g., both rooks plus their pawns still in initial positions) and then artificially place one rook/bishop/queen beyond that barrier on a square that requires going through the wall.
- C2: Template-based constructions inspired by retro analysis problems, e.g., a bishop on a1 with all dark-squared diagonals from c3 to g7 blocked by unmoved pawns that are still present.[^6]

These are harder to get right and might require restricted search or offline proof-game tools (e.g., Natch, Texel) for validation.[^7]

Dataset sizing for Category C:

- Around **10–20% of unreachable samples** in early iterations (it is the hardest generator to get correct).
- Start conservative with hand-crafted templates plus simple barrier-logic, and increase complexity later.

### Category D – Check/Turn Consistency Impossibilities (Local Rules)

These are simpler retro constraints that tablebases routinely exclude:

- **Both kings in check** simultaneously is impossible after a single legal move.[^7]
- **Side not to move is in check** (e.g., it is White to move but Black’s king is in check) if you define legality as “existence of a previous legal move,” rather than FIDE’s weaker static legality notion.

Generators:

- D1: Take a legal position and apply a move that delivers check to both kings (e.g., double-check plus discovered check) but then **retract only one side’s checks** in the diagram.
- D2: Flip the “side to move” bit of a legal check position, so that the side in check is the side that just moved.

Dataset sizing:

- Around **10–20% of unreachable samples**.
- These are good for teaching the model tablebase-style legality notions and are relatively easy to generate.

### Category E – Castling/En Passant Ambiguities (Optional v2)

Tablebases generally avoid these by dropping castling rights and handling en passant partially, because history is not encoded in the static diagram.[^7]

For an initial scope, it is reasonable to **exclude these from training and labeling** by:

- Clearing castling rights and en passant flags in all FENs.
- Treating all such positions as “no special rights,” thereby avoiding subtle historical questions.

Once the base classifier is working, a v2 project could expand to **augmented FEN** that explicitly includes “ground truth” castling and EP rights from a simulated game history and then classifies possibility including those.


## How to Generate Synthetic Unreachable Positions

This section outlines a concrete pipeline to produce unreachable labels that are still structurally legal.

### Step 1 – Start from Clean Legal Positions

- Use a strong legality checker (e.g., python-chess) and discard any positions where kings are missing, in check illegally, or other surface-level violations.
- Optionally, restrict to positions with full FIDE legality (exactly one side in check at most).

Sources:

- Random-play generators such as `random-chess` benchmark positions.[^3]
- Game databases (e.g., Lichess HF datasets in Parquet format).[^8][^5]

### Step 2 – Apply Targeted Perturbations

For each category A–D above:

- Define a set of **local edit operations** (e.g., move pawn one rank backward, duplicate pawn, move rook through pawn wall, flip side-to-move in a check position).
- Apply 1–3 such edits to a base legal position.
- After editing, re-run legality checks:
  - Ensure exactly one white and one black king exist.
  - Ensure no piece stands off-board, no side has more than 8 pawns, etc.
  - Ensure the move generator considers the position legal in the static sense.

Any positions that fail these checks are discarded or optionally relabeled as “structurally illegal” and stored for a separate auxiliary task.

### Step 3 – Validate a Small Gold Subset via Retro Tools

To anchor the labels, construct a **small gold test set** (e.g., a few thousand positions) where reachability/unreachability is verified by more exact methods.

- Use proof-game solvers like Natch and Texel for smaller board or low-piece-count positions to conclusively classify positions.[^7]
- For endgame positions (≤7 pieces), cross-check with tablebases to exclude positions that tablebases label as impossible or that cannot be generated from legal sequences.[^7]

This gold set becomes your **absolute validation** for measuring model accuracy on truly-hard retrograde cases.


## Endgames and Tablebases – Include or Not?

Endgames (≤7 pieces) are special because they intersect directly with endgame tablebases and retrograde analysis.

### Recommended v1 Policy

For the **first iteration of Simurgh**, it is reasonable to **exclude explicit endgame specialization**:

- Keep positions with **8+ pieces** only in the main training/validation/test splits.
- Allow endgames to appear incidentally in the random-play generation but do not overweight them.
- Do **not** build a dedicated endgame-only submodel yet.

Reasons:

- Reachability of low-piece-count positions is highly constrained and well studied; building a dedicated retro-grade or tablebase-based labeler is an entire subproject.[^7]
- The main goal is to train a model with **global intuition** about reachability, not to exactly match tablebase logic.

### When to Bring in Endgames

Endgames become interesting once:

- The main model performs well on midgame positions (e.g., 10–32 pieces) and recognizes common impossibility motifs.
- You want an **absolute correctness benchmark**: a set of positions where reachability is fully decidable.

At that point:

- Construct a separate **7-piece-or-less dataset** with labels sourced from tablebases and retro tools.
- Use it as an **evaluation-only benchmark**, not necessarily as training data, to see whether your learned model extrapolates down to simple positions.


## Suggested Dataset Breakdown by Difficulty

To shape the learning curriculum, it helps to mark synthetic unreachable positions by **difficulty tier** (even if the model only sees binary labels). You can use this for curriculum learning or for analysis.

Example proportions within unreachable training samples:

| Tier | Description | Examples | Share |
|------|-------------|----------|-------|
| Easy | Obvious local rule contradictions | Both kings in check, side-not-to-move in check, pawn on rank 1/8 unpromoted | 30% |
| Medium | Local structural impossibilities | Simple pawn-structure anomalies, rooks behind untouched pawn walls, simple tempo violations | 50% |
| Hard | Deep history contradictions | Subtle pawn-capture histories, complex tempo bounds, puzzly retro constructions | 20% |

You can map Categories A–D into these tiers, and optionally reserve hard cases mostly for validation/test to avoid overfitting to puzzle-like artifacts.


## Practical Numbers and Scaling Plan

A concrete v1 scope that is large but doable on a single-GPU training setup:

- 3M total training positions (1.5M reachable, 1.5M unreachable synthetic).
- 300k validation (balanced).
- 300k synthetic test (balanced) + 300k natural-only test.

If training is stable and the model underfits, you can double these numbers; if training is slow or memory-limited, halve them while keeping the **relative composition** and category proportions similar.


## Summary of Recommendations

- Use `lapp0/100M_random_chess_boards` and/or Lichess HF datasets as your main reachable source, downsampled to ~2M–3M positions for v1.[^8][^5][^2][^3]
- Generate unreachable positions via **parametric perturbations** in four main categories: pawn-structure, time/tempo, piece-mobility history, and check/turn inconsistencies, with optional castling/EP history left for later.
- Maintain a roughly **50–50 reachable/unreachable** split in synthetic training, plus a **natural-only** test set to measure realistic false positive rates.
- Defer dedicated endgame/tablebase integration to a **second phase**, using ≤7-piece positions primarily as an evaluation benchmark grounded in retrograde analysis tools and tablebases.[^7]

---

## References

1. [[PDF] Complexity of Retrograde and Helpmate Chess Problems - arXiv](https://arxiv.org/pdf/2010.09271.pdf)

2. [lapp0/100M_random_chess_boards · Datasets at Hugging Face](https://huggingface.co/datasets/lapp0/100M_random_chess_boards) - We’re on a journey to advance and democratize artificial intelligence through open source and open s...

3. [GitHub - gcp/random-chess: A simple yet effective LLM reasoning benchmark](https://github.com/gcp/random-chess) - A simple yet effective LLM reasoning benchmark. Contribute to gcp/random-chess development by creati...

4. [Prove that a position is legal and reachable. - Page 2 - TalkChess.com](https://talkchess.com/viewtopic.php?t=43732&start=10) - For one there are a few easy heuristics to establish that a structure is not reachable, (e.g. white ...

5. [Lichess/chess-puzzles · Datasets at Hugging Face](https://hf.rst.im/datasets/Lichess/chess-puzzles) - We’re on a journey to advance and democratize artificial intelligence through open source and open s...

6. [Retrograde analysis - Wikipedia](https://en.wikipedia.org/wiki/Retrograde_analysis)

7. [Retrograde Analysis](https://www.chessprogramming.org/Retrograde_Analysis)

8. [The Lichess database of games, puzzles, and engine evaluations is now on Hugging Face: Billions of chess data points to download, query, and stream!](https://www.reddit.com/r/chess/comments/1h7y1bf/the_lichess_database_of_games_puzzles_and_engine/) - The Lichess database of games, puzzles, and engine evaluations is now on Hugging Face: Billions of c...

