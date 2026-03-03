# Simurgh: Chess Position Possibility Classification

## Overview

I know what you are thinking — "oh no, not another chess project." But hear me out. 

While lots of work is dedicated to evaluating positions or legal moves, **I have yet to see a project evaluating possibilities**. 

Chess grandmasters can look at a board and determine whether such a game could exist. Their experience and pattern recognition allow them to assess whether a board state is **possible**. The distinction between **legal** and **possible** is crucial here.

> **Key Insight:** A state can be legal but unreachable. For example, a specific pawn structure might follow all the rules of chess, but given the turn number, there's no possible sequence of moves that could produce this arrangement.

**Core Challenge:** Can I create a system that accurately classifies whether a given board state is reachable?


## Problem Statement

| Aspect         | Details                                                                                                      |
| -------------- | ------------------------------------------------------------------------------------------------------------ |
| **Task**       | Binary classification - Determine if a chess board state is possible or unreachable                          |
| **Scope**      | Identify states that are legal but unreachable (not structural illegality such as improper number of pieces) |
| **Key Inputs** | Board state, Turn number                                                                                     |

---

## Related Work

There exist approaches close to what I'm looking for, most notably- retrograde analysis, where mathematically one searches and computes for a sequence of moves to reach a given board state. However this is not computationally possible beyond around 25 turns. A similar story can be seen on the other end of the spectrum - chess endgames are a mature area of research, and chess is considered "solved" when there are 7 or fewer pieces on the board, that is not to say there do not exist unreachable arrangements with such piece counts, but my paper will not delve into this area, but rather seek to ammend the mid game gap between these two fields.

The problem as a whole is PSPACE-complete. A learned approximator cuts out the long computational process for each check that the standard retrograde analysis methods use.

### Existing Approaches
- **Retrograde Analysis** - Focuses on pawn structure analysis
- **Proof Games** - Generates a history of how a game reached a position (computationally infeasible at scale)
- **Exhaustive Search** - Mathematical methods that search all possible game arrangements (again - computationally infeasible at scale)

### Key Resources
- [Natch Proof Game Solver](http://natch.free.fr/Natch.html)
- [Texel Proof Game Documentation](https://github.com/peterosterlund2/texel/blob/master/doc/proofgame.md)
- [Chess Neural Networks Research](https://theses.liacs.nl/pdf/2022-2023-AlwerSaleh.pdf)
- [Problem Size Estimation](https://univ-avignon.hal.science/hal-03483904)
- [Chess 960 Lichess Dataset](https://www.kaggle.com/datasets/alexmolas/chess-960-lichess) - All back row pieces randomized at start

### Problem Complexity Estimates

We know unreachable boards exist, but just how many are there? Thankfully others have calculated this, giving us a good picture of the scale.

| Estimate                                   | Value        | Notes                                   |
| ------------------------------------------ | ------------ | --------------------------------------- |
| Shannon's piece-count-respecting estimate  | ~4.63 × 10⁴² | Correct piece types, ignores illegality |
| Legal diagrams (upper bound, no promotion) | ~4 × 10³⁷    | Gourion's result                        |

Meaning there are roughly 100 000 times more possible boards than reachable boards

## Dataset

| Type                   | Source                   | Status |
| ---------------------- | ------------------------ | ------ |
| **Possible states**    | download lichess dataset |
| **Unreachable states** | synthetically generated  |

### Unreachable States Generation

Apply perturbations to **reachable** positions (e.g. from simulated or real games) to produce synthetic unreachable states. Core perturbation ideas by category:

#### 1. Pawn structure

- **Pawn swap** — Swap two same-color pawns on different files (breaks file history).
- **Double pawn** — Copy a pawn to an adjacent file so two same-color pawns share a file.
- **Pawn teleport** — Move a pawn to another file or rank without a legal path (e.g. e2→e5 in one step).
- **Wrong count / placement** — Add/remove pawns, or put a pawn on rank 1/8 (without promotion), or create impossible passed/blocked structures.

#### 2. Turn number

- **Turn truncation** — Take a position from move 40 and label it as move 10 (or 5); many arrangements are impossible that early.
- **Turn inflation** — Label an early-looking position (e.g. many pawns, few pieces moved) as move 60.
- **Minimum-move violation** — Set turn number below the minimum moves needed (e.g. for knights to leave the back rank, or for the given piece layout).

#### 3. Piece mobility / placement

- **Piece swap (same type)** — Swap two identical pieces (e.g. two knights) so at least one couldn’t have reached its square.
- **Knight jump** — Move a knight to a non-knight square (same colour as origin).
- **Bishop colour** — Put a bishop on the wrong colour complex.
- **King in check (illegal last move)** — Add or move a piece so the side not to move has their king in check.
- **Promotion contradiction** — Add an extra queen (or piece) that implies promotion, but remove/block pawns so that many promotions are impossible; or wrong piece counts (e.g. two queens, one promotion possible).
- **Castling / en passant** — Set castling rights or en passant when the board doesn’t allow it (e.g. king/rook moved, or no pawn just moved two).

#### 4. Structural / cross-cutting

- **Side swap** — Mirror the board or swap colours and keep the same turn number.
- **FEN corruption** — Small edits to a valid FEN (castling, en passant, a digit) to get parseable but unreachable positions.

*Implementation priority: start with pawn swap, turn truncation/inflation, knight square and bishop colour — then add double-pawn, king-in-check, and promotion contradictions for stronger signal.*

---

## Predicted Impossibility Types

The system should detect the following types of unreachable board states:

1. **Pawn Structure Violation**
   - Unreachable pawn configurations given the move history (e.g. doubled pawns, wrong file, impossible passed pawns).

2. **State Beyond Turn Number Possibility**
   - Board state unreachable within the given number of turns (too many pieces moved for the count, or too few).

3. **Piece Mobility Violations**
   - Piece configurations that contradict piece movement rules over time (wrong squares for knights/bishops, illegal last move, impossible castling or promotions).

### Alternative Approaches

A potential alternative is simulated chess variant games. These would naturally tend towards unreachable states, however these is no guarantee that for example a crazyhouse chess game state would not be reachable via normal chess play. So this approach is falling to the way side in terms of priority.



## Data Pipeline

```mermaid
graph TD
    A["Raw Possible Dataset<br/>(100M boards)"] --> B["Clean Possible Dataset"]
    B --> C["Analyze Dataset<br/>Distributions & Characteristics"]
    
    D["Generate unreachable<br/>Dataset"] --> E["Clean unreachable Dataset"]
    E --> C
    
    C --> F["Split for<br/>Cross Validation"]
    F --> G["Train Model"]
    G --> H["Test Model"]
    H --> I["Evaluate Results"]
    
    style A fill:#e1f5ff
    style D fill:#fff3e0
    style I fill:#e8f5e9
```

---

## Additional Goals

- [ ] Create a web interface for position evaluation
- [ ] Deploy as a hosted website for public access
- [ ] Integrate legal move validation alongside possibility checks

Feature Importance Visualization: Use SHAP or LIME to show which squares or pieces the Neural Network is looking at when it decides a position is unreachable.



move number is normalized to 0-1, this is done by dividing the move number by an arbitrary maximum turn number (200), this exact figure should not matter that much, as we will cut off endgames based off of pieces left anyway.