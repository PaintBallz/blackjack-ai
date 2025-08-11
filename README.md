# Blackjack AI (Console)

This repo contains a single-file Blackjack engine plus three AI agents:
- MCTS-Profit – Monte Carlo Tree Search maximizing expected chips (EV)
- MCTS-Win – Monte Carlo Tree Search maximizing win probability
- Expectiminimax-Win – Expectiminimax maximizing win probability

The program runs in the console and prints per-round actions, results, chip deltas, and final stacks.

## Features

- Standard Blackjack rules
  - Dealer stands on all 17 (S17).
  - Aces count as 11, then convert to 1 as needed to avoid busting.
  - K/Q/J/10 count as 10.
  - Actions: HIT, STAND, DOUBLE (no splits), Taking and Skipping Insurance.
  - Naturals (player blackjack) pay 3:2 on the base bet only (double does not apply).
  - If dealer busts and the player ≤ 21, the player wins automatically.
  - 
- Shared dealer per roun
  - All agents in a round face the same dealer hand (upcard + committed hole + precompute - hisequence)
  - Dealer’s hole is hidden from agents; it’s revealed in output after play
- Two distinct MCTS agent
  - Profit rollout is a bit aggressive (may DOUBLE on 9–11)
  - Win rollout is conservative (never DOUBLE), biases away from busts
  - Independent RNGs so tie situations don’t mirror decisions
- Expectiminimax agen
  - Chance nodes on HIT expand by card probability from the shoe
  - Objective is win indicator (+1 / 0 / −1)
- Configurable tournamen
  - rounds (hands to play)
  - iters (MCTS iterations per decision)
  - depth (Expectiminimax depth)
  - decks (shoe size)
  - starting_chips (per agent)
  - base_bet (static bet; DOUBLE uses 2× base_bet)
- Clear console output
  - Per round: dealer upcard, shared final dealer hand/total, each agent’s starting hand/total, chosen actions, WIN/LOSS/PUSH, chip Δ, running stack, and final player total.

## Files
- Single file (e.g., blackjack_ai.py) containing:
  - Core engine (Blackjack, BJState, rules & payouts)
  - MCTS-Profit and MCTS-Win implementations
  - Expectiminimax-Win
  - Tournament runner run_comparison(...)

## Requirements
- Python 3.9+
- No external dependencies (standard library only)

## How to Run
  1. Download the code file.
  2. Run:
  ```bash
  python blackjack_ai.py
  ```

## Configuration

Edit the call at the bottom of the file:

```bash
if __name__ == "__main__":
    run_comparison(
        rounds=25,              # number of hands to play
        iters=3000,             # MCTS iterations per decision
        depth=6,                # Expectiminimax search depth
        decks=6,                # number of decks in the shoe
        starting_chips=1000,    # starting chip count
        base_bet=100            # static bet per hand; DOUBLE uses 2× base_bet
    )
```
## Notes
- The dealer hand is precomputed once per round so every agent faces the same final dealer outcome—apple-to-apples comparisons.
