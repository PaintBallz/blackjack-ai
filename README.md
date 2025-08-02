# Blackjack AI

This project implements a Blackjack game with AI players using **Expectiminimax** and **Monte Carlo** strategies. It includes a GUI to watch the game and player stats live.

## Features

- Standard Blackjack game engine with dealer and multiple players
- AI players using:
  - Expectiminimax search to find the best moves
  - Monte Carlo simulations to maximize wins or earnings
- GUI with live chip count updates and game stats visualization
- Betting with minimum $10 and maximum $50 per hand

## Files

- `Blackjack.py`: Core game classes like `Deck` and `Player`
- `PlayerClass.py`: AI player implementations (`Player_EMM`, `Player_MCWin`, `Player_MCEarn`)
- `blackjack_gui_stats.py`: GUI code to display stats and graphs
- `run.py`: Main script to run the game with AI and GUI

## How to Run

1. Install required packages (Tkinter usually included, install matplotlib if needed):

```bash
pip install matplotlib
