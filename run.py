import tkinter as tk
from PlayerClass import Player_EMM, Player_MCWin, Player_MCEarn
from Blackjack import Deck, Blackjack
from blackjack_gui_stats import BlackjackStatsGUI

def main():
    # Setup tkinter root window
    root = tk.Tk()

    # Create AI players with initial chips
    emm_player = Player_EMM()
    mc_win_player = Player_MCWin()
    mc_earn_player = Player_MCEarn()

    # Wrap players in a list
    players = [emm_player, mc_win_player, mc_earn_player]

    # Initialize deck with 1 standard deck
    deck = Deck(1)

    # Create GUI stats window
    gui = BlackjackStatsGUI(root, [p.__class__.__name__ for p in players])

    # Create Blackjack game manager with AI players and GUI
    game = Blackjack(players, deck, gui)

    def play_round():
        game.PlayHand()  # Play a hand with all AI players
        root.after(2000, play_round)  # Run next hand in 2 seconds

    # Start playing rounds automatically every 2 seconds
    root.after(1000, play_round)
    root.mainloop()

if __name__ == "__main__":
    main()
