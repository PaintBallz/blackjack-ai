import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from collections import defaultdict


class BlackjackStatsGUI:
    def __init__(self, root, player_names):
        self.root = root
        self.root.title("Blackjack AI Strategy Visualization")

        self.stats_frame = ttk.LabelFrame(root, text="Statistics")
        self.stats_frame.pack(fill='both', expand=True)

        self.chip_history = defaultdict(list)
        self.result_counters = {name: {'Win': 0, 'Lose': 0, 'Push': 0} for name in player_names}

        self.fig, self.axs = plt.subplots(2, 1, figsize=(6, 6))
        self.fig.tight_layout(pad=3.0)

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.stats_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill='both', expand=True)

    def update_stats(self, chip_counts, game_results):
        # Update chip history and win/loss counters
        for name, chip in chip_counts.items():
            self.chip_history[name].append(chip)

        for name, result in game_results.items():
            if result in self.result_counters[name]:
                self.result_counters[name][result] += 1

        self.redraw()

    def redraw(self):
        # Clear plots
        self.axs[0].clear()
        self.axs[1].clear()

        # Chip Count Plot
        for name, chips in self.chip_history.items():
            self.axs[0].plot(chips, label=f"{name} Chips")
        self.axs[0].set_title("Chip Balance Over Time")
        self.axs[0].set_ylabel("Chips")
        self.axs[0].legend()
        self.axs[0].grid(True)

        # Win/Loss/Push Bar Plot
        names = list(self.result_counters.keys())
        win_vals = [self.result_counters[n]['Win'] for n in names]
        lose_vals = [self.result_counters[n]['Lose'] for n in names]
        push_vals = [self.result_counters[n]['Push'] for n in names]

        x = range(len(names))
        self.axs[1].bar(x, win_vals, width=0.2, label='Win', color='green', align='center')
        self.axs[1].bar([i + 0.2 for i in x], lose_vals, width=0.2, label='Lose', color='red', align='center')
        self.axs[1].bar([i + 0.4 for i in x], push_vals, width=0.2, label='Push', color='gray', align='center')
        self.axs[1].set_xticks([i + 0.2 for i in x])
        self.axs[1].set_xticklabels(names)
        self.axs[1].set_title("Game Outcomes")
        self.axs[1].legend()
        self.axs[1].grid(True)

        self.canvas.draw()

# Example usage (to be called from the Blackjack game engine loop)
if __name__ == '__main__':
    root = tk.Tk()
    gui = BlackjackStatsGUI(root, ["EMM", "MC-Win", "MC-Earn"])

    # Simulated test updates
    import random
    chip_counts = {"EMM": 100, "MC-Win": 100, "MC-Earn": 100}
    outcomes = ["Win", "Lose", "Push"]

    def simulate_game():
        for name in chip_counts:
            change = random.choice([-10, 0, 10])
            chip_counts[name] += change
        game_results = {name: random.choice(outcomes) for name in chip_counts}
        gui.update_stats(chip_counts, game_results)
        root.after(1000, simulate_game)  # repeat every 1s

    simulate_game()
    root.mainloop()
