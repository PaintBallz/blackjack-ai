from collections import Counter
from copy import deepcopy
import random

# Base Player class for human or baseline AI
class Player:
    def __init__(self, initialchip: int, name="Player"):
        self.name = name
        self.ChipCount = initialchip
        self.Hand = []
        self.CurrentBet = 10  # default minimum
        self.results = {"win": 0, "loss": 0, "push": 0}
        self.earnings_history = []

    def add_card(self, card):
        self.Hand.append(card)

    def reset_hand(self):
        self.Hand = []

    def get_sum(self):
        total = 0
        aces = 0
        for rank, _ in self.Hand:
            if rank in ['J', 'Q', 'K']:
                total += 10
            elif rank == 'A':
                total += 11
                aces += 1
            else:
                total += int(rank)
        while total > 21 and aces:
            total -= 10
            aces -= 1
        return total

    def is_bust(self):
        return self.get_sum() > 21

    def win(self):
        self.ChipCount += self.CurrentBet
        self.results["win"] += 1
        self.earnings_history.append(self.ChipCount)

    def lose(self):
        self.ChipCount -= self.CurrentBet
        self.results["loss"] += 1
        self.earnings_history.append(self.ChipCount)

    def push(self):
        self.results["push"] += 1
        self.earnings_history.append(self.ChipCount)

    def choose_action(self, dealer_card):
        # Default player always stands if >= 17
        return 'hit' if self.get_sum() < 17 else 'stand'


# Expectiminimax-based strategy
class Player_EMM(Player):
    def __init__(self, name="EMM", depth=3):
        super().__init__(initialchip=1000, name=name)
        self.depth = depth
        self.full_deck = [2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10, 11] * 4

    def best_hand(self, hand):
        total = sum(hand)
        aces = hand.count(11)
        while total > 21 and aces:
            total -= 10
            aces -= 1
        return total

    def evaluate(self, player_hand, dealer_hand):
        ps = self.best_hand(player_hand)
        ds = self.best_hand(dealer_hand)
        if ps > 21:
            return -1
        elif ds > 21:
            return 1
        elif ps > ds:
            return 1
        elif ps < ds:
            return -1
        return 0

    def get_deck_counter(self, deck, exclude):
        deck_counter = Counter(deck)
        for card in exclude:
            deck_counter[card] -= 1
        return {k: v for k, v in deck_counter.items() if v > 0}

    def expectiminimax(self, player_hand, dealer_hand, deck, depth, is_player_turn):
        if depth == 0 or self.best_hand(player_hand) > 21:
            return self.evaluate(player_hand, dealer_hand)
        if is_player_turn:
            hit = self.expect_hit(player_hand, dealer_hand, deck, depth)
            stand = self.expectiminimax(player_hand, dealer_hand, deck, depth - 1, False)
            return max(hit, stand)
        else:
            dealer_total = self.best_hand(dealer_hand)
            if dealer_total >= 17:
                return self.evaluate(player_hand, dealer_hand)
            expected_value = 0
            counter = self.get_deck_counter(deck, player_hand + dealer_hand)
            total_cards = sum(counter.values())
            for card, count in counter.items():
                prob = count / total_cards
                new_hand = dealer_hand + [card]
                new_deck = deepcopy(deck)
                new_deck.remove(card)
                expected_value += prob * self.expectiminimax(player_hand, new_hand, new_deck, depth, False)
            return expected_value

    def expect_hit(self, player_hand, dealer_hand, deck, depth):
        expected = 0
        counter = self.get_deck_counter(deck, player_hand + dealer_hand)
        total_cards = sum(counter.values())
        for card, count in counter.items():
            prob = count / total_cards
            new_hand = player_hand + [card]
            new_deck = deepcopy(deck)
            new_deck.remove(card)
            expected += prob * self.expectiminimax(new_hand, dealer_hand, new_deck, depth - 1, True)
        return expected

    def choose_action(self, dealer_card):
        # Translate hand into simplified values
        mapped = []
        for rank, _ in self.Hand:
            if rank in ['J', 'Q', 'K']:
                mapped.append(10)
            elif rank == 'A':
                mapped.append(11)
            else:
                mapped.append(int(rank))

        deck = deepcopy(self.full_deck)
        for v in mapped + [dealer_card]:
            deck.remove(v)

        hit_val = self.expect_hit(mapped, [dealer_card], deck, self.depth)
        stand_val = self.expectiminimax(mapped, [dealer_card], deck, self.depth - 1, False)
        return 'hit' if hit_val > stand_val else 'stand'


# Monte Carlo simulation player (win probability)
class Player_MCWin(Player):
    def __init__(self, name="MC-Win"):
        super().__init__(initialchip=1000, name=name)

    def simulate(self, hand, dealer_card, trials=1000):
        wins = 0
        for _ in range(trials):
            temp_hand = list(hand)
            while sum(temp_hand) < 17:
                temp_hand.append(random.choice([2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10, 11]))
                if sum(temp_hand) > 21 and 11 in temp_hand:
                    temp_hand[temp_hand.index(11)] = 1
            dealer = [dealer_card]
            while sum(dealer) < 17:
                dealer.append(random.choice([2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10, 11]))
                if sum(dealer) > 21 and 11 in dealer:
                    dealer[dealer.index(11)] = 1
            ps = sum(temp_hand)
            ds = sum(dealer)
            if ps > 21:
                continue
            if ds > 21 or ps > ds:
                wins += 1
        return wins / trials

    def choose_action(self, dealer_card):
        base = []
        for rank, _ in self.Hand:
            if rank in ['J', 'Q', 'K']:
                base.append(10)
            elif rank == 'A':
                base.append(11)
            else:
                base.append(int(rank))

        win_prob_hit = self.simulate(base + [random.choice([2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10, 11])], dealer_card)
        win_prob_stand = self.simulate(base, dealer_card)
        return 'hit' if win_prob_hit > win_prob_stand else 'stand'


# Monte Carlo simulation player (max earnings)
class Player_MCEarn(Player_MCWin):
    def __init__(self, name="MC-Earn"):
        super().__init__(name)
        self.ChipCount = 1000

    def simulate_earnings(self, hand, dealer_card, trials=1000):
        outcomes = []
        for _ in range(trials):
            temp_hand = list(hand)
            while sum(temp_hand) < 17:
                temp_hand.append(random.choice([2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10, 11]))
                if sum(temp_hand) > 21 and 11 in temp_hand:
                    temp_hand[temp_hand.index(11)] = 1
            dealer = [dealer_card]
            while sum(dealer) < 17:
                dealer.append(random.choice([2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10, 11]))
                if sum(dealer) > 21 and 11 in dealer:
                    dealer[dealer.index(11)] = 1
            ps = sum(temp_hand)
            ds = sum(dealer)
            if ps > 21:
                outcomes.append(-10)
            elif ds > 21 or ps > ds:
                outcomes.append(10)
            elif ps == ds:
                outcomes.append(0)
            else:
                outcomes.append(-10)
        return sum(outcomes) / trials

    def choose_action(self, dealer_card):
        base = []
        for rank, _ in self.Hand:
            if rank in ['J', 'Q', 'K']:
                base.append(10)
            elif rank == 'A':
                base.append(11)
            else:
                base.append(int(rank))

        hit_val = self.simulate_earnings(base + [random.choice(self.full_deck)], dealer_card)
        stand_val = self.simulate_earnings(base, dealer_card)
        return 'hit' if hit_val > stand_val else 'stand'
