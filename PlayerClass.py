"""
Imports
"""
from collections import Counter
from copy import deepcopy
"""
Blackjack Player Class
"""
class Player:

    # Constructor
    def __init__(self, initialchip: int):
        self.ChipCount = initialchip

    # Hit action
    def Hit(self):
        pass

    # Stay action
    def Stay(self):
        pass

    # Double action
    def Double(self):
        pass

    # Split action
    def Split(self):
        pass

    # Calculate Sum
    def GetSum(self):
        pass

    # Chip count
    def GetChipCoint(self):
        pass


# Expectiminimax Player

class Player_EMM:
    def __init__(self, depth=3):
        self.depth = depth
        self.full_deck = [2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10, 11] * 4

    def best_hand(self, hand):
        total = sum(hand)
        aces = hand.count(11)
        while total > 21 and aces > 0:
            total -= 10
            aces -= 1
        return total

    def is_bust(self, hand):
        return self.best_hand(hand) > 21

    def evaluate(self, player_hand, dealer_hand):
        player_score = self.best_hand(player_hand)
        dealer_score = self.best_hand(dealer_hand)

        if player_score > 21:
            return -1
        elif dealer_score > 21:
            return 1
        elif player_score > dealer_score:
            return 1
        elif player_score < dealer_score:
            return -1
        else:
            return 0

    def get_deck_counter(self, deck, exclude):
        # Count remaining cards excluding those already drawn
        deck_counter = Counter(deck)
        for card in exclude:
            deck_counter[card] -= 1
        return {k: v for k, v in deck_counter.items() if v > 0}

    def expectiminimax(self, player_hand, dealer_hand, deck, depth, is_player_turn):
        if depth == 0 or self.is_bust(player_hand):
            return self.evaluate(player_hand, dealer_hand)

        if is_player_turn:
            # Player's decision: choose best between hit or stand
            hit_value = self.expect_hit(player_hand, dealer_hand, deck, depth)
            stand_value = self.expectiminimax(player_hand, dealer_hand, deck, depth - 1, False)
            return max(hit_value, stand_value)
        else:
            # Dealer's turn: draw all possible cards
            dealer_score = self.best_hand(dealer_hand)
            if dealer_score >= 17:
                return self.evaluate(player_hand, dealer_hand)
            else:
                expected_value = 0
                deck_counter = self.get_deck_counter(deck, player_hand + dealer_hand)
                total_cards = sum(deck_counter.values())

                for card, count in deck_counter.items():
                    prob = count / total_cards
                    new_dealer_hand = dealer_hand + [card]
                    new_deck = deepcopy(deck)
                    new_deck.remove(card)
                    value = self.expectiminimax(player_hand, new_dealer_hand, new_deck, depth, False)
                    expected_value += prob * value
                return expected_value

    def expect_hit(self, player_hand, dealer_hand, deck, depth):
        expected_value = 0
        deck_counter = self.get_deck_counter(deck, player_hand + dealer_hand)
        total_cards = sum(deck_counter.values())

        for card, count in deck_counter.items():
            prob = count / total_cards
            new_player_hand = player_hand + [card]
            new_deck = deepcopy(deck)
            new_deck.remove(card)
            value = self.expectiminimax(new_player_hand, dealer_hand, new_deck, depth - 1, True)
            expected_value += prob * value

        return expected_value

    def choose_action(self, player_hand, dealer_card):
        # Initialize deck with removed known cards
        known_cards = player_hand + [dealer_card]
        deck = deepcopy(self.full_deck)
        for card in known_cards:
            deck.remove(card)

        # Expectiminimax decision
        hit_value = self.expect_hit(player_hand, [dealer_card], deck, self.depth)
        stand_value = self.expectiminimax(player_hand, [dealer_card], deck, self.depth - 1, False)

        print(f"Hit Value: {hit_value:.4f}, Stand Value: {stand_value:.4f}")
        return 'hit' if hit_value > stand_value else 'stand'


# Testing

if __name__ == "__main__":
    player = Player_EMM(depth=3)
    player_hand = [10, 6]          # Example hand: 10 and 6
    dealer_card = 6                # Dealer's visible card: 9

    action = player.choose_action(player_hand, dealer_card)
    print(f"Chosen action: {action}")