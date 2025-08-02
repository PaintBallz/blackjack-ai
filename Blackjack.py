import random
from copy import deepcopy
from PlayerClass import Player, Player_EMM
from PlayerClass import Player_MCWin, Player_MCEarn

class Deck:
    def __init__(self, numdeck=1):
        self.Deck = self.BuildDeck(numdeck=numdeck)

    def DrawCard(self):
        if len(self.Deck) == 0:
            return None
        return self.Deck.pop(random.randrange(len(self.Deck)))

    def BuildDeck(self, numdeck: int) -> list:
        suits = ['Hearts', 'Diamonds', 'Clubs', 'Spades']
        ranks = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
        cardlist = []

        for _ in range(numdeck):
            for suit in suits:
                for rank in ranks:
                    cardlist.append((rank, suit))
        random.shuffle(cardlist)
        return cardlist

    def RemoveCard(self, cardnum):
        if 0 <= cardnum < len(self.Deck):
            return self.Deck.pop(cardnum)
        return None

    def CardsLeft(self):
        return len(self.Deck)

class Blackjack:
    def __init__(self, players: list, deck: Deck, gui=None):
        self.players = players
        self.Deck = deck
        self.gui = gui  # Optional GUI reference

        self.min_bet = 10
        self.max_bet = 50

    def Deal(self):
        """
        Deal two cards to each player and the dealer.
        """
        for _ in range(2):
            for player in self.players:
                card = self.Deck.DrawCard()
                player.add_card(card)
            self.dealer.add_card(self.Deck.DrawCard())

    def PlayHand(self):
        """
        Plays a single round of Blackjack for all players using their assigned AI strategies.
        """
        self.Deal()

        dealer_visible_card = self.dealer.hand[0]
        dealer_total = self.dealer.get_sum()

        for player in self.players:
            strategy = player.strategy
            visible_hand = player.get_hand_values()

            # Use the appropriate strategy
            if isinstance(strategy, Player_EMM):
                action = strategy.choose_action(visible_hand, dealer_visible_card)
            elif isinstance(strategy, Player_MCWin):
                action = strategy.choose_action(visible_hand, dealer_visible_card)
            elif isinstance(strategy, Player_MCEarn):
                action = strategy.choose_action(visible_hand, dealer_visible_card)
            else:
                action = "stay"  # Default fallback

            while action == "hit":
                player.add_card(self.Deck.DrawCard())
                visible_hand = player.get_hand_values()
                if player.get_sum() >= 21:
                    break
                action = strategy.choose_action(visible_hand, dealer_visible_card)

            # Determine and process result
            dealer_sum = self.dealer_play()
            self.resolve_outcome(player, dealer_sum)

        # GUI: Update stats window if attached
        if self.gui:
            self.gui.update_stats(self.players)

        # Reset hands
        for player in self.players:
            player.reset_hand()
        self.dealer.reset_hand()

    def dealer_play(self):
        """
        Dealer hits until total is 17 or more (soft 17 rules can be added if needed).
        """
        while self.dealer.get_sum() < 17:
            self.dealer.add_card(self.Deck.DrawCard())
        return self.dealer.get_sum()

    def resolve_outcome(self, player, dealer_sum):
        """
        Resolves the outcome of a single player vs dealer and updates chip count.
        """
        player_sum = player.get_sum()
        bet = random.randint(self.min_bet, self.max_bet)  # Simple random betting
        player.bet_history.append(bet)

        if player_sum > 21:
            player.chip_count -= bet
            player.result_history.append("Loss")
        elif dealer_sum > 21 or player_sum > dealer_sum:
            player.chip_count += bet
            player.result_history.append("Win")
        elif player_sum == dealer_sum:
            player.result_history.append("Push")
        else:
            player.chip_count -= bet
            player.result_history.append("Loss")