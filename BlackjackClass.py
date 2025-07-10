
"""
Imports here
"""
import PlayerClass


class Deck:

    def __init__(self, numdeck):
        self.Deck = self.BuildDeck(numdeck=numdeck)

    def DrawCard(self):
        pass


    def BuildDeck(self, numdeck: int) -> list:
        """
        Build deck function
        :param numdeck:
        :return:
        """

        # List of cards
        cardlist = []

        # Loop through deck number and add to cardList


        # Return here
        return cardlist

    def RemoveCard(self, cardnum):
        pass


"""
Blackjack class that handles the game play.
"""
class Blackjack:

    # Constructor
    def __init__(self, players: PlayerClass, dealer: PlayerClass, deck: Deck):
        self.players = players
        self.Deck = deck

    # Deal Cards
    def Deal(self):
        """
        Deal 2 cards per player
        :return:
        """

        # Loop through players twice!
        for player in self.players:
            pass

    # Update State
    def UpdateState(self):
        pass

    # Get State
    def GetState(self):
        pass

    # Play hand
    def PlayHand(self):
        pass

