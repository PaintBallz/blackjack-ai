from dataclasses import dataclass, replace
from collections import Counter, defaultdict
from typing import Dict, Tuple, List, Optional, Iterable
import random, math, secrets

# =========================
# Blackjack core
# =========================

RANKS = ['2','3','4','5','6','7','8','9','10','J','Q','K','A']

def make_shoe(decks: int = 6) -> Counter:
    shoe = Counter()
    for rank in RANKS:
        shoe[rank] = 4 * decks
    return shoe

def card_value(rank: str) -> int:
    if rank == 'A': return 11
    if rank in ['10','J','Q','K']: return 10
    return int(rank)

def hand_value(cards: Tuple[str, ...]) -> Tuple[int, bool]:
    total, ace_count = 0, 0
    for rank in cards:
        if rank == 'A':
            total += 11; ace_count += 1
        elif rank in {'10','J','Q','K'}:
            total += 10
        else:
            total += int(rank)
    while total > 21 and ace_count:
        total -= 10; ace_count -= 1
    return total, (ace_count > 0)

def is_blackjack(cards: Tuple[str, ...]) -> bool:
    return len(cards) == 2 and 'A' in cards and any(c in {'10','J','Q','K'} for c in cards)

A_HIT, A_STAND, A_DOUBLE, A_INSURANCE, A_SKIP_INSURANCE = (
    "HIT","STAND","DOUBLE","INSURANCE","SKIP_INSURANCE"
)

@dataclass(frozen=True)
class BlackjackState:
    to_move: str
    player_cards: Tuple[str, ...]
    dealer_cards: Tuple[str, ...]
    shoe: Tuple[Tuple[str,int], ...]
    base_bet: int
    bet_multiplier: int
    can_double: bool
    resolved: bool
    insurance_bet: int = 0
    insurance_allowed: bool = False
    def shoe_counter(self) -> Counter: return Counter(dict(self.shoe))

class Blackjack:
    def __init__(self, decks: int = 6, dealer_hits_soft_17: bool = False):
        self.decks = decks
        self.dealer_hits_soft_17 = dealer_hits_soft_17
        self._round_hole: Optional[str] = None
        self._round_final_dealer: Optional[Tuple[str, ...]] = None

    def _draw_from_shoe(self, shoe: Counter, rank: Optional[str] = None) -> Tuple[str, Counter]:
        if rank is None:
            total_cards = sum(shoe.values())
            draw_index = random.randrange(total_cards)
            cumulative = 0
            for candidate_rank, count in shoe.items():
                cumulative += count
                if draw_index < cumulative:
                    rank = candidate_rank
                    break
        if rank is None or shoe[rank] <= 0: raise RuntimeError("Invalid draw")
        updated_shoe = shoe.copy(); updated_shoe[rank] -= 1
        if updated_shoe[rank] == 0: del updated_shoe[rank]
        return rank, updated_shoe

    def _precompute_dealer_hand(self, upcard: str, shoe_after_dealer: Counter):
        dealer_hand = [upcard, self._round_hole]; shoe = shoe_after_dealer.copy()
        while True:
            total, soft = hand_value(tuple(dealer_hand))
            if total > 17 or (total==17 and not (soft and self.dealer_hits_soft_17)): break
            r, shoe = self._draw_from_shoe(shoe); dealer_hand.append(r)
        self._round_final_dealer = tuple(dealer_hand)

    def actions(self, state: BlackjackState) -> Iterable[str]:
        if self.is_terminal(state) or state.to_move!='Player': return []
        if state.insurance_allowed: return [A_INSURANCE, A_SKIP_INSURANCE]
        acts=[A_HIT,A_STAND]; 
        if state.can_double: acts.append(A_DOUBLE)
        return acts

    def is_terminal(self, state: BlackjackState) -> bool: return state.resolved

    def _maybe_resolve_naturals(self, state: BlackjackState) -> BlackjackState:
        if state.insurance_allowed: return state
        if is_blackjack(state.player_cards) or is_blackjack((state.dealer_cards[0], self._round_hole)):
            return replace(state, resolved=True, to_move='Dealer')
        return state

    def result(self, state: BlackjackState, action: str) -> BlackjackState:
        shoe_counter = state.shoe_counter(); cards=list(state.player_cards)
        if action==A_INSURANCE:
            return self._maybe_resolve_naturals(replace(state, insurance_bet=state.base_bet//2, insurance_allowed=False))
        if action==A_SKIP_INSURANCE:
            return self._maybe_resolve_naturals(replace(state, insurance_allowed=False))
        if action==A_HIT:
            r, shoe_counter = self._draw_from_shoe(shoe_counter); cards.append(r)
            total,_=hand_value(tuple(cards))
            if total>21:
                return replace(state, player_cards=tuple(cards), shoe=tuple(sorted(shoe_counter.items())),
                               to_move='Dealer', resolved=True, can_double=False)
            return replace(state, player_cards=tuple(cards), shoe=tuple(sorted(shoe_counter.items())),
                           to_move='Player', can_double=False)
        if action==A_DOUBLE:
            r, shoe_counter = self._draw_from_shoe(shoe_counter); cards.append(r)
            return self._dealer_play(replace(state, player_cards=tuple(cards),
                                             shoe=tuple(sorted(shoe_counter.items())),
                                             bet_multiplier=2, to_move='Dealer', can_double=False))
        if action==A_STAND:
            return self._dealer_play(replace(state, to_move='Dealer', can_double=False))
        raise RuntimeError("Illegal")

    def _dealer_play(self, state: BlackjackState) -> BlackjackState:
        if state.resolved: return state
        return replace(state, dealer_cards=self._round_final_dealer or (state.dealer_cards[0], self._round_hole),
                       resolved=True)

    def utility_ev(self, state: BlackjackState) -> float:
        base=state.base_bet; mult=state.bet_multiplier
        player_total,_=hand_value(state.player_cards)
        dealer_tuple=self._round_final_dealer or (state.dealer_cards[0], self._round_hole)
        dealer_total,_=hand_value(dealer_tuple)
        pBJ=is_blackjack(state.player_cards); dBJ=is_blackjack(dealer_tuple)
        ins = 0.0
        if state.insurance_bet: ins=(2*state.insurance_bet) if dBJ else -state.insurance_bet
        if pBJ or dBJ:
            if pBJ and not dBJ: return 1.5*base + ins
            if dBJ and not pBJ: return -base + ins
            return 0+ins
        wager=base*mult
        if player_total>21: return -wager+ins
        if dealer_total>21: return +wager+ins
        if player_total>dealer_total: return +wager+ins
        if player_total<dealer_total: return -wager+ins
        return 0+ins

    def utility_win(self,state:BlackjackState)->float:
        d=self.utility_ev(state); return 1 if d>0 else (-1 if d<0 else 0)

# ==================================================
# Action logging
# ==================================================

action_log = defaultdict(lambda: defaultdict(lambda: defaultdict(Counter)))

def record_action(agent: str, dealer_up: str, player_cards: Tuple[str,...], action: str):
    total,_=hand_value(player_cards)
    if 2<=total<=20: action_log[agent][dealer_up][total][action]+=1

def print_strategy_tables():
    for agent, dealer_dict in action_log.items():
        print(f"\n=== Strategy Table: {agent} ===")
        header=["Dealer\\Player"]+[str(t) for t in range(2,21)]
        print(" | ".join(f"{h:>6s}" for h in header))
        print("-"*(8*len(header)))
        for dealer_up in RANKS:
            row=[f"{dealer_up:>6s}"]
            for total in range(2,21):
                counts=dealer_dict.get(dealer_up,{}).get(total,{})
                if not counts:
                    cell="  -  "
                else:
                    # Pick the single most common action
                    cell=max(counts.items(), key=lambda x:x[1])[0][0].upper()
                row.append(f"{cell:>6s}")
            print(" | ".join(row))

# ==================================================
# Simple Expectiminimax & MCTS (Win / Profit)
# ==================================================

def expectiminimax_win(game: Blackjack, state: BlackjackState, depth=2) -> str:
    if depth==0 or game.is_terminal(state): return None
    best_val=-999; best_act=None
    for a in game.actions(state):
        new=game.result(state,a)
        val=game.utility_win(new) if game.is_terminal(new) else -expectiminimax_val(game,new,depth-1)
        if val>best_val: best_val=val; best_act=a
    return best_act or A_STAND

def expectiminimax_val(game,state,depth):
    if game.is_terminal(state): return game.utility_win(state)
    best=-999
    for a in game.actions(state):
        new=game.result(state,a)
        val=game.utility_win(new) if game.is_terminal(new) else -expectiminimax_val(game,new,depth-1)
        best=max(best,val)
    return best

class MCTSNode:
    def __init__(self,state:BlackjackState,parent=None,action=None):
        self.state=state; self.parent=parent; self.action=action
        self.children=[]; self.visits=0; self.value=0.0

def mcts_policy(game: Blackjack, state: BlackjackState, utility_fn, iters=200) -> str:
    root=MCTSNode(state)
    for _ in range(iters):
        node=root
        while node.children and not game.is_terminal(node.state):
            node=max(node.children,key=lambda c:c.value/(c.visits+1e-9)+1.4*math.sqrt(math.log(node.visits+1)/(c.visits+1e-9)))
        if not game.is_terminal(node.state):
            for a in game.actions(node.state):
                node.children.append(MCTSNode(game.result(node.state,a),node,a))
        leaf=random.choice(node.children) if node.children else node
        s=leaf.state
        while not game.is_terminal(s):
            acts=list(game.actions(s))
            if not acts: break
            s=game.result(s,random.choice(acts))
        reward=utility_fn(s)
        while leaf:
            leaf.visits+=1; leaf.value+=reward; leaf=leaf.parent
    if not root.children: return A_STAND
    return max(root.children,key=lambda c:c.visits).action

def mcts_profit(game,state): return mcts_policy(game,state,game.utility_ev,200)
def mcts_win(game,state): return mcts_policy(game,state,game.utility_win,200)

# ==================================================
# Runner
# ==================================================

def play_full_hand(game: Blackjack, start: BlackjackState, policy_fn, *, agent_name:str, auto_skip_insurance=False):
    s=start
    while not game.is_terminal(s):
        if s.to_move=='Dealer': s=game._dealer_play(s); break
        if auto_skip_insurance and s.insurance_allowed:
            s=game.result(s,A_SKIP_INSURANCE); continue
        a=policy_fn(game,s)
        record_action(agent_name,s.dealer_cards[0],s.player_cards,a)
        s=game.result(s,a)
    return s

def run_comparison(rounds=100):
    agents=[("Expectiminimax-Win",expectiminimax_win),
            ("MCTS-Profit",mcts_profit),
            ("MCTS-Win",mcts_win)]
    bankroll={name:0 for name,_ in agents}
    wins={name:0 for name,_ in agents}
    losses={name:0 for name,_ in agents}
    for _ in range(rounds):
        game=Blackjack(decks=1)
        shoe=make_shoe(1)
        dealer_up,shoe=game._draw_from_shoe(shoe)
        dealer_hole,shoe=game._draw_from_shoe(shoe)
        game._round_hole=dealer_hole
        game._precompute_dealer_hand(dealer_up,shoe)
        for name,policy in agents:
            pc1,shoe=game._draw_from_shoe(shoe)
            pc2,shoe=game._draw_from_shoe(shoe)
            start=BlackjackState("Player",(pc1,pc2),(dealer_up,),tuple(sorted(shoe.items())),10,1,True,False,insurance_allowed=(dealer_up=="A"))
            end=play_full_hand(game,start,policy,agent_name=name,auto_skip_insurance=True)
            u=game.utility_ev(end); bankroll[name]+=u
            if u>0:wins[name]+=1
            elif u<0:losses[name]+=1
    for name in bankroll:
        print(f"{name}: bankroll {bankroll[name]:.2f}, W {wins[name]}, L {losses[name]}, T {rounds-wins[name]-losses[name]}")
    print_strategy_tables()

if __name__=="__main__":
    run_comparison(5000)