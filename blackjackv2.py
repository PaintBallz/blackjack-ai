from __future__ import annotations
from dataclasses import dataclass, replace
from collections import Counter
import random, math
from typing import Dict, Tuple, List, Optional, Iterable
import secrets

# =========================
# Blackjack core
# EV is used to measure effective value of search results.
# =========================

RANKS = ['2','3','4','5','6','7','8','9','10','J','Q','K','A']
# Per deck: 2-9 (4 each), 10/J/Q/K (16 tens + 4 J + 4 Q + 4 K = 28 tens/faces), A (4)
# We'll explicitly count: 10 has 16, J/Q/K each 4
DECK_COUNTS = {r:4 for r in RANKS}
DECK_COUNTS['10'] = 16  # override for tens

def make_shoe(decks: int = 6) -> Counter:
    shoe = Counter()
    for r, n in DECK_COUNTS.items():
        shoe[r] = n * decks
    return shoe

def card_value(rank: str) -> int:
    if rank == 'A': return 11
    if rank in ['10','J','Q','K']: return 10
    return int(rank)

def hand_value(cards: Tuple[str, ...]) -> Tuple[int, bool]:
    total = 0
    aces = 0
    for r in cards:
        if r == 'A':
            total += 11
            aces += 1
        else:
            total += card_value(r)
    while total > 21 and aces:
        total -= 10
        aces -= 1
    soft = aces > 0
    return total, soft

def is_blackjack(cards: Tuple[str, ...]) -> bool:
    return len(cards) == 2 and 'A' in cards and any(c in {'10','J','Q','K'} for c in cards)

A_HIT = "HIT"
A_STAND = "STAND"
A_DOUBLE = "DOUBLE"

@dataclass(frozen=True)
class BJState:
    to_move: str                       # 'Player' or 'Dealer'
    player_cards: Tuple[str, ...]
    dealer_cards: Tuple[str, ...]      # includes hole card
    shoe: Tuple[Tuple[str,int], ...]   # immutable Counter snapshot
    bet: int
    can_double: bool
    resolved: bool

    def shoe_counter(self) -> Counter:
        return Counter(dict(self.shoe))

class Blackjack:
    def __init__(self, decks: int = 6, dealer_hits_soft_17: bool = False):
        self.decks = decks
        self.H17 = dealer_hits_soft_17

    # --- dealing ---
    def _draw_from_shoe(self, shoe: Counter, rank: Optional[str] = None) -> Tuple[str, Counter]:
        if rank is None:
            total = sum(shoe.values())
            r = random.randrange(total)
            cum = 0
            for k, v in shoe.items():
                cum += v
                if r < cum:
                    rank = k
                    break
        if rank is None or shoe[rank] <= 0:
            raise RuntimeError("Bad draw")
        shoe2 = shoe.copy()
        shoe2[rank] -= 1
        if shoe2[rank] == 0:
            del shoe2[rank]
        return rank, shoe2

    def deal_initial(self) -> BJState:
        shoe = make_shoe(self.decks)
        p1, shoe = self._draw_from_shoe(shoe)
        d1, shoe = self._draw_from_shoe(shoe)
        p2, shoe = self._draw_from_shoe(shoe)
        d2, shoe = self._draw_from_shoe(shoe)
        s = BJState(
            to_move='Player',
            player_cards=(p1, p2),
            dealer_cards=(d1, d2),
            shoe=tuple(sorted(shoe.items())),
            bet=1,
            can_double=True,
            resolved=False
        )
        return self._maybe_resolve_naturals(s)

    def _maybe_resolve_naturals(self, s: BJState) -> BJState:
        if is_blackjack(s.player_cards) or is_blackjack(s.dealer_cards):
            return replace(s, resolved=True, to_move='Dealer')
        return s

    # --- rules interface ---
    def actions(self, s: BJState) -> Iterable[str]:
        if self.is_terminal(s) or s.to_move != 'Player':
            return []
        acts = [A_HIT, A_STAND]
        if s.can_double:
            acts.append(A_DOUBLE)
        return acts

    def is_terminal(self, s: BJState) -> bool:
        return s.resolved

    def utility_ev(self, s: BJState) -> float:
        """Payout in base-bet units (+1 win, -1 loss, +1.5 natural, push 0; double scales bet)."""
        pv, _ = hand_value(s.player_cards)
        dv, _ = hand_value(s.dealer_cards)
        p_nat = is_blackjack(s.player_cards)
        d_nat = is_blackjack(s.dealer_cards)
        if p_nat or d_nat:
            if p_nat and not d_nat: return 1.5 * s.bet
            if d_nat and not p_nat: return -1.0 * s.bet
            return 0.0
        if pv > 21: return -1.0 * s.bet
        if dv > 21: return 1.0 * s.bet
        if pv > dv: return 1.0 * s.bet
        if pv < dv: return -1.0 * s.bet
        return 0.0

    def utility_win(self, s: BJState) -> float:
        """Win-loss indicator (+1 win, 0 push, -1 loss)."""
        ev = self.utility_ev(s)
        if ev > 0: return 1.0
        if ev < 0: return -1.0
        return 0.0

    # --- transitions ---
    def result(self, s: BJState, move: str) -> BJState:
        if move not in self.actions(s):
            raise ValueError("Illegal move")
        shoe = s.shoe_counter()
        player = list(s.player_cards)

        if move == A_HIT:
            rank, shoe = self._draw_from_shoe(shoe)
            player.append(rank)
            pv, _ = hand_value(tuple(player))
            if pv > 21:
                # bust -> terminal
                return replace(s, player_cards=tuple(player),
                               shoe=tuple(sorted(shoe.items())),
                               to_move='Dealer', resolved=True, can_double=False)
            else:
                return replace(s, player_cards=tuple(player),
                               shoe=tuple(sorted(shoe.items())),
                               to_move='Player', can_double=False)

        if move == A_DOUBLE:
            rank, shoe = self._draw_from_shoe(shoe)
            player.append(rank)
            s2 = replace(s, player_cards=tuple(player),
                         shoe=tuple(sorted(shoe.items())),
                         bet=2, to_move='Dealer', can_double=False)
            return self._dealer_play(s2)

        if move == A_STAND:
            return self._dealer_play(replace(s, to_move='Dealer', can_double=False))

        raise RuntimeError("Unreachable")

    def _dealer_play(self, s: BJState) -> BJState:
        if s.resolved: return s
        shoe = s.shoe_counter()
        dealer = list(s.dealer_cards)
        while True:
            dv, soft = hand_value(tuple(dealer))
            if dv > 21: break
            if dv > 17: break
            if dv == 17 and not self.H17: break
            if dv == 17 and self.H17 and not soft: break
            # hit
            rank, shoe = self._draw_from_shoe(shoe)
            dealer.append(rank)
        return replace(s, dealer_cards=tuple(dealer), shoe=tuple(sorted(shoe.items())), resolved=True)

    # Chance expansion for expectiminimax HIT
    def chance_children_hit(self, s: BJState) -> List[Tuple[float, BJState]]:
        shoe = s.shoe_counter()
        total = sum(shoe.values())
        out = []
        for r, cnt in shoe.items():
            p = cnt / total
            new_shoe = shoe.copy()
            new_shoe[r] -= 1
            if new_shoe[r] == 0: del new_shoe[r]
            new_cards = tuple(list(s.player_cards) + [r])
            pv, _ = hand_value(new_cards)
            if pv > 21:
                ns = replace(s, player_cards=new_cards, shoe=tuple(sorted(new_shoe.items())),
                             to_move='Dealer', resolved=True, can_double=False)
            else:
                ns = replace(s, player_cards=new_cards, shoe=tuple(sorted(new_shoe.items())),
                             to_move='Player', can_double=False)
            out.append((p, ns))
        return out

# =========================
# Expectiminimax (maximize win probability)
# =========================

def expectiminimax_win(game: Blackjack, state: BJState, depth_limit: int = 6) -> Tuple[float, Optional[str]]:
    cache = {}
    NEG_INF = -1e9

    def eval_ev(s: BJState, depth: int) -> Tuple[float, Optional[str]]:
        key = (s, depth)
        if key in cache: return cache[key]

        # Cutoff: if at player node and depth exhausted, approximate with a quick heuristic
        if game.is_terminal(s) or (depth <= 0 and s.to_move == 'Player'):
            if not game.is_terminal(s) and s.to_move == 'Player':
                # quick heuristic: compare stand win-indicator vs. one-card-hit expectation
                stand_win = game.utility_win(game._dealer_play(replace(s, to_move='Dealer')))
                hit_exp = 0.0
                for p, ns in game.chance_children_hit(s):
                    if not ns.resolved and ns.to_move == 'Player':
                        # approximate by standing immediately after a safe hit
                        hit_exp += p * game.utility_win(game._dealer_play(replace(ns, to_move='Dealer')))
                    else:
                        hit_exp += p * game.utility_win(ns)
                val = max(stand_win, hit_exp)
                cache[key] = (val, None)
                return cache[key]
            # terminal
            val = game.utility_win(s)
            cache[key] = (val, None)
            return cache[key]

        if s.to_move == 'Player':
            best = NEG_INF
            best_a = None
            for a in game.actions(s):
                if a == A_HIT:
                    expv = 0.0
                    for p, ns in game.chance_children_hit(s):
                        v, _ = eval_ev(ns, depth)  # chance doesn't reduce depth
                        expv += p * v
                    val = expv
                else:
                    ns = game.result(s, a)
                    v, _ = eval_ev(ns, depth - 1)
                    val = v
                if val > best:
                    best, best_a = val, a
            cache[key] = (best, best_a)
            return cache[key]

        # Dealer
        ns = game._dealer_play(s)
        v, _ = eval_ev(ns, depth)
        cache[key] = (v, None)
        return cache[key]

    return eval_ev(state, depth_limit)

# =========================
# MCTS (UCT) with pluggable reward
# =========================

class MCTSNode:
    __slots__ = ("state","parent","children","N","W","untried")
    def __init__(self, state: BJState, parent: Optional[MCTSNode], actions: List[str]):
        self.state = state
        self.parent = parent
        self.children: Dict[str, MCTSNode] = {}
        self.N = 0
        self.W = 0.0
        self.untried = list(actions)

def mcts_choose(game: Blackjack, root_state: BJState, iters: int, reward_fn) -> str:
    root = MCTSNode(root_state, None, list(game.actions(root_state)))
    C = math.sqrt(2)

    def ucb(node: MCTSNode, child: MCTSNode) -> float:
        if child.N == 0: return float('inf')
        return (child.W / child.N) + C * math.sqrt(math.log(node.N + 1) / child.N)

    for _ in range(iters):
        # 1) Selection
        node = root
        state = node.state
        # auto-play dealer if it's dealer turn
        if state.to_move == 'Dealer' and not game.is_terminal(state):
            state = game._dealer_play(state)
            node = MCTSNode(state, node, list(game.actions(state)))

        while not game.is_terminal(state) and not node.untried and node.children:
            # best child by UCB
            a, child = max(node.children.items(), key=lambda kv: ucb(node, kv[1]))
            state = stochastic_step(game, state, a)
            node = child

        # 2) Expansion
        if not game.is_terminal(state) and node.untried:
            a = node.untried.pop()
            next_state = stochastic_step(game, state, a)
            child = MCTSNode(next_state, node, list(game.actions(next_state)))
            node.children[a] = child
            node = child
            state = next_state

        # 3) Simulation
        reward = rollout(game, state, reward_fn)

        # 4) Backprop
        while node is not None:
            node.N += 1
            node.W += reward
            node = node.parent

    if not root.children:
        return A_STAND
    # choose most visited
    return max(root.children.items(), key=lambda kv: kv[1].N)[0]

def stochastic_step(game: Blackjack, s: BJState, action: str) -> BJState:
    if action == A_HIT:
        shoe = s.shoe_counter()
        total = sum(shoe.values())
        r = random.randrange(total)
        cum = 0
        rank = None
        for k, v in shoe.items():
            cum += v
            if r < cum:
                rank = k; break
        player = list(s.player_cards)
        player.append(rank)
        shoe[rank] -= 1
        if shoe[rank] == 0: del shoe[rank]
        pv, _ = hand_value(tuple(player))
        if pv > 21:
            return replace(s, player_cards=tuple(player),
                           shoe=tuple(sorted(shoe.items())),
                           to_move='Dealer', resolved=True, can_double=False)
        else:
            return replace(s, player_cards=tuple(player),
                           shoe=tuple(sorted(shoe.items())),
                           to_move='Player', can_double=False)
    elif action in (A_STAND, A_DOUBLE):
        return game.result(s, action)
    else:
        raise ValueError("Unknown action")

def rollout(game: Blackjack, s: BJState, reward_fn) -> float:
    state = s
    while not game.is_terminal(state):
        if state.to_move == 'Dealer':
            state = game._dealer_play(state)
            break
        # Simple threshold policy; favors reasonable realism without lookup tables
        pv, soft = hand_value(state.player_cards)
        if state.can_double and 9 <= pv <= 11 and random.random() < 0.25:
            state = stochastic_step(game, state, A_DOUBLE)
        elif pv <= 11:
            state = stochastic_step(game, state, A_HIT)
        elif 12 <= pv <= 16:
            up = state.dealer_cards[0]
            upv = 11 if up == 'A' else card_value(up)
            if up == 'A' or upv >= 7:
                state = stochastic_step(game, state, A_HIT)
            else:
                state = stochastic_step(game, state, A_STAND)
        else:
            state = stochastic_step(game, state, A_STAND)
    return reward_fn(game, state)

# Reward functions
def reward_profit(game: Blackjack, s: BJState) -> float:
    return game.utility_ev(s)

def reward_win(game: Blackjack, s: BJState) -> float:
    return game.utility_win(s)

# =========================
# Policies for each agent
# =========================

def policy_mcts_profit(game: Blackjack, state: BJState, iters=3000) -> str:
    return mcts_choose(game, state, iters=iters, reward_fn=reward_profit)

def policy_mcts_win(game: Blackjack, state: BJState, iters=3000) -> str:
    return mcts_choose(game, state, iters=iters, reward_fn=reward_win)

def policy_expecti_win(game: Blackjack, state: BJState, depth=6) -> str:
    _, a = expectiminimax_win(game, state, depth_limit=depth)
    # Fallback if None at terminal (shouldn't be called)
    return a or A_STAND

# =========================
# Runner: play a full hand given a policy
# =========================

def play_full_hand(game: Blackjack, start: BJState, chooser) -> Tuple[BJState, List[str]]:
    s = start
    actions_taken: List[str] = []
    while not game.is_terminal(s):
        if s.to_move == 'Dealer':
            s = game._dealer_play(s)
            break
        a = chooser(game, s)
        actions_taken.append(a)
        if a == A_HIT:
            s = stochastic_step(game, s, A_HIT)
        else:
            s = game.result(s, a)
    return s, actions_taken

# =========================
# Tournament: 10 rounds, same initial state per round
# =========================

def clone_state(s: BJState) -> BJState:
    # dataclass is immutable; re-wrap to ensure distinct object identity
    return BJState(s.to_move, s.player_cards, s.dealer_cards, s.shoe, s.bet, s.can_double, s.resolved)

def run_comparison(rounds=10, iters=2500, depth=6, decks=4):
    game = Blackjack(decks=decks, dealer_hits_soft_17=False)

    stats = {
        'MCTS-Profit': {'ev':0.0,'win':0,'loss':0,'push':0},
        'MCTS-Win'   : {'ev':0.0,'win':0,'loss':0,'push':0},
        'Expecti-Win': {'ev':0.0,'win':0,'loss':0,'push':0},
    }

    for rnd in range(1, rounds+1):
        random.seed(secrets.randbits(64))
        base_shoe = make_shoe(decks)

        d1, shoe_after = game._draw_from_shoe(base_shoe)
        d2, shoe_after = game._draw_from_shoe(shoe_after)
        dealer_cards = (d1, d2)

        def deal_player_from(cloned_shoe):
            p1, cs = game._draw_from_shoe(cloned_shoe)
            p2, cs = game._draw_from_shoe(cs)
            s = BJState(
                to_move='Player',
                player_cards=(p1, p2),
                dealer_cards=dealer_cards,      # <- same dealer for all agents
                shoe=tuple(sorted(cs.items())),
                bet=1,
                can_double=True,
                resolved=False
            )
            return game._maybe_resolve_naturals(s)

        shoe1 = shoe_after.copy()
        shoe2 = shoe_after.copy()
        shoe3 = shoe_after.copy()

        s1 = deal_player_from(shoe1)  # MCTS-Profit
        s2 = deal_player_from(shoe2)  # MCTS-Win
        s3 = deal_player_from(shoe3)  # Expecti-Win

        # Play out each hand independently (same dealer cards)
        f1, a1 = play_full_hand(game, s1, lambda g, s: policy_mcts_profit(g, s, iters))
        f2, a2 = play_full_hand(game, s2, lambda g, s: policy_mcts_win(g, s, iters))
        f3, a3 = play_full_hand(game, s3, lambda g, s: policy_expecti_win(g, s, depth))

        # Update stats
        def upd(tag, final_state):
            ev = game.utility_ev(final_state)
            stats[tag]['ev'] += ev
            if ev > 0: stats[tag]['win'] += 1
            elif ev < 0: stats[tag]['loss'] += 1
            else: stats[tag]['push'] += 1

        upd('MCTS-Profit', f1)
        upd('MCTS-Win',    f2)
        upd('Expecti-Win', f3)

        # Round output (dealer printed separately) 
        print(f"\n=== Round {rnd} ===")
        dv,_ = hand_value(dealer_cards)
        print(f"Dealer: {dealer_cards} (value now unknown to player; totals shown post-round)")

        pv1,_ = hand_value(s1.player_cards)
        pv2,_ = hand_value(s2.player_cards)
        pv3,_ = hand_value(s3.player_cards)

        print(f"MCTS-Profit    start {s1.player_cards} ({pv1}) | actions {a1} -> EV {game.utility_ev(f1):+0.2f}, final={f1.player_cards} vs {f1.dealer_cards}")
        print(f"MCTS-Win       start {s2.player_cards} ({pv2}) | actions {a2} -> EV {game.utility_ev(f2):+0.2f}, final={f2.player_cards} vs {f2.dealer_cards}")
        print(f"Expecti-Win    start {s3.player_cards} ({pv3}) | actions {a3} -> EV {game.utility_ev(f3):+0.2f}, final={f3.player_cards} vs {f3.dealer_cards}")

    # Totals
    print("\n==== Totals ====")
    for tag, d in stats.items():
        print(f"{tag:13s} | EV sum={d['ev']:+0.2f} | W-L-P = {d['win']}-{d['loss']}-{d['push']}")

if __name__ == "__main__":
    # Tweak iters/depth as you like. More iterations => stronger MCTS.
    # Round count is also adjustable.
    run_comparison(rounds=50, iters=3000, depth=8, decks=6)

