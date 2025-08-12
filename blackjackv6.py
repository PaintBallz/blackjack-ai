from dataclasses import dataclass, replace
from collections import Counter
from typing import Dict, Tuple, List, Optional, Iterable
import random, math, secrets

# =========================
# Blackjack core
# Insuance Implementation
# Same Player Hands for Players
# =========================

RANKS = ['2','3','4','5','6','7','8','9','10','J','Q','K','A']

def make_shoe(decks: int = 6) -> Counter:
    """Standard 52-card deck counts per rank, times `decks`."""
    shoe = Counter()
    for r in RANKS:
        shoe[r] = 4 * decks  # each rank has 4 per deck
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
        elif r in {'10', 'J', 'Q', 'K'}:
            total += 10
        else:
            total += int(r)
    while total > 21 and aces:
        total -= 10
        aces -= 1
    is_soft = (aces > 0)
    return total, is_soft

def is_blackjack(cards: Tuple[str, ...]) -> bool:
    return len(cards) == 2 and 'A' in cards and any(c in {'10','J','Q','K'} for c in cards)

A_HIT = "HIT"
A_STAND = "STAND"
A_DOUBLE = "DOUBLE"
A_INSURANCE = "INSURANCE"           # NEW: take insurance (half bet)
A_SKIP_INSURANCE = "SKIP_INSURANCE" # NEW: decline insurance

@dataclass(frozen=True)
class BJState:
    to_move: str
    player_cards: Tuple[str, ...]
    dealer_cards: Tuple[str, ...]      # (upcard, None) during play; final tuple after resolution
    shoe: Tuple[Tuple[str,int], ...]   # immutable snapshot of remaining shoe
    base_bet: int                      # static bet (configurable in run_comparison)
    bet_mult: int                      # 1 normally, 2 after DOUBLE
    can_double: bool
    resolved: bool
    insurance_bet: int = 0             # 0 if no insurance; otherwise base_bet // 2
    insurance_allowed: bool = False    # whether insurance decision is pending now

    def shoe_counter(self) -> Counter:
        return Counter(dict(self.shoe))

class Blackjack:
    """Blackjack environment. Dealer stands on all 17 (S17 by default)."""
    def __init__(self, decks: int = 6, dealer_hits_soft_17: bool = False):
        self.decks = decks
        self.H17 = dealer_hits_soft_17
        self._round_hole: Optional[str] = None
        self._round_dealer_hits: Tuple[str, ...] = ()
        self._round_final_dealer: Optional[Tuple[str, ...]] = None

    # --- dealing & shoes ---
    def _draw_from_shoe(self, shoe: Counter, rank: Optional[str] = None) -> Tuple[str, Counter]:
        """Draw specific rank (if provided) or sample proportional to counts."""
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
            raise RuntimeError("Invalid draw")
        shoe2 = shoe.copy()
        shoe2[rank] -= 1
        if shoe2[rank] == 0:
            del shoe2[rank]
        return rank, shoe2

    def _precompute_dealer_hand(self, upcard: str, shoe_after_dealer: Counter):
        dealer = [upcard, self._round_hole]
        shoe = shoe_after_dealer.copy()
        # Dealer hits while total < 17; stands on 17+
        while True:
            dv, _ = hand_value(tuple(dealer))
            if dv >= 17:
                break
            rank, shoe = self._draw_from_shoe(shoe)
            dealer.append(rank)
        self._round_dealer_hits = tuple(dealer[2:])
        self._round_final_dealer = tuple(dealer)

    # --- rules interface ---
    def actions(self, s: BJState) -> Iterable[str]:
        if self.is_terminal(s) or s.to_move != 'Player':
            return []
        # If insurance decision is pending, only offer insurance choices
        if s.insurance_allowed:
            return [A_INSURANCE, A_SKIP_INSURANCE]
        acts = [A_HIT, A_STAND]
        if s.can_double:
            acts.append(A_DOUBLE)
        return acts

    def is_terminal(self, s: BJState) -> bool:
        return s.resolved

    def _maybe_resolve_naturals(self, s: BJState) -> BJState:
        # Do not resolve naturals until insurance decision has been made (if applicable)
        if s.insurance_allowed:
            return s
        p_nat = is_blackjack(s.player_cards)
        hole = s.dealer_cards[1] if len(s.dealer_cards) > 1 else None
        if hole is None:
            hole = self._round_hole
        d_nat = is_blackjack((s.dealer_cards[0], hole)) if hole is not None else False
        if p_nat or d_nat:
            return replace(s, resolved=True, to_move='Dealer')
        return s

    def result(self, s: BJState, move: str) -> BJState:
        if move not in self.actions(s):
            raise ValueError("Illegal move")
        shoe = s.shoe_counter()
        player = list(s.player_cards)

        # --- Insurance decision branch ---
        if move == A_INSURANCE:
            s2 = replace(
                s,
                insurance_bet=s.base_bet // 2,
                insurance_allowed=False  # decision made; now we may resolve naturals immediately
            )
            return self._maybe_resolve_naturals(s2)

        if move == A_SKIP_INSURANCE:
            s2 = replace(
                s,
                insurance_bet=0,
                insurance_allowed=False
            )
            return self._maybe_resolve_naturals(s2)

        # --- Regular actions ---
        if move == A_HIT:
            rank, shoe = self._draw_from_shoe(shoe)
            player.append(rank)
            pv, _ = hand_value(tuple(player))
            if pv > 21:
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
                         bet_mult=2, to_move='Dealer', can_double=False)
            return self._dealer_play(s2)

        if move == A_STAND:
            return self._dealer_play(replace(s, to_move='Dealer', can_double=False))

        raise RuntimeError("Unreachable")

    def _dealer_play(self, s: BJState) -> BJState:
        if s.resolved:
            return s
        final = self._round_final_dealer
        if final is None:  # safety fallback
            final = (s.dealer_cards[0], self._round_hole)
        return replace(s, dealer_cards=final, resolved=True)

    # --- payouts ---
    def utility_ev(self, s: BJState) -> float:
        """
        Returns chip delta for the player:
        - Naturals: +1.5 * base_bet (no double), or -base_bet if dealer natural only.
        - Otherwise: +/- (base_bet * bet_mult) or 0 on push.
        - Insurance (if taken): pays 2:1 if dealer has blackjack; else it's lost.
        """
        base = s.base_bet
        mult = s.bet_mult
        pv, _ = hand_value(s.player_cards)

        # Dealer tuple using committed hole if needed
        if len(s.dealer_cards) > 1 and s.dealer_cards[1] is not None:
            dealer_tuple = s.dealer_cards
        else:
            dealer_tuple = (s.dealer_cards[0], self._round_hole)

        dv, _ = hand_value(dealer_tuple)
        p_nat = is_blackjack(s.player_cards)
        d_nat = is_blackjack(dealer_tuple)

        # Insurance resolution
        ins = 0.0
        if s.insurance_bet:
            ins = (2.0 * s.insurance_bet) if d_nat else (-1.0 * s.insurance_bet)

        # Natural cases (resolved immediately)
        if p_nat or d_nat:
            if p_nat and not d_nat:
                main = 1.5 * base
            elif d_nat and not p_nat:
                main = -1.0 * base
            else:
                main = 0.0  # both have blackjack -> push main
            return main + ins

        # Regular play (no naturals)
        wager = base * mult
        if pv > 21: return -wager + ins
        if dv > 21: return +wager + ins
        if pv > dv: return +wager + ins
        if pv < dv: return -wager + ins
        return 0.0 + ins

    def utility_win(self, s: BJState) -> float:
        """+1 on win, 0 on push, -1 on loss."""
        delta = self.utility_ev(s)
        return 1.0 if delta > 0 else (-1.0 if delta < 0 else 0.0)

# =========================
# Expectiminimax (maximize win probability)
# =========================

def expectiminimax_win(game: Blackjack, state: BJState, depth_limit: int = 6) -> Tuple[float, Optional[str]]:
    cache: Dict[Tuple[BJState, int], Tuple[float, Optional[str]]] = {}
    NEG_INF = -1e9

    def chance_children_hit(s: BJState) -> List[Tuple[float, BJState]]:
        shoe = s.shoe_counter()
        total = sum(shoe.values())
        out: List[Tuple[float, BJState]] = []
        for r, cnt in shoe.items():
            p = cnt / total
            new_shoe = shoe.copy()
            new_shoe[r] -= 1
            if new_shoe[r] == 0: del new_shoe[r]
            new_cards = tuple(list(s.player_cards) + [r])
            pv, _ = hand_value(new_cards)
            if pv > 21:
                ns = replace(s, player_cards=new_cards,
                             shoe=tuple(sorted(new_shoe.items())),
                             to_move='Dealer', resolved=True, can_double=False)
            else:
                ns = replace(s, player_cards=new_cards,
                             shoe=tuple(sorted(new_shoe.items())),
                             to_move='Player', can_double=False)
            out.append((p, ns))
        return out

    def eval_ev(s: BJState, depth: int) -> Tuple[float, Optional[str]]:
        key = (s, depth)
        if key in cache:
            return cache[key]

        # Cutoff heuristic at player node
        if game.is_terminal(s) or (depth <= 0 and s.to_move == 'Player'):
            if not game.is_terminal(s) and s.to_move == 'Player':
                available = list(game.actions(s))
                # If insurance decision is pending at cutoff, evaluate both choices,
                # then from each resulting state approximate by (stand vs hit) heuristic.
                if A_INSURANCE in available or A_SKIP_INSURANCE in available:
                    best = -1e9
                    for a in available:
                        ns = game.result(s, a)
                        if game.is_terminal(ns):
                            v = game.utility_win(ns)
                        else:
                            stand_win = game.utility_win(game._dealer_play(replace(ns, to_move='Dealer')))
                            hit_exp = 0.0
                            for p, cns in chance_children_hit(ns):
                                if not cns.resolved and cns.to_move == 'Player':
                                    hit_exp += p * game.utility_win(game._dealer_play(replace(cns, to_move='Dealer')))
                                else:
                                    hit_exp += p * game.utility_win(cns)
                            v = max(stand_win, hit_exp)
                        if v > best: best = v
                    cache[key] = (best, None)
                    return cache[key]
                # Otherwise, approximate by stand vs hit from current state
                stand_win = game.utility_win(game._dealer_play(replace(s, to_move='Dealer')))
                hit_exp = 0.0
                for p, ns in chance_children_hit(s):
                    if not ns.resolved and ns.to_move == 'Player':
                        hit_exp += p * game.utility_win(game._dealer_play(replace(ns, to_move='Dealer')))
                    else:
                        hit_exp += p * game.utility_win(ns)
                val = max(stand_win, hit_exp)
                cache[key] = (val, None)
                return cache[key]
            val = game.utility_win(s)
            cache[key] = (val, None)
            return cache[key]

        if s.to_move == 'Player':
            best = NEG_INF
            best_a: Optional[str] = None
            for a in game.actions(s):
                if a == A_HIT:
                    expv = 0.0
                    for p, ns in chance_children_hit(s):
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

        # Dealer turn: deterministic (precomputed)
        ns = game._dealer_play(s)
        v, _ = eval_ev(ns, depth)
        cache[key] = (v, None)
        return cache[key]

    return eval_ev(state, depth_limit)

# =========================
# Monte Carlo Tree Search (two distinct variants)
# =========================

class MCTSNode:
    __slots__ = ("state","parent","children","N","W","untried")
    def __init__(self, state: BJState, parent: Optional['MCTSNode'], actions: List[str]):
        self.state = state
        self.parent = parent
        self.children: Dict[str, 'MCTSNode'] = {}
        self.N = 0
        self.W = 0.0
        self.untried = list(actions)

def stochastic_step_rng(game: Blackjack, s: BJState, action: str, rng: random.Random) -> BJState:
    """Apply action; sample draws from shoe using a provided RNG."""
    if action == A_HIT:
        shoe = s.shoe_counter()
        total = sum(shoe.values())
        r = rng.randrange(total)
        cum = 0
        rank = None
        for k, v in shoe.items():
            cum += v
            if r < cum:
                rank = k
                break
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
    elif action in (A_STAND, A_DOUBLE, A_INSURANCE, A_SKIP_INSURANCE):
        return game.result(s, action)
    else:
        raise ValueError("Unknown action")

# -------- Distinct rollout policies --------

def rollout_profit(game: Blackjack, s: BJState, rng: random.Random) -> float:
    """Profit-focused rollout: occasionally DOUBLE on 9–11; thresholdy HIT/STAND; insurance: even-money only."""
    state = s
    while not game.is_terminal(state):
        if state.to_move == 'Dealer':
            state = game._dealer_play(state)
            break

        # Respect insurance gate if pending
        acts = list(game.actions(state))
        if A_INSURANCE in acts or A_SKIP_INSURANCE in acts:
            if is_blackjack(state.player_cards):
                state = stochastic_step_rng(game, state, A_INSURANCE, rng)
            else:
                state = stochastic_step_rng(game, state, A_SKIP_INSURANCE, rng)
            continue

        pv, soft = hand_value(state.player_cards)
        if state.can_double and 9 <= pv <= 11 and rng.random() < 0.30:
            state = stochastic_step_rng(game, state, A_DOUBLE, rng)
        elif pv <= 11:
            state = stochastic_step_rng(game, state, A_HIT, rng)
        elif 12 <= pv <= 16:
            up = state.dealer_cards[0]
            upv = 11 if up == 'A' else card_value(up)
            if up == 'A' or upv >= 7:
                state = stochastic_step_rng(game, state, A_HIT, rng)
            else:
                state = stochastic_step_rng(game, state, A_STAND, rng)
        else:
            state = stochastic_step_rng(game, state, A_STAND, rng)
    return game.utility_ev(state)

def rollout_win(game: Blackjack, s: BJState, rng: random.Random) -> float:
    """Win-probability rollout: never DOUBLE; avoid busts; insurance: even-money only."""
    state = s
    while not game.is_terminal(state):
        if state.to_move == 'Dealer':
            state = game._dealer_play(state)
            break

        # Respect insurance gate if pending
        acts = list(game.actions(state))
        if A_INSURANCE in acts or A_SKIP_INSURANCE in acts:
            if is_blackjack(state.player_cards):
                state = stochastic_step_rng(game, state, A_INSURANCE, rng)
            else:
                state = stochastic_step_rng(game, state, A_SKIP_INSURANCE, rng)
            continue

        pv, soft = hand_value(state.player_cards)
        if pv <= 11:
            state = stochastic_step_rng(game, state, A_HIT, rng)
        elif 12 <= pv <= 16:
            up = state.dealer_cards[0]
            upv = 11 if up == 'A' else card_value(up)
            if up == 'A' or upv >= 7:
                state = stochastic_step_rng(game, state, A_HIT, rng)
            else:
                state = stochastic_step_rng(game, state, A_STAND, rng)
        else:
            state = stochastic_step_rng(game, state, A_STAND, rng)
    return game.utility_win(state)

# -------- Core UCT engine (parametrized) --------

def mcts_core(game: Blackjack,
              root_state: BJState,
              iters: int,
              reward_rollout_fn,              # (game, state, rng) -> scalar
              C: float = math.sqrt(2),
              rng: Optional[random.Random] = None) -> str:
    """
    Generic UCT with pluggable reward+rollout and independent RNG.
    """
    rng = rng or random.Random(secrets.randbits(64))
    root = MCTSNode(root_state, None, list(game.actions(root_state)))

    def ucb(node: MCTSNode, child: MCTSNode) -> float:
        if child.N == 0: return float('inf')
        return (child.W / child.N) + C * math.sqrt(math.log(node.N + 1) / child.N)

    for _ in range(iters):
        # 1) Selection
        node = root
        state = node.state

        # Auto-play dealer if needed
        if state.to_move == 'Dealer' and not game.is_terminal(state):
            state = game._dealer_play(state)
            node = MCTSNode(state, node, list(game.actions(state)))

        while not game.is_terminal(state) and not node.untried and node.children:
            a, child = max(node.children.items(), key=lambda kv: ucb(node, kv[1]))
            state = stochastic_step_rng(game, state, a, rng)
            node = child

        # 2) Expansion
        if not game.is_terminal(state) and node.untried:
            a = node.untried.pop()
            next_state = stochastic_step_rng(game, state, a, rng)
            child = MCTSNode(next_state, node, list(game.actions(next_state)))
            node.children[a] = child
            node = child
            state = next_state

        # 3) Simulation
        reward = reward_rollout_fn(game, state, rng)

        # 4) Backprop
        while node is not None:
            node.N += 1
            node.W += reward
            node = node.parent

    if not root.children:
        return A_STAND
    maxN = max(ch.N for ch in root.children.values())
    best_actions = [a for a, ch in root.children.items() if ch.N == maxN]
    return rng.choice(best_actions)

# -------- Public distinct MCTS wrappers --------

def mcts_choose_profit(game: Blackjack, root_state: BJState, iters: int) -> str:
    rng = random.Random(secrets.randbits(64))
    return mcts_core(game, root_state, iters, reward_rollout_fn=rollout_profit, C=math.sqrt(2), rng=rng)

def mcts_choose_win(game: Blackjack, root_state: BJState, iters: int) -> str:
    rng = random.Random(secrets.randbits(64))
    return mcts_core(game, root_state, iters, reward_rollout_fn=rollout_win, C=math.sqrt(2), rng=rng)

# -------- Policies --------

def policy_mcts_profit(game: Blackjack, state: BJState, iters=3000) -> str:
    return mcts_choose_profit(game, state, iters=iters)

def policy_mcts_win(game: Blackjack, state: BJState, iters=3000) -> str:
    return mcts_choose_win(game, state, iters=iters)

def policy_expecti_win(game: Blackjack, state: BJState, depth=6) -> str:
    _, a = expectiminimax_win(game, state, depth_limit=depth)
    return a or A_STAND

# =========================
# Runner utilities
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
        s = game.result(s, a)  # unified transition
    return s, actions_taken

def reconstruct_final_dealer(game: Blackjack) -> Tuple[str, ...]:
    return game._round_final_dealer or ('?', '?')

# =========================
# Tournament: shared dealer, static bet, hidden hole, insurance action on A
# =========================

def run_comparison(rounds=10, iters=2500, depth=6, decks=4, starting_chips=1000, base_bet=50):
    game = Blackjack(decks=decks, dealer_hits_soft_17=False)

    stacks = {'MCTS-Profit': float(starting_chips),
              'MCTS-Win'   : float(starting_chips),
              'Expecti-Win': float(starting_chips)}
    rec = {k:{'win':0,'loss':0,'push':0} for k in stacks}

    for rnd in range(1, rounds+1):
        random.seed(secrets.randbits(64))
        base_shoe = make_shoe(decks)

        # Deal dealer once
        d1, shoe_after = game._draw_from_shoe(base_shoe)
        d2, shoe_after = game._draw_from_shoe(shoe_after)
        game._round_hole = d2

        dealer_public = (d1, None)                    # upcard visible, hole hidden to agents
        game._precompute_dealer_hand(d1, shoe_after)  # strict shared dealer
        final_dealer = reconstruct_final_dealer(game)

        # ------ SINGLE player deal used for ALL algorithms ------
        # Draw the player's two cards ONCE so all algorithms get the same starting hand.
        p1, cs = game._draw_from_shoe(shoe_after)
        p2, cs = game._draw_from_shoe(cs)
        common_player_cards = (p1, p2)
        common_shoe_after_player = tuple(sorted(cs.items()))
        insurance_gate = (dealer_public[0] == 'A')

        def make_state() -> BJState:
            s = BJState(
                to_move='Player',
                player_cards=common_player_cards,
                dealer_cards=dealer_public,
                shoe=common_shoe_after_player,   # identical shoe snapshot for all players
                base_bet=base_bet,
                bet_mult=1,
                can_double=True,
                resolved=False,
                insurance_bet=0,
                insurance_allowed=insurance_gate
            )
            # If insurance isn't pending, resolve naturals immediately; otherwise delay.
            return s if s.insurance_allowed else game._maybe_resolve_naturals(s)

        s1 = make_state()  # MCTS-Profit
        s2 = make_state()  # MCTS-Win
        s3 = make_state()  # Expecti-Win
        # --------------------------------------------------------

        f1, a1 = play_full_hand(game, s1, lambda g, s: policy_mcts_profit(g, s, iters))
        f2, a2 = play_full_hand(game, s2, lambda g, s: policy_mcts_win(g, s, iters))
        f3, a3 = play_full_hand(game, s3, lambda g, s: policy_expecti_win(g, s, depth))

        def settle(tag, final_state):
            delta = game.utility_ev(final_state)
            if   delta > 0: rec[tag]['win']  += 1; result = "WIN"
            elif delta < 0: rec[tag]['loss'] += 1; result = "LOSS"
            else:           rec[tag]['push'] += 1; result = "PUSH"
            stacks[tag] += delta
            return delta, result

        d1c, r1 = settle('MCTS-Profit', f1)
        d2c, r2 = settle('MCTS-Win',    f2)
        d3c, r3 = settle('Expecti-Win', f3)

        print(f"\n=== Round {rnd} ===")
        print(f"Dealer Upcard: {dealer_public[0]}   (Hole Hidden)")
        dv, _ = hand_value(final_dealer)
        print(f"Final Dealer Hand: {final_dealer} (Total={dv})")

        def show(tag, start_state, actions, final_state, delta, result):
            sv, _ = hand_value(start_state.player_cards)
            fv, _ = hand_value(final_state.player_cards)
            doubled = "(Doubled)" if final_state.bet_mult == 2 else ""
            ins_txt = f" Ins={final_state.insurance_bet}" if final_state.insurance_bet else ""
            print(
                f"{tag:13s} Result={result:5s}  Bet={start_state.base_bet} {doubled:9s}"
                f"{ins_txt:>8s} Start {start_state.player_cards} ({sv}) | Actions {actions} "
                f"-> Chips={delta:+.2f} | Stack={stacks[tag]:.2f} | "
                f"Final={final_state.player_cards} (Total={fv})"
            )

        show("MCTS-Profit", s1, a1, f1, d1c, r1)
        show("MCTS-Win",    s2, a2, f2, d2c, r2)
        show("Expecti-Win", s3, a3, f3, d3c, r3)

    print("\n           ==== Final Chip Stacks ====")
    for tag in ('MCTS-Profit','MCTS-Win','Expecti-Win'):
        w, l, p = rec[tag]['win'], rec[tag]['loss'], rec[tag]['push']
        print(f"{tag:13s}   | Stack={stacks[tag]:.2f}   | W-L-P = {w}-{l}-{p}")

if __name__ == "__main__":
    run_comparison(
        rounds=500,
        iters=8000,
        depth=6,
        decks=8,
        starting_chips=1000,
        base_bet=100
    )
