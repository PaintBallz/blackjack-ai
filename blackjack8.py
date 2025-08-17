from dataclasses import dataclass, replace
from collections import Counter
from typing import Dict, Tuple, List, Optional, Iterable
import random, math, secrets

# =========================
# Blackjack core
# Insurance Implementation
# Same Player Hands for Players
# =========================

RANKS = ['2','3','4','5','6','7','8','9','10','J','Q','K','A']

def make_shoe(decks: int = 6) -> Counter:
    """Standard 52-card deck counts per rank, times `decks`."""
    shoe = Counter()
    for rank in RANKS:
        shoe[rank] = 4 * decks  # each rank has 4 per deck
    return shoe

def card_value(rank: str) -> int:
    if rank == 'A':
        return 11
    if rank in ['10','J','Q','K']:
        return 10
    return int(rank)

def hand_value(cards: Tuple[str, ...]) -> Tuple[int, bool]:
    total = 0
    ace_count = 0
    for rank in cards:
        if rank == 'A':
            total += 11
            ace_count += 1
        elif rank in {'10', 'J', 'Q', 'K'}:
            total += 10
        else:
            total += int(rank)
    while total > 21 and ace_count:
        total -= 10
        ace_count -= 1
    is_soft = (ace_count > 0)
    return total, is_soft

def is_blackjack(cards: Tuple[str, ...]) -> bool:
    return len(cards) == 2 and 'A' in cards and any(c in {'10','J','Q','K'} for c in cards)

A_HIT = "HIT"
A_STAND = "STAND"
A_DOUBLE = "DOUBLE"
A_INSURANCE = "INSURANCE"
A_SKIP_INSURANCE = "SKIP_INSURANCE"

@dataclass(frozen=True)
class BlackjackState:
    to_move: str
    player_cards: Tuple[str, ...]
    dealer_cards: Tuple[str, ...]      # (upcard, None) during play; final tuple after resolution
    shoe: Tuple[Tuple[str,int], ...]   # immutable snapshot of remaining shoe
    base_bet: int                      # static bet (configurable in run_comparison)
    bet_multiplier: int                # 1 normally, 2 after DOUBLE
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
        self.dealer_hits_soft_17 = dealer_hits_soft_17
        self._round_hole: Optional[str] = None
        self._round_dealer_hits: Tuple[str, ...] = ()
        self._round_final_dealer: Optional[Tuple[str, ...]] = None

    # dealing & shoes
    def _draw_from_shoe(self, shoe: Counter, rank: Optional[str] = None) -> Tuple[str, Counter]:
        """Draw specific rank (if provided) or sample proportional to counts."""
        if rank is None:
            total_cards = sum(shoe.values())
            draw_index = random.randrange(total_cards)
            cumulative = 0
            for candidate_rank, count in shoe.items():
                cumulative += count
                if draw_index < cumulative:
                    rank = candidate_rank
                    break
        if rank is None or shoe[rank] <= 0:
            raise RuntimeError("Invalid draw")
        updated_shoe = shoe.copy()
        updated_shoe[rank] -= 1
        if updated_shoe[rank] == 0:
            del updated_shoe[rank]
        return rank, updated_shoe

    def _precompute_dealer_hand(self, upcard: str, shoe_after_dealer: Counter):
        dealer_hand = [upcard, self._round_hole]
        shoe = shoe_after_dealer.copy()
        # Dealer hits while total < 17; stands on 17+
        while True:
            dealer_total, _ = hand_value(tuple(dealer_hand))
            if dealer_total >= 17:
                break
            drawn_rank, shoe = self._draw_from_shoe(shoe)
            dealer_hand.append(drawn_rank)
        self._round_dealer_hits = tuple(dealer_hand[2:])
        self._round_final_dealer = tuple(dealer_hand)

    # rules interface
    def actions(self, state: BlackjackState) -> Iterable[str]:
        if self.is_terminal(state) or state.to_move != 'Player':
            return []
        # If insurance decision is pending, only offer insurance choices
        if state.insurance_allowed:
            return [A_INSURANCE, A_SKIP_INSURANCE]
        actions_list = [A_HIT, A_STAND]
        if state.can_double:
            actions_list.append(A_DOUBLE)
        return actions_list

    def is_terminal(self, state: BlackjackState) -> bool:
        return state.resolved

    def _maybe_resolve_naturals(self, state: BlackjackState) -> BlackjackState:
        # Do not resolve naturals until insurance decision has been made (if applicable)
        if state.insurance_allowed:
            return state
        player_blackjack = is_blackjack(state.player_cards)
        dealer_hole = state.dealer_cards[1] if len(state.dealer_cards) > 1 else None
        if dealer_hole is None:
            dealer_hole = self._round_hole
        dealer_blackjack = is_blackjack((state.dealer_cards[0], dealer_hole)) if dealer_hole is not None else False
        if player_blackjack or dealer_blackjack:
            return replace(state, resolved=True, to_move='Dealer')
        return state

    def result(self, state: BlackjackState, action: str) -> BlackjackState:
        if action not in self.actions(state):
            raise ValueError("Illegal move")
        shoe_counter = state.shoe_counter()
        updated_player_cards = list(state.player_cards)

        # Insurance decision branch
        if action == A_INSURANCE:
            next_state = replace(
                state,
                insurance_bet=state.base_bet // 2,
                insurance_allowed=False  # decision made; now we may resolve naturals immediately
            )
            return self._maybe_resolve_naturals(next_state)

        if action == A_SKIP_INSURANCE:
            next_state = replace(
                state,
                insurance_bet=0,
                insurance_allowed=False
            )
            return self._maybe_resolve_naturals(next_state)

        # Regular actions
        if action == A_HIT:
            drawn_rank, shoe_counter = self._draw_from_shoe(shoe_counter)
            updated_player_cards.append(drawn_rank)
            player_total, _ = hand_value(tuple(updated_player_cards))
            if player_total > 21:
                return replace(state, player_cards=tuple(updated_player_cards),
                               shoe=tuple(sorted(shoe_counter.items())),
                               to_move='Dealer', resolved=True, can_double=False)
            else:
                return replace(state, player_cards=tuple(updated_player_cards),
                               shoe=tuple(sorted(shoe_counter.items())),
                               to_move='Player', can_double=False)

        if action == A_DOUBLE:
            drawn_rank, shoe_counter = self._draw_from_shoe(shoe_counter)
            updated_player_cards.append(drawn_rank)
            next_state = replace(state, player_cards=tuple(updated_player_cards),
                                 shoe=tuple(sorted(shoe_counter.items())),
                                 bet_multiplier=2, to_move='Dealer', can_double=False)
            return self._dealer_play(next_state)

        if action == A_STAND:
            return self._dealer_play(replace(state, to_move='Dealer', can_double=False))

        raise RuntimeError("Unreachable")

    def _dealer_play(self, state: BlackjackState) -> BlackjackState:
        if state.resolved:
            return state
        final_dealer_tuple = self._round_final_dealer
        if final_dealer_tuple is None:  # safety fallback
            final_dealer_tuple = (state.dealer_cards[0], self._round_hole)
        return replace(state, dealer_cards=final_dealer_tuple, resolved=True)

    # payouts 
    def utility_ev(self, state: BlackjackState) -> float:
        """
        Returns chip delta for the player:
        - Naturals: +1.5 * base_bet (no double), or -base_bet if dealer natural only.
        - Otherwise: +/- (base_bet * bet_multiplier) or 0 on push.
        - Insurance (if taken): pays 2:1 if dealer has blackjack; else it's lost.
        """
        base = state.base_bet
        multiplier = state.bet_multiplier
        player_total, _ = hand_value(state.player_cards)

        # Dealer tuple using committed hole if needed
        if len(state.dealer_cards) > 1 and state.dealer_cards[1] is not None:
            dealer_tuple = state.dealer_cards
        else:
            dealer_tuple = (state.dealer_cards[0], self._round_hole)

        dealer_total, _ = hand_value(dealer_tuple)
        player_blackjack = is_blackjack(state.player_cards)
        dealer_blackjack = is_blackjack(dealer_tuple)

        # Insurance resolution
        insurance_delta = 0.0
        if state.insurance_bet:
            insurance_delta = (2.0 * state.insurance_bet) if dealer_blackjack else (-1.0 * state.insurance_bet)

        # Natural cases (resolved immediately)
        if player_blackjack or dealer_blackjack:
            if player_blackjack and not dealer_blackjack:
                main_delta = 1.5 * base
            elif dealer_blackjack and not player_blackjack:
                main_delta = -1.0 * base
            else:
                main_delta = 0.0  # both have blackjack -> push main
            return main_delta + insurance_delta

        # Regular play (no naturals)
        wager = base * multiplier
        if player_total > 21: return -wager + insurance_delta
        if dealer_total > 21: return +wager + insurance_delta
        if player_total > dealer_total: return +wager + insurance_delta
        if player_total < dealer_total: return -wager + insurance_delta
        return 0.0 + insurance_delta

    def utility_win(self, state: BlackjackState) -> float:
        """+1 on win, 0 on push, -1 on loss."""
        delta = self.utility_ev(state)
        return 1.0 if delta > 0 else (-1.0 if delta < 0 else 0.0)

# Expectiminimax (maximize win probability, HIT/STAND only)

def expectiminimax_win(game: Blackjack, root_state: BlackjackState, depth_limit: int = 6) -> Tuple[float, Optional[str]]:
    """Return (value, best_action) maximizing win utility; actions restricted to HIT/STAND."""
    transposition_cache: Dict[Tuple[BlackjackState, int], Tuple[float, Optional[str]]] = {}
    NEG_INF = -1e9

    def chance_children_for_hit(state: BlackjackState) -> List[Tuple[float, BlackjackState]]:
        shoe_counter = state.shoe_counter()
        total_cards = sum(shoe_counter.values())
        children: List[Tuple[float, BlackjackState]] = []
        for drawn_rank, remaining_count in shoe_counter.items():
            prob = remaining_count / total_cards
            new_shoe = shoe_counter.copy()
            new_shoe[drawn_rank] -= 1
            if new_shoe[drawn_rank] == 0:
                del new_shoe[drawn_rank]
            new_player_cards = tuple(list(state.player_cards) + [drawn_rank])
            player_total, _ = hand_value(new_player_cards)
            if player_total > 21:
                next_state = replace(state, player_cards=new_player_cards,
                                     shoe=tuple(sorted(new_shoe.items())),
                                     to_move='Dealer', resolved=True, can_double=False)
            else:
                next_state = replace(state, player_cards=new_player_cards,
                                     shoe=tuple(sorted(new_shoe.items())),
                                     to_move='Player', can_double=False)
            children.append((prob, next_state))
        return children

    def eval_recursive(state: BlackjackState, depth_remaining: int) -> Tuple[float, Optional[str]]:
        cache_key = (state, depth_remaining)
        if cache_key in transposition_cache:
            return transposition_cache[cache_key]

        # Cutoff heuristic at player node
        if game.is_terminal(state) or (depth_remaining <= 0 and state.to_move == 'Player'):
            if not game.is_terminal(state) and state.to_move == 'Player':
                # One-step lookahead: compare STAND vs expected HIT then resolve
                stand_value = game.utility_win(game._dealer_play(replace(state, to_move='Dealer')))
                hit_expectation = 0.0
                for prob, child_state in chance_children_for_hit(state):
                    if not child_state.resolved and child_state.to_move == 'Player':
                        hit_expectation += prob * game.utility_win(game._dealer_play(replace(child_state, to_move='Dealer')))
                    else:
                        hit_expectation += prob * game.utility_win(child_state)
                value = max(stand_value, hit_expectation)
                transposition_cache[cache_key] = (value, None)
                return transposition_cache[cache_key]
            terminal_value = game.utility_win(state)
            transposition_cache[cache_key] = (terminal_value, None)
            return transposition_cache[cache_key]

        if state.to_move == 'Player':
            best_value = NEG_INF
            best_action: Optional[str] = None
            # Filter actions to HIT/STAND only (DOUBLE/INSURANCE excluded for this agent)
            for action in [A_HIT, A_STAND]:
                if action not in game.actions(state):
                    continue
                if action == A_HIT:
                    exp_value = 0.0
                    for prob, child_state in chance_children_for_hit(state):
                        child_value, _ = eval_recursive(child_state, depth_remaining)  # chance doesn't reduce depth
                        exp_value += prob * child_value
                    candidate_value = exp_value
                else:
                    child_state = game.result(state, action)
                    candidate_value, _ = eval_recursive(child_state, depth_remaining - 1)
                if candidate_value > best_value:
                    best_value, best_action = candidate_value, action
            transposition_cache[cache_key] = (best_value, best_action)
            return transposition_cache[cache_key]

        # Dealer turn: deterministic (precomputed)
        dealer_resolved_state = game._dealer_play(state)
        value, _ = eval_recursive(dealer_resolved_state, depth_remaining)
        transposition_cache[cache_key] = (value, None)
        return transposition_cache[cache_key]

    return eval_recursive(root_state, depth_limit)

# Monte Carlo Tree Search (EV and Win variants)

class MCTSNode:
    __slots__ = ("state","parent","children","visit_count","total_reward","untried_actions")
    def __init__(self, state: BlackjackState, parent: Optional['MCTSNode'], actions: List[str]):
        self.state = state
        self.parent = parent
        self.children: Dict[str, 'MCTSNode'] = {}
        self.visit_count = 0
        self.total_reward = 0.0
        self.untried_actions = list(actions)

def stochastic_step_with_rng(game: Blackjack, state: BlackjackState, action: str, rng: random.Random) -> BlackjackState:
    """Apply action; sample draws from shoe using a provided RNG."""
    if action == A_HIT:
        shoe_counter = state.shoe_counter()
        total_cards = sum(shoe_counter.values())
        draw_index = rng.randrange(total_cards)
        cumulative = 0
        drawn_rank = None
        for rank, count in shoe_counter.items():
            cumulative += count
            if draw_index < cumulative:
                drawn_rank = rank
                break
        player_cards = list(state.player_cards)
        player_cards.append(drawn_rank)
        shoe_counter[drawn_rank] -= 1
        if shoe_counter[drawn_rank] == 0:
            del shoe_counter[drawn_rank]
        player_total, _ = hand_value(tuple(player_cards))
        if player_total > 21:
            return replace(state, player_cards=tuple(player_cards),
                           shoe=tuple(sorted(shoe_counter.items())),
                           to_move='Dealer', resolved=True, can_double=False)
        else:
            return replace(state, player_cards=tuple(player_cards),
                           shoe=tuple(sorted(shoe_counter.items())),
                           to_move='Player', can_double=False)
    elif action in (A_STAND, A_DOUBLE, A_INSURANCE, A_SKIP_INSURANCE):
        return game.result(state, action)
    else:
        raise ValueError("Unknown action")

# Random rollouts (purely random actions until terminal)

def _random_rollout(game: Blackjack,
                    start_state: BlackjackState,
                    rng: random.Random,
                    reward_function) -> float:
    """
    Take uniformly random legal actions until the state is terminal.
    Dealer steps are handled deterministically via _dealer_play.
    `reward_function` is either game.utility_ev or game.utility_win.
    """
    current_state = start_state
    while not game.is_terminal(current_state):
        if current_state.to_move == 'Dealer':
            current_state = game._dealer_play(current_state)
            break

        legal_actions = list(game.actions(current_state))
        if not legal_actions:
            current_state = game._dealer_play(current_state)
            break

        chosen_action = rng.choice(legal_actions)
        current_state = stochastic_step_with_rng(game, current_state, chosen_action, rng)

    return reward_function(current_state)

def rollout_random_ev(game: Blackjack, state: BlackjackState, rng: random.Random) -> float:
    """Random rollout scored by expected chips (EV)."""
    return _random_rollout(game, state, rng, game.utility_ev)

def rollout_random_win(game: Blackjack, state: BlackjackState, rng: random.Random) -> float:
    """Random rollout scored by win/loss (+1/0/-1)."""
    return _random_rollout(game, state, rng, game.utility_win)

# Core UCT engine (with optional debug)

def mcts_core(game: Blackjack,
              root_state: BlackjackState,
              iteration_budget: int,
              reward_rollout_fn,              # (game, state, rng) -> scalar
              exploration_constant: float = math.sqrt(2),
              rng: Optional[random.Random] = None,
              *,
              debug: bool = False,
              debug_every: int = 500,
              reward_metric: str = "ev") -> str:

    rng = rng or random.Random(secrets.randbits(64))
    root_node = MCTSNode(root_state, None, list(game.actions(root_state)))

    def ucb(parent: MCTSNode, child: MCTSNode) -> float:
        if child.visit_count == 0:
            return float('inf')
        return (child.total_reward / child.visit_count) + \
               exploration_constant * math.sqrt(math.log(parent.visit_count + 1) / child.visit_count)

    def print_children_stats(prefix: str):
        if not root_node.children:
            print(f"{prefix} | root has no children yet")
            return
        parts = []
        for action, child in root_node.children.items():
            parts.append(f"{action}:visits={child.visit_count},reward={child.total_reward:.3f}")
        print(f"{prefix} | root.visits={root_node.visit_count} | " + " | ".join(parts))

    for i in range(1, iteration_budget + 1):
        node = root_node
        state = node.state

        # If it happens to be dealer's turn at the root, resolve once
        if state.to_move == 'Dealer' and not game.is_terminal(state):
            state = game._dealer_play(state)
            node = MCTSNode(state, node, list(game.actions(state)))

        # Selection
        while not game.is_terminal(state) and not node.untried_actions and node.children:
            selected_action, child_node = max(node.children.items(), key=lambda kv: ucb(node, kv[1]))
            state = stochastic_step_with_rng(game, state, selected_action, rng)
            node = child_node

        # Expansion
        if not game.is_terminal(state) and node.untried_actions:
            expand_action = node.untried_actions.pop()
            next_state = stochastic_step_with_rng(game, state, expand_action, rng)
            child_node = MCTSNode(next_state, node, list(game.actions(next_state)))
            node.children[expand_action] = child_node
            node = child_node
            state = next_state

        # Simulation
        reward = reward_rollout_fn(game, state, rng)

        # Backpropagation
        while node is not None:
            node.visit_count += 1
            node.total_reward += reward
            node = node.parent

        if debug and (i % debug_every == 0 or i == 1):
            print_children_stats(f"[MCTS] iter {i}/{iteration_budget}")

    if not root_node.children:
        if debug:
            print("[MCTS] No children expanded; defaulting to STAND")
        return A_STAND

    # Choose action with max visit count; break ties randomly
    max_visits = max(child.visit_count for child in root_node.children.values())
    best_actions = [action for action, child in root_node.children.items() if child.visit_count == max_visits]
    chosen = rng.choice(best_actions)

    if debug:
        print_children_stats("[MCTS] final")
        print(f"[MCTS] chosen action: {chosen}")

    return chosen

# Public distinct MCTS wrappers (random rollouts)

def mcts_profit(game: Blackjack, root_state: BlackjackState, iteration_budget: int, *,
                debug: bool = False, debug_every: int = 500) -> str:
    rng = random.Random(secrets.randbits(64))
    return mcts_core(game, root_state, iteration_budget,
                     reward_rollout_fn=rollout_random_ev,
                     exploration_constant=math.sqrt(2),
                     rng=rng,
                     debug=debug, debug_every=debug_every, reward_metric="ev")

def mcts_win(game: Blackjack, root_state: BlackjackState, iteration_budget: int, *,
             debug: bool = False, debug_every: int = 500) -> str:
    rng = random.Random(secrets.randbits(64))
    return mcts_core(game, root_state, iteration_budget,
                     reward_rollout_fn=rollout_random_win,
                     exploration_constant=math.sqrt(2),
                     rng=rng,
                     debug=debug, debug_every=debug_every, reward_metric="win")

# Policies

def policy_mcts_profit(game: Blackjack, state: BlackjackState, iteration_budget=3000, *, debug=False, debug_every=500) -> str:
    return mcts_profit(game, state, iteration_budget=iteration_budget, debug=debug, debug_every=debug_every)

def policy_mcts_win(game: Blackjack, state: BlackjackState, iteration_budget=3000, *, debug=False, debug_every=500) -> str:
    # state for win-agent has no double; insurance is skipped by the runner before policy is queried
    return mcts_win(game, state, iteration_budget=iteration_budget, debug=debug, debug_every=debug_every)

def policy_expectiminimax_win(game: Blackjack, state: BlackjackState, depth=6) -> str:
    # DOUBLE and INSURANCE are excluded for this agent (handled by runner & action filter in expectiminimax)
    _, best_action = expectiminimax_win(game, state, depth_limit=depth)
    return best_action or A_STAND

# Runner utilities

def play_full_hand(game: Blackjack, start_state: BlackjackState, policy_fn, *,
                   auto_skip_insurance: bool = False) -> Tuple[BlackjackState, List[str]]:
    """
    Plays a hand. If auto_skip_insurance=True, any insurance gate is resolved by auto-declining
    BEFORE asking the agent, so the agent never performs SKIP_INSURANCE or INSURANCE.
    """
    state = start_state
    actions_taken: List[str] = []
    while not game.is_terminal(state):
        if state.to_move == 'Dealer':
            state = game._dealer_play(state)
            break

        # For restricted agents, resolve insurance outside the agent
        if auto_skip_insurance and state.insurance_allowed:
            state = game.result(state, A_SKIP_INSURANCE)
            # Not counted as an agent action
            continue

        chosen_action = policy_fn(game, state)
        actions_taken.append(chosen_action)
        state = game.result(state, chosen_action)  # unified transition
    return state, actions_taken

def reconstruct_final_dealer(game: Blackjack) -> Tuple[str, ...]:
    return game._round_final_dealer or ('?', '?')

# Game Policy: shared dealer, static bet

def run_comparison(rounds=10, iteration_budget=2500, depth=6, decks=4, starting_chips=1000, base_bet=50, *,
                   debug=False, debug_every=500):
    game = Blackjack(decks=decks, dealer_hits_soft_17=False)

    bankrolls = {'MCTS-Profit': float(starting_chips),
                 'MCTS-Win'   : float(starting_chips),
                 'Expecti-Win': float(starting_chips)}
    record = {k:{'win':0,'loss':0,'push':0} for k in bankrolls}

    for round_index in range(1, rounds+1):
        random.seed(secrets.randbits(64))
        base_shoe = make_shoe(decks)

        # Deal dealer once
        dealer_upcard, shoe_after = game._draw_from_shoe(base_shoe)
        dealer_hole, shoe_after = game._draw_from_shoe(shoe_after)
        game._round_hole = dealer_hole

        dealer_public = (dealer_upcard, None)                    # upcard visible, hole hidden to agents
        game._precompute_dealer_hand(dealer_upcard, shoe_after)  # strict shared dealer
        final_dealer = reconstruct_final_dealer(game)

        # ------ SINGLE player deal used for ALL algorithms ------
        # Draw the player's two cards ONCE so all algorithms get the same starting hand.
        p1, shoe_after_player = game._draw_from_shoe(shoe_after)
        p2, shoe_after_player = game._draw_from_shoe(shoe_after_player)
        common_player_cards = (p1, p2)
        common_shoe_after_player = tuple(sorted(shoe_after_player.items()))
        insurance_gate_open = (dealer_public[0] == 'A')

        def make_start_state() -> BlackjackState:
            state = BlackjackState(
                to_move='Player',
                player_cards=common_player_cards,
                dealer_cards=dealer_public,
                shoe=common_shoe_after_player,   # identical shoe snapshot for all players
                base_bet=base_bet,
                bet_multiplier=1,
                can_double=True,
                resolved=False,
                insurance_bet=0,
                insurance_allowed=insurance_gate_open
            )
            # If insurance isn't pending, resolve naturals immediately; otherwise delay.
            return state if state.insurance_allowed else game._maybe_resolve_naturals(state)

        start_profit = make_start_state()      # MCTS-Profit (full rules)
        start_win_mcts = make_start_state()    # MCTS-Win     (restricted: no insurance, no double)
        start_win_expecti = make_start_state() # Expecti-Win  (restricted: no insurance, no double)

        # Disable doubling for the two win-probability agents at the start
        start_win_mcts = replace(start_win_mcts, can_double=False)
        start_win_expecti = replace(start_win_expecti, can_double=False)

        final_profit, actions_profit = play_full_hand(
            game, start_profit,
            lambda g, s: policy_mcts_profit(g, s, iteration_budget, debug=debug, debug_every=debug_every),
            auto_skip_insurance=False
        )
        final_win_mcts, actions_win_mcts = play_full_hand(
            game, start_win_mcts,
            lambda g, s: policy_mcts_win(g, s, iteration_budget, debug=debug, debug_every=debug_every),
            auto_skip_insurance=True
        )
        final_win_expecti, actions_win_expecti = play_full_hand(
            game, start_win_expecti,
            lambda g, s: policy_expectiminimax_win(g, s, depth),
            auto_skip_insurance=True
        )

        def settle(agent_tag: str, final_state: BlackjackState):
            delta = game.utility_ev(final_state)
            if   delta > 0: record[agent_tag]['win']  += 1; result_str = "WIN"
            elif delta < 0: record[agent_tag]['loss'] += 1; result_str = "LOSS"
            else:           record[agent_tag]['push'] += 1; result_str = "PUSH"
            bankrolls[agent_tag] += delta
            return delta, result_str

        delta_profit, result_profit = settle('MCTS-Profit', final_profit)
        delta_win_mcts, result_win_mcts = settle('MCTS-Win', final_win_mcts)
        delta_win_expecti, result_win_expecti = settle('Expecti-Win', final_win_expecti)

        print(f"\n=== Round {round_index} ===")
        print(f"Dealer Upcard: {dealer_public[0]}   (Hole Hidden)")
        dealer_total, _ = hand_value(final_dealer)
        print(f"Final Dealer Hand: {final_dealer} (Total={dealer_total})")

        def show(agent_tag: str, start_state: BlackjackState, actions: List[str], final_state: BlackjackState, delta: float, result_str: str):
            start_total, _ = hand_value(start_state.player_cards)
            final_total, _ = hand_value(final_state.player_cards)
            doubled_flag = "(Doubled)" if final_state.bet_multiplier == 2 else ""
            insurance_text = f" Ins={final_state.insurance_bet}" if final_state.insurance_bet else ""
            print(
                f"{agent_tag:13s} Result={result_str:5s}  Bet={start_state.base_bet} {doubled_flag:9s}"
                f"{insurance_text:>8s} Start {start_state.player_cards} ({start_total}) | Actions {actions} "
                f"-> Chips={delta:+.2f} | Stack={bankrolls[agent_tag]:.2f} | "
                f"Final={final_state.player_cards} (Total={final_total})"
            )

        show("MCTS-Profit", start_profit, actions_profit, final_profit, delta_profit, result_profit)
        show("MCTS-Win",    start_win_mcts, actions_win_mcts, final_win_mcts, delta_win_mcts, result_win_mcts)
        show("Expecti-Win", start_win_expecti, actions_win_expecti, final_win_expecti, delta_win_expecti, result_win_expecti)

    print("\n           ==== Final Chip Stacks ====")
    for tag in ('MCTS-Profit','MCTS-Win','Expecti-Win'):
        w, l, p = record[tag]['win'], record[tag]['loss'], record[tag]['push']
        print(f"{tag:13s}   | Stack={bankrolls[tag]:.2f}   | W-L-P = {w}-{l}-{p}")

if __name__ == "__main__":
    # Defaults tuned down for readable console output. Increase rounds/iterations as desired.
    run_comparison(
        rounds=25,
        iteration_budget=9000,
        depth=6,
        decks=1,
        starting_chips=1000,
        base_bet=100,
        debug=True,
        debug_every=500
    )
