import random
import numpy as np
from collections import deque


class FairnessTracker:
    """
    Tracks each fairness agent's agreement with the delivered (social choice) output
    over a sliding window of recently processed users. Used as the agent's
    fairness-so-far score m_i in [0, 1] (SCRUF-D notation).
    """

    def __init__(self, agents, window):
        self.window = window
        self.history = {agent: deque(maxlen=window) for agent in agents}

    def update(self, agreement_scores):
        """agreement_scores: {agent_name: agent_agreement(agent_list, delivered_list)}"""
        for agent, score in agreement_scores.items():
            if agent in self.history:
                self.history[agent].append(score)

    def mi(self):
        """
        Each agent's current fairness-so-far score. Expanding window: averages over
        however much history exists so far. Defaults to 1.0 (optimistic) with no
        history yet, so only the very first processed user is affected.
        """
        return {
            agent: (float(np.mean(scores)) if scores else 1.0)
            for agent, scores in self.history.items()
        }


def raw_unfairness(mi_scores, fairness_agents, ci_scores=None, floor=0.0):
    """
    Shared raw (1 - m_i) * c_i score per fairness agent -- the common input to all
    three SCRUF-D allocation mechanisms (Weighted, Lottery, Least Fair).
    ci_scores=None means compatibility is left out (c_i treated as 1.0/neutral for all agents).
    """
    raw = {}
    for agent in fairness_agents:
        unfairness = 1.0 - mi_scores.get(agent, 1.0)
        compatibility = ci_scores[agent] if ci_scores else 1.0
        raw[agent] = max(unfairness * compatibility, floor)
    return raw


def compute_weights(mi_scores, fairness_agents, ci_scores=None, baseline_weight=1.0, floor=0.0):
    """
    SCRUF-D "Weighted" mechanism: beta_i ~ (1 - m_i) * c_i, normalized so the
    fairness agents' weights always sum to len(fairness_agents) -- i.e. the same
    total ballot mass as an equal-weight run, just redistributed among them.
    baseline is always allocated at a fixed weight, outside this computation.
    """
    raw = raw_unfairness(mi_scores, fairness_agents, ci_scores, floor)

    total = sum(raw.values())
    n = len(fairness_agents)
    weights = {agent: (raw[agent] / total) * n if total > 0 else 1.0 for agent in raw}
    weights["baseline"] = baseline_weight

    return weights


def select_least_fair(mi_scores, fairness_agents):
    """
    SCRUF-D "Least Fair" mechanism: deterministically pick the fairness agent
    with the lowest fairness-so-far score m_i (ties broken by fairness_agents order).
    Ignores compatibility entirely, per the mechanism's definition.
    """
    return min(fairness_agents, key=lambda agent: mi_scores.get(agent, 1.0))


def select_lottery(mi_scores, fairness_agents, rng, ci_scores=None):
    """
    SCRUF-D "Lottery" mechanism: draw a single fairness agent with probability
    p(f_i) ~ (1 - m_i) * c_i, normalized to sum to 1. Falls back to a uniform
    draw if every agent's raw score is 0 (e.g. cold start, m_i defaults to 1.0
    for everyone with no history yet).
    `rng` must be a persistent random.Random instance, reused across calls so
    draws form one reproducible sequence for the whole run.
    """
    raw = raw_unfairness(mi_scores, fairness_agents, ci_scores, floor=0.0)
    total = sum(raw.values())

    if total <= 0:
        return rng.choice(fairness_agents)

    weights = [raw[agent] for agent in fairness_agents]
    return rng.choices(fairness_agents, weights=weights, k=1)[0]



def stream_order(user_ids, seed):
    """Fixed-seed shuffle of user IDs, to simulate a stable but non-artifactual processing order."""
    ordered = list(user_ids)
    random.Random(seed).shuffle(ordered)
    return ordered
