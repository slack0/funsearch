"""FunSearch-inspired prototype for Neural Architecture Search (NAS).

This module adapts key ideas from the FunSearch paper/codebase to NAS:
- Evolve only a critical function (architecture scoring policy), not the whole pipeline.
- Best-shot prompting analogue: sample high performers to seed new candidates.
- Island-based diversity: keep multiple sub-populations to avoid local optima.
- Strict evaluator loop: every proposed program is executed and scored.

The implementation is self-contained and intentionally lightweight so it can be
used with any model backend (OpenAI, local LLM, templated mutations, etc.).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import random
from typing import Dict, List, Tuple


Architecture = Dict[str, float]

SEARCH_SPACE = {
    "depth": (2, 24),
    "width": (16, 1024),
    "kernel": (1, 9),
    "expansion": (1, 8),
    "attention": (0, 1),
}


def random_architecture(rng: random.Random) -> Architecture:
    return {
        "depth": rng.randint(*SEARCH_SPACE["depth"]),
        "width": rng.randint(*SEARCH_SPACE["width"]),
        "kernel": rng.randint(*SEARCH_SPACE["kernel"]),
        "expansion": rng.randint(*SEARCH_SPACE["expansion"]),
        "attention": rng.randint(*SEARCH_SPACE["attention"]),
    }


@dataclass
class Candidate:
    """A candidate 'program' represented by symbolic coefficients."""

    coeffs: Dict[str, float]
    bias: float
    score: float | None = None

    def policy(self, arch: Architecture) -> float:
        raw = self.bias
        for k, v in self.coeffs.items():
            raw += v * arch[k]
        # Non-linearity keeps values in stable range and mimics heuristic policies.
        return math.tanh(raw / 100.0)


@dataclass
class Island:
    members: List[Candidate] = field(default_factory=list)

    def best(self) -> Candidate:
        return max(self.members, key=lambda c: c.score if c.score is not None else -1e9)


class NASFunSearch:
    """FunSearch-style search loop for NAS objective functions."""

    def __init__(
        self,
        *,
        rng_seed: int = 7,
        n_islands: int = 4,
        island_size: int = 16,
        top_k: int = 4,
        eval_budget: int = 200,
    ) -> None:
        self.rng = random.Random(rng_seed)
        self.n_islands = n_islands
        self.island_size = island_size
        self.top_k = top_k
        self.eval_budget = eval_budget
        self.islands = [Island() for _ in range(n_islands)]
        self.history: List[Tuple[int, float]] = []

    def _new_candidate(self) -> Candidate:
        coeffs = {
            "depth": self.rng.uniform(-2.0, 2.0),
            "width": self.rng.uniform(-0.02, 0.02),
            "kernel": self.rng.uniform(-1.0, 1.0),
            "expansion": self.rng.uniform(-1.0, 1.0),
            "attention": self.rng.uniform(-4.0, 4.0),
        }
        return Candidate(coeffs=coeffs, bias=self.rng.uniform(-2.0, 2.0))

    def _mutate(self, parent: Candidate, temperature: float) -> Candidate:
        coeffs = dict(parent.coeffs)
        for key in coeffs:
            if self.rng.random() < 0.8:
                coeffs[key] += self.rng.gauss(0, temperature)
        return Candidate(coeffs=coeffs, bias=parent.bias + self.rng.gauss(0, temperature))

    def _objective(self, arch: Architecture, policy_score: float) -> float:
        """Proxy objective: accuracy minus cost penalty.

        Replace with your training-based evaluator for production use:
        e.g., train for N epochs and evaluate validation score.
        """
        depth = arch["depth"]
        width = arch["width"]
        kernel = arch["kernel"]
        expansion = arch["expansion"]
        attention = arch["attention"]

        # Fake 'accuracy' landscape (non-convex, with interactions).
        accuracy = (
            0.68
            + 0.06 * math.exp(-((depth - 12) ** 2) / 36)
            + 0.04 * math.exp(-((kernel - 5) ** 2) / 6)
            + 0.05 * math.tanh((expansion - 3) / 2)
            + 0.02 * attention
            + 0.03 * policy_score
        )
        # Approximate compute cost penalty.
        flops = depth * width * kernel * (1 + 0.3 * attention) * (1 + 0.2 * expansion)
        cost_penalty = min(0.15, flops / 2_000_000)
        return accuracy - cost_penalty

    def evaluate_candidate(self, candidate: Candidate, n_samples: int = 64) -> float:
        values: List[float] = []
        for _ in range(n_samples):
            arch = random_architecture(self.rng)
            p = candidate.policy(arch)
            values.append(self._objective(arch, p))
        candidate.score = sum(values) / len(values)
        return candidate.score

    def initialize(self) -> None:
        for island in self.islands:
            island.members = []
            for _ in range(self.island_size):
                c = self._new_candidate()
                self.evaluate_candidate(c)
                island.members.append(c)

    def step(self, step_id: int) -> None:
        for idx, island in enumerate(self.islands):
            ranked = sorted(island.members, key=lambda c: c.score or -1e9, reverse=True)
            parents = ranked[: self.top_k]
            parent = self.rng.choice(parents)
            temperature = max(0.05, 0.4 * (1.0 - step_id / max(1, self.eval_budget)))
            child = self._mutate(parent, temperature)
            self.evaluate_candidate(child)

            island.members.append(child)
            island.members = sorted(island.members, key=lambda c: c.score or -1e9, reverse=True)[: self.island_size]

            # Occasional migration (island-based evolution).
            if step_id % 10 == 0 and idx < self.n_islands - 1:
                migrant = island.best()
                self.islands[idx + 1].members.append(migrant)
                self.islands[idx + 1].members = sorted(
                    self.islands[idx + 1].members,
                    key=lambda c: c.score or -1e9,
                    reverse=True,
                )[: self.island_size]

        global_best = self.best_candidate()
        self.history.append((step_id, global_best.score or float("-inf")))

    def best_candidate(self) -> Candidate:
        return max((i.best() for i in self.islands), key=lambda c: c.score if c.score is not None else -1e9)

    def run(self) -> Candidate:
        self.initialize()
        for step_id in range(1, self.eval_budget + 1):
            self.step(step_id)
        return self.best_candidate()


def format_candidate(candidate: Candidate) -> str:
    terms = ", ".join(f"{k}={v:.3f}" for k, v in sorted(candidate.coeffs.items()))
    return f"score={candidate.score:.4f}; bias={candidate.bias:.3f}; {terms}"


if __name__ == "__main__":
    search = NASFunSearch(rng_seed=42, eval_budget=80)
    best = search.run()
    print("Best discovered NAS policy:")
    print(format_candidate(best))
