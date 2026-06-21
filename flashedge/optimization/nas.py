"""Mobile-aware Neural Architecture Search with latency constraints.

Searches for efficient architectures under FLOPs, parameter count, and
latency budgets suitable for edge/mobile deployment.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from flashedge.registry import OPTIMIZERS


@dataclass
class SearchSpace:
    """Defines the architecture search space for mobile NAS.

    Attributes:
        kernel_sizes: Candidate kernel sizes for conv layers.
        expand_ratios: Candidate expansion ratios for inverted residuals.
        depths: Candidate block depths per stage.
        widths: Candidate channel widths per stage.
    """

    kernel_sizes: List[int] = field(default_factory=lambda: [3, 5, 7])
    expand_ratios: List[float] = field(default_factory=lambda: [1.0, 2.0, 4.0, 6.0])
    depths: List[int] = field(default_factory=lambda: [1, 2, 3, 4])
    widths: List[int] = field(default_factory=lambda: [16, 24, 32, 48, 64, 96, 128, 160])

    def sample_architecture(self, num_stages: int = 5) -> Dict[str, List[Any]]:
        """Sample a random architecture from the search space."""
        arch = {
            "kernel_sizes": [random.choice(self.kernel_sizes) for _ in range(num_stages)],
            "expand_ratios": [random.choice(self.expand_ratios) for _ in range(num_stages)],
            "depths": [random.choice(self.depths) for _ in range(num_stages)],
            "widths": [random.choice(self.widths) for _ in range(num_stages)],
        }
        return arch


@dataclass
class ArchitectureCandidate:
    """A candidate architecture with its evaluated metrics."""

    config: Dict[str, List[Any]]
    flops: float = 0.0
    params: float = 0.0
    latency_ms: float = 0.0
    accuracy: float = 0.0
    fitness: float = 0.0


@OPTIMIZERS.register("mobile_nas")
class MobileNAS:
    """Latency-constrained Neural Architecture Search for edge devices.

    Uses an evolutionary strategy to search for architectures that meet
    FLOPs, parameter count, and latency constraints.

    Args:
        search_space: Architecture search space definition.
        population_size: Number of candidates per generation.
        generations: Number of evolutionary generations.
        mutation_prob: Probability of mutating a gene.
        target_latency_ms: Maximum acceptable latency in milliseconds.
        max_flops: Maximum acceptable FLOPs.
        max_params: Maximum acceptable parameters.
        num_stages: Number of stages in the architecture.
    """

    def __init__(
        self,
        search_space: Optional[SearchSpace] = None,
        population_size: int = 50,
        generations: int = 30,
        mutation_prob: float = 0.1,
        target_latency_ms: float = 10.0,
        max_flops: float = 300e6,
        max_params: float = 3e6,
        num_stages: int = 5,
    ) -> None:
        self.search_space = search_space or SearchSpace()
        self.population_size = population_size
        self.generations = generations
        self.mutation_prob = mutation_prob
        self.target_latency_ms = target_latency_ms
        self.max_flops = max_flops
        self.max_params = max_params
        self.num_stages = num_stages

    def search(
        self,
        eval_fn: Optional[Callable[[Dict[str, List[Any]]], Dict[str, float]]] = None,
    ) -> ArchitectureCandidate:
        """Run the evolutionary search.

        Args:
            eval_fn: Function that evaluates an architecture config and returns
                     a dict with keys: flops, params, latency_ms, accuracy.
                     If None, uses a synthetic estimator.

        Returns:
            The best architecture candidate found.
        """
        if eval_fn is None:
            eval_fn = self._synthetic_eval

        population = [self._create_candidate(eval_fn) for _ in range(self.population_size)]

        best = max(population, key=lambda c: c.fitness)
        print(f"  NAS Generation 0 — Best fitness: {best.fitness:.4f}")

        for gen in range(1, self.generations + 1):
            population.sort(key=lambda c: c.fitness, reverse=True)
            top_k = population[: self.population_size // 4]

            offspring = []
            while len(offspring) < self.population_size - len(top_k):
                p1, p2 = random.sample(top_k, 2)
                child_config = self._crossover(p1.config, p2.config)
                child_config = self._mutate(child_config)
                child = self._evaluate_candidate(child_config, eval_fn)
                offspring.append(child)

            population = top_k + offspring
            best = max(population, key=lambda c: c.fitness)

            if gen % 5 == 0 or gen == self.generations:
                print(
                    f"  NAS Generation {gen} — Best fitness: {best.fitness:.4f} "
                    f"(FLOPs: {best.flops / 1e6:.1f}M, Params: {best.params / 1e6:.2f}M, "
                    f"Latency: {best.latency_ms:.2f}ms)"
                )

        return best

    def _create_candidate(
        self,
        eval_fn: Callable[[Dict[str, List[Any]]], Dict[str, float]],
    ) -> ArchitectureCandidate:
        config = self.search_space.sample_architecture(self.num_stages)
        return self._evaluate_candidate(config, eval_fn)

    def _evaluate_candidate(
        self,
        config: Dict[str, List[Any]],
        eval_fn: Callable[[Dict[str, List[Any]]], Dict[str, float]],
    ) -> ArchitectureCandidate:
        metrics = eval_fn(config)
        candidate = ArchitectureCandidate(
            config=config,
            flops=metrics.get("flops", 0.0),
            params=metrics.get("params", 0.0),
            latency_ms=metrics.get("latency_ms", 0.0),
            accuracy=metrics.get("accuracy", 0.0),
        )
        candidate.fitness = self._compute_fitness(candidate)
        return candidate

    def _compute_fitness(self, candidate: ArchitectureCandidate) -> float:
        """Compute fitness score with penalty for constraint violations."""
        fitness = candidate.accuracy

        if candidate.flops > self.max_flops:
            fitness *= 0.5 * (self.max_flops / candidate.flops)
        if candidate.params > self.max_params:
            fitness *= 0.5 * (self.max_params / candidate.params)
        if candidate.latency_ms > self.target_latency_ms:
            fitness *= 0.5 * (self.target_latency_ms / candidate.latency_ms)

        return fitness

    def _crossover(
        self,
        config1: Dict[str, List[Any]],
        config2: Dict[str, List[Any]],
    ) -> Dict[str, List[Any]]:
        """Single-point crossover between two architecture configs."""
        child = {}
        for key in config1:
            point = random.randint(1, len(config1[key]) - 1)
            child[key] = config1[key][:point] + config2[key][point:]
        return child

    def _mutate(self, config: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        """Randomly mutate genes in an architecture config."""
        mutated = {}
        for key, values in config.items():
            new_values = []
            for v in values:
                if random.random() < self.mutation_prob:
                    candidates = getattr(self.search_space, key, [v])
                    new_values.append(random.choice(candidates))
                else:
                    new_values.append(v)
            mutated[key] = new_values
        return mutated

    def _synthetic_eval(self, config: Dict[str, List[Any]]) -> Dict[str, float]:
        """Synthetic evaluator for testing — estimates metrics from config."""
        total_flops = 0.0
        total_params = 0.0

        in_channels = 3
        spatial = 224
        for i in range(len(config["widths"])):
            out_channels = config["widths"][i]
            kernel = config["kernel_sizes"][i]
            expand = config["expand_ratios"][i]
            depth = config["depths"][i]

            mid_channels = int(in_channels * expand)
            for _ in range(depth):
                flops_layer = mid_channels * out_channels * kernel * kernel * spatial * spatial
                params_layer = mid_channels * out_channels * kernel * kernel + out_channels
                total_flops += flops_layer
                total_params += params_layer
                spatial = max(spatial // 2, 1) if _ == 0 else spatial

            in_channels = out_channels

        flops_noise = random.uniform(0.9, 1.1)
        accuracy = max(0.0, min(1.0, 0.9 - total_flops / 5e9 + random.uniform(-0.05, 0.05)))
        latency = total_flops / 1e8 + random.uniform(0.5, 2.0)

        return {
            "flops": total_flops * flops_noise,
            "params": total_params,
            "latency_ms": latency,
            "accuracy": accuracy,
        }
