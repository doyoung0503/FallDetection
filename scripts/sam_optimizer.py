from __future__ import annotations

import numpy as np


class SGDOptimizer:
    def __init__(self, learning_rate: float, momentum: float = 0.0) -> None:
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.velocity: dict[str, np.ndarray] = {}

    def step(self, params: dict[str, np.ndarray], grads: dict[str, np.ndarray]) -> None:
        if not self.velocity:
            self.velocity = {
                name: np.zeros_like(value) for name, value in params.items()
            }

        for name, grad in grads.items():
            if self.momentum > 0.0:
                self.velocity[name] = self.momentum * self.velocity[name] - self.learning_rate * grad
                params[name] += self.velocity[name]
            else:
                params[name] -= self.learning_rate * grad


class SAMOptimizer:
    def __init__(
        self,
        params: dict[str, np.ndarray],
        base_optimizer: SGDOptimizer,
        rho: float = 0.05,
        eps: float = 1e-12,
    ) -> None:
        self.base_optimizer = base_optimizer
        self.rho = rho
        self.eps = eps
        self.perturbations = {
            name: np.zeros_like(value) for name, value in params.items()
        }

    def first_step(
        self, params: dict[str, np.ndarray], grads: dict[str, np.ndarray]
    ) -> None:
        grad_norm_sq = 0.0
        for grad in grads.values():
            grad_norm_sq += float(np.sum(grad * grad))

        grad_norm = np.sqrt(grad_norm_sq)
        scale = self.rho / (grad_norm + self.eps)

        for name, grad in grads.items():
            perturbation = grad * scale
            params[name] += perturbation
            self.perturbations[name] = perturbation

    def second_step(
        self, params: dict[str, np.ndarray], grads: dict[str, np.ndarray]
    ) -> None:
        for name, perturbation in self.perturbations.items():
            params[name] -= perturbation

        self.base_optimizer.step(params, grads)

        for perturbation in self.perturbations.values():
            perturbation.fill(0.0)
