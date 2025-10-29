# --------------------------------------------------------
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from simanneal import Annealer as sa

from pfo_base import PFOBase

# --------------------------------------------------------


class HP3DSimulatedAnnealing(sa):
    """Simulated Annealing extension for HP3D protein folding"""

    def __init__(self, pfo_base: PFOBase, initial_moves: Optional[np.ndarray] = None):
        self.pfo_base = pfo_base

        # Energy tracking for convergence plots
        # [TO DO] Remove this and use only on PFOBase
        self.energy_history: list[float] = []
        self.iteration_count = 0
        self.best_energy_value = float("inf")
        self.best_conformation = None
        self.get_results_summary = pfo_base.get_results_summary

        # Generate initial valid moves if not provided
        if initial_moves is None:
            initial_moves = self.generate_random_valid_moves()
            if initial_moves is None:
                raise ValueError("Could not generate initial valid conformation")

        self.state = initial_moves
        super(HP3DSimulatedAnnealing, self).__init__(self.state)

    def energy(self) -> float:
        self.pfo_base.evaluation_count += 1
        self.iteration_count += 1

        conformation = self.pfo_base.moves_to_conformation(self.state)
        if conformation is None:
            energy_val = 1000
        else:
            energy_val = self.pfo_base.calculate_energy(conformation)

        # Update best solution
        if energy_val < self.best_energy_value:
            self.best_energy_value = energy_val
            if conformation is not None:
                self.best_conformation = conformation.copy()
                if energy_val < self.pfo_base.best_energy:
                    self.pfo_base.best_energy = energy_val
                    self.pfo_base.best_conformation = conformation.copy()

        # [TO DO] update energy history method + tests
        self.energy_history.append(self.best_energy_value)

        return energy_val

    def move(self):
        """Make a random move to neighboring state (required by simanneal)"""
        moves = self.state
        new_moves = self.perturb_conformation(
            moves, perturbation_strength=0.1, max_attempts=50
        )

        if new_moves is not None:
            self.state = new_moves
        else:
            # Fallback: single random change
            pos = np.random.randint(0, len(self.state))
            self.state[pos] = np.random.randint(0, 6)

    # SIMULATED ANNEALING SUPPORT
    # [TO DO] Tests: assert is not None
    def generate_random_valid_moves(
        self, max_attempts: int = 1000
    ) -> Optional[np.ndarray]:
        """Generate a random valid move sequence"""
        for attempt in range(max_attempts):
            moves = np.random.randint(0, 6, size=self.pfo_base.length - 1)
            conformation = self.pfo_base.moves_to_conformation(moves)
            if conformation is not None:
                return moves
        return None

    def perturb_conformation(
        self,
        moves: np.ndarray,
        perturbation_strength: float = 0.1,
        max_attempts: int = 50,
    ) -> Optional[np.ndarray]:
        """Perturb a conformation for simulated annealing"""
        if moves is None or len(moves) != self.pfo_base.length - 1:
            return None

        for attempt in range(max_attempts):
            new_moves = moves.copy()
            num_to_change = max(1, int(perturbation_strength * len(moves)))
            positions = np.random.choice(len(moves), size=num_to_change, replace=False)

            for pos in positions:
                new_moves[pos] = np.random.randint(0, 6)

            conformation = self.pfo_base.moves_to_conformation(new_moves)
            if conformation is not None:
                return new_moves

        return self.generate_random_valid_moves()

    def copy_state(self, state):
        """Return copy of state (required by simanneal)"""
        return state.copy()

    def get_sa_convergence_data(self):
        """Get convergence data for SA"""
        iterations = list(range(len(self.energy_history)))
        return iterations, self.energy_history
