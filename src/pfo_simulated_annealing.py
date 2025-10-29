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

    def __init__(
        self, hp_model: PFOBase, initial_moves: Optional[np.ndarray] = None
    ):
        self.hp_model = hp_model
        self.best_energy_value = float("inf")
        self.best_conformation = None
        self.get_results_summary = hp_model.get_results_summary

        # Energy tracking for convergence plots
        self.energy_history: list[float] = []
        self.label = ""
        self.iteration_count = 0

        # Generate initial valid moves if not provided
        if initial_moves is None:
            initial_moves = hp_model.generate_random_valid_moves()
            if initial_moves is None:
                raise ValueError("Could not generate initial valid conformation")

        self.state = initial_moves
        super(HP3DSimulatedAnnealing, self).__init__(self.state)

    def energy(self) -> float:
        self.hp_model.evaluation_count += 1
        self.iteration_count += 1

        conformation = self.hp_model.moves_to_conformation(self.state)
        if conformation is None:
            energy_val = 1000
        else:
            energy_val = self.hp_model.calculate_energy(conformation)

        # Update best solution
        if energy_val < self.best_energy_value:
            self.best_energy_value = energy_val
            if conformation is not None:
                self.best_conformation = conformation.copy()
                if energy_val < self.hp_model.best_energy:
                    self.hp_model.best_energy = energy_val
                    self.hp_model.best_conformation = conformation.copy()

        self.energy_history.append(self.best_energy_value)

        return energy_val

    def move(self):
        """Make a random move to neighboring state (required by simanneal)"""
        new_moves = self.hp_model.perturb_conformation(
            self.state, perturbation_strength=0.1, max_attempts=50
        )

        if new_moves is not None:
            self.state = new_moves
        else:
            # Fallback: single random change
            pos = np.random.randint(0, len(self.state))
            self.state[pos] = np.random.randint(0, 6)

    def copy_state(self, state):
        """Return copy of state (required by simanneal)"""
        return state.copy()

    def get_sa_convergence_data(self):
        """Get convergence data for SA"""
        iterations = list(range(len(self.energy_history)))
        return iterations, self.energy_history
