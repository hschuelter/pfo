from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest

from src.pfo_lattice_model import HP3DLatticeModel
from src.pfo_simulated_annealing import HP3DSimulatedAnnealing


@pytest.fixture
def simple_model():
    return HP3DLatticeModel("HPHPH")


class TestParametrized:
    @pytest.mark.parametrize(
        "sequence", ["HP", "HPH", "HPHP", "HPHPH", "HHHHHH", "PPPPPP", "HPHPHPHPHP"]
    )
    def test_various_sequences(self, sequence):
        model = HP3DLatticeModel(sequence)
        assert model.sequence == sequence
        assert model.length == len(sequence)

    @pytest.mark.parametrize("sequence", ["HP", "HPH", "HPHPH"])
    def test_random_generation_various_lengths(self, sequence):
        model = HP3DLatticeModel(sequence)
        moves = model.generate_random_valid_moves()

        assert moves is not None
        assert len(moves) == len(sequence) - 1

    @pytest.mark.parametrize(
        "direction_idx,expected",
        [
            (0, [1, 0, 0]),
            (1, [-1, 0, 0]),
            (2, [0, 1, 0]),
            (3, [0, -1, 0]),
            (4, [0, 0, 1]),
            (5, [0, 0, -1]),
        ],
    )
    def test_direction_vectors(self, simple_model, direction_idx, expected):
        np.testing.assert_array_equal(simple_model.directions[direction_idx], expected)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
