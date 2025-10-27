import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock

from src.pfo_lattice_model import HP3DLatticeModel
from src.pfo_simulated_annealing import HP3DSimulatedAnnealing


@pytest.fixture
def simple_model():
    return HP3DLatticeModel("HPHPH")

class TestInitialization:
    def test_initialization(self, simple_model):
        assert simple_model.sequence == "HPHPH"
        assert simple_model.length == 5
        assert len(simple_model.var_bound) == 4
        assert simple_model.evaluation_count == 0
        assert simple_model.best_energy == float('inf')

    def test_invalid_sequence_with_another_character(self):
        with pytest.raises(ValueError, match="only 'H' and 'P'"):
            HP3DLatticeModel("HPXPH")

    def test_lowercase_sequence(self):
        model = HP3DLatticeModel("hphph")
        assert model.sequence == "HPHPH"

    def test_mixed_case_sequence(self):
        model = HP3DLatticeModel("HpHpH")
        assert model.sequence == "HPHPH"

    def test_directions_shape(self, simple_model):
        assert simple_model.directions.shape == (6, 3)

    def test_directions_are_unit_vectors(self, simple_model):
        for direction in simple_model.directions:
            assert np.linalg.norm(direction) == 1.0

    def test_variable_bounds(self, simple_model):
        for bound in simple_model.var_bound:
            assert bound == (0, 5)

    def test_variable_bounds_length(self, simple_model):
        assert len(simple_model.var_bound) == simple_model.length - 1

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])