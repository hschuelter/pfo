# python3 -m venv env
# source env/bin/activate
# pip3 install -r requirements.txt
# python3 main.py
# pip3 freeze > requirements.txt
# deactivate
# --------------------------------------------------------
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from geneticalgorithm2 import geneticalgorithm2 as ga

from pfo_base import PFOBase
from pfo_simulated_annealing import HP3DSimulatedAnnealing

# --------------------------------------------------------


def optimize_ga(sequence: str) -> PFOBase:
    print("Optimizing with Genetic Algorithm...")
    hp_model = PFOBase(sequence, "Genetic Algorithm")
    hp_model.print_header()
    ga_model = instantiate_ga(hp_model)

    ga_model.run(function=hp_model.fitness_function, no_plot=True)
    hp_model.energy_history = ga_model.report

    if hp_model.best_conformation is not None:
        hp_model.visualize_best("Best 3D HP Conformation (GA)")

    hp_model.model = ga_model
    hp_model.get_results_summary("GENETIC ALGORITHM")

    return hp_model


def instantiate_ga(hp_model: PFOBase):
    default_params = {
        "max_num_iteration": 1000,
        "population_size": 100,
        "mutation_probability": 0.1,
        "elit_ratio": 0.01,
        "parents_portion": 0.3,
        "crossover_type": "uniform",
        "mutation_type": "uniform_by_center",
        "selection_type": "roulette",
        "max_iteration_without_improv": None,
    }

    return ga(
        dimension=hp_model.length - 1,
        variable_type="int",
        variable_boundaries=hp_model.var_bound,
        algorithm_parameters=default_params,
    )


def optimize_sa(sequence):
    """Optimize using Simulated Annealing"""
    print("Optimizing with Simulated Annealing...")
    hp_model = PFOBase(sequence, "Simulated Annealing")
    hp_model.print_header()

    sa_solver = HP3DSimulatedAnnealing(hp_model)

    # Configure annealing schedule
    sa_solver.Tmax = 100.0  # Maximum temperature
    sa_solver.Tmin = 0.01  # Minimum temperature
    sa_solver.steps = 10000  # Reduced for faster comparison
    sa_solver.updates = 500  # Update frequency

    best_state, best_energy = sa_solver.anneal()

    sa_solver.get_results_summary("SIMULATED ANNEALING")

    hp_model.best_energy = sa_solver.best_energy_value
    hp_model.energy_history = sa_solver.energy_history

    if sa_solver.best_conformation is not None:
        hp_model.visualize_best("Best 3D HP Conformation (SA)")
        pass

    return hp_model


def compare_algorithms(sequence):
    """Compare GA and SA performance"""
    print(f"\n{'='*80}")
    print(f"COMPARING ALGORITHMS FOR SEQUENCE: {sequence}")
    print(f"{'='*80}")

    # Run GA
    print("\n" + "-" * 40)
    ga_result = optimize_ga(sequence)
    ga_energy = ga_result.best_energy
    ga_evaluations = ga_result.evaluation_count

    # Reset evaluation counter for fair comparison
    print("\n" + "-" * 40)
    sa_result = optimize_sa(sequence)
    sa_energy = sa_result.best_energy
    sa_evaluations = sa_result.evaluation_count
    sa_iterations, sa_energies = len(sa_result.energy_history), sa_result.energy_history

    # Comparison summary
    print(f"\n{'='*60}")
    print("ALGORITHM COMPARISON SUMMARY")
    print(f"{'='*60}")
    print(f"Sequence: {sequence}")
    print(f"Length: {len(sequence)}")
    print()
    print(
        f"{'Algorithm':<15} {'Best Energy':<12} {'H-H Contacts':<12} {'Evaluations':<12}"
    )
    print("-" * 60)
    print(f"{'GA':<15} {ga_energy:<12.2f} {-int(ga_energy):<12} {ga_evaluations:<12}")
    print(f"{'SA':<15} {sa_energy:<12.2f} {-int(sa_energy):<12} {sa_evaluations:<12}")
    print()

    if sa_energy < ga_energy:
        print("🏆 Simulated Annealing found better solution!")
    elif ga_energy < sa_energy:
        print("🏆 Genetic Algorithm found better solution!")
    else:
        print("🤝 Both algorithms found equally good solutions!")

    print(f"{'='*60}")

    return [ga_result, sa_result]


def main():
    sequences = [
        "HPHPPHHPHPPHPHHPPHPH",
        # "HHPPHPPHPPHPPHPPHPPHPPHH",
        # "PPPHHPPHHPPPPPHHHHHHHPPHHPPPPHHPPHPP"
    ]

    for sequence in sequences:
        # Choose optimization method
        method = "compare"  # Options: "ga", "sa", "compare"
        models = []
        labels = []

        if method == "ga":
            ga_model = optimize_ga(sequence)
            models.append(ga_model.energy_history)
            labels.append(ga_model.label)
        elif method == "sa":
            sa_model = optimize_sa(sequence)
            models.append(sa_model.energy_history)
            labels.append(sa_model.label)
        elif method == "compare":
            compare_models = compare_algorithms(sequence)
            models = [m.energy_history for m in compare_models]
            labels = [m.label for m in compare_models]

        plot_energy_history(models, labels, sequence)

        # Future algorithms can be added here:
        # elif method == "pso":
        #     optimize_pso(sequence)
        # elif method == "de":
        #     optimize_differential_evolution(sequence)


def plot_energy_history(history, labels, sequence):
    print("history length:", len(history))
    print("labels length:", len(labels))
    print(labels)
    for h, l in zip(history, labels):
        plt.plot(h, label=l)

    plt.xlabel("Generation")
    plt.ylabel("Energy")
    plt.title(f"Convergence for {sequence}")
    plt.legend(loc="best")
    plt.show()


if __name__ == "__main__":
    main()
