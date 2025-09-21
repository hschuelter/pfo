# python3 -m venv env
# source env/bin/activate
# pip3 install -r requirements.txt
# python3 main.py
# pip3 freeze > requirements.txt
# deactivate
# --------------------------------------------------------
from geneticalgorithm2 import geneticalgorithm2 as ga

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pfo_lattice_model import HP3DLatticeModel
from pfo_simulated_annealing import HP3DSimulatedAnnealing
# --------------------------------------------------------

def optimize_ga(sequence):
    """Optimize using Genetic Algorithm"""
    print("Optimizing with Genetic Algorithm...")
    hp_model = HP3DLatticeModel(sequence)
    hp_model.label = 'Genetic Algorithm'

    # Default GA parameters
    default_params = {
        'max_num_iteration': 1000,  # Reduced for faster comparison
        'population_size': 100,
        'mutation_probability': 0.1,
        'elit_ratio': 0.01,
        'parents_portion': 0.3,
        'crossover_type': 'uniform',
        'mutation_type': 'uniform_by_center',
        'selection_type': 'roulette',
        'max_iteration_without_improv': None
    }

    # Create GA instance
    ga_model = ga(
        dimension=hp_model.length - 1,
        variable_type='int',
        variable_boundaries=hp_model.var_bound,
        algorithm_parameters=default_params
    )

    # Run optimization
    print(f"Starting GA optimization for sequence: {hp_model.sequence}")
    print(f"Sequence length: {hp_model.length}")
    print(f"Search space size: 6^{hp_model.length-1} = {6**(hp_model.length-1):.2e}")
    
    ga_model.run(function=hp_model.fitness_function, no_plot=True)
    hp_model.get_results_summary(ga_model)
    hp_model.energy_history = ga_model.report

    if hp_model.best_conformation is not None:
        # visualize_best(hp_model.best_conformation, hp_model, "Best 3D HP Conformation (GA)")
        pass
    
    # Store convergence data in the model for comparison
    hp_model.ga_model = ga_model
    
    return hp_model


def optimize_sa(sequence):
    """Optimize using Simulated Annealing"""
    print("Optimizing with Simulated Annealing...")
    hp_model = HP3DLatticeModel(sequence)
    hp_model.label = 'Simulated Annealing'
    
    # Create SA extension
    sa_solver = HP3DSimulatedAnnealing(hp_model)
    
    # Configure annealing schedule
    sa_solver.Tmax = 100.0       # Maximum temperature
    sa_solver.Tmin = 0.01        # Minimum temperature  
    sa_solver.steps = 10000      # Reduced for faster comparison
    sa_solver.updates = 500      # Update frequency
    
    print(f"Starting SA optimization for sequence: {hp_model.sequence}")
    print(f"Sequence length: {hp_model.length}")
    print(f"Search space size: 6^{hp_model.length-1} = {6**(hp_model.length-1):.2e}")
    
    # Run optimization
    best_state, best_energy = sa_solver.anneal()
    
    # Print results
    sa_solver.get_results_summary()

    hp_model.best_energy = sa_solver.best_energy_value
    hp_model.energy_history = sa_solver.energy_history
    
    # if sa_solver.best_conformation is not None:
    #     visualize_best(sa_solver.best_conformation, hp_model, "Best 3D HP Conformation (SA)")
    
    return hp_model

def compare_algorithms(sequence):
    """Compare GA and SA performance"""
    print(f"\n{'='*80}")
    print(f"COMPARING ALGORITHMS FOR SEQUENCE: {sequence}")
    print(f"{'='*80}")
    
    # Run GA
    print("\n" + "-"*40)
    ga_result = optimize_ga(sequence)
    ga_energy = ga_result.best_energy
    ga_evaluations = ga_result.evaluation_count
    ga_iterations, ga_energies = ga_result.get_ga_convergence_data(None)  # Will be handled in optimize_ga
    
    # Reset evaluation counter for fair comparison
    print("\n" + "-"*40)
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
    print(f"{'Algorithm':<15} {'Best Energy':<12} {'H-H Contacts':<12} {'Evaluations':<12}")
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
    
    # Plot convergence comparison
    # plot_convergence_comparison(ga_iterations, ga_energies, sa_iterations, sa_energies, sequence)
    
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

def visualize_best(best_conformation, hp_model, title: str = "Best 3D HP Conformation"):
    """Visualize the best conformation found"""
    if best_conformation is None:
        print("No valid conformation found yet. Run optimization first.")
        return
    
    energy = hp_model.calculate_energy(best_conformation)
    hh_contacts = -int(energy)
    compactness = hp_model.get_compactness(best_conformation)
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot backbone
    ax.plot(best_conformation[:, 0], 
            best_conformation[:, 1], 
            best_conformation[:, 2], 
            'k-', linewidth=3, alpha=0.7, label='Backbone')
    
    # Plot residues with colors
    for i, (x, y, z) in enumerate(best_conformation):
        color = 'red' if hp_model.sequence[i] == 'H' else 'blue'
        ax.scatter(x, y, z, c=color, s=300, alpha=0.8, 
                    edgecolors='black', linewidth=2)
        ax.text(x+0.1, y+0.1, z+0.1, f'{i}', fontsize=8)
    
    # Highlight H-H contacts
    contact_pairs = []
    for i in range(hp_model.length):
        for j in range(i + 2, hp_model.length):
            if hp_model.are_neighbors(best_conformation[i], best_conformation[j]):
                if hp_model.sequence[i] == 'H' and hp_model.sequence[j] == 'H':
                    ax.plot([best_conformation[i, 0], best_conformation[j, 0]], 
                            [best_conformation[i, 1], best_conformation[j, 1]],
                            [best_conformation[i, 2], best_conformation[j, 2]], 
                            'g--', linewidth=3, alpha=0.7)
                    contact_pairs.append((i, j))
    
    ax.set_title(f"{title}\nSequence: {hp_model.sequence}\n"
                f"Energy: {energy:.1f} | H-H Contacts: {hh_contacts} | "
                f"Compactness: {compactness:.2f}")
    ax.set_xlabel('X')
    ax.set_ylabel('Y') 
    ax.set_zlabel('Z')
    
    # Set equal aspect ratio
    coords = best_conformation
    max_range = np.array([coords[:, 0].max() - coords[:, 0].min(),
                            coords[:, 1].max() - coords[:, 1].min(),
                            coords[:, 2].max() - coords[:, 2].min()]).max() / 2.0
    mid_x = (coords[:, 0].max() + coords[:, 0].min()) * 0.5
    mid_y = (coords[:, 1].max() + coords[:, 1].min()) * 0.5
    mid_z = (coords[:, 2].max() + coords[:, 2].min()) * 0.5
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    # Legend
    h_scatter = ax.scatter([], [], [], c='red', s=100, label='H (Hydrophobic)')
    p_scatter = ax.scatter([], [], [], c='blue', s=100, label='P (Polar)')
    if hh_contacts > 0:
        contact_line = plt.Line2D([0], [0], color='green', linestyle='--', 
                                    linewidth=3, label='H-H Contact')
        ax.legend(handles=[h_scatter, p_scatter, contact_line])
    else:
        ax.legend(handles=[h_scatter, p_scatter])
    
    plt.tight_layout()
    plt.show()
    
    print(f"H-H contact pairs: {contact_pairs}")
    print(f"Total evaluations: {hp_model.evaluation_count}")

def plot_energy_history(history, labels, sequence):
    print('history length:', len(history))
    print('labels length:', len(labels))
    print(labels)
    for h, l in zip(history, labels):
        plt.plot(h, label=l)

    plt.xlabel("Generation")
    plt.ylabel("Best Energy")
    plt.title(f"Convergence for {sequence}")
    plt.legend(loc="best")
    plt.show()

if __name__ == "__main__":
    main()