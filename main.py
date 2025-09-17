# python3 -m venv env
# source env/bin/activate
# pip3 install -r requirements.txt
# python3 main.py
# pip3 freeze > requirements.txt
# deactivate
# --------------------------------------------------------
from geneticalgorithm2 import geneticalgorithm2 as ga
from simanneal import Annealer as sa


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pfo_lattice_model import HP3DLatticeModel
# --------------------------------------------------------

def optimize_ga(sequence):
    print("Optimizing with Genetic Algorithm...")
    hp_model = HP3DLatticeModel(sequence)

    # Default GA parameters
    default_params = {
        'max_num_iteration': 10000,
        'population_size': 100,
        'mutation_probability': 0.1,
        'elit_ratio': 0.01,
        # 'crossover_probability': 0.5,
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
    
    ga_model.run(function=hp_model.fitness_function)
    print(ga_model)
    hp_model.get_results_summary(ga_model)
    visualize_best(hp_model.best_conformation, hp_model)


def optimize_sa(sequence):
    print("Optimizing with Simulated Annealing...")
    hp_model = HP3DLatticeModel(sequence)
    return ''

def main():
    sequences = [
        "HPHPPHHPHPPHPHHPPHPH", 
        # "HHPPHPPHPPHPPHPPHPPHPPHH", 
        # "PPPHHPPHHPPPPPHHHHHHHPPHHPPPPHHPPHPP"
        ]
    for sequence in sequences:
        optimize_sa(sequence)
        # optimize_ga(sequence)


def visualize_best(best_conformation, hp_model, title: str = "Best 3D HP Conformation (GA)"):
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

if __name__ == "__main__":
    main()
