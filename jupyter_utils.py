import matplotlib.pyplot as plt
import numpy as np

perturb_name_dict = {
    "perturb_move_wrapper": "Move",
    "perturb_switch_wrapper": "Switch",
    "perturb_invert_wrapper": "Reverse",
}
ls_name_dict = {
    "ls_first_improvement": "First-Improve",
    "ls_best_improvement": "Best-Improve",
}
init_name_dict = {
    "random_init": "Random",
    "better_init": "Better",
    "constructive_init": "Constructive",
}


def print_results(results):
    best_fitness = results["best_fitness"]
    best_order = results["best_order"]
    iteration_history = results["history"]

    print("Best solution found:")
    print(best_order)
    print("With fitness: ", best_fitness)
    print("Total steps taken: ", len(iteration_history))


def simple_graph(array, xlabel, ylabel, title):
    plt.plot(range(len(array)), array)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()


def weights_to_matrix(weights):
    n = len(weights) + 1
    matrix = np.zeros((n, n))
    for i in range(n - 1):
        matrix[i, i + 1 :] = weights[i]
    mirror = np.triu(matrix) + np.triu(matrix, 1).T

    assert is_mirror(mirror), "Matrix is not symmetric"
    assert (mirror.diagonal() == 0).all, "Diagonal is not 0"

    return mirror


def is_mirror(matrix):
    return np.allclose(matrix, matrix.T)
