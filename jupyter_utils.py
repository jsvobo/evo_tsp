import matplotlib.pyplot as plt
import numpy as np


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


def blocky_weights_to_sym_matrix(weights, n):
    weights = np.array(weights).flatten()
    matrix = np.zeros((n, n))
    start_idx = 0
    for i in range(n - 1):
        len_row = n - i - 1
        matrix[i, i + 1 :] = weights[start_idx : start_idx + len_row]
        start_idx += len_row
    mirror = np.triu(matrix) + np.triu(matrix, 1).T

    assert is_mirror(mirror), "Matrix is not symmetric"
    assert (mirror.diagonal() == 0).all, "Diagonal is not 0"

    return mirror


def blocky_weights_to_asym_matrix(matrix, n):
    weights = []
    for row in matrix:
        weights.extend(row)
    weights = np.array(weights)
    weights = weights.reshape(n, n)

    return weights


def is_mirror(matrix):
    return np.allclose(matrix, matrix.T)


def draw_tour(coords, order, title, generace):
    plt.figure()
    plt.title(title + f" Gen: {generace}")

    plt.plot(coords[:, 0], coords[:, 1], "o")
    for i in range(len(order) - 1):
        plt.plot(
            [coords[order[i], 0], coords[order[i + 1], 0]],
            [coords[order[i], 1], coords[order[i + 1], 1]],
            "k-",
        )
    plt.plot(
        [coords[order[-1], 0], coords[order[0], 0]],
        [coords[order[-1], 1], coords[order[0], 1]],
        "k-",
    )
    plt.show()
