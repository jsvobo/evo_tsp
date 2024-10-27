import numpy as np


def order_is_valid(order):
    taken = np.zeros(len(order), dtype=bool)
    for i in order:
        if taken[i]:
            return False  # already been through, visiting again
        taken[i] = True  # mark as visited
    return taken.all()


def random_distances(num_cities):
    # random distance, only care about upper triangular and diag, then mirror
    matrix = np.random.randint(1, 100, (num_cities, num_cities))
    matrix = np.triu(matrix) + np.triu(matrix).T
    for i in range(num_cities):
        matrix[i, i] = 0
    return matrix


# more intelligent init, not random TODO
def better_init_wrapper(size, dist_matrix):
    def better_init(dist, num_cities):
        rng = np.arange(num_cities)
        fitness_fn = fitness_wrapper(dist)

        best = None
        best_fitness = float("inf")
        for i in range(50):
            order = np.random.permutation(num_cities)
            fitness = fitness_fn(order)
            if fitness < best_fitness:
                best = order
                best_fitness = fitness
        assert order_is_valid(order), "invalid order created by the better init"
        return best

    return lambda: better_init(dist_matrix, size)


# random order of cities initialisation and wrapper
def random_init_wrapper(size):
    def random_init(num_cities):
        rng = np.arange(num_cities)
        order = np.random.permutation(num_cities)
        assert order_is_valid(order), "invalid order created by random init??"
        return order

    return lambda: random_init(size)


def constructive_heuristics_init_wrapper(size, dist_matrix):
    def constructive_init(num_cities):
        idx = np.random.randint(num_cities)
        taken = np.zeros(num_cities, dtype=bool)
        taken[idx] = True
        order = [idx]
        while len(order) < num_cities:
            indices = np.argsort(dist_matrix[idx])
            for i in indices:
                if not taken[i]:
                    order.append(i)
                    taken[i] = True
                    idx = i
                    break
        return np.array(order)

    return lambda: constructive_init(size)


# fitness fn wrapper
def fitness_wrapper(distances):
    def fitness(distances, order):
        # from the start, sum the distances from one to the next
        sum_distances = 0
        prev = None
        for i, city_idx in enumerate(order):
            if i == 0:
                prev = city_idx
                continue

            current = city_idx
            dist = distances[prev, current]
            sum_distances += dist
            prev = current
        sum_distances += distances[prev, order[0]]  # back to first city
        return sum_distances

    return lambda x: fitness(distances, order=x)
