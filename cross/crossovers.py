import numpy as np
from representation import order_is_valid


def crossover_scx(distance_matrix, parent_a, parent_b):
    # sequential constructive crossover
    searched_city = parent_a[0]
    new_order = [searched_city]
    visited = np.zeros(len(parent_a))
    visited[searched_city] = 1  # mark as visited

    while not (visited == 1).all():
        city_a = -1
        city_b = -1

        idx_a = np.where(parent_a == searched_city)[0]
        idx_b = np.where(parent_b == searched_city)[0]

        for next_val in np.roll(parent_a, -idx_a):  # iterate to the end from the idx_a
            if not visited[next_val]:
                city_a = next_val
                break

        for next_val in np.roll(
            parent_a, -idx_b
        ):  # iterate through the parent_b from the index where searched_city is located around
            if not visited[next_val]:
                city_b = next_val
                break

        if city_a == city_b:
            next_city = city_a
        else:
            w_a = distance_matrix[searched_city][city_a]
            w_b = distance_matrix[searched_city][city_b]

            if w_a < w_b:
                next_city = city_a
            else:
                next_city = city_b

        new_order.append(next_city)
        visited[next_city] = 1  # cannot go here again
        searched_city = next_city

    assert order_is_valid(new_order), "invalid order created by SCX crossover"
    return np.array(new_order)


def crossover_pmx(distance_matrix, parent_a, parent_b):
    # partial mapped crossover
    n = len(parent_a)
    idx_a, idx_b = np.random.choice(n, 2, replace=False)
    idx_a, idx_b = min(idx_a, idx_b), max(idx_a, idx_b)

    child = parent_a.copy()
    for i in range(idx_a, idx_b + 1):
        to_insert = parent_b[i]
        was_before = child[i]
        idx_instead = np.where(child == to_insert)[0][0]
        child[i] = to_insert
        child[idx_instead] = was_before

    assert order_is_valid(child), "invalid order created by PMX crossover"
    return child


if __name__ == "__main__":
    s = np.array([8, 1, 3, 2, 7, 5, 4, 6, 0, 9])
    t = np.array([9, 3, 7, 8, 2, 6, 5, 1, 0, 4])
    print("s: ", s)
    print("t: ", t)

    d = None  # for compatibility
    print(crossover_pmx(d, s, t))
