import numpy as np


# different ways to generate indices for changes
def random_idcs(n):  # random indices uniformly
    first = np.random.randint(0, n)
    while True:
        second = np.random.randint(0, n)
        if second != first:
            break
    return first, second


def close_random_indices(n):
    # dfirst index is uniformly sampled from 0..n
    first = np.random.randint(0, n)
    # i then sample from binomial distribution with p = first/n
    p = float(first) / n
    hits = np.random.binomial(n, p)
    return first, int(hits)


# custom change operations
def switch_cities(order, idx_from, idx_to):
    order = order.copy()
    order[idx_from], order[idx_to] = order[idx_to], order[idx_from]
    return order


def move_cities(order, idx_from, idx_to):
    new_order = []
    num_moving = order[idx_from]  # which city do I move
    for i, city in enumerate(order):
        if i == idx_from:
            continue
        if i == idx_to:
            new_order.append(num_moving)
        new_order.append(city)
    return np.array(new_order)


def invert_subseq(order, idx_from, idx_to):
    before, middle, after = order[:idx_from], order[idx_from:idx_to], order[idx_to:]
    return np.concatenate((before, middle[::-1], after))
