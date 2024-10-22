import numpy as np
from perturbations.order_changes import (
    close_random_indices,
    invert_subseq,
    move_cities,
    random_idcs,
    switch_cities,
)
from representation import order_is_valid


# generic perturbation generating indices and then applying the custom op. on prev_order
def perturb(idx_gen, op, prev_order):
    # move a city to i different part in the order, at random
    n = len(prev_order)
    idx_from, idx_to = idx_gen(n)
    idx_from, idx_to = min(idx_from, idx_to), max(idx_from, idx_to)
    order = op(prev_order, idx_from, idx_to)  # new np.array
    assert order_is_valid(
        order
    ), f"New order is invalid: idcs: {idx_from} to {idx_to},\n{prev_order},\n{order}"
    return order  # np.array from list


def perturb_move(x):
    # move a city to i different part in the order, at random
    return perturb(random_idcs, move_cities, x)


def perturb_switch(x):
    # move a city to i different part in the order, at random
    return perturb(random_idcs, switch_cities, x)


def perturb_invert(x):
    # move a city to i different part in the order, at random
    return perturb(random_idcs, invert_subseq, x)


def all_perturbations(op, prev_order):
    n = len(prev_order)
    orders = [op(prev_order, i, j) for i in range(n) for j in range(n) if i != j]

    for i, o in enumerate(orders):
        assert order_is_valid(o), f"Order {i} is invalid"
    return orders


def parallel_move(x):
    return all_perturbations(move_cities, x)


def parallel_switch(x):
    return all_perturbations(switch_cities, x)


def parallel_invert(x):
    return all_perturbations(invert_subseq, x)
