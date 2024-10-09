import numpy as np
from perturbations import *
from representation import *


def test_representation(size=10):
    # generate random problem:
    distances = random_distances(size)
    # instantiate fitness fn and random order generator
    random_init = random_init_wrapper(size)
    order = random_init()
    fitness = fitness_wrapper(distances)

    print("generated distance matrix\n", distances)
    print("generated cities: ", order)
    print("fitness fn: ", fitness(order))

    # test other stuff, like validity check fn
    test_assertions(order)


def test_assertions(order):
    valid = order
    print("\n", valid)
    print("generated, should be True (valid order): ", order_is_valid(valid))

    invalid = order.tolist()
    invalid.append(0)
    print(invalid)
    print("handpicked, should be False (invalid order): ", order_is_valid(invalid))


def test_perturbation_fns():
    a = np.arange(10)
    b = perturb_switch(a)
    c = perturb_move(a)
    d = perturb_invert(a)
    print("base order: ", a)
    print("switched: ", b)
    print("moved: ", c)
    print("reversed: ", d)


if __name__ == "__main__":
    test_representation()
    test_perturbation_fns()
    # not testing local search, that is a bit more complicated
